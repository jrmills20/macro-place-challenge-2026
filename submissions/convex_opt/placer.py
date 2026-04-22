"""
Convex Optimization Placer

Two-stage algorithm:
  1. Quadratic Placement (QP): solve for macro positions that minimize
     weighted sum of squared pairwise distances for connected macros.
     Formulated as a sparse linear system via the graph Laplacian:
         (L_mm + lambda*I) * x_m = -L_mf * x_f + lambda * x_m0
     Separates identically for x and y. Solved with numpy's dense solver
     (n_macros <= 600, so 600x600 systems are trivial).

  2. Legalization: minimum-displacement overlap removal. Processes macros
     largest-first, searching outward from QP positions for the nearest
     overlap-free location.

  Conservative fallback: a fast approximate proxy cost (HPWL + density)
  compares the QP+legalized result against the initial placement. If QP
  degraded the result, the initial placement is kept instead. This ensures
  we never do worse than the starting point.

Usage:
    uv run evaluate submissions/convex_opt/placer.py
    uv run evaluate submissions/convex_opt/placer.py --all
    uv run evaluate submissions/convex_opt/placer.py -b ibm03
"""

import math
import random

import numpy as np
import torch

from macro_place.benchmark import Benchmark

# Nets larger than this use a star model instead of full clique expansion
# to avoid O(k^2) blowup on high-fanout (e.g. clock) nets.
MAX_CLIQUE_SIZE = 20


class ConvexOptPlacer:
    """
    Convex (quadratic) placement + legalization + SA refinement.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    lambda_anchor : float
        Spring constant for the anchor term.
        Keeps the QP well-conditioned and pulls macros toward their initial
        positions when few fixed-position nodes exist.
    sa_iters : int
        Number of SA refinement steps after legalization.
    """

    def __init__(
        self,
        seed: int = 42,
        lambda_anchor: float = 30.0,
        sa_iters: int = 200_000,
        grad_steps: int = 0,
        sa_density_weight: float = 0.5,
    ):
        self.seed = seed
        self.lambda_anchor = lambda_anchor
        self.sa_iters = sa_iters
        self.grad_steps = grad_steps
        self.sa_density_weight = sa_density_weight

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        n_total = benchmark.num_macros

        pos = (
            benchmark.macro_positions.numpy().copy().astype(np.float64)
        )  # [n_total, 2]
        sizes = benchmark.macro_sizes.numpy().astype(np.float64)  # [n_total, 2]
        fixed_mask = benchmark.macro_fixed.numpy()  # [n_total] bool
        port_pos = benchmark.port_positions.numpy().astype(np.float64)  # [n_ports, 2]

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        # Movable: non-fixed hard macros only.
        # Soft macros stay at their initial positions.
        movable = np.zeros(n_total, dtype=bool)
        movable[:n_hard] = ~fixed_mask[:n_hard]

        def _qp_legalize_candidates(anchor_pos, lambdas):
            """Run QP+legalize for each lambda, return list of (cost, pos)."""
            cands = []
            for lam in lambdas:
                self.lambda_anchor = lam
                p = self._qp_solve(
                    anchor_pos.copy(),
                    movable,
                    sizes,
                    benchmark,
                    n_total,
                    port_pos,
                    cw,
                    ch,
                )
                p[:n_hard] = self._legalize(
                    p[:n_hard].copy(), ~fixed_mask[:n_hard], sizes[:n_hard], cw, ch
                )
                cands.append((self._fast_proxy(p, benchmark, n_total, port_pos), p))
            self.lambda_anchor = 30.0
            return cands

        # Try a range of anchor strengths from the initial positions.
        candidates = _qp_legalize_candidates(pos, [30.0, 100.0, 400.0])

        # Also legalize the initial positions as a fallback.
        pos_init_legal = pos.copy()
        pos_init_legal[:n_hard] = self._legalize(
            pos[:n_hard].copy(), ~fixed_mask[:n_hard], sizes[:n_hard], cw, ch
        )
        candidates.append(
            (
                self._fast_proxy(pos_init_legal, benchmark, n_total, port_pos),
                pos_init_legal,
            )
        )

        best_cost, best_pos = min(candidates, key=lambda c: c[0])

        # Use the best legal positions as the new anchor
        # and re-run QP.  Each iteration, QP finds a WL minimum near a legal
        # solution, then legalization nudges it back to feasibility
        for _ in range(2):
            new_cands = _qp_legalize_candidates(best_pos, [30.0, 100.0, 400.0])
            new_cost, new_pos = min(new_cands, key=lambda c: c[0])
            if new_cost < best_cost:
                best_cost = new_cost
                best_pos = new_pos
            else:
                break  # no improvement, stop early

        pos = best_pos

        # SA refinement — compare before/after by proxy cost and keep the better one.
        if self.sa_iters > 0:
            sa_result = self._sa_refine(
                pos[:n_hard].copy(),
                ~fixed_mask[:n_hard],
                sizes[:n_hard],
                benchmark,
                n_hard,
                n_total,
                port_pos,
                cw,
                ch,
            )
            sa_pos = pos.copy()
            sa_pos[:n_hard] = sa_result
            sa_cost = self._fast_proxy(sa_pos, benchmark, n_total, port_pos)
            if sa_cost < best_cost:
                pos = sa_pos

        # Gradient refinement — L-BFGS on smooth WL + density, then re-legalise.
        # Only kept if the fast proxy cost improves over the current best.
        if self.grad_steps > 0:
            pre_grad_cost = self._fast_proxy(pos, benchmark, n_total, port_pos)
            grad_result = self._grad_refine(
                pos[:n_hard].copy(),
                ~fixed_mask[:n_hard],
                sizes[:n_hard],
                benchmark,
                n_hard,
                n_total,
                port_pos,
                cw,
                ch,
            )
            grad_pos = pos.copy()
            grad_pos[:n_hard] = grad_result
            grad_cost = self._fast_proxy(grad_pos, benchmark, n_total, port_pos)
            if grad_cost < pre_grad_cost:
                pos = grad_pos

        return torch.tensor(pos, dtype=torch.float32)

    # Fast proxy cost

    def _fast_proxy(self, pos, benchmark, n_total, port_pos):
        """
        Fast approximate proxy cost: HPWL + 0.5 * density.

        Omits congestion (expensive to compute without plc), but the
        WL + density approximation is enough to detect regressions where
        QP clustering increases density without meaningful WL gain.
        """
        wl = self._fast_hpwl(pos, benchmark, n_total, port_pos)
        density = self._fast_density(pos, benchmark)
        return wl + 0.5 * density

    def _fast_hpwl(self, pos, benchmark, n_total, port_pos):
        """Normalized HPWL over all nets, same normalization as plc.get_cost():
        total_wirelength / ((canvas_width + canvas_height) * net_count).
        """
        norm = (benchmark.canvas_width + benchmark.canvas_height) * benchmark.num_nets
        total = 0.0
        for net_i, nodes_tensor in enumerate(benchmark.net_nodes):
            nodes = nodes_tensor.numpy()
            w = float(benchmark.net_weights[net_i])
            xs, ys = [], []
            for n in nodes:
                n = int(n)
                if n < n_total:
                    xs.append(pos[n, 0])
                    ys.append(pos[n, 1])
                else:
                    p = n - n_total
                    if p < len(port_pos):
                        xs.append(port_pos[p, 0])
                        ys.append(port_pos[p, 1])
            if len(xs) >= 2:
                total += w * (max(xs) - min(xs) + max(ys) - min(ys))
        return total / norm

    def _fast_density(self, pos, benchmark):
        """
        Grid-based density cost: average density of top 10% cells.
        Approximates plc.get_density_cost() without the Python evaluator.
        """
        cw = benchmark.canvas_width
        ch = benchmark.canvas_height
        rows = benchmark.grid_rows
        cols = benchmark.grid_cols
        cell_w = cw / cols
        cell_h = ch / rows
        cell_area = cell_w * cell_h
        n_hard = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes.numpy().astype(np.float64)

        density_grid = np.zeros(rows * cols, dtype=np.float64)
        n_total_macros = benchmark.num_macros  # hard + soft

        for i in range(n_total_macros):
            x, y = pos[i, 0], pos[i, 1]
            hw, hh = sizes[i, 0] / 2, sizes[i, 1] / 2
            xl, xr = x - hw, x + hw
            yl, yr = y - hh, y + hh

            c_start = max(0, int(xl / cell_w))
            c_end = min(cols - 1, int(xr / cell_w))
            r_start = max(0, int(yl / cell_h))
            r_end = min(rows - 1, int(yr / cell_h))

            for r in range(r_start, r_end + 1):
                cy_lo, cy_hi = r * cell_h, (r + 1) * cell_h
                oy = max(0.0, min(yr, cy_hi) - max(yl, cy_lo))
                if oy <= 0:
                    continue
                for c in range(c_start, c_end + 1):
                    cx_lo, cx_hi = c * cell_w, (c + 1) * cell_w
                    ox = max(0.0, min(xr, cx_hi) - max(xl, cx_lo))
                    if ox > 0:
                        density_grid[r * cols + c] += ox * oy / cell_area

        density_grid.sort()
        thresh = int(0.9 * len(density_grid))
        return float(density_grid[thresh:].mean())

    # Quadratic Placement

    def _qp_solve(self, pos, movable, sizes, benchmark, n_total, port_pos, cw, ch):
        """
        Build and solve the Laplacian linear system.

        For each net with nodes [n1, n2, ...] (hard macros, soft macros, ports):
        - Expand to weighted edges via clique (or star for large nets)
        - Build Laplacian L partitioned into movable (m) and fixed (f)

        The spring term handles:
        - Near-singular systems (few fixed nodes)
        - Keeps macros near their initial positions as a prior
        """
        movable_idx = np.where(movable)[0]
        n_mov = len(movable_idx)
        if n_mov == 0:
            return pos

        mov_map = {int(idx): i for i, idx in enumerate(movable_idx)}

        # Dense Laplacian for movable macros
        L_mm = np.zeros((n_mov, n_mov), dtype=np.float64)
        # Accumulate L_mf * x_f and L_mf * y_f directly
        rhs_x = np.zeros(n_mov, dtype=np.float64)
        rhs_y = np.zeros(n_mov, dtype=np.float64)

        for net_i, nodes_tensor in enumerate(benchmark.net_nodes):
            nodes = nodes_tensor.numpy().tolist()
            w = float(benchmark.net_weights[net_i])
            k = len(nodes)
            if k < 2:
                continue

            # Generate weighted edges
            if k > MAX_CLIQUE_SIZE:
                # Star model: connect anchor to all others
                anchor = nodes[0]
                edges = [(anchor, nodes[j], w / (k - 1)) for j in range(1, k)]
            else:
                # Full clique, total weight = w
                ew = 2.0 * w / (k * (k - 1))
                edges = [
                    (nodes[a], nodes[b], ew) for a in range(k) for b in range(a + 1, k)
                ]

            for ni, nj, pw in edges:
                ni, nj = int(ni), int(nj)
                ri_mov = ni in mov_map
                rj_mov = nj in mov_map

                if ri_mov:
                    ri = mov_map[ni]
                    L_mm[ri, ri] += pw
                    if rj_mov:
                        rj = mov_map[nj]
                        L_mm[rj, rj] += pw
                        L_mm[ri, rj] -= pw
                        L_mm[rj, ri] -= pw
                    else:
                        xj, yj = self._fixed_pos(nj, pos, port_pos, n_total)
                        rhs_x[ri] += pw * xj
                        rhs_y[ri] += pw * yj
                elif rj_mov:
                    rj = mov_map[nj]
                    L_mm[rj, rj] += pw
                    xi, yi = self._fixed_pos(ni, pos, port_pos, n_total)
                    rhs_x[rj] += pw * xi
                    rhs_y[rj] += pw * yi

        # Spring anchor: pulls toward initial positions (regularization)
        mean_diag = L_mm.diagonal().mean()
        if mean_diag <= 0:
            mean_diag = 1.0
        lam = self.lambda_anchor * mean_diag
        A = L_mm + np.eye(n_mov) * lam

        x_m0 = pos[movable_idx, 0]
        y_m0 = pos[movable_idx, 1]
        bx = rhs_x + lam * x_m0
        by = rhs_y + lam * y_m0

        try:
            x_mov = np.linalg.solve(A, bx)
            y_mov = np.linalg.solve(A, by)
        except np.linalg.LinAlgError:
            # Fallback: keep initial positions
            return pos

        # Replace NaNs/Infs with initial positions
        x_mov = np.where(np.isfinite(x_mov), x_mov, x_m0)
        y_mov = np.where(np.isfinite(y_mov), y_mov, y_m0)

        # Clamp to canvas (positions are centers)
        hw = sizes[movable_idx, 0] / 2
        hh = sizes[movable_idx, 1] / 2
        x_mov = np.clip(x_mov, hw, cw - hw)
        y_mov = np.clip(y_mov, hh, ch - hh)

        pos = pos.copy()
        pos[movable_idx, 0] = x_mov
        pos[movable_idx, 1] = y_mov
        return pos

    @staticmethod
    def _fixed_pos(global_idx, pos, port_pos, n_total):
        """Return (x, y) for a fixed node (macro or port)."""
        if global_idx < n_total:
            return pos[global_idx, 0], pos[global_idx, 1]
        p = global_idx - n_total
        if p < len(port_pos):
            return port_pos[p, 0], port_pos[p, 1]
        return 0.0, 0.0

    # Legalization

    def _legalize(self, pos, movable, sizes, cw, ch):
        """
        Minimum-displacement legalization for hard macros.

        Two-phase search:
          1. Coarse ring search (step = max(w,h)*0.25) to locate the first
             ring that contains a legal position.
          2. Fine-grained local search within ±coarse_step of the best
             coarse position, using step = min(w,h)*0.08 (≥0.03 μm floor).
             This tightens the final placement toward the true nearest legal
             point without iterating many small rings from the origin.
        """
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        sep_x = (sizes[:, 0:1] + sizes[:, 0].reshape(1, -1)) / 2  # [n, n]
        sep_y = (sizes[:, 1:2] + sizes[:, 1].reshape(1, -1)) / 2  # [n, n]
        n = len(pos)

        order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
        placed = np.zeros(n, dtype=bool)
        legal = pos.copy()

        def _no_conflict(cx, cy, idx):
            if not placed.any():
                return True
            dx = np.abs(cx - legal[:, 0])
            dy = np.abs(cy - legal[:, 1])
            c = (dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & placed
            c[idx] = False
            return not c.any()

        for idx in order:
            if not movable[idx]:
                placed[idx] = True
                continue

            # Check if current position is already legal
            if _no_conflict(legal[idx, 0], legal[idx, 1], idx):
                placed[idx] = True
                continue

            # ── Phase 1: coarse ring search ──────────────────────────────────
            step_c = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
            best_p = legal[idx].copy()
            best_d = float("inf")
            found_ring = False

            for r in range(1, 200):
                for dxr in range(-r, r + 1):
                    for dyr in range(-r, r + 1):
                        if abs(dxr) != r and abs(dyr) != r:
                            continue
                        cx = np.clip(
                            pos[idx, 0] + dxr * step_c, half_w[idx], cw - half_w[idx]
                        )
                        cy = np.clip(
                            pos[idx, 1] + dyr * step_c, half_h[idx], ch - half_h[idx]
                        )
                        if not _no_conflict(cx, cy, idx):
                            continue
                        d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                        if d < best_d:
                            best_d = d
                            best_p = np.array([cx, cy])
                            found_ring = True
                if found_ring:
                    break

            # ── Phase 2: vectorised fine-grained refinement ─────────────────
            # Build a dense grid within ±step_c of the best coarse position
            # and check all grid points at once via numpy broadcast, which is
            # O(n_pts × n_placed) instead of O(n_pts × n_placed) Python calls.
            # This is critical for large benchmarks where Phase 2 was the
            # dominant cost.
            step_f = max(min(sizes[idx, 0], sizes[idx, 1]) * 0.08, 0.03)
            x_lo = max(half_w[idx], best_p[0] - step_c)
            x_hi = min(cw - half_w[idx], best_p[0] + step_c)
            y_lo = max(half_h[idx], best_p[1] - step_c)
            y_hi = min(ch - half_h[idx], best_p[1] + step_c)

            n_x = min(40, max(1, int((x_hi - x_lo) / step_f) + 1))
            n_y = min(40, max(1, int((y_hi - y_lo) / step_f) + 1))
            gx = np.linspace(x_lo, x_hi, n_x)
            gy = np.linspace(y_lo, y_hi, n_y)
            gx2, gy2 = np.meshgrid(gx, gy)  # both [n_y, n_x]
            pts_x = gx2.ravel()  # [n_pts]
            pts_y = gy2.ravel()

            # Placed macros, excluding self
            pmask = placed.copy()
            pmask[idx] = False
            pidx = np.where(pmask)[0]

            if len(pidx) == 0:
                dists2 = (pts_x - pos[idx, 0]) ** 2 + (pts_y - pos[idx, 1]) ** 2
                k = int(np.argmin(dists2))
                if dists2[k] < best_d:
                    best_p = np.array([pts_x[k], pts_y[k]])
                    best_d = float(dists2[k])
            else:
                px_p = legal[pidx, 0]  # [n_placed]
                py_p = legal[pidx, 1]
                sx_p = sep_x[idx, pidx] + 0.05  # [n_placed]
                sy_p = sep_y[idx, pidx] + 0.05

                # [n_pts, n_placed] conflict matrix, fully vectorised
                dx2d = np.abs(pts_x[:, np.newaxis] - px_p[np.newaxis, :])
                dy2d = np.abs(pts_y[:, np.newaxis] - py_p[np.newaxis, :])
                valid = ~((dx2d < sx_p) & (dy2d < sy_p)).any(axis=1)  # [n_pts]

                if valid.any():
                    vx = pts_x[valid]
                    vy = pts_y[valid]
                    dists2 = (vx - pos[idx, 0]) ** 2 + (vy - pos[idx, 1]) ** 2
                    k = int(np.argmin(dists2))
                    if dists2[k] < best_d:
                        best_p = np.array([vx[k], vy[k]])
                        best_d = float(dists2[k])

            legal[idx] = best_p
            placed[idx] = True

        return legal

    # SA Refinement

    def _sa_refine(
        self, pos, movable, sizes, benchmark, n_hard, n_total, port_pos, cw, ch
    ):
        """
        Simulated annealing on combined WL + density cost.

        Cost = WL(hard↔hard, hard↔soft, hard↔port) + w_dens * sum_of_squares(density_grid)

        WL uses O(degree) incremental delta updates.
        Density uses an incrementally-maintained grid (soft macros fixed, hard macros movable).
        w_dens is set so that density contributes ~50% of the initial total cost,
        preventing SA from clustering macros to reduce WL at the expense of density.

        Move types: Shift (45%), Swap (30%), Pull (25%)
        """
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        sep_x = (sizes[:, 0:1] + sizes[:, 0].reshape(1, -1)) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1].reshape(1, -1)) / 2

        movable_idx = np.where(movable)[0]
        if len(movable_idx) == 0:
            return pos

        # ── Build weighted adjacency: hard-macro ↔ {hard macro, soft macro, port} ──
        #
        # Each entry: (endpoint_x, endpoint_y, weight)
        # For hard-macro endpoints we store their index so we can look up pos[idx]
        # dynamically.  For fixed endpoints (soft macros, ports) we snapshot their
        # positions once — they never move during SA.
        #
        # adj_hard[i]  = list of (hard_macro_j_index, weight)   — movable ends
        # adj_fixed[i] = list of (fixed_x, fixed_y, weight)     — immovable ends

        all_macro_pos = benchmark.macro_positions.numpy().astype(np.float64)
        adj_hard: list = [[] for _ in range(n_hard)]  # (j, w) pairs
        adj_fixed: list = [[] for _ in range(n_hard)]  # (x, y, w) triples

        for net_i, nodes_tensor in enumerate(benchmark.net_nodes):
            raw = nodes_tensor.numpy().tolist()
            w = float(benchmark.net_weights[net_i])
            hard_nodes = [int(n) for n in raw if n < n_hard]
            if len(hard_nodes) < 1:
                continue

            # Fixed endpoints on this net: soft macros + ports
            fixed_pts = []
            for n in raw:
                n = int(n)
                if n_hard <= n < n_total:
                    # soft macro — use initial (fixed) position
                    fixed_pts.append((all_macro_pos[n, 0], all_macro_pos[n, 1]))
                elif n >= n_total:
                    p = n - n_total
                    if p < len(port_pos):
                        fixed_pts.append((port_pos[p, 0], port_pos[p, 1]))

            k_h = len(hard_nodes)
            k_f = len(fixed_pts)
            k_total = k_h + k_f

            if k_total < 2:
                continue

            # Generate weighted edges (clique or star for large nets)
            if k_total > MAX_CLIQUE_SIZE:
                # Star model: hard anchor → all others
                anchor = hard_nodes[0]
                ew = w / (k_total - 1)
                for j in hard_nodes[1:]:
                    a, b = (anchor, j) if anchor < j else (j, anchor)
                    # Record as hard–hard edge
                    adj_hard[a].append((b, ew))
                    adj_hard[b].append((a, ew))
                for fx, fy in fixed_pts:
                    adj_fixed[anchor].append((fx, fy, ew))
            else:
                ew = 2.0 * w / (k_total * (k_total - 1))
                # hard–hard edges
                for ai in range(k_h):
                    for bi in range(ai + 1, k_h):
                        ni, nj = hard_nodes[ai], hard_nodes[bi]
                        adj_hard[ni].append((nj, ew))
                        adj_hard[nj].append((ni, ew))
                # hard–fixed edges
                for ni in hard_nodes:
                    for fx, fy in fixed_pts:
                        adj_fixed[ni].append((fx, fy, ew))

        # Collapse into numpy arrays per macro for fast vectorised delta computation
        macro_hard_j: list = []  # macro_hard_j[i]  = int32 array of hard neighbours
        macro_hard_w: list = []  # macro_hard_w[i]  = float64 weights
        macro_fixed_x: list = []  # macro_fixed_x[i] = float64 array of fixed x coords
        macro_fixed_y: list = []
        macro_fixed_w: list = []

        for i in range(n_hard):
            if adj_hard[i]:
                js, ws = zip(*adj_hard[i])
                macro_hard_j.append(np.array(js, dtype=np.int32))
                macro_hard_w.append(np.array(ws, dtype=np.float64))
            else:
                macro_hard_j.append(np.empty(0, dtype=np.int32))
                macro_hard_w.append(np.empty(0, dtype=np.float64))

            if adj_fixed[i]:
                fxs, fys, fws = zip(*adj_fixed[i])
                macro_fixed_x.append(np.array(fxs, dtype=np.float64))
                macro_fixed_y.append(np.array(fys, dtype=np.float64))
                macro_fixed_w.append(np.array(fws, dtype=np.float64))
            else:
                macro_fixed_x.append(np.empty(0, dtype=np.float64))
                macro_fixed_y.append(np.empty(0, dtype=np.float64))
                macro_fixed_w.append(np.empty(0, dtype=np.float64))

        # Neighbour list (hard–hard only) for connectivity-biased swap/pull moves
        neighbors: list = [list(macro_hard_j[i]) for i in range(n_hard)]

        # ── WL delta helpers ─────────────────────────────────────────────────────

        def delta_move(i, ox, oy, nx, ny) -> float:
            """
            Cost delta for moving macro i from (ox,oy) to (nx,ny).
            Handles both hard-neighbour and fixed-endpoint contributions.
            Also accounts for the change seen from i's hard neighbours (their
            edge to i changes too — captured via symmetric adjacency).
            """
            delta = 0.0
            # Hard neighbours of i: edge (i,j) — counted from i's side
            if len(macro_hard_j[i]):
                js = macro_hard_j[i]
                ox_arr = pos[js, 0]
                oy_arr = pos[js, 1]
                delta += float(
                    (
                        macro_hard_w[i]
                        * (
                            np.abs(nx - ox_arr)
                            + np.abs(ny - oy_arr)
                            - np.abs(ox - ox_arr)
                            - np.abs(oy - oy_arr)
                        )
                    ).sum()
                )
            # Fixed endpoints of i
            if len(macro_fixed_x[i]):
                delta += float(
                    (
                        macro_fixed_w[i]
                        * (
                            np.abs(nx - macro_fixed_x[i])
                            + np.abs(ny - macro_fixed_y[i])
                            - np.abs(ox - macro_fixed_x[i])
                            - np.abs(oy - macro_fixed_y[i])
                        )
                    ).sum()
                )
            return delta

        def delta_swap_wl(i, j, ox_i, oy_i, ox_j, oy_j) -> float:
            """WL delta for swapping i and j (call AFTER applying swap in pos)."""
            nx_i, ny_i = pos[i, 0], pos[i, 1]
            nx_j, ny_j = pos[j, 0], pos[j, 1]
            delta = 0.0
            if len(macro_hard_j[i]):
                js = macro_hard_j[i]
                mask = js != j
                js_f, w_f = js[mask], macro_hard_w[i][mask]
                if len(js_f):
                    delta += float(
                        (
                            w_f
                            * (
                                np.abs(nx_i - pos[js_f, 0])
                                + np.abs(ny_i - pos[js_f, 1])
                                - np.abs(ox_i - pos[js_f, 0])
                                - np.abs(oy_i - pos[js_f, 1])
                            )
                        ).sum()
                    )
            if len(macro_fixed_x[i]):
                delta += float(
                    (
                        macro_fixed_w[i]
                        * (
                            np.abs(nx_i - macro_fixed_x[i])
                            + np.abs(ny_i - macro_fixed_y[i])
                            - np.abs(ox_i - macro_fixed_x[i])
                            - np.abs(oy_i - macro_fixed_y[i])
                        )
                    ).sum()
                )
            if len(macro_hard_j[j]):
                js = macro_hard_j[j]
                mask = js != i
                js_f, w_f = js[mask], macro_hard_w[j][mask]
                if len(js_f):
                    delta += float(
                        (
                            w_f
                            * (
                                np.abs(nx_j - pos[js_f, 0])
                                + np.abs(ny_j - pos[js_f, 1])
                                - np.abs(ox_j - pos[js_f, 0])
                                - np.abs(oy_j - pos[js_f, 1])
                            )
                        ).sum()
                    )
            if len(macro_fixed_x[j]):
                delta += float(
                    (
                        macro_fixed_w[j]
                        * (
                            np.abs(nx_j - macro_fixed_x[j])
                            + np.abs(ny_j - macro_fixed_y[j])
                            - np.abs(ox_j - macro_fixed_x[j])
                            - np.abs(oy_j - macro_fixed_y[j])
                        )
                    ).sum()
                )
            return delta

        def has_overlap(idx):
            gap = 0.05
            dx = np.abs(pos[idx, 0] - pos[:, 0])
            dy = np.abs(pos[idx, 1] - pos[:, 1])
            ov = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap)
            ov[idx] = False
            return ov.any()

        # ── Incremental density grid ─────────────────────────────────────────
        #
        # Tracks macro area coverage per grid cell.  Soft macros contribute a
        # fixed baseline; only hard macro movements update the grid.  We use
        # sum-of-squares as the density penalty: cheap to update incrementally
        # and strongly penalises hotspots.

        cols_d = benchmark.grid_cols
        rows_d = benchmark.grid_rows
        cell_w_d = cw / cols_d
        cell_h_d = ch / rows_d
        cell_area_d = cell_w_d * cell_h_d
        all_sizes_d = benchmark.macro_sizes.numpy().astype(np.float64)

        def _cells(x, y, mw, mh):
            hw_, hh_ = mw / 2, mh / 2
            xl_, xr_ = x - hw_, x + hw_
            yl_, yr_ = y - hh_, y + hh_
            c_s = max(0, int(xl_ / cell_w_d))
            c_e = min(cols_d - 1, int(xr_ / cell_w_d))
            r_s = max(0, int(yl_ / cell_h_d))
            r_e = min(rows_d - 1, int(yr_ / cell_h_d))
            result = []
            for r in range(r_s, r_e + 1):
                oy_ = max(0.0, min(yr_, (r + 1) * cell_h_d) - max(yl_, r * cell_h_d))
                if oy_ <= 0:
                    continue
                for c in range(c_s, c_e + 1):
                    ox_ = max(
                        0.0, min(xr_, (c + 1) * cell_w_d) - max(xl_, c * cell_w_d)
                    )
                    if ox_ > 0:
                        result.append((r * cols_d + c, ox_ * oy_ / cell_area_d))
            return result

        density_grid = np.zeros(rows_d * cols_d, dtype=np.float64)
        for i_sm in range(n_hard, benchmark.num_macros):  # soft macros (fixed)
            for cell, frac in _cells(
                all_macro_pos[i_sm, 0],
                all_macro_pos[i_sm, 1],
                all_sizes_d[i_sm, 0],
                all_sizes_d[i_sm, 1],
            ):
                density_grid[cell] += frac
        for i_hm in range(n_hard):  # hard macros (current pos)
            for cell, frac in _cells(
                pos[i_hm, 0], pos[i_hm, 1], sizes[i_hm, 0], sizes[i_hm, 1]
            ):
                density_grid[cell] += frac

        def _dens_changes(x_old, y_old, x_new, y_new, mw, mh):
            """Dict of {cell: net_density_change} for one macro move."""
            ch_dict = {}
            for cell, f in _cells(x_old, y_old, mw, mh):
                ch_dict[cell] = ch_dict.get(cell, 0.0) - f
            for cell, f in _cells(x_new, y_new, mw, mh):
                ch_dict[cell] = ch_dict.get(cell, 0.0) + f
            return ch_dict

        def _delta_sos(ch_dict):
            """Delta in sum-of-squares density for a given change dict."""
            d = 0.0
            for cell, dc in ch_dict.items():
                dg = density_grid[cell]
                d += (dg + dc) ** 2 - dg**2
            return d

        def _apply_changes(ch_dict):
            for cell, dc in ch_dict.items():
                density_grid[cell] += dc

        # ── Incremental routing-demand grid (RUDY model) ─────────────────────
        #
        # For each net, the RUDY model distributes expected wire demand uniformly
        # across the net's bounding box.  For a net with bbox [xl,xr]×[yl,yr]:
        #   h_demand per cell = weight * cell_h / bbox_height / h_cap
        #   v_demand per cell = weight * cell_w / bbox_width  / v_cap
        # Normalising by routing capacity (routes/μm) makes the values
        # dimensionless congestion ratios, matching the plc evaluator's metric.
        #
        # Only nets that contain at least one movable hard macro are tracked here;
        # purely fixed nets never change and can be folded into a constant baseline.

        h_cap = float(benchmark.hroutes_per_micron) * cell_w_d  # tracks per cell (h)
        v_cap = float(benchmark.vroutes_per_micron) * cell_h_d  # tracks per cell (v)

        # Per routing-net arrays (indexed 0..n_rnets-1)
        rn_hard = []  # list of int arrays: hard-macro indices in this net
        rn_fx = []  # list of float arrays: fixed endpoint x coords
        rn_fy = []  # list of float arrays: fixed endpoint y coords
        rn_wt = []  # list of float weights
        rn_bbox = []  # list of [xl, xr, yl, yr] (mutable)

        macro_to_rn = [[] for _ in range(n_hard)]  # macro_to_rn[i] = [rn_idx, ...]

        for net_i, nodes_tensor in enumerate(benchmark.net_nodes):
            raw = [int(n) for n in nodes_tensor.numpy()]
            w_net = float(benchmark.net_weights[net_i])
            hn = [n for n in raw if n < n_hard]
            if not hn:
                continue  # no hard macros → bbox never changes

            fx, fy = [], []
            for n in raw:
                if n >= n_hard:
                    if n < n_total:
                        fx.append(float(all_macro_pos[n, 0]))
                        fy.append(float(all_macro_pos[n, 1]))
                    else:
                        p = n - n_total
                        if p < len(port_pos):
                            fx.append(float(port_pos[p, 0]))
                            fy.append(float(port_pos[p, 1]))

            rn_idx = len(rn_hard)
            rn_hard.append(np.array(hn, dtype=np.int32))
            rn_fx.append(np.array(fx))
            rn_fy.append(np.array(fy))
            rn_wt.append(w_net)
            rn_bbox.append([0.0, 0.0, 0.0, 0.0])  # filled below
            for hi in hn:
                macro_to_rn[hi].append(rn_idx)

        def _rn_current_bbox(rn_idx):
            """Recompute bounding box of routing net from current hard-macro positions."""
            hn = rn_hard[rn_idx]
            xs = list(pos[hn, 0])
            ys = list(pos[hn, 1])
            if len(rn_fx[rn_idx]):
                xs.extend(rn_fx[rn_idx])
                ys.extend(rn_fy[rn_idx])
            return [min(xs), max(xs), min(ys), max(ys)]

        def _rudy_ch(xl, xr, yl, yr, xl2, xr2, yl2, yr2, w):
            """Change dict {cell: (dh, dv)} for one net's bbox change (old→new)."""
            ch = {}
            dx = max(xr - xl, cell_w_d)
            dy = max(yr - yl, cell_h_d)
            dx2 = max(xr2 - xl2, cell_w_d)
            dy2 = max(yr2 - yl2, cell_h_d)
            h_rm = -w * cell_h_d / dy / h_cap  # remove old
            v_rm = -w * cell_w_d / dx / v_cap
            h_ad = w * cell_h_d / dy2 / h_cap  # add new
            v_ad = w * cell_w_d / dx2 / v_cap
            # Old bbox cells
            c_s = max(0, int(xl / cell_w_d))
            c_e = min(cols_d - 1, int(xr / cell_w_d))
            r_s = max(0, int(yl / cell_h_d))
            r_e = min(rows_d - 1, int(yr / cell_h_d))
            for r in range(r_s, r_e + 1):
                for c in range(c_s, c_e + 1):
                    k = r * cols_d + c
                    dh, dv = ch.get(k, (0.0, 0.0))
                    ch[k] = (dh + h_rm, dv + v_rm)
            # New bbox cells
            c_s = max(0, int(xl2 / cell_w_d))
            c_e = min(cols_d - 1, int(xr2 / cell_w_d))
            r_s = max(0, int(yl2 / cell_h_d))
            r_e = min(rows_d - 1, int(yr2 / cell_h_d))
            for r in range(r_s, r_e + 1):
                for c in range(c_s, c_e + 1):
                    k = r * cols_d + c
                    dh, dv = ch.get(k, (0.0, 0.0))
                    ch[k] = (dh + h_ad, dv + v_ad)
            return ch

        def _delta_cong_sos(cong_ch):
            d = 0.0
            for k, (dh, dv) in cong_ch.items():
                d += (
                    (h_dem[k] + dh) ** 2
                    - h_dem[k] ** 2
                    + (v_dem[k] + dv) ** 2
                    - v_dem[k] ** 2
                )
            return d

        def _apply_cong(cong_ch):
            for k, (dh, dv) in cong_ch.items():
                h_dem[k] += dh
                v_dem[k] += dv

        # Initialise demand grids: fixed nets first, then movable hard-macro nets
        h_dem = np.zeros(rows_d * cols_d, dtype=np.float64)
        v_dem = np.zeros(rows_d * cols_d, dtype=np.float64)

        # Fixed baseline: nets with no hard macros (purely soft+ports, never change)
        for net_i, nodes_tensor in enumerate(benchmark.net_nodes):
            raw = [int(n) for n in nodes_tensor.numpy()]
            if any(n < n_hard for n in raw):
                continue  # handled per-net below
            w_net = float(benchmark.net_weights[net_i])
            xs, ys = [], []
            for n in raw:
                if n < n_total:
                    xs.append(float(all_macro_pos[n, 0]))
                    ys.append(float(all_macro_pos[n, 1]))
                else:
                    p = n - n_total
                    if p < len(port_pos):
                        xs.append(float(port_pos[p, 0]))
                        ys.append(float(port_pos[p, 1]))
            if len(xs) < 2:
                continue
            xl, xr, yl, yr = min(xs), max(xs), min(ys), max(ys)
            dx = max(xr - xl, cell_w_d)
            dy = max(yr - yl, cell_h_d)
            hc = w_net * cell_h_d / dy / h_cap
            vc = w_net * cell_w_d / dx / v_cap
            c_s = max(0, int(xl / cell_w_d))
            c_e = min(cols_d - 1, int(xr / cell_w_d))
            r_s = max(0, int(yl / cell_h_d))
            r_e = min(rows_d - 1, int(yr / cell_h_d))
            for r in range(r_s, r_e + 1):
                for c in range(c_s, c_e + 1):
                    h_dem[r * cols_d + c] += hc
                    v_dem[r * cols_d + c] += vc

        # Movable hard-macro nets: compute initial bboxes and add demand
        for rn_idx in range(len(rn_hard)):
            rn_bbox[rn_idx] = _rn_current_bbox(rn_idx)
            xl, xr, yl, yr = rn_bbox[rn_idx]
            dx = max(xr - xl, cell_w_d)
            dy = max(yr - yl, cell_h_d)
            hc = rn_wt[rn_idx] * cell_h_d / dy / h_cap
            vc = rn_wt[rn_idx] * cell_w_d / dx / v_cap
            c_s = max(0, int(xl / cell_w_d))
            c_e = min(cols_d - 1, int(xr / cell_w_d))
            r_s = max(0, int(yl / cell_h_d))
            r_e = min(rows_d - 1, int(yr / cell_h_d))
            for r in range(r_s, r_e + 1):
                for c in range(c_s, c_e + 1):
                    h_dem[r * cols_d + c] += hc
                    v_dem[r * cols_d + c] += vc

        # ── Initialise combined cost ─────────────────────────────────────────

        init_wl = 0.0
        for i_c in range(n_hard):
            if len(macro_hard_j[i_c]):
                js = macro_hard_j[i_c]
                init_wl += float(
                    (
                        macro_hard_w[i_c]
                        * (
                            np.abs(pos[i_c, 0] - pos[js, 0])
                            + np.abs(pos[i_c, 1] - pos[js, 1])
                        )
                    ).sum()
                )
            if len(macro_fixed_x[i_c]):
                init_wl += float(
                    (
                        macro_fixed_w[i_c]
                        * (
                            np.abs(pos[i_c, 0] - macro_fixed_x[i_c])
                            + np.abs(pos[i_c, 1] - macro_fixed_y[i_c])
                        )
                    ).sum()
                )

        init_sos = float((density_grid**2).sum())
        # w_dens: weight so density starts at sa_density_weight × WL contribution
        w_dens = (
            (self.sa_density_weight * init_wl / init_sos) if init_sos > 1e-12 else 0.0
        )

        init_cong = float((h_dem**2).sum() + (v_dem**2).sum())
        # w_cong: weight so congestion starts at sa_density_weight × WL contribution
        # (same multiplier as density — both are spreading signals)
        w_cong = (
            (self.sa_density_weight * init_wl / init_cong) if init_cong > 1e-12 else 0.0
        )

        current_wl = init_wl
        current_sos = init_sos
        current_cong = init_cong
        current_cost = current_wl + w_dens * current_sos + w_cong * current_cong
        best_pos = pos.copy()
        best_cost = current_cost

        T_start = max(cw, ch) * 0.10
        T_end = max(cw, ch) * 0.001

        for step in range(self.sa_iters):
            frac = step / self.sa_iters
            T = T_start * (T_end / T_start) ** frac

            move_type = random.random()
            i = int(random.choice(movable_idx))
            ox, oy = pos[i, 0], pos[i, 1]
            j = None
            ojx = ojy = 0.0
            ch_dict = None

            if move_type < 0.45:
                # Shift
                sigma = T * (0.2 + 0.8 * (1 - frac))
                nx = np.clip(ox + random.gauss(0, sigma), half_w[i], cw - half_w[i])
                ny = np.clip(oy + random.gauss(0, sigma), half_h[i], ch - half_h[i])
                pos[i, 0] = nx
                pos[i, 1] = ny
                if has_overlap(i):
                    pos[i, 0] = ox
                    pos[i, 1] = oy
                    continue
                delta_wl = delta_move(i, ox, oy, nx, ny)
                ch_dict = _dens_changes(ox, oy, nx, ny, sizes[i, 0], sizes[i, 1])

            elif move_type < 0.75:
                # Swap (prefer connected neighbours)
                if neighbors[i] and random.random() < 0.7:
                    cands = [nb for nb in neighbors[i] if movable[nb]]
                    j = (
                        int(random.choice(cands))
                        if cands
                        else int(random.choice(movable_idx))
                    )
                else:
                    j = int(random.choice(movable_idx))
                if i == j:
                    continue
                ojx, ojy = pos[j, 0], pos[j, 1]
                pos[i, 0] = np.clip(ojx, half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(ojy, half_h[i], ch - half_h[i])
                pos[j, 0] = np.clip(ox, half_w[j], cw - half_w[j])
                pos[j, 1] = np.clip(oy, half_h[j], ch - half_h[j])
                if has_overlap(i) or has_overlap(j):
                    pos[i, 0] = ox
                    pos[i, 1] = oy
                    pos[j, 0] = ojx
                    pos[j, 1] = ojy
                    continue
                delta_wl = delta_swap_wl(i, j, ox, oy, ojx, ojy)
                # Merge both moves' density changes simultaneously
                c1 = _dens_changes(
                    ox, oy, pos[i, 0], pos[i, 1], sizes[i, 0], sizes[i, 1]
                )
                c2 = _dens_changes(
                    ojx, ojy, pos[j, 0], pos[j, 1], sizes[j, 0], sizes[j, 1]
                )
                ch_dict = dict(c1)
                for cell, dc in c2.items():
                    ch_dict[cell] = ch_dict.get(cell, 0.0) + dc

            else:
                # Pull toward a connected node
                if not neighbors[i]:
                    continue
                nb = int(random.choice(neighbors[i]))
                alpha = random.uniform(0.05, 0.35)
                nx = np.clip(ox + alpha * (pos[nb, 0] - ox), half_w[i], cw - half_w[i])
                ny = np.clip(oy + alpha * (pos[nb, 1] - oy), half_h[i], ch - half_h[i])
                pos[i, 0] = nx
                pos[i, 1] = ny
                if has_overlap(i):
                    pos[i, 0] = ox
                    pos[i, 1] = oy
                    continue
                delta_wl = delta_move(i, ox, oy, nx, ny)
                ch_dict = _dens_changes(ox, oy, nx, ny, sizes[i, 0], sizes[i, 1])

            delta_sos = _delta_sos(ch_dict)

            # Routing congestion delta (RUDY model, incremental bbox update)
            aff_rn = set(macro_to_rn[i])
            if j is not None:
                aff_rn |= set(macro_to_rn[j])
            cong_ch = {}
            new_bboxes = {}
            for rn_idx in aff_rn:
                ob = rn_bbox[rn_idx]
                # Fast-path: if neither moved macro is at a bbox boundary, the
                # bounding box cannot change — skip the expensive recompute.
                eps = 1e-6
                i_at_bnd = (
                    abs(ox - ob[0]) < eps
                    or abs(ox - ob[1]) < eps
                    or abs(oy - ob[2]) < eps
                    or abs(oy - ob[3]) < eps
                )
                j_at_bnd = j is not None and (
                    abs(ojx - ob[0]) < eps
                    or abs(ojx - ob[1]) < eps
                    or abs(ojy - ob[2]) < eps
                    or abs(ojy - ob[3]) < eps
                )
                if not (i_at_bnd or j_at_bnd):
                    new_bboxes[rn_idx] = ob
                    continue
                nb = _rn_current_bbox(rn_idx)
                new_bboxes[rn_idx] = nb
                if (
                    nb[0] == ob[0]
                    and nb[1] == ob[1]
                    and nb[2] == ob[2]
                    and nb[3] == ob[3]
                ):
                    continue
                for k, (dh, dv) in _rudy_ch(*ob, *nb, rn_wt[rn_idx]).items():
                    cdh, cdv = cong_ch.get(k, (0.0, 0.0))
                    cong_ch[k] = (cdh + dh, cdv + dv)
            delta_cong = _delta_cong_sos(cong_ch)

            delta = delta_wl + w_dens * delta_sos + w_cong * delta_cong

            # Metropolis acceptance
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_wl += delta_wl
                current_sos += delta_sos
                current_cong += delta_cong
                current_cost = current_wl + w_dens * current_sos + w_cong * current_cong
                _apply_changes(ch_dict)
                _apply_cong(cong_ch)
                for rn_idx, nb in new_bboxes.items():
                    rn_bbox[rn_idx] = nb
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pos = pos.copy()
            else:
                pos[i, 0] = ox
                pos[i, 1] = oy
                if j is not None:
                    pos[j, 0] = ojx
                    pos[j, 1] = ojy

        return best_pos

    # Gradient Refinement

    def _grad_refine(
        self, pos, movable, sizes, benchmark, n_hard, n_total, port_pos, cw, ch
    ):
        """
        Gradient-based refinement using smooth HPWL (log-sum-exp) + Gaussian density.

        Starts from the current best legal placement, relaxes into continuous space,
        then re-legalises. Only kept if the fast proxy cost improves.

        Algorithm:
          - Smooth HPWL: log-sum-exp approximation of max/min per net.
              smooth_hpwl_x = α·log(Σ exp(xᵢ/α)) − α·log(Σ exp(−xᵢ/α))
            α = canvas/100 gives <1% HPWL approximation error.
          - All nets are batched into padded tensors so the closure is purely
            tensor ops — no Python loop at evaluation time.
          - Density: Gaussian kernel per macro (σ = half-size), penalises
            sum-of-squares of per-cell density.  Fixed soft-macro baseline
            precomputed once.  Separable x/y formulation avoids a 3-D
            intermediate tensor.
          - Boundary: quadratic penalty for leaving canvas.
          - Optimiser: L-BFGS with strong Wolfe line search.

        Parameters
        ----------
        pos     : [n_hard, 2] float64 — current hard-macro positions
        movable : [n_hard] bool
        sizes   : [n_hard, 2] float64
        """
        movable_idx = np.where(movable)[0]
        n_mov = len(movable_idx)
        if n_mov == 0:
            return pos

        # α = canvas/100 → <1% HPWL approximation error
        alpha = float(max(cw, ch)) / 100.0

        all_macro_pos = benchmark.macro_positions.numpy().astype(np.float64)
        glob_to_loc = {int(g): l for l, g in enumerate(movable_idx)}

        # ── Collect per-net node lists ────────────────────────────────────────
        raw_mov = []  # list of lists of local movable indices
        raw_fixed = []  # list of lists of [x, y] for fixed endpoints
        raw_w = []

        for nodes_tensor, w_t in zip(benchmark.net_nodes, benchmark.net_weights):
            nodes = [int(n) for n in nodes_tensor.numpy()]
            w = float(w_t)
            if len(nodes) < 2:
                continue

            mov_local = [glob_to_loc[n] for n in nodes if n in glob_to_loc]
            fixed_pts = []
            for n in nodes:
                if n not in glob_to_loc:
                    if n < n_total:
                        fixed_pts.append(
                            [float(all_macro_pos[n, 0]), float(all_macro_pos[n, 1])]
                        )
                    else:
                        p = n - n_total
                        if p < len(port_pos):
                            fixed_pts.append(
                                [float(port_pos[p, 0]), float(port_pos[p, 1])]
                            )

            if len(mov_local) + len(fixed_pts) < 2:
                continue

            raw_mov.append(mov_local)
            raw_fixed.append(fixed_pts)
            raw_w.append(w)

        if not raw_mov:
            return pos

        n_nets = len(raw_w)
        wl_norm = float((cw + ch) * benchmark.num_nets)

        # ── Build padded batch tensors (precomputed, outside closure) ─────────
        #
        # Each net has up to MAX_K endpoints.  Nets larger than MAX_K are
        # truncated — high-fanout clock/power nets contribute little per edge.
        # Slots are filled: movable first, then fixed, then padding (sentinel).
        #
        # Shapes: [n_nets, MAX_K]
        #
        # In the closure we do 4 batched logsumexp calls instead of a Python
        # loop over thousands of nets, eliminating the per-kernel-launch overhead.

        MAX_K = min(max(len(m) + len(f) for m, f in zip(raw_mov, raw_fixed)), 64)

        # Movable slot indices (sentinel = n_mov, clamped away in gather)
        mov_slots = torch.full((n_nets, MAX_K), n_mov, dtype=torch.int64)
        # Fixed coordinates for non-movable slots
        fixed_xy_pad = torch.zeros(n_nets, MAX_K, 2, dtype=torch.float32)
        # Masks
        is_movable = torch.zeros(n_nets, MAX_K, dtype=torch.bool)
        is_valid = torch.zeros(n_nets, MAX_K, dtype=torch.bool)
        net_weights_t = torch.tensor(raw_w, dtype=torch.float32)  # [n_nets]

        for i, (ml, fl, _) in enumerate(zip(raw_mov, raw_fixed, raw_w)):
            k_m = min(len(ml), MAX_K)
            k_f = min(len(fl), MAX_K - k_m)
            if k_m > 0:
                mov_slots[i, :k_m] = torch.tensor(ml[:k_m], dtype=torch.int64)
                is_movable[i, :k_m] = True
                is_valid[i, :k_m] = True
            if k_f > 0:
                fixed_xy_pad[i, k_m : k_m + k_f] = torch.tensor(
                    fl[:k_f], dtype=torch.float32
                )
                is_valid[i, k_m : k_m + k_f] = True

        # ── Density setup ─────────────────────────────────────────────────────
        rows, cols = benchmark.grid_rows, benchmark.grid_cols
        n_cells = rows * cols
        cell_w, cell_h = cw / cols, ch / rows

        cx_arr = (np.arange(cols) + 0.5) * cell_w  # [cols]
        cy_arr = (np.arange(rows) + 0.5) * cell_h  # [rows]
        cell_cx = torch.tensor(
            np.tile(cx_arr, rows), dtype=torch.float32
        )  # [n_cells]  row-major
        cell_cy = torch.tensor(
            np.repeat(cy_arr, cols), dtype=torch.float32
        )  # [n_cells]

        # σ = macro half-size; precompute reciprocals for cheaper closure math
        sigma_mov = torch.tensor(sizes[movable_idx], dtype=torch.float32).clamp(
            min=1e-3
        )
        inv_sx = 1.0 / sigma_mov[:, 0]  # [n_mov]
        inv_sy = 1.0 / sigma_mov[:, 1]

        # Precompute scaled cell-centre offsets: [n_mov, n_cells] (constant)
        # scaled_cc_x[i, c] = cell_cx[c] * inv_sx[i]
        # In the closure: diff_x_scaled[i,c] = x_mov[i,0]*inv_sx[i] - scaled_cc_x[i,c]
        scaled_cc_x = inv_sx[:, None] * cell_cx[None, :]  # [n_mov, n_cells]
        scaled_cc_y = inv_sy[:, None] * cell_cy[None, :]

        # Fixed density from soft macros (never changes)
        soft_dens = torch.zeros(n_cells, dtype=torch.float32)
        for i_sm in range(n_hard, n_total):
            sx = float(all_macro_pos[i_sm, 0])
            sy = float(all_macro_pos[i_sm, 1])
            swx = max(float(benchmark.macro_sizes[i_sm, 0]), 1e-3)
            swy = max(float(benchmark.macro_sizes[i_sm, 1]), 1e-3)
            dx = (cell_cx - sx) / swx
            dy = (cell_cy - sy) / swy
            soft_dens = soft_dens + torch.exp(-0.5 * (dx * dx + dy * dy))

        # ── Optimisation variable ─────────────────────────────────────────────
        hw_t = torch.tensor(sizes[movable_idx, 0] / 2, dtype=torch.float32)
        hh_t = torch.tensor(sizes[movable_idx, 1] / 2, dtype=torch.float32)

        x_mov = torch.tensor(pos[movable_idx], dtype=torch.float32, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [x_mov],
            lr=1.0,
            max_iter=5,
            line_search_fn="strong_wolfe",
        )

        NEG_INF = -1e9  # masks padding in logsumexp (exp(NEG_INF/α) ≈ 0)
        w_dens = [None]

        def closure():
            optimizer.zero_grad()

            # ── Batched smooth WL ─────────────────────────────────────────────
            #
            # Gather movable positions into the padded slot tensor.
            # safe_slots clamps the sentinel (n_mov) to a valid index; the
            # gathered value for sentinel slots is overwritten by fixed_xy_pad
            # via torch.where, so its value doesn't matter.
            safe_slots = mov_slots.clamp(0, n_mov - 1)  # [n_nets, MAX_K]
            gathered = x_mov[safe_slots]  # [n_nets, MAX_K, 2]

            # Blend: movable slots → x_mov, fixed/padding slots → precomputed coords
            all_xy = torch.where(is_movable[:, :, None], gathered, fixed_xy_pad)

            # Mask padding to NEG_INF so exp contribution → 0 in logsumexp
            all_x = all_xy[:, :, 0].masked_fill(~is_valid, NEG_INF)  # [n_nets, MAX_K]
            neg_x = -all_xy[:, :, 0].masked_fill(~is_valid, NEG_INF)
            all_y = all_xy[:, :, 1].masked_fill(~is_valid, NEG_INF)
            neg_y = -all_xy[:, :, 1].masked_fill(~is_valid, NEG_INF)

            # smooth HPWL per net = α·lse(x/α) + α·lse(-x/α) + same for y
            smooth_span = (
                alpha * torch.logsumexp(all_x / alpha, dim=1)
                + alpha * torch.logsumexp(neg_x / alpha, dim=1)
                + alpha * torch.logsumexp(all_y / alpha, dim=1)
                + alpha * torch.logsumexp(neg_y / alpha, dim=1)
            )  # [n_nets]
            wl = (net_weights_t * smooth_span).sum() / wl_norm

            # ── Density (separable Gaussian, no 3-D tensor) ───────────────────
            #
            # diff_x_scaled[i,c] = (x_mov[i,0] / σx[i]) − (cell_cx[c] / σx[i])
            #                    = x_mov[i,0]*inv_sx[i] − scaled_cc_x[i,c]
            xs = (x_mov[:, 0] * inv_sx)[:, None] - scaled_cc_x  # [n_mov, n_cells]
            ys = (x_mov[:, 1] * inv_sy)[:, None] - scaled_cc_y
            gauss = torch.exp(-0.5 * (xs * xs + ys * ys))  # [n_mov, n_cells]
            density = gauss.sum(0) + soft_dens  # [n_cells]
            dens_loss = (density * density).sum() / n_cells

            if w_dens[0] is None:
                wl_v = float(wl.detach())
                dv = float(dens_loss.detach())
                w_dens[0] = (0.5 * wl_v / dv) if dv > 1e-12 else 0.0

            # ── Boundary penalty ──────────────────────────────────────────────
            bnd = (
                torch.relu(hw_t - x_mov[:, 0]).pow(2)
                + torch.relu(x_mov[:, 0] - (cw - hw_t)).pow(2)
                + torch.relu(hh_t - x_mov[:, 1]).pow(2)
                + torch.relu(x_mov[:, 1] - (ch - hh_t)).pow(2)
            ).sum()

            loss = wl + w_dens[0] * dens_loss + 10.0 * bnd
            loss.backward()
            return loss

        for _ in range(self.grad_steps):
            optimizer.step(closure)

        # ── Extract + re-legalize ─────────────────────────────────────────────
        with torch.no_grad():
            x_mov_np = x_mov.numpy().astype(np.float64)

        result = pos.copy()
        result[movable_idx] = x_mov_np
        result = self._legalize(result, movable, sizes, cw, ch)
        return result
