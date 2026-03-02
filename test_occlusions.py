#!/usr/bin/env python
"""
Cross-comparison of P2P, P3P, and P4P solvers under partial occlusion.

A rigid body with 4 world points is projected through a pinhole camera with
a known FoV.  We then systematically test all partial-occlusion scenarios
(all C(4,2), C(4,3), and the full C(4,4) subsets) and compare the chosen
solvers:

  P2P  – Sweeney (gravity-aware, needs known rotation axis)
  P3P  – Nakano
  P4P  – Lehavi (algebraic)

For each scenario we report rotation / translation errors, reprojection
quality, and number of valid solutions.

At the end we produce a 3D plot showing:
  • world points (black)
  • ground-truth camera (green)
  • estimated cameras from every solver & every valid solution (coloured)
"""

import itertools
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import time

from recipnps.model import Model
from recipnps.p2p import sweeney as p2p_sweeney
from recipnps.p3p import nakano as p3p_nakano
from recipnps.p4p import lehavi_p4p_algebraic as p4p_lehavi


# ──────────────────────────────────────────────────────────────────────
#  Camera helpers
# ──────────────────────────────────────────────────────────────────────

def intrinsic_from_fov(fov_deg, img_w, img_h):
    """Build a 3×3 intrinsic matrix from horizontal FoV (degrees)."""
    f = (img_w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    K = np.array([[f, 0.0, img_w / 2.0],
                  [0.0, f, img_h / 2.0],
                  [0.0, 0.0, 1.0]])
    return K


def project(K, R, t, pw):
    """Project 3×N world points → 2×N pixel coords.  Returns (uv, depths)."""
    pc = R @ pw + t[:, None]
    depths = pc[2, :]
    uv_h = K @ pc
    uv = uv_h[:2, :] / uv_h[2:, :]
    return uv, depths


def bearing_vectors(K, uv):
    """Pixel coords (2×N) → unit bearing vectors (3×N) in camera frame."""
    ones = np.ones((1, uv.shape[1]))
    pts_h = np.vstack([uv, ones])
    rays = np.linalg.inv(K) @ pts_h
    rays /= np.linalg.norm(rays, axis=0)
    return rays


def random_rotation(rng):
    """Uniform random SO(3) via QR decomposition."""
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def rotation_error_deg(R_est, R_gt):
    delta = R_est @ R_gt.T
    cos_theta = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def translation_error(t_est, t_gt):
    return float(np.linalg.norm(t_est - t_gt))


# ──────────────────────────────────────────────────────────────────────
#  Solver wrappers (uniform interface)
# ──────────────────────────────────────────────────────────────────────

def run_p2p(pw_subset, bi_subset, R_gt, v_world):
    """Run P2P solver.  Needs gravity vectors derived from ground truth."""
    v_cam = R_gt @ v_world  # gravity in camera frame
    return p2p_sweeney(pw_subset, bi_subset, v_cam, v_world)


def run_p3p(pw_subset, bi_subset, R_gt, v_world):
    return p3p_nakano(pw_subset, bi_subset)


def run_p4p(pw_subset, bi_subset, R_gt, v_world):
    return p4p_lehavi(pw_subset, bi_subset, reprojection_threshold=1e-4)


SOLVER_MAP = {
    2: ("P2P Sweeney", run_p2p),
    3: ("P3P Nakano",  run_p3p),
    4: ("P4P Lehavi",  run_p4p),
}

SOLVER_COLOURS = {
    2: "tab:blue",
    3: "tab:orange",
    4: "tab:red",
}


# ──────────────────────────────────────────────────────────────────────
#  Main cross-comparison
# ──────────────────────────────────────────────────────────────────────

def main(seed=time.time_ns()):
    rng = np.random.default_rng(seed)

    # ---- scene setup ------------------------------------------------
    # Rigid body (4 world points – a small tetrahedron-like shape)
    pw = np.array([
        [ 0.0,  1.0,  0.5, -0.3],   # x
        [ 0.0,  0.0,  0.8,  0.4],   # y
        [ 0.0,  0.0,  0.0,  0.6],   # z
    ], dtype=np.float64)

    # Camera parameters
    fov_deg = 60.0
    img_w, img_h = 640, 480
    K = intrinsic_from_fov(fov_deg, img_w, img_h)

    # Ground-truth pose
    R_gt = random_rotation(rng)
    t_gt = np.array([0.1, -0.2, 4.0]) + rng.uniform(-0.3, 0.3, size=3)

    # Gravity direction in world frame (arbitrary but fixed)
    v_world = np.array([0.0, -1.0, 0.0])

    # Project all 4 points
    uv_all, depths_all = project(K, R_gt, t_gt, pw)
    assert np.all(depths_all > 0), "Some points behind camera — adjust t_gt"

    bi_all = bearing_vectors(K, uv_all)          # 3 × 4

    # ---- enumerate occlusion subsets --------------------------------
    point_indices = list(range(4))

    # All subsets of size 2, 3, 4
    subsets = []
    for k in (2, 3, 4):
        for combo in itertools.combinations(point_indices, k):
            subsets.append(combo)

    print("=" * 72)
    print("Cross-comparison: P2P / P3P / P4P under partial occlusion")
    print("=" * 72)
    print(f"Camera FoV = {fov_deg}°,  image = {img_w}×{img_h}")
    print(f"World points:\n{pw.T}")
    print(f"R_gt:\n{R_gt}")
    print(f"t_gt: {t_gt}")
    print(f"Projected pixels:\n{uv_all.T}")
    print()

    # Collect results for plotting  {(n, combo_idx): [list of Model]}
    all_results = {}

    for combo in subsets:
        k = len(combo)
        idx = list(combo)
        pw_sub = pw[:, idx]
        bi_sub = bi_all[:, idx]

        solver_name, solver_fn = SOLVER_MAP[k]

        try:
            solutions = solver_fn(pw_sub, bi_sub, R_gt, v_world)
        except Exception as exc:
            solutions = []
            print(f"  [{solver_name}] pts {combo} — EXCEPTION: {exc}")

        all_results[(k, combo)] = solutions

        # Report
        print(f"  [{solver_name}]  visible pts = {combo}  →  "
              f"{len(solutions)} solution(s)")
        for si, sol in enumerate(solutions):
            re = rotation_error_deg(sol.rotation, R_gt)
            te = translation_error(sol.translation, t_gt)
            reproj = sol.reprojection_dists_of(pw_sub, bi_sub)
            is_gt = (re < 0.01 and te < 1e-4)
            tag = " ★ GT" if is_gt else ""
            print(f"      sol[{si}]  rot_err={re:8.4f}°  trans_err={te:.2e}  "
                  f"max_reproj={np.max(reproj):.2e}{tag}")

    # ---- 3D Visualisation -------------------------------------------
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("P2P / P3P / P4P — estimated camera poses under partial occlusion")

    # World points
    ax.scatter(*pw, c="black", s=80, marker="o", label="World points")
    for i in range(pw.shape[1]):
        ax.text(pw[0, i], pw[1, i], pw[2, i], f"  P{i}", fontsize=8)

    # draw camera frustum helper
    def draw_camera(ax, R, t, color, label, alpha=0.8, scale=0.3):
        """Draw a small camera frustum at position -R^T @ t."""
        C = -R.T @ t  # camera centre in world frame
        ax.scatter(*C, c=color, s=60, marker="^", alpha=alpha)

        # axes of the camera
        dirs = R.T  # columns = world-frame directions of cam x, y, z
        for j, (c_ax, lbl) in enumerate(zip(["r", "g", "b"], ["x", "y", "z"])):
            end = C + scale * dirs[:, j]
            ax.plot([C[0], end[0]], [C[1], end[1]], [C[2], end[2]],
                    color=c_ax, alpha=alpha * 0.7, linewidth=1)

        if label:
            ax.text(C[0], C[1], C[2], f"  {label}", fontsize=6,
                    color=color, alpha=alpha)

    # Ground truth camera
    draw_camera(ax, R_gt, t_gt, "green", "GT", alpha=1.0, scale=0.5)

    # Estimated cameras
    drawn_labels = set()
    for (k, combo), solutions in all_results.items():
        solver_label, _ = SOLVER_MAP[k]
        col = SOLVER_COLOURS[k]
        for si, sol in enumerate(solutions):
            re = rotation_error_deg(sol.rotation, R_gt)
            te = translation_error(sol.translation, t_gt)
            lbl = solver_label if solver_label not in drawn_labels else ""
            drawn_labels.add(solver_label)
            tag = f"{solver_label[0:3]}{''.join(str(c) for c in combo)}[{si}]"
            alpha = 0.9 if (re < 0.1 and te < 1e-3) else 0.35
            draw_camera(ax, sol.rotation, sol.translation, col, tag,
                        alpha=alpha, scale=0.25)

    # Cosmetics
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Manual legend entries (one per solver class)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="green",
               markersize=10, label="Ground truth"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor=SOLVER_COLOURS[2], markersize=10,
               label="P2P Sweeney"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor=SOLVER_COLOURS[3], markersize=10,
               label="P3P Nakano"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor=SOLVER_COLOURS[4], markersize=10,
               label="P4P Lehavi"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=8, label="World points"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
