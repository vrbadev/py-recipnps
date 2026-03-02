"""
Perspective-2-Point (P2P) solvers with a known rotation axis (gravity direction).

Given two 3D-2D point correspondences and the gravity/vertical direction in both
the camera and world frames, solve for the full 6-DOF camera pose R, t such that:

    λ_i * p_i = R @ P_w_i + t,    i = 1, 2

where p_i are unit bearing vectors, P_w_i are 3D world points, and λ_i > 0 are
unknown depths.

The known gravity direction constrains R to be a rotation about that axis, reducing
the rotation unknowns from 3 DOF to 1 DOF (the angle about the gravity axis).

Implemented methods:
  - sweeney: Sweeney et al. (ISMAR 2015) — depth-based formulation
  - li:      Li et al. (IEEE Access 2023) — rotation-based formulation
"""

import numpy as np
from .model import Model


# ---------------------------------------------------------------------------
#  Helper utilities
# ---------------------------------------------------------------------------

def _rotation_between(v_from, v_to):
    """Rotation matrix that maps unit vector *v_from* to unit vector *v_to*.

    Uses Rodrigues' rotation formula.  Handles the near-parallel and
    near-antiparallel cases gracefully.
    """
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)

    c = float(np.dot(v_from, v_to))

    if c > 1.0 - 1e-12:              # nearly identical
        return np.eye(3)

    if c < -1.0 + 1e-12:             # nearly opposite
        # pick any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(v_from, perp)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(v_from, perp)
        axis /= np.linalg.norm(axis)
        # 180-degree rotation about *axis*
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    k = np.cross(v_from, v_to)       # sin(θ) * axis
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def _rodrigues(axis, angle):
    """Rotation matrix for rotation of *angle* radians about unit *axis*
    (Rodrigues' formula)."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


# ---------------------------------------------------------------------------
#  Sweeney et al. 2015 — depth-based P2P solver
# ---------------------------------------------------------------------------

def sweeney(p_w, p_i, v_cam, v_world):
    r"""Absolute pose from two 2D-3D correspondences with known gravity direction.

    **Reference:** C. Sweeney *et al.*, "Efficient Computation of Absolute Pose
    for Gravity-Aware Augmented Reality", ISMAR 2015.

    The method first solves for the two unknown depths λ₁, λ₂ by forming:

    1. A *distance* constraint:  ‖P₁ᵂ − P₂ᵂ‖ = ‖λ₁p₁ − λ₂p₂‖
    2. An *axis-projection* constraint:  (P₁ᵂ − P₂ᵂ)·v = (λ₁p₁ − λ₂p₂)·v

    yielding a quadratic in λ₂.  Then it recovers the rotation angle α about
    the gravity axis and the translation vector.

    Parameters
    ----------
    p_w : ndarray, shape (3, 2)
        World-frame 3D points.
    p_i : ndarray, shape (3, 2)
        Unit bearing vectors in the camera frame.
    v_cam : ndarray, shape (3,)
        Unit gravity direction in the camera frame.
    v_world : ndarray, shape (3,)
        Unit gravity direction in the world frame.

    Returns
    -------
    list of Model
        Up to two solutions (R, t) with P_cam = R @ P_world + t.
    """
    v_c = v_cam / np.linalg.norm(v_cam)
    v_w = v_world / np.linalg.norm(v_world)

    # Pre-align: rotate camera so that gravity -> [0,0,1], same for world
    R_c = _rotation_between(v_c, np.array([0.0, 0.0, 1.0]))
    R_w = _rotation_between(v_w, np.array([0.0, 0.0, 1.0]))

    # Transformed bearing vectors and world points (in aligned frames)
    p1 = R_c @ p_i[:, 0]
    p2 = R_c @ p_i[:, 1]
    Pw1 = R_w @ p_w[:, 0]
    Pw2 = R_w @ p_w[:, 1]

    v = np.array([0.0, 0.0, 1.0])  # aligned gravity direction

    # World-frame distances
    dw_vec = Pw1 - Pw2
    D_sq = float(np.dot(dw_vec, dw_vec))

    # Axis-projection constraint → λ₁ = m + n·λ₂
    p1v = float(np.dot(p1, v))
    p2v = float(np.dot(p2, v))
    dw  = float(np.dot(dw_vec, v))        # projection of world diff on gravity

    if abs(p1v) < 1e-15:
        return []  # degenerate — both bearing vectors orthogonal to gravity

    m = dw / p1v
    n = p2v / p1v

    # Distance constraint → quadratic in λ₂
    cos12 = float(np.dot(p1, p2))

    A = n * n - 2.0 * n * cos12 + 1.0
    B = 2.0 * m * (n - cos12)
    C = m * m - D_sq

    disc = B * B - 4.0 * A * C

    solutions = []

    if abs(A) < 1e-15:
        if abs(B) > 1e-15:
            lam2_vals = [-C / B]
        else:
            return solutions
    elif disc < -1e-10:
        return solutions
    else:
        disc = max(0.0, disc)
        sd = np.sqrt(disc)
        lam2_vals = [(-B + sd) / (2.0 * A), (-B - sd) / (2.0 * A)]

    for lam2 in lam2_vals:
        lam1 = m + n * lam2
        if lam1 <= 1e-12 or lam2 <= 1e-12:
            continue

        # 3D points in the aligned camera frame
        Pc1 = lam1 * p1
        Pc2 = lam2 * p2

        # Rotation angle α about [0,0,1]:
        # R'(α) rotates dw_vec (in aligned world) to dc_vec (in aligned camera)
        dc_vec = Pc1 - Pc2

        # Project onto the x-y plane (perpendicular to gravity)
        dw_proj = dw_vec.copy(); dw_proj[2] = 0.0
        dc_proj = dc_vec.copy(); dc_proj[2] = 0.0

        dw_pn = np.linalg.norm(dw_proj)
        dc_pn = np.linalg.norm(dc_proj)

        if dw_pn < 1e-14 or dc_pn < 1e-14:
            # Degenerate: world-point difference lies along gravity axis
            alpha = 0.0
        else:
            cos_a = np.dot(dw_proj, dc_proj) / (dw_pn * dc_pn)
            sin_a = (dw_proj[0] * dc_proj[1] - dw_proj[1] * dc_proj[0]) / (dw_pn * dc_pn)
            alpha = np.arctan2(sin_a, cos_a)

        ca, sa = np.cos(alpha), np.sin(alpha)
        R_prime = np.array([[ca, -sa, 0.0],
                            [sa,  ca, 0.0],
                            [0.0, 0.0, 1.0]])

        # Full pose recovery
        R_full = R_c.T @ R_prime @ R_w
        t_full = R_c.T @ (Pc1 - R_prime @ Pw1)

        solutions.append(Model(rotation=R_full, translation=t_full))

    return solutions


# ---------------------------------------------------------------------------
#  Li et al. 2023 — rotation-based P2P solver
# ---------------------------------------------------------------------------

def li(p_w, p_i, v_cam, v_world):
    r"""Absolute pose from two 2D-3D correspondences with known gravity direction.

    **Reference:** B. Li *et al.*, "A Generalized 2-Point Solution for Absolute
    Camera Pose With Known Rotation Axis", IEEE Access 2023.

    The method formulates a quadratic in s = tan(α/2) directly from
    a cross-product constraint that eliminates both depth and translation,
    then recovers depth and translation by back-substitution.

    Parameters
    ----------
    p_w : ndarray, shape (3, 2)
        World-frame 3D points.
    p_i : ndarray, shape (3, 2)
        Unit bearing vectors in the camera frame.
    v_cam : ndarray, shape (3,)
        Unit gravity direction in the camera frame.
    v_world : ndarray, shape (3,)
        Unit gravity direction in the world frame.

    Returns
    -------
    list of Model
        Up to two solutions (R, t) with P_cam = R @ P_world + t.
    """
    v_c = v_cam / np.linalg.norm(v_cam)
    v_w = v_world / np.linalg.norm(v_world)

    # Pre-align so that gravity → [0, 0, 1] in both frames
    R_c = _rotation_between(v_c, np.array([0.0, 0.0, 1.0]))
    R_w = _rotation_between(v_w, np.array([0.0, 0.0, 1.0]))

    # Bearing vectors and world points in their aligned intermediate frames
    p1 = R_c @ p_i[:, 0]
    p2 = R_c @ p_i[:, 1]
    Pw1 = R_w @ p_w[:, 0]
    Pw2 = R_w @ p_w[:, 1]

    # ---- Quadratic in s = tan(α/2) from cross-product constraint ----
    # (R'·Δ)ᵀ (p₁ × p₂) = 0   where Δ = Pw1 − Pw2
    delta = Pw1 - Pw2                     # [Δx, Δy, Δz]
    p_cross = np.cross(p1, p2)            # p₁ × p₂ = [px, py, pz]

    dx, dy, dz = delta
    px, py, pz = p_cross

    # Coefficients (Eq. 11–12 in the paper)
    a1 = -dx * px - dy * py + dz * pz
    a2 =  2.0 * dx * py - 2.0 * dy * px
    a3 =  dx * px + dy * py + dz * pz

    # Solve  a1·s² + a2·s + a3 = 0
    solutions = []
    if abs(a1) < 1e-15:
        if abs(a2) > 1e-15:
            s_vals = [-a3 / a2]
        else:
            return solutions
    else:
        disc = a2 * a2 - 4.0 * a1 * a3
        if disc < -1e-10:
            return solutions
        disc = max(0.0, disc)
        sd = np.sqrt(disc)
        s_vals = [(-a2 + sd) / (2.0 * a1),
                  (-a2 - sd) / (2.0 * a1)]

    for s in s_vals:
        # Rotation angle
        alpha = 2.0 * np.arctan(s)
        ca, sa = np.cos(alpha), np.sin(alpha)

        R_prime = np.array([[ca, -sa, 0.0],
                            [sa,  ca, 0.0],
                            [0.0, 0.0, 1.0]])

        # ---- Depth recovery (Eq. 15-16) ----
        # λ₁·(p₂ × p₁) = p₂ × (R'·Δ)
        R_delta = R_prime @ delta
        lhs = np.cross(p2, p1)           # p₂ × p₁
        rhs = np.cross(p2, R_delta)      # p₂ × (R'·Δ)

        lhs_sq = float(np.dot(lhs, lhs))
        if lhs_sq < 1e-30:
            continue

        lam1 = float(np.dot(rhs, lhs)) / lhs_sq

        if lam1 <= 1e-12:
            continue

        # Check λ₂ > 0:  λ₁·p₁ − R'·Δ = λ₂·p₂  →  λ₂ = (λ₁·p₁ − R'·Δ)·p₂
        lam2 = float(np.dot(lam1 * p1 - R_delta, p2))
        if lam2 <= 1e-12:
            continue

        # ---- Translation (Eq. 17) ----
        t_prime = lam1 * p1 - R_prime @ Pw1

        # Full pose recovery (Eq. 7 with tc=0, tw=0)
        R_full = R_c.T @ R_prime @ R_w
        t_full = R_c.T @ t_prime

        solutions.append(Model(rotation=R_full, translation=t_full))

    return solutions
