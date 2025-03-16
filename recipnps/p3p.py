import numpy as np

from .model import Model


def arun(p, p_prime):
    # find centroids
    p_centroid = np.mean(p, axis=1)
    p_prime_centroid = np.mean(p_prime, axis=1)

    # calculate the vectors from centroids
    q = p - p_centroid[:, None]
    q_prime = p_prime - p_prime_centroid[:, None]

    # rotation estimation
    H = q @ q_prime.T
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    
    # translation estimation
    t = p_prime_centroid - R @ p_centroid
    
    results = list()
    results.append(Model(rotation=R, translation=t))
    return results


"""
B. M. Haralick, C.-N. Lee, K. Ottenberg, and M. Nölle, “Review and analysis of solutions of the three point perspective pose estimation problem,” Int J Comput Vision, vol. 13, no. 3, pp. 331–356, Dec. 1994, doi: 10.1007/BF02028352.
"""
def grunert(p_w, p_i):
    # 1. Calculate the known lengths of p_w
    p1 = p_w[:, 0]
    p2 = p_w[:, 1]
    p3 = p_w[:, 2]
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    a_sq = a ** 2
    b_sq = b ** 2
    c_sq = c ** 2

    # 2. Get directional vectors j_i (j_i points to p_w(i))
    assert np.abs(np.linalg.norm(p_i[:, 0]) - 1.0) < 1e-7
    assert np.abs(np.linalg.norm(p_i[:, 1]) - 1.0) < 1e-7
    assert np.abs(np.linalg.norm(p_i[:, 2]) - 1.0) < 1e-7
    j1 = p_i[:, 0]
    j2 = p_i[:, 1]
    j3 = p_i[:, 2]

    # 3. Calculate cos(alpha) cos(beta) cos(gamma)
    # note: cosines need to be within [-1, 1]
    cos_alpha = np.dot(j2, j3)
    cos_beta = np.dot(j1, j3)
    cos_gamma = np.dot(j1, j2)
    cos_alpha_sq = cos_alpha ** 2
    cos_beta_sq = cos_beta ** 2
    cos_gamma_sq = cos_gamma ** 2

    # 4. Solve polynomial
    a_sq_minus_c_sq_div_b_sq = (a_sq - c_sq) / b_sq
    a_sq_plus_c_sq_div_b_sq = (a_sq + c_sq) / b_sq
    b_sq_minus_c_sq_div_b_sq = (b_sq - c_sq) / b_sq
    b_sq_minus_a_sq_div_b_sq = (b_sq - a_sq) / b_sq

    a4 = (a_sq_minus_c_sq_div_b_sq - 1.0) ** 2 - 4.0 * c_sq / b_sq * cos_alpha_sq
    a3 = 4.0 * (a_sq_minus_c_sq_div_b_sq * (1.0 - a_sq_minus_c_sq_div_b_sq) * cos_beta
                - (1.0 - a_sq_plus_c_sq_div_b_sq) * cos_alpha * cos_gamma
                + 2.0 * c_sq / b_sq * cos_alpha_sq * cos_beta)
    a2 = 2.0 * ((a_sq_minus_c_sq_div_b_sq) ** 2 - 1.0
                + 2.0 * (a_sq_minus_c_sq_div_b_sq) ** 2 * cos_beta_sq
                + 2.0 * (b_sq_minus_c_sq_div_b_sq) * cos_alpha_sq
                - 4.0 * (a_sq_plus_c_sq_div_b_sq) * cos_alpha * cos_beta * cos_gamma
                + 2.0 * (b_sq_minus_a_sq_div_b_sq) * cos_gamma_sq)
    a1 = 4.0 * (-(a_sq_minus_c_sq_div_b_sq) * (1.0 + a_sq_minus_c_sq_div_b_sq) * cos_beta
                + 2.0 * a_sq / b_sq * cos_gamma_sq * cos_beta
                - (1.0 - (a_sq_plus_c_sq_div_b_sq)) * cos_alpha * cos_gamma)
    a0 = (1.0 + a_sq_minus_c_sq_div_b_sq) ** 2 - 4.0 * a_sq / b_sq * cos_gamma_sq

    def get_points_in_cam_frame_from_v(v):
        u = ((-1.0 + a_sq_minus_c_sq_div_b_sq) * v ** 2
             - 2.0 * (a_sq_minus_c_sq_div_b_sq) * cos_beta * v + 1.0 + a_sq_minus_c_sq_div_b_sq) / (2.0 * (cos_gamma - v * cos_alpha))
        s1 = np.sqrt(c_sq / (1.0 + u ** 2 - 2.0 * u * cos_gamma))
        s2 = u * s1
        s3 = v * s1
        p_cam = np.column_stack((s1 * j1, s2 * j2, s3 * j3))
        return p_cam

    all_roots = np.roots([a4, a3, a2, a1, a0])
    results = []
    for root in all_roots:
        p_cam = get_points_in_cam_frame_from_v(root)
        results.append(arun(p_w, p_cam)[0])

    return results


"""
B. M. Haralick, C.-N. Lee, K. Ottenberg, and M. Nölle, “Review and analysis of solutions of the three point perspective pose estimation problem,” Int J Comput Vision, vol. 13, no. 3, pp. 331–356, Dec. 1994, doi: 10.1007/BF02028352.
"""
def fischler(p_w, p_i):
    # 1. Calculate the known lengths of p_w
    p1 = p_w[:, 0]
    p2 = p_w[:, 1]
    p3 = p_w[:, 2]
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    a_sq = a ** 2
    b_sq = b ** 2
    c_sq = c ** 2

    # 2. Get directional vectors j_i (j_i points to p_w(i))
    assert np.abs(np.linalg.norm(p_i[:, 0]) - 1.0) < 1e-7
    assert np.abs(np.linalg.norm(p_i[:, 1]) - 1.0) < 1e-7
    assert np.abs(np.linalg.norm(p_i[:, 2]) - 1.0) < 1e-7
    j1 = p_i[:, 0]
    j2 = p_i[:, 1]
    j3 = p_i[:, 2]

    # 3. Calculate cos(alpha) cos(beta) cos(gamma)
    # note: cosines need to be within [-1, 1]
    cos_alpha = np.dot(j2, j3)
    cos_beta = np.dot(j1, j3)
    cos_gamma = np.dot(j1, j2)
    cos_alpha_sq = cos_alpha ** 2
    cos_beta_sq = cos_beta ** 2
    cos_gamma_sq = cos_gamma ** 2

    # 4. Solve polynomial
    d4 = 4 * b_sq * c_sq * cos_alpha_sq - (a_sq - b_sq - c_sq) ** 2
    d3 = -4 * c_sq * (a_sq + b_sq - c_sq) * cos_alpha * cos_beta \
         - 8 * b_sq * c_sq * cos_alpha_sq * cos_gamma \
         + 4 * (a_sq - b_sq - c_sq) * (a_sq - b_sq) * cos_gamma
    d2 = 4 * c_sq * (a_sq - c_sq) * cos_beta_sq \
         + 8 * c_sq * (a_sq + b_sq) * cos_alpha * cos_beta * cos_gamma \
         + 4 * c_sq * (b_sq - c_sq) * cos_alpha_sq \
         - 2 * (a_sq - b_sq - c_sq) * (a_sq - b_sq + c_sq) \
         - 4 * (a_sq - b_sq) ** 2 * cos_gamma_sq
    d1 = -8 * a_sq * c_sq * cos_beta_sq * cos_gamma \
         - 4 * c_sq * (b_sq - c_sq) * cos_alpha * cos_beta \
         - 4 * a_sq * c_sq * cos_alpha * cos_beta \
         + 4 * (a_sq - b_sq) * (a_sq - b_sq + c_sq) * cos_gamma
    d0 = 4 * a_sq * c_sq * cos_beta_sq - (a_sq - b_sq + c_sq) ** 2

    def get_points_in_cam_frame_from_u(u):
        v = (-(a_sq - b_sq - c_sq) * u ** 2 - 2 * (b_sq - a_sq) * cos_gamma * u
             - a_sq + b_sq - c_sq) / (2 * c_sq * (cos_alpha * u - cos_beta))
        s1 = np.sqrt(c_sq / (1.0 + u ** 2 - 2.0 * u * cos_gamma))
        s2 = u * s1
        s3 = v * s1
        p_cam = np.column_stack((s1 * j1, s2 * j2, s3 * j3))
        return p_cam

    all_roots = np.roots([d4, d3, d2, d1, d0])
    results = []
    for root in all_roots:
        p_cam = get_points_in_cam_frame_from_u(root)
        results.append(arun(p_w, p_cam)[0])

    return results


"""
L. Kneip, D. Scaramuzza, and R. Siegwart, “A novel parametrization of the perspective-three-point problem for a direct computation of absolute camera position and orientation,” in CVPR 2011, Jun. 2011, pp. 2969–2976. doi: 10.1109/CVPR.2011.5995464.
"""
def kneip(p_w, p_i):
    p1 = p_w[:, 0]
    p2 = p_w[:, 1]
    p3 = p_w[:, 2]
    if np.linalg.norm(np.cross(p2 - p1, p3 - p1)) == 0.0:
        return []

    assert np.abs(np.linalg.norm(p_i[:, 0]) - 1.0) < 1e-7
    assert np.abs(np.linalg.norm(p_i[:, 1]) - 1.0) < 1e-7
    assert np.abs(np.linalg.norm(p_i[:, 2]) - 1.0) < 1e-7
    f1_og = p_i[:, 0]
    f2_og = p_i[:, 1]
    f3_og = p_i[:, 2]
    f1 = f1_og.copy()
    f2 = f2_og.copy()
    f3 = f3_og.copy()

    e1 = f1 / np.linalg.norm(f1)
    e3 = np.cross(f1, f2) / np.linalg.norm(np.cross(f1, f2))
    e2 = np.cross(e3, e1)
    tt = np.vstack((e1, e2, e3))

    f3 = tt @ f3
    if f3[2] > 0.0:
        f1 = f2_og
        f2 = f1_og
        f3 = f3_og

        e1 = f1 / np.linalg.norm(f1)
        e3 = np.cross(f1, f2) / np.linalg.norm(np.cross(f1, f2))
        e2 = np.cross(e3, e1)
        tt = np.vstack((e1, e2, e3))

        f3 = tt @ f3

        p1 = p_w[:, 1]
        p2 = p_w[:, 0]
        p3 = p_w[:, 2]

    n1 = (p2 - p1) / np.linalg.norm(p2 - p1)
    n3 = np.cross(n1, p3 - p1) / np.linalg.norm(np.cross(n1, p3 - p1))
    n2 = np.cross(n3, n1)

    nn = np.vstack((n1, n2, n3))
    p3 = nn @ (p3 - p1)

    d_12 = np.linalg.norm(p2 - p1)
    f_1 = f3[0] / f3[2]
    f_2 = f3[1] / f3[2]
    p_1 = p3[0]
    p_2 = p3[1]

    cos_beta = np.dot(f1, f2)
    b = 1.0 / (1.0 - cos_beta ** 2) - 1.0
    if cos_beta < 0.0:
        b = -np.sqrt(b)
    else:
        b = np.sqrt(b)

    f_1_pw2 = f_1 ** 2
    f_2_pw2 = f_2 ** 2
    p_1_pw2 = p_1 ** 2
    p_1_pw3 = p_1_pw2 * p_1
    p_1_pw4 = p_1_pw3 * p_1
    p_2_pw2 = p_2 ** 2
    p_2_pw3 = p_2_pw2 * p_2
    p_2_pw4 = p_2_pw3 * p_2
    d_12_pw2 = d_12 ** 2
    b_pw2 = b ** 2

    a4 = -f_2_pw2 * p_2_pw4 - p_2_pw4 * f_1_pw2 - p_2_pw4
    a3 = 2.0 * p_2_pw3 * d_12 * b + 2.0 * f_2_pw2 * p_2_pw3 * d_12 * b \
         - 2.0 * f_2 * p_2_pw3 * f_1 * d_12
    a2 = -f_2_pw2 * p_2_pw2 * p_1_pw2 \
         - f_2_pw2 * p_2_pw2 * d_12_pw2 * b_pw2 \
         - f_2_pw2 * p_2_pw2 * d_12_pw2 \
         + f_2_pw2 * p_2_pw4 \
         + p_2_pw4 * f_1_pw2 \
         + 2.0 * p_1 * p_2_pw2 * d_12 \
         + 2.0 * f_1 * f_2 * p_1 * p_2_pw2 * d_12 * b \
         - p_2_pw2 * p_1_pw2 * f_1_pw2 \
         + 2.0 * p_1 * p_2_pw2 * f_2_pw2 * d_12 \
         - p_2_pw2 * d_12_pw2 * b_pw2 \
         - 2.0 * p_1_pw2 * p_2_pw2
    a1 = 2.0 * p_1_pw2 * p_2 * d_12 * b \
         + 2.0 * f_2 * p_2_pw3 * f_1 * d_12 \
         - 2.0 * f_2_pw2 * p_2_pw3 * d_12 * b \
         - 2.0 * p_1 * p_2 * d_12_pw2 * b
    a0 = -2.0 * f_2 * p_2_pw2 * f_1 * p_1 * d_12 * b \
         + f_2_pw2 * p_2_pw2 * d_12_pw2 \
         + 2.0 * p_1_pw3 * d_12 \
         - p_1_pw2 * d_12_pw2 \
         + f_2_pw2 * p_2_pw2 * p_1_pw2 \
         - p_1_pw4 \
         - 2.0 * f_2_pw2 * p_2_pw2 * p_1 * d_12 \
         + p_2_pw2 * f_1_pw2 * p_1_pw2 \
         + f_2_pw2 * p_2_pw2 * d_12_pw2 * b_pw2
         
    all_roots = np.roots([a4, a3, a2, a1, a0])
    results = []
    for cos_theta in all_roots:
        cot_alpha = (-f_1 * p_1 / f_2 - cos_theta * p_2 + d_12 * b) / (-f_1 * cos_theta * p_2 / f_2 + p_1 - d_12)
        sin_theta = np.sqrt(1.0 - cos_theta ** 2)
        sin_alpha = np.sqrt(1.0 / (cot_alpha ** 2 + 1.0))
        cos_alpha = np.sqrt(1.0 - sin_alpha ** 2)
        if cot_alpha < 0.0:
            cos_alpha = -cos_alpha

        rr = np.array([
            [-cos_alpha, -sin_alpha * cos_theta, -sin_alpha * sin_theta],
            [sin_alpha, -cos_alpha * cos_theta, -cos_alpha * sin_theta],
            [0.0, -sin_theta, cos_theta]
        ])
        rotation_est = tt.T @ rr @ nn

        t = np.array([
            d_12 * cos_alpha * (sin_alpha * b + cos_alpha),
            cos_theta * d_12 * sin_alpha * (sin_alpha * b + cos_alpha),
            sin_theta * d_12 * sin_alpha * (sin_alpha * b + cos_alpha)
        ])
        t = p1 + nn.T @ t
        t_est = -rotation_est @ t

        results.append(Model(rotation=rotation_est, translation=t_est))

    return results

