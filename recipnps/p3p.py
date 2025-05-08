import numpy as np

from .model import Model
from numpy.linalg import svd


def arun(p, p_prime):
    # find centroids
    p_centroid = np.mean(p, axis=1)
    p_prime_centroid = np.mean(p_prime, axis=1)

    # calculate the vectors from centroids
    q = p - p_centroid[:, None]
    q_prime = p_prime - p_prime_centroid[:, None]

    # rotation estimation
    H = q @ q_prime.T
    
    U, _, Vt = svd(H)
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
B. M. Haralick, C.-N. Lee, K. Ottenberg, and M. Nölle, “Review and analysis of solutions of the three point perspective pose estimation problem,” Int J Comput Vision, vol. 13, no. 3, pp. 331-356, Dec. 1994, doi: 10.1007/BF02028352.
"""
def grunert(p_w, p_i):
    # p_w: 3x3 world points, p_i: 3x3 bearing vectors (normalized)
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
B. M. Haralick, C.-N. Lee, K. Ottenberg, and M. Nölle, “Review and analysis of solutions of the three point perspective pose estimation problem,” Int J Comput Vision, vol. 13, no. 3, pp. 331-356, Dec. 1994, doi: 10.1007/BF02028352.
"""
def fischler(p_w, p_i):
    # p_w: 3x3 world points, p_i: 3x3 bearing vectors (normalized)
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
L. Kneip, D. Scaramuzza, and R. Siegwart, “A novel parametrization of the perspective-three-point problem for a direct computation of absolute camera position and orientation,” in CVPR 2011, Jun. 2011, pp. 2969-2976. doi: 10.1109/CVPR.2011.5995464.
"""
def kneip(p_w, p_i):
    # p_w: 3x3 world points, p_i: 3x3 bearing vectors (normalized)
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


"""
G. Nakano, “A Simple Direct Solution to the Perspective-Three-Point Problem,” in 30th British Machine Vision Conference 2019, BMVC 2019, Cardiff, UK, September 9-12, 2019. pages 26, BMVA Press, 2019.
"""
def nakano(p_w, p_i, polishing=1):
    # p_w: 3x3 world points, p_i: 3x3 bearing vectors (normalized)
    # polishing: int, number of root polishing iterations (default 1).
    d = np.linalg.norm(np.column_stack([p_w[:,1]-p_w[:,0], p_w[:,2]-p_w[:,1], p_w[:,0]-p_w[:,2]]), axis=0)
    idx = np.argmax(d)
    if idx == 1:
        p_w = p_w[:,[1,2,0]]
        p_i = p_i[:,[1,2,0]]
    elif idx == 2:
        p_w = p_w[:,[0,2,1]]
        p_i = p_i[:,[0,2,1]]

    # Rigid transformation: all points on plane z=0
    X21 = p_w[:,1] - p_w[:,0]
    X31 = p_w[:,2] - p_w[:,0]
    nx = X21 / np.linalg.norm(X21)
    nz = np.cross(nx, X31)
    nz = nz / np.linalg.norm(nz)
    ny = np.cross(nz, nx)
    N = np.column_stack((nx, ny, nz))

    a = N[:,0].dot(X21)
    b = N[:,0].dot(X31)
    c = N[:,1].dot(X31)

    M12 = p_i[:,0].dot(p_i[:,1])
    M13 = p_i[:,0].dot(p_i[:,2])
    M23 = p_i[:,1].dot(p_i[:,2])
    p = b/a
    q = (b**2 + c**2) / a**2

    f = np.array([p, -M23, 0, -M12*(2*p-1), M13, p-1])
    g = np.array([q, 0, -1, -2*M12*q, 2*M13, q-1])

    h = np.zeros(5)
    h[0] = -f[0]**2 + g[0]*f[1]**2
    h[1] = f[1]**2*g[3] - 2*f[0]*f[3] - 2*f[0]*f[1]*f[4] + 2*f[1]*f[4]*g[0]
    h[2] = f[4]**2*g[0] - 2*f[0]*f[4]**2 - 2*f[0]*f[5] + f[1]**2*g[5] - f[3]**2 - 2*f[1]*f[3]*f[4] + 2*f[1]*f[4]*g[3]
    h[3] = f[4]**2*g[3] - 2*f[3]*f[4]**2 - 2*f[3]*f[5] - 2*f[1]*f[4]*f[5] + 2*f[1]*f[4]*g[5]
    h[4] = -2*f[4]**2*f[5] + g[5]*f[4]**2 - f[5]**2

    x_roots = np.roots(h)
    # Only keep real, positive roots (imaginary part < 1e-8)
    x_roots = x_roots[np.abs(np.imag(x_roots)) < 1e-8]
    x_roots = np.real(x_roots)
    x_roots = x_roots[x_roots > 0]

    # Compute y for each x
    denom = (f[4] + f[1]*x_roots)
    # Avoid division by zero
    valid = np.abs(denom) > 1e-12
    x_roots = x_roots[valid]
    denom = denom[valid]
    y_roots = - ((f[0]*x_roots + f[3])*x_roots + f[5]) / denom

    # Inline root polishing (Gauss-Newton)
    if polishing > 0 and len(x_roots) > 0:
        x = np.array(x_roots, dtype=np.float64)
        y = np.array(y_roots, dtype=np.float64)
        for _ in range(polishing):
            x2 = x**2
            xy = x*y
            y2 = y**2

            fv = f[0]*x2 + f[1]*xy + f[3]*x + f[4]*y + f[5]
            gv = g[0]*x2 - y2 + g[3]*x + g[4]*y + g[5]

            dfdx = 2*f[0]*x + f[1]*y + f[3]
            dfdy = f[1]*x + f[4]
            dgdx = 2*g[0]*x + g[3]
            dgdy = -2*y + g[4]

            detJ = dfdx*dgdy - dfdy*dgdx
            mask = np.abs(detJ) > 1e-15
            dx = np.zeros_like(x)
            dy = np.zeros_like(y)
            if np.any(mask):
                inv_detJ = 1.0 / detJ[mask]
                dx[mask] = (dgdy[mask]*fv[mask] - dfdy[mask]*gv[mask]) * inv_detJ
                dy[mask] = (-dgdx[mask]*fv[mask] + dfdx[mask]*gv[mask]) * inv_detJ
            x = x - dx
            y = y - dy
        x_roots, y_roots = x, y

    results = []
    nsols = len(x_roots)
    A = p_i @ np.diag([-1, 1, 0])
    B = p_i @ np.diag([-1, 0, 1])
    C = B - p*A

    for i in range(nsols):
        lam = np.array([1, x_roots[i], y_roots[i]])
        s = np.linalg.norm(A @ lam) / a
        d = lam / s

        r1 = (A @ d) / a
        r2 = (C @ d) / c
        r3 = np.cross(r1, r2)
        # Ensure right-handed coordinate system
        if np.dot(np.cross(r1, r2), r3) < 0:
            r3 = -r3
        Rc = np.column_stack((r1, r2, r3))
        tc = d[0]*p_i[:,0]

        R = Rc @ N.T
        t = tc - R @ p_w[:,0]
        results.append(Model(rotation=R, translation=t))
    return results

"""
M. Persson and K. Nordberg, “Lambda Twist: An Accurate Fast Robust Perspective Three Point (P3P) Solver,” Lecture Notes in Computer Science. Springer International Publishing, pp. 334-349, 2018. doi: 10.1007/978-3-030-01225-0_20. 
"""
def lambdatwist(p_w, p_i, polishing=True):
    # p_w: 3x3 world points, p_i: 3x3 bearing vectors (normalized)
    y1 = p_i[:, 0] / np.linalg.norm(p_i[:, 0])
    y2 = p_i[:, 1] / np.linalg.norm(p_i[:, 1])
    y3 = p_i[:, 2] / np.linalg.norm(p_i[:, 2])

    b12 = -2 * np.dot(y1, y2)
    b13 = -2 * np.dot(y1, y3)
    b23 = -2 * np.dot(y2, y3)

    x1 = p_w[:, 0]
    x2 = p_w[:, 1]
    x3 = p_w[:, 2]

    d12 = x1 - x2
    d13 = x1 - x3
    d23 = x2 - x3
    d12xd13 = np.cross(d12, d13)

    a12 = np.dot(d12, d12)
    a13 = np.dot(d13, d13)
    a23 = np.dot(d23, d23)

    c31 = -0.5 * b13
    c23 = -0.5 * b23
    c12 = -0.5 * b12
    blob = c12 * c23 * c31 - 1.0

    s31_squared = 1.0 - c31 * c31
    s23_squared = 1.0 - c23 * c23
    s12_squared = 1.0 - c12 * c12

    p3 = a13 * (a23 * s31_squared - a13 * s23_squared)
    p2 = 2.0 * blob * a23 * a13 + a13 * (2.0 * a12 + a13) * s23_squared + a23 * (a23 - a12) * s31_squared
    p1 = a23 * (a13 - a23) * s12_squared - a12 * a12 * s23_squared - 2.0 * a12 * (blob * a23 + a13 * s23_squared)
    p0 = a12 * (a12 * s23_squared - a23 * s12_squared)

    # Cubic root selection as in C++/MATLAB
    if abs(p3) >= abs(p0):
        p3inv = 1.0 / p3
        p2 *= p3inv
        p1 *= p3inv
        p0 *= p3inv
        # Cubic root selection (see cubick in MATLAB/C++)
        coeffs = [1.0, p2, p1, p0]
        roots = np.roots(coeffs)
        # Select real root with largest gradient (sharpest)
        real_roots = roots[np.abs(np.imag(roots)) < 1e-8].real
        if len(real_roots) == 0:
            return []
        g = real_roots[np.argmax(np.abs(3*real_roots**2 + 2*p2*real_roots + p1))]
    else:
        # Lower numerical performance branch
        p0inv = 1.0 / p0
        p1 *= p0inv
        p2 *= p0inv
        p3 *= p0inv
        coeffs = [1.0, p1, p2, p3]
        roots = np.roots(coeffs)
        real_roots = roots[np.abs(np.imag(roots)) < 1e-8].real
        if len(real_roots) == 0:
            return []
        g = 1.0 / real_roots[np.argmax(np.abs(3*real_roots**2 + 2*p1*real_roots + p2))]

    # Build matrix A as in MATLAB/C++
    A00 = a23 * (1.0 - g)
    A01 = (a23 * b12) * 0.5
    A02 = (a23 * b13 * g) * (-0.5)
    A11 = a23 - a12 + a13 * g
    A12 = b23 * (a13 * g - a12) * 0.5
    A22 = g * (a13 - a23) - a12
    A = np.array([[A00, A01, A02],
                  [A01, A11, A12],
                  [A02, A12, A22]])

    # Eigen decomposition with one eigenvalue known to be zero
    # (see eigwithknown0 in MATLAB)
    # The zero eigenvector is the nullspace of A
    v3 = np.array([
        A[0,1]*A[1,2] - A[0,2]*A[1,1],
        A[0,2]*A[0,1] - A[0,0]*A[1,2],
        A[0,0]*A[1,1] - A[0,1]*A[0,1]
    ])
    v3 = v3 / np.linalg.norm(v3)
    # The other two eigenvalues (roots of quadratic)
    b = -A[0,0] - A[1,1] - A[2,2]
    c = -A[0,1]**2 - A[0,2]**2 - A[1,2]**2 + A[0,0]*(A[1,1] + A[2,2]) + A[1,1]*A[2,2]
    quad_roots = np.roots([1.0, b, c])
    e1, e2 = quad_roots
    if abs(e1) < abs(e2):
        e1, e2 = e2, e1
    L = np.array([e1, e2, 0.0])
    # Eigenvectors for e1, e2
    def eigvec_for_eig(A, e):
        tmp = 1.0 / (e*(A[0,0] + A[1,1]) - A[0,0]*A[1,1] - e*e + A[0,1]**2)
        a1 = -(e*A[0,2] + (A[0,1]*A[1,2] - A[0,2]*A[1,1])) * tmp
        a2 = -(e*A[1,2] + (A[0,1]*A[0,2] - A[0,0]*A[1,2])) * tmp
        rnorm = 1.0 / np.sqrt(a1*a1 + a2*a2 + 1.0)
        return np.array([a1*rnorm, a2*rnorm, rnorm])
    v1 = eigvec_for_eig(A, e1)
    v2 = eigvec_for_eig(A, e2)
    V = np.column_stack((v1, v2, v3))

    v = np.sqrt(max(0, -L[1]/L[0])) if L[0] != 0 else 0.0
    solutions = []
    for s in [v, -v]:
        w2 = 1.0 / (s*V[0,1] - V[0,0]) if abs(s*V[0,1] - V[0,0]) > 1e-12 else None
        if w2 is None:
            continue
        w0 = (V[1,0] - s*V[1,1]) * w2
        w1 = (V[2,0] - s*V[2,1]) * w2

        a = 1.0 / ((a13 - a12)*w1*w1 - a12*b13*w1 - a12)
        b_ = (a13*b12*w1 - a12*b13*w0 - 2*w0*w1*(a12 - a13)) * a
        c_ = ((a13 - a12)*w0*w0 + a13*b12*w0 + a13) * a

        tau_roots = np.roots([1.0, b_, c_])
        for tau in tau_roots:
            if np.abs(np.imag(tau)) > 1e-8 or tau.real <= 0:
                continue
            tau = tau.real
            d = a23 / (tau*(b23 + tau) + 1.0)
            if d <= 0:
                continue
            l2 = np.sqrt(d)
            l3 = tau * l2
            l1 = w0*l2 + w1*l3
            if l1 < 0:
                continue
            Ls = np.array([l1, l2, l3])
            # Optional root polishing (Gauss-Newton)
            if polishing:
                for _ in range(5):
                    l1, l2, l3 = Ls
                    r1 = l1*l1 + l2*l2 + b12*l1*l2 - a12
                    r2 = l1*l1 + l3*l3 + b13*l1*l3 - a13
                    r3 = l2*l2 + l3*l3 + b23*l2*l3 - a23
                    if abs(r1) + abs(r2) + abs(r3) < 1e-10:
                        break
                    dr1dl1 = 2*l1 + b12*l2
                    dr1dl2 = 2*l2 + b12*l1
                    dr2dl1 = 2*l1 + b13*l3
                    dr2dl3 = 2*l3 + b13*l1
                    dr3dl2 = 2*l2 + b23*l3
                    dr3dl3 = 2*l3 + b23*l2
                    J = np.array([
                        [dr1dl1, dr1dl2, 0],
                        [dr2dl1, 0, dr2dl3],
                        [0, dr3dl2, dr3dl3]
                    ])
                    r = np.array([r1, r2, r3])
                    try:
                        delta = np.linalg.lstsq(J, r, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        break
                    Ls = Ls - delta
            l1, l2, l3 = Ls
            if l1 < 0 or l2 < 0 or l3 < 0:
                continue
            # Compute rotation and translation
            ry1 = y1 * l1
            ry2 = y2 * l2
            ry3 = y3 * l3
            yd1 = ry1 - ry2
            yd2 = ry1 - ry3
            yd1xd2 = np.cross(yd1, yd2)
            Y = np.column_stack((yd1, yd2, yd1xd2))
            X = np.column_stack((d12, d13, d12xd13))
            Xinv = np.linalg.inv(X)
            R = Y @ Xinv
            t = ry1 - R @ x1
            solutions.append(Model(rotation=R, translation=t))
    return solutions




