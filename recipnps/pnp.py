from .sac import ransac
from .p3p import grunert, fischler, kneip


def pnp_ransac_fischler(world_points, bearing_vectors, max_iterations, inlier_dist_threshold, probability):
    return ransac(world_points, bearing_vectors, fischler, max_iterations, inlier_dist_threshold, probability)


def pnp_ransac_grunert(world_points, bearing_vectors, max_iterations, inlier_dist_threshold, probability):
    return ransac(world_points, bearing_vectors, grunert, max_iterations, inlier_dist_threshold, probability)


def pnp_ransac_kneip(world_points, bearing_vectors, max_iterations, inlier_dist_threshold, probability):
    return ransac(world_points, bearing_vectors, kneip, max_iterations, inlier_dist_threshold, probability)

