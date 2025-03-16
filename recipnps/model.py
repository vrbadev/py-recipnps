import numpy as np


class Model:
    def __init__(self, rotation, translation):
        self.rotation = rotation
        self.translation = translation

    def reprojection_dist_of(self, world_point, bearing_vector):
        reprojection = (self.rotation @ world_point + self.translation) / np.linalg.norm(self.rotation @ world_point + self.translation)
        return 1.0 - np.dot(reprojection, bearing_vector / np.linalg.norm(bearing_vector))

    def reprojection_dists_of(self, world_points, bearing_vectors):
        reprojections = self.rotation @ world_points
        for i in range(reprojections.shape[1]):
            reprojections[:, i] += self.translation
            reprojections[:, i] /= np.linalg.norm(reprojections[:, i])

        cos_angles = np.einsum('ij,ij->j', reprojections, bearing_vectors)
        dists = 1.0 - cos_angles
        return dists
    