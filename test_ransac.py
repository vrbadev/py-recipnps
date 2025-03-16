import unittest
import numpy as np

from recipnps.sac import ransac
from recipnps.p3p import fischler, grunert, kneip


class TestRansac(unittest.TestCase):
    def test_ransac(self):
        np.random.seed()
        t_gt = np.random.rand(3)
        rotation_gt, _ = np.linalg.qr(np.random.rand(3, 3))

        n_points = 10
        p_src = np.random.rand(3, n_points)
        p_tgt = rotation_gt @ p_src
        for i in range(p_tgt.shape[1]):
            p_tgt[:, i] += t_gt

        n_outliers = 1
        for i in range(n_outliers):
            p_tgt[:, i] += np.array([10.0 * (i + 1), -10.0 * (i + 1), 0.0])

        p_tgt /= np.linalg.norm(p_tgt, axis=0)

        result = ransac(p_src, p_tgt, grunert, 50, 0.1, 0.95)
        self.assertIsNotNone(result)
        self.assertTrue(np.allclose(result.rotation, rotation_gt, atol=1e-7))
        self.assertTrue(np.allclose(result.translation, t_gt, atol=1e-7))


if __name__ == '__main__':
    unittest.main()
    