import unittest
import numpy as np

from recipnps.model import Model


class TestModel(unittest.TestCase):
    def test_reprojection_cost(self):
        np.random.seed()
        t_gt = np.random.rand(3)
        r_gt, _ = np.linalg.qr(np.random.rand(3, 3))
        rotation_gt = r_gt

        p_src = np.random.rand(3, 3)
        p_tgt = rotation_gt @ p_src
        for i in range(p_tgt.shape[1]):
            p_tgt[:, i] += t_gt

        model_to_test = Model(rotation_gt, t_gt)
        cost = model_to_test.reprojection_dist_of(p_src[:, 0], p_tgt[:, 0])
        self.assertTrue(cost < 1e-7)


if __name__ == '__main__':
    unittest.main()
    