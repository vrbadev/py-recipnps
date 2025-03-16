import unittest
import numpy as np

from recipnps.p3p import arun, fischler, grunert, kneip


class TestP3P(unittest.TestCase):
    def random_test_case(self):
        np.random.seed()
        p_cam = np.random.rand(3, 3)
        p_cam /= np.linalg.norm(p_cam, axis=0)

        t_gt = np.random.rand(3)
        rotation_gt, _ = np.linalg.qr(np.random.rand(3, 3))

        p_world = np.linalg.inv(rotation_gt) @ (p_cam - t_gt[:, None])
        return p_cam, p_world, t_gt, rotation_gt
    
    def fixed_test_case(self):
        p_cam = np.array([[0.6453839206344928, 0.6551415125227705, 0.39277117200200334], [0.5027152459826345, 0.836171970335917, 0.21930302661196782], [0.9908391145824398, 0.08503578769276794, 0.10491312513197464]]).T
        t_gt = np.array([0.17216231844648533, 0.8968470516910476, 0.7639868514400336])
        rotation_gt = np.array([[-0.5871671330204742, -0.6943436357845219, 0.4160789268228421], [0.5523730621371405, 0.032045330116620696, 0.8329808503458861], [-0.5917083387326525, 0.7189297686584192, 0.364720755662461]]).T
        p_world = rotation_gt.T @ (p_cam - t_gt[:, None])
        return p_cam, p_world, t_gt, rotation_gt

    def test_arun(self):
        p_cam, p_world, t_gt, rotation_gt = self.random_test_case()
        p_i = p_cam / np.linalg.norm(p_cam, axis=0)
        solutions = arun(p_world, p_i)
        flag = any(np.allclose(sol.translation, t_gt, atol=1e-7) and np.allclose(sol.rotation, rotation_gt, atol=1e-7) for sol in solutions)
        self.assertTrue(flag)

    def test_grunert(self):
        p_cam, p_world, t_gt, rotation_gt = self.random_test_case()
        p_i = p_cam / np.linalg.norm(p_cam, axis=0)
        solutions = grunert(p_world, p_i)
        flag = any(np.allclose(sol.translation, t_gt, atol=1e-7) and np.allclose(sol.rotation, rotation_gt, atol=1e-7) for sol in solutions)
        self.assertTrue(flag)

    def test_fischler(self):
        p_cam, p_world, t_gt, rotation_gt = self.random_test_case()
        p_i = p_cam / np.linalg.norm(p_cam, axis=0)
        solutions = fischler(p_world, p_i)
        flag = any(np.allclose(sol.translation, t_gt, atol=1e-7) and np.allclose(sol.rotation, rotation_gt, atol=1e-7) for sol in solutions)
        self.assertTrue(flag)

    def test_kneip(self):
        p_cam, p_world, t_gt, rotation_gt = self.random_test_case()
        p_i = p_cam / np.linalg.norm(p_cam, axis=0)
        solutions = kneip(p_world, p_i)
        flag = any(np.allclose(sol.translation, t_gt, atol=1e-7) and np.allclose(sol.rotation, rotation_gt, atol=1e-7) for sol in solutions)
        self.assertTrue(flag)


if __name__ == '__main__':
    unittest.main()
