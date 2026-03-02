import unittest
import numpy as np
import time

from recipnps.p4p import p4p_distance, p4p_liu_wong, lehavi_p4p, lehavi_p4p_algebraic


def _random_rotation(rng):
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


class TestP4P(unittest.TestCase):
    def _make_case(self, seed=time.time_ns()):
        rng = np.random.default_rng(seed)

        while True:
            p_world = rng.uniform(-1.0, 1.0, size=(3, 4))
            if np.linalg.matrix_rank(p_world - p_world[:, [0]]) >= 3:
                break

        rotation_gt = _random_rotation(rng)
        translation_gt = np.array([0.2, -0.3, 3.0]) + rng.uniform(-0.2, 0.2, size=3)

        p_cam = rotation_gt @ p_world + translation_gt[:, None]
        p_i = p_cam / np.linalg.norm(p_cam, axis=0)
        return p_world, p_i, translation_gt, rotation_gt

    def _make_coplanar_case(self, seed=time.time_ns()):
        rng = np.random.default_rng(seed)

        while True:
            xy = rng.uniform(-1.0, 1.0, size=(2, 4))
            z = np.zeros((1, 4), dtype=np.float64)
            p_world = np.vstack((xy, z))
            if np.linalg.matrix_rank(p_world - p_world[:, [0]]) >= 2:
                break

        rotation_gt = _random_rotation(rng)
        translation_gt = np.array([0.1, 0.2, 3.0]) + rng.uniform(-0.2, 0.2, size=3)
        p_cam = rotation_gt @ p_world + translation_gt[:, None]
        p_i = p_cam / np.linalg.norm(p_cam, axis=0)
        return p_world, p_i, translation_gt, rotation_gt

    def _rotation_geodesic_error(self, r_est, r_gt):
        delta = r_est @ r_gt.T
        tr = np.trace(delta)
        cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
        return float(np.arccos(cos_theta))

    def _report_solutions(self, solver_name, solutions, p_world, p_i, translation_gt, rotation_gt):
        print(f"\n[{solver_name}] n_solutions={len(solutions)}")
        for idx, sol in enumerate(solutions):
            rot_err = self._rotation_geodesic_error(sol.rotation, rotation_gt)
            trans_err = float(np.linalg.norm(sol.translation - translation_gt))
            reproj = sol.reprojection_dists_of(p_world, p_i)
            print(
                f"  sol[{idx}] rot_err(rad)={rot_err:.3e}, "
                f"trans_err={trans_err:.3e}, max_reproj={np.max(reproj):.3e}, "
                f"mean_reproj={np.mean(reproj):.3e}"
            )

        if len(solutions) == 0:
            print("  no solutions returned")

    def _assert_contains_gt(self, solutions, translation_gt, rotation_gt, atol=1e-4):
        self.assertTrue(len(solutions) > 0)
        found = any(
            np.allclose(sol.translation, translation_gt, atol=atol) and
            np.allclose(sol.rotation, rotation_gt, atol=atol)
            for sol in solutions
        )
        self.assertTrue(found)

    def test_p4p_distance_fixed(self):
        p_world, p_i, translation_gt, rotation_gt = self._make_case()
        solutions = p4p_distance(p_world, p_i, reprojection_threshold=1e-8)
        self._report_solutions("p4p_distance_fixed", solutions, p_world, p_i, translation_gt, rotation_gt)
        self._assert_contains_gt(solutions, translation_gt, rotation_gt)

    def test_p4p_liu_wong_fixed(self):
        p_world, p_i, translation_gt, rotation_gt = self._make_case()
        solutions = p4p_liu_wong(p_world, p_i, reprojection_threshold=1e-8)
        self._report_solutions("p4p_liu_wong_fixed", solutions, p_world, p_i, translation_gt, rotation_gt)
        self._assert_contains_gt(solutions, translation_gt, rotation_gt)

    def test_lehavi_p4p_fixed(self):
        p_world, p_i, translation_gt, rotation_gt = self._make_case()
        solutions = lehavi_p4p(p_world, p_i, reprojection_threshold=1e-8)
        self._report_solutions("lehavi_p4p_fixed", solutions, p_world, p_i, translation_gt, rotation_gt)
        self._assert_contains_gt(solutions, translation_gt, rotation_gt, atol=2e-4)

    def test_lehavi_p4p_algebraic_fixed(self):
        p_world, p_i, translation_gt, rotation_gt = self._make_case()
        solutions = lehavi_p4p_algebraic(p_world, p_i, reprojection_threshold=1e-8)
        self._report_solutions("lehavi_p4p_algebraic_fixed", solutions, p_world, p_i, translation_gt, rotation_gt)
        self._assert_contains_gt(solutions, translation_gt, rotation_gt, atol=2e-4)

    def test_p4p_coplanar_points(self):
        p_world, p_i, translation_gt, rotation_gt = self._make_coplanar_case()

        solutions_distance = p4p_distance(p_world, p_i, reprojection_threshold=1e-8)
        self._report_solutions("p4p_distance_coplanar", solutions_distance, p_world, p_i, translation_gt, rotation_gt)
        self._assert_contains_gt(solutions_distance, translation_gt, rotation_gt, atol=2e-4)

        solutions_liu = p4p_liu_wong(p_world, p_i, reprojection_threshold=1e-8)
        self._report_solutions("p4p_liu_wong_coplanar", solutions_liu, p_world, p_i, translation_gt, rotation_gt)
        self._assert_contains_gt(solutions_liu, translation_gt, rotation_gt, atol=2e-4)

        solutions_lehavi = lehavi_p4p(p_world, p_i, reprojection_threshold=1e-8)
        self._report_solutions("lehavi_p4p_coplanar", solutions_lehavi, p_world, p_i, translation_gt, rotation_gt)
        self._assert_contains_gt(solutions_lehavi, translation_gt, rotation_gt, atol=2e-4)

        solutions_lehavi_alg = lehavi_p4p_algebraic(p_world, p_i, reprojection_threshold=1e-8)
        self._report_solutions("lehavi_p4p_algebraic_coplanar", solutions_lehavi_alg, p_world, p_i, translation_gt, rotation_gt)
        self._assert_contains_gt(solutions_lehavi_alg, translation_gt, rotation_gt, atol=2e-4)

    def test_p4p_collinear_world_points_raise(self):
        p_world = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        rotation_gt = np.eye(3)
        translation_gt = np.array([0.0, 0.0, 3.0])
        p_cam = rotation_gt @ p_world + translation_gt[:, None]
        p_i = p_cam / np.linalg.norm(p_cam, axis=0)

        with self.assertRaises(ValueError):
            p4p_distance(p_world, p_i)
        with self.assertRaises(ValueError):
            p4p_liu_wong(p_world, p_i)
        with self.assertRaises(ValueError):
            lehavi_p4p(p_world, p_i)
        with self.assertRaises(ValueError):
            lehavi_p4p_algebraic(p_world, p_i)

    def test_p4p_timing_comparison(self):
        n_cases = 10
        seeds = [10_000 + i for i in range(n_cases)]

        solvers = {
            "p4p_distance": p4p_distance,
            "p4p_liu_wong": p4p_liu_wong,
            "lehavi_p4p": lehavi_p4p,
            "lehavi_p4p_algebraic": lehavi_p4p_algebraic,
        }

        elapsed = {name: 0.0 for name in solvers}
        solved = {name: 0 for name in solvers}

        for seed in seeds:
            p_world, p_i, _, _ = self._make_case(seed=seed)
            for name, solver in solvers.items():
                t0 = time.perf_counter()
                sols = solver(p_world, p_i, reprojection_threshold=1e-8)
                elapsed[name] += time.perf_counter() - t0
                solved[name] += int(len(sols) > 0)

        print("\n[P4P timing comparison over", n_cases, "cases]")
        for name in solvers:
            avg_us = 1e6 * elapsed[name] / n_cases
            print(f"  {name:24s} avg={avg_us:9.1f} us, solved={solved[name]}/{n_cases}")

        for name in solvers:
            self.assertEqual(solved[name], n_cases)


if __name__ == '__main__':
    unittest.main()
