import unittest
import numpy as np
import time

from recipnps.p2p import sweeney, li


def _random_rotation(rng):
    """Generate a uniformly random SO(3) rotation matrix."""
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


class TestP2P(unittest.TestCase):
    """Unit tests for gravity-aware Perspective-2-Point solvers."""

    # -- helpers ---------------------------------------------------------

    def _make_case(self, seed=None):
        """Generate a random P2P test case.

        Returns (p_world, p_i, t_gt, R_gt, v_cam, v_world).
        """
        if seed is None:
            seed = time.time_ns() % (2**32)
        rng = np.random.default_rng(seed)

        # Random world points in front of camera
        p_world = rng.uniform(-1.0, 1.0, size=(3, 2))

        # Random rotation and translation (ensure points are in front)
        R_gt = _random_rotation(rng)
        t_gt = np.array([0.1, -0.2, 3.0]) + rng.uniform(-0.3, 0.3, size=3)

        # Camera-frame points
        p_cam = R_gt @ p_world + t_gt[:, None]

        # Ensure positive depth (all z-coordinates positive isn't strictly
        # needed—any positive λ suffices—but let's just require that all
        # norms are positive, which is always true for finite points)
        p_i = p_cam / np.linalg.norm(p_cam, axis=0)

        # Gravity direction: choose an arbitrary world gravity, compute
        # its image in camera frame via R_gt
        v_world = np.array([0.0, -1.0, 0.0])
        v_cam = R_gt @ v_world          # gravity in camera frame

        return p_world, p_i, t_gt, R_gt, v_cam, v_world

    def _rotation_geodesic_error(self, R_est, R_gt):
        delta = R_est @ R_gt.T
        tr = np.trace(delta)
        cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
        return float(np.arccos(cos_theta))

    def _report_solutions(self, solver_name, solutions, p_world, p_i,
                          t_gt, R_gt):
        print(f"\n[{solver_name}] n_solutions={len(solutions)}")
        for idx, sol in enumerate(solutions):
            rot_err = self._rotation_geodesic_error(sol.rotation, R_gt)
            trans_err = float(np.linalg.norm(sol.translation - t_gt))
            reproj = sol.reprojection_dists_of(p_world, p_i)
            print(
                f"  sol[{idx}] rot_err(rad)={rot_err:.3e}, "
                f"trans_err={trans_err:.3e}, max_reproj={np.max(reproj):.3e}, "
                f"mean_reproj={np.mean(reproj):.3e}"
            )
        if len(solutions) == 0:
            print("  no solutions returned")

    def _assert_contains_gt(self, solutions, t_gt, R_gt, atol=1e-6):
        self.assertTrue(len(solutions) > 0,
                        msg="solver returned no solutions")
        found = any(
            np.allclose(sol.translation, t_gt, atol=atol)
            and np.allclose(sol.rotation, R_gt, atol=atol)
            for sol in solutions
        )
        if not found:
            # Provide some diagnostics
            for idx, sol in enumerate(solutions):
                rot_err = self._rotation_geodesic_error(sol.rotation, R_gt)
                trans_err = float(np.linalg.norm(sol.translation - t_gt))
                print(f"  sol[{idx}] rot_err={rot_err:.3e}, trans_err={trans_err:.3e}")
        self.assertTrue(found, msg="ground truth not among solutions")

    # -- sweeney tests ---------------------------------------------------

    def test_sweeney_random(self):
        p_w, p_i, t_gt, R_gt, v_cam, v_world = self._make_case()
        solutions = sweeney(p_w, p_i, v_cam, v_world)
        self._report_solutions("sweeney_random", solutions, p_w, p_i, t_gt, R_gt)
        self._assert_contains_gt(solutions, t_gt, R_gt)

    def test_sweeney_fixed(self):
        p_w, p_i, t_gt, R_gt, v_cam, v_world = self._make_case(seed=42)
        solutions = sweeney(p_w, p_i, v_cam, v_world)
        self._report_solutions("sweeney_fixed", solutions, p_w, p_i, t_gt, R_gt)
        self._assert_contains_gt(solutions, t_gt, R_gt)

    def test_sweeney_many_random(self):
        n_cases = 50
        n_ok = 0
        for i in range(n_cases):
            p_w, p_i, t_gt, R_gt, v_cam, v_world = self._make_case(seed=1000 + i)
            solutions = sweeney(p_w, p_i, v_cam, v_world)
            ok = any(
                np.allclose(sol.translation, t_gt, atol=1e-6)
                and np.allclose(sol.rotation, R_gt, atol=1e-6)
                for sol in solutions
            )
            if ok:
                n_ok += 1
        print(f"\n[sweeney_many_random] solved {n_ok}/{n_cases}")
        self.assertEqual(n_ok, n_cases)

    # -- li tests --------------------------------------------------------

    def test_li_random(self):
        p_w, p_i, t_gt, R_gt, v_cam, v_world = self._make_case()
        solutions = li(p_w, p_i, v_cam, v_world)
        self._report_solutions("li_random", solutions, p_w, p_i, t_gt, R_gt)
        self._assert_contains_gt(solutions, t_gt, R_gt)

    def test_li_fixed(self):
        p_w, p_i, t_gt, R_gt, v_cam, v_world = self._make_case(seed=42)
        solutions = li(p_w, p_i, v_cam, v_world)
        self._report_solutions("li_fixed", solutions, p_w, p_i, t_gt, R_gt)
        self._assert_contains_gt(solutions, t_gt, R_gt)

    def test_li_many_random(self):
        n_cases = 50
        n_ok = 0
        for i in range(n_cases):
            p_w, p_i, t_gt, R_gt, v_cam, v_world = self._make_case(seed=1000 + i)
            solutions = li(p_w, p_i, v_cam, v_world)
            ok = any(
                np.allclose(sol.translation, t_gt, atol=1e-6)
                and np.allclose(sol.rotation, R_gt, atol=1e-6)
                for sol in solutions
            )
            if ok:
                n_ok += 1
        print(f"\n[li_many_random] solved {n_ok}/{n_cases}")
        self.assertEqual(n_ok, n_cases)

    # -- cross-solver agreement tests ------------------------------------

    def test_both_solvers_agree(self):
        """Both solvers should find the same ground truth on the same case."""
        p_w, p_i, t_gt, R_gt, v_cam, v_world = self._make_case(seed=99)
        sols_s = sweeney(p_w, p_i, v_cam, v_world)
        sols_l = li(p_w, p_i, v_cam, v_world)
        self._assert_contains_gt(sols_s, t_gt, R_gt)
        self._assert_contains_gt(sols_l, t_gt, R_gt)

    # -- edge cases ------------------------------------------------------

    def test_points_along_gravity_axis(self):
        """World points differ only along the gravity axis (degenerate for
        depth-based method but may still work)."""
        rng = np.random.default_rng(777)
        R_gt = _random_rotation(rng)
        t_gt = np.array([0.0, 0.0, 5.0])
        v_world = np.array([0.0, -1.0, 0.0])
        v_cam = R_gt @ v_world

        # Two points that differ only along the gravity axis
        p_w = np.array([[0.0, 0.0],
                        [1.0, 2.0],
                        [0.0, 0.0]])
        p_cam = R_gt @ p_w + t_gt[:, None]
        p_i = p_cam / np.linalg.norm(p_cam, axis=0)

        # Li should still work (cross-product constraint is not degenerate
        # unless the projected directions are parallel)
        sols_l = li(p_w, p_i, v_cam, v_world)
        if len(sols_l) > 0:
            self._assert_contains_gt(sols_l, t_gt, R_gt, atol=1e-5)

    # -- timing comparison -----------------------------------------------

    def test_timing_comparison(self):
        n_cases = 100
        seeds = [5000 + i for i in range(n_cases)]

        solvers = {
            "sweeney": sweeney,
            "li": li,
        }

        elapsed = {name: 0.0 for name in solvers}
        solved = {name: 0 for name in solvers}

        for seed in seeds:
            p_w, p_i, t_gt, R_gt, v_cam, v_world = self._make_case(seed=seed)
            for name, solver in solvers.items():
                t0 = time.perf_counter()
                sols = solver(p_w, p_i, v_cam, v_world)
                elapsed[name] += time.perf_counter() - t0
                ok = any(
                    np.allclose(sol.translation, t_gt, atol=1e-6)
                    and np.allclose(sol.rotation, R_gt, atol=1e-6)
                    for sol in sols
                )
                solved[name] += int(ok)

        print(f"\n[P2P timing comparison over {n_cases} cases]")
        for name in solvers:
            avg_us = 1e6 * elapsed[name] / n_cases
            print(f"  {name:12s} avg={avg_us:9.1f} µs, solved={solved[name]}/{n_cases}")

        for name in solvers:
            self.assertEqual(solved[name], n_cases,
                             msg=f"{name} failed {n_cases - solved[name]}/{n_cases}")


if __name__ == '__main__':
    unittest.main()
