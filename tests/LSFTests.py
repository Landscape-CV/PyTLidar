"""
Python adaptation and extension of TREEQSM.

Version: 0.0.4
Date: 4 March 2025
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

from Utils.Utils import Utils
from LeastSquaresFitting.LSF import LSF
import unittest
import numpy as np


class TestLSF(unittest.TestCase):
    def test_rotate_to_z_axis_nonzero(self):
        # Test with a vector not already aligned with z-axis.
        Vec = [1, 0, 0]
        R, D, a = LSF.rotate_to_z_axis(Vec)
        # After rotation, the resulting vector should be along positive z
        Vec_rotated = R @ np.array(Vec)
        # Due to numerical issues, we test using np.allclose.
        expected = np.array([0, 0, np.linalg.norm(Vec)])
        self.assertTrue(np.allclose(Vec_rotated, expected, atol=1e-6),
                        msg=f"Rotated vector {Vec_rotated} does not match expected {expected}")
        # Also check that the angle is about pi/2.
        self.assertAlmostEqual(a, np.pi / 2, places=6,
                               msg=f"Rotation angle {a} is not close to pi/2")

    def test_rotate_to_z_axis_already_aligned(self):
        # Test with a vector that is already aligned with the z-axis.
        Vec = [0, 0, 1]
        R, D, a = LSF.rotate_to_z_axis(Vec)
        Vec_rotated = R @ np.array(Vec)
        expected = np.array([0, 0, 1])
        self.assertTrue(np.allclose(Vec_rotated, expected, atol=1e-6),
                        msg=f"Rotated vector {Vec_rotated} does not match expected {expected}")
        self.assertAlmostEqual(a, 0.0, places=6,
                               msg=f"Rotation angle {a} should be 0 for a vector aligned with z-axis")
        self.assertTrue(np.allclose(R, np.eye(3), atol=1e-6),
                        msg="Rotation matrix should be identity when the vector is already aligned with z-axis")

    def test_rotation_matrix_product(self):
        # Test with known angles.
        theta = [np.pi/4, np.pi/6]  # t1 = 45 degrees, t2 = 30 degrees
        R, dR1, dR2 = LSF.form_rotation_matrices(theta)

        # Verify that R has shape (3,3)
        self.assertEqual(R.shape, (3, 3))

        # Verify that dR1 and dR2 have the expected shapes.
        self.assertEqual(dR1.shape, (3, 3))
        self.assertEqual(dR2.shape, (3, 3))

        # For extra verification, check that small perturbations in t1 or t2 are approximated by the derivatives.
        delta = 1e-6

        # Approximate derivative for R1 with respect to t1.
        theta_perturbed = [theta[0] + delta, theta[1]]
        R_perturbed, _, _ = LSF.form_rotation_matrices(theta_perturbed)
        R_approx = (R_perturbed - R) / delta

        # Since R = R2*R1, the derivative with respect to t1 is R2 * dR1.
        dR1_numeric = R2_dR1 = LSF.form_rotation_matrices(theta)[0]  # get R from original theta
        # Compute analytical derivative for R change due to t1: R2 * dR1.
        theta = np.array(theta)
        c2 = np.cos(theta[1])
        s2 = np.sin(theta[1])
        R2 = np.array([
            [c2,  0,  s2],
            [0,   1,   0],
            [-s2, 0,  c2]
        ])
        dR_analytical_t1 = R2 @ dR1
        # Compare a few elements; note that this is an approximate check.
        self.assertTrue(np.allclose(R_approx, dR_analytical_t1, atol=1e-3),
                        msg="Numerical derivative approximation for t1 does not match analytical derivative.")

    def test_invalid_theta(self):
        # Test that an error is raised if theta has fewer than two elements.
        with self.assertRaises(ValueError):
            LSF.form_rotation_matrices([np.pi/4])

    def test_func_grad_axis_unweighted(self):
        # Create a simple point cloud: 5 points in 3D.
        P = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 0.0, 1.0],
            [0.5, 1.5, 2.5],
            [1.5, 2.5, 3.5],
            [2.5, 1.0, 0.5]
        ])
        # Cylinder parameters: x0, y0, alpha, beta, r
        par = [0.5, 1.0, 0.1, 0.2, 1.0]
        dist, J = LSF.func_grad_axis(P, par)

        # Check shapes of outputs.
        self.assertEqual(dist.shape, (P.shape[0],))
        self.assertEqual(J.shape, (P.shape[0], 2))

        # Simple sanity checks (e.g., distances should be finite numbers).
        self.assertTrue(np.all(np.isfinite(dist)))
        self.assertTrue(np.all(np.isfinite(J)))

    def test_func_grad_axis_weighted(self):
        # Create a simple point cloud.
        P = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        par = [0.0, 0.0, 0.0, 0.0, 1.0]  # With no rotation, the distance is radial distance minus 1.
        weight = np.array([2.0, 3.0, 4.0])
        dist, J = LSF.func_grad_axis(P, par, weight)

        # For points on x-y plane with no rotation, the distances are:
        # sqrt(x^2+y^2)-1, multiplied by weight.
        expected_dist = weight * (np.sqrt(np.sum(P[:, :2]**2, axis=1)) - 1.0)
        self.assertTrue(np.allclose(dist, expected_dist, atol=1e-6))

        # Check shapes.
        self.assertEqual(J.shape, (P.shape[0], 2))

    def test_func_grad_circle_unweighted(self):
        # Create a simple 2D point cloud (points in the xy-plane).
        P = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        # Define circle parameters: center at (1,1) and radius 1.
        par = [1.0, 1.0, 1.0]
        # Expected: distances = sqrt((x-1)^2+(y-1)^2) - 1.
        expected_dist = np.sqrt((P[:,0]-1.0)**2 + (P[:,1]-1.0)**2) - 1.0

        dist, J = LSF.func_grad_circle(P, par)
        self.assertTrue(np.allclose(dist, expected_dist, atol=1e-6),
                        msg="Unweighted distances do not match expected values.")
        self.assertEqual(J.shape, (P.shape[0], 3))

    def test_func_grad_circle_weighted(self):
        # Create a simple 2D point cloud.
        P = np.array([
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        # Define circle parameters: center at (0,0) and radius 5.
        par = [0.0, 0.0, 5.0]
        # Weight vector.
        weight = np.array([2.0, 3.0])
        expected_dist = weight * (np.sqrt(P[:,0]**2 + P[:,1]**2) - 5.0)
        dist, J = LSF.func_grad_circle(P, par, weight)
        self.assertTrue(np.allclose(dist, expected_dist, atol=1e-6),
                        msg="Weighted distances do not match expected values.")
        self.assertEqual(J.shape, (P.shape[0], 3))

    def test_func_grad_circle_centre_unweighted(self):
        # Create a simple point cloud in the xy-plane.
        P = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        # Circle parameters: center at (1,1) and radius = 1.
        par = [1.0, 1.0, 1.0]
        # Expected distances: sqrt((x-1)^2+(y-1)^2) - 1.
        expected_dist = np.sqrt((P[:, 0] - 1.0)**2 + (P[:, 1] - 1.0)**2) - 1.0

        dist, J = LSF.func_grad_circle_centre(P, par)
        self.assertTrue(np.allclose(dist, expected_dist, atol=1e-6),
                        msg="Unweighted distances do not match expected values.")
        self.assertEqual(J.shape, (P.shape[0], 2),
                         msg="Jacobian must have shape (n,2) for circle centre derivatives.")

    def test_func_grad_circle_centre_weighted(self):
        # Create a simple point cloud.
        P = np.array([
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        # Circle parameters: center at (0,0) and radius = 5.
        par = [0.0, 0.0, 5.0]
        weight = np.array([2.0, 3.0])
        expected_dist = weight * (np.sqrt(P[:, 0]**2 + P[:, 1]**2) - 5.0)

        dist, J = LSF.func_grad_circle_centre(P, par, weight)
        self.assertTrue(np.allclose(dist, expected_dist, atol=1e-6),
                        msg="Weighted distances do not match expected values.")
        self.assertEqual(J.shape, (P.shape[0], 2),
                         msg="Jacobian must have shape (n,2) for weighted circle centre derivatives.")

    def test_func_grad_cylinder_unweighted(self):
        # Create a simple point cloud with 5 points.
        P = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 0.0, 1.0],
            [0.5, 1.5, 2.5],
            [1.5, 2.5, 3.5],
            [2.5, 1.0, 0.5]
        ])
        # Define cylinder parameters: x0, y0, alpha, beta, r.
        par = [0.5, 1.0, 0.1, 0.2, 1.0]
        dist, J = LSF.func_grad_cylinder(par, P)

        # Check output shapes.
        self.assertEqual(dist.shape, (P.shape[0],))
        self.assertEqual(J.shape, (P.shape[0], 5))

        # Basic sanity: All computed values should be finite.
        self.assertTrue(np.all(np.isfinite(dist)))
        self.assertTrue(np.all(np.isfinite(J)))

    def test_func_grad_cylinder_weighted(self):
        # Create a simple point cloud.
        P = np.array([
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        # Cylinder parameters: center at (0,0), no rotation, radius = 5.
        par = [0.0, 0.0, 0.0, 0.0, 5.0]
        weight = np.array([2.0, 3.0])
        expected_dist = weight * (np.sqrt(P[:,0]**2 + P[:,1]**2) - 5.0)
        dist, J = LSF.func_grad_cylinder(par, P, weight)

        self.assertTrue(np.allclose(dist, expected_dist, atol=1e-6),
                        msg="Weighted distances do not match expected values.")
        self.assertEqual(J.shape, (P.shape[0], 5))

    def test_least_squares_circle_no_weight(self):
        # Create a perfect circle: points on a circle of radius 1 centered at (1,1)
        theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        radius = 1.0
        x = 1.0 + radius * np.cos(theta)
        y = 1.0 + radius * np.sin(theta)
        P = np.column_stack((x, y))

        Point0 = [1.0, 1.0]
        Rad0 = 1.0
        cir = LSF.least_squares_circle(P, Point0, Rad0)

        # For a perfect circle, the fitted centre should be very close to the input,
        # the radius should be near 1, and residuals (mad) near zero.
        self.assertTrue(np.allclose(cir['point'], Point0, atol=1e-4),
                        msg="Fitted centre is not close to the input centre.")
        self.assertAlmostEqual(cir['radius'], Rad0, places=4,
                               msg="Fitted radius is not close to the input radius.")
        self.assertTrue(cir['mad'] < 1e-4,
                        msg="Mean absolute deviation should be near zero for a perfect circle.")
        self.assertTrue(cir['conv'], msg="Solver did not converge in the unweighted case.")
        self.assertTrue(cir['rel'], msg="Solver reported unreliable result in the unweighted case.")

    def test_least_squares_circle_weighted(self):
        # Create a circle with slight noise.
        theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        radius = 1.0
        x = 1.0 + radius * np.cos(theta) + 0.01 * np.random.randn(len(theta))
        y = 1.0 + radius * np.sin(theta) + 0.01 * np.random.randn(len(theta))
        P = np.column_stack((x, y))

        Point0 = [1.0, 1.0]
        Rad0 = 1.0
        weight = np.linspace(1, 2, len(theta))
        cir = LSF.least_squares_circle(P, Point0, Rad0, weight)

        self.assertTrue(np.allclose(cir['point'], Point0, atol=1e-2),
                        msg="Fitted centre (weighted) is not close to the input centre.")
        self.assertAlmostEqual(cir['radius'], Rad0, places=2,
                               msg="Fitted radius (weighted) is not close to the input radius.")
        self.assertTrue(cir['conv'], msg="Solver did not converge in the weighted case.")
        self.assertTrue(cir['rel'], msg="Solver reported unreliable result in the weighted case.")

    def test_least_squares_circle_centre_perfect(self):
        # Create a perfect circle: points on a circle of radius 1 centered at (1,1)
        theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        radius = 1.0
        x = 1.0 + radius * np.cos(theta)
        y = 1.0 + radius * np.sin(theta)
        P = np.column_stack((x, y))

        Point0 = [1.0, 1.0]
        Rad0 = 1.0
        cir = LSF.least_squares_circle_centre(P, Point0, Rad0)

        # For a perfect circle, the fitted centre should be very close to the input,
        # the radius remains fixed, and the mean absolute deviation (mad) should be nearly zero.
        self.assertTrue(np.allclose(cir['point'], Point0, atol=1e-4),
                        msg="Fitted centre is not close to the input centre.")
        self.assertAlmostEqual(cir['radius'], Rad0, places=4,
                               msg="Fitted radius is not equal to the input radius.")
        self.assertTrue(cir['mad'] < 1e-4,
                        msg="Mean absolute deviation should be near zero for a perfect circle.")
        self.assertTrue(cir['conv'], msg="Solver did not converge in the perfect circle case.")
        self.assertTrue(cir['rel'], msg="Solver reported unreliable result in the perfect circle case.")

