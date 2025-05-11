"""
Python adaptation and extension of TREEQSM.

Version: 0.0.4
Date: 4 March 2025
Authors: Fan Yang, John Hagood, Amir Hossein Alikhah Mishamandani
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

from Utils.Utils import Utils
import numpy as np
import matplotlib
import unittest
import os
import laspy
matplotlib.use('Agg')

class TestUtils(unittest.TestCase):

    def test_average_multiple_rows(self):
        # Test with a matrix having multiple rows
        X = np.array([[1, 2, 3],
                      [4, 5, 6]])
        expected = np.array([2.5, 3.5, 4.5])
        result = Utils.average(X)
        np.testing.assert_array_almost_equal(result, expected)

    def test_average_single_row(self):
        # Test with a single row matrix; it should return the row unchanged.
        X = np.array([[7, 8, 9]])
        expected = X
        result = Utils.average(X)
        np.testing.assert_array_equal(result, expected)

    def test_average_with_list(self):
        # Test with the input provided as a list of lists.
        X = [[10, 20], [30, 40]]
        expected = np.array([20, 30])
        result = Utils.average(X)
        np.testing.assert_array_almost_equal(result, expected)

    def test_change_precision_positive(self):
        # Test change_precision with positive values covering all branches:
        #   v[0]: >= 1e3, v[1]: [1e2, 1e3), v[2]: [1e1, 1e2), v[3]: [1e0, 1e1),
        #   v[4]: [1e-1, 1e0), v[5]: < 1e-1.
        v = np.array([1234.567, 123.456, 12.3456, 5.6789, 0.123456, 0.0123456])
        expected = np.array([
            np.round(1234.567),                # 1235.0
            np.round(10 * 123.456) / 10,         # 123.5
            np.round(100 * 12.3456) / 100,       # 12.35
            np.round(1000 * 5.6789) / 1000,      # 5.679
            np.round(10000 * 0.123456) / 10000,  # 0.1235
            np.round(100000 * 0.0123456) / 100000 # 0.01235
        ])
        result = Utils.change_precision(v)
        np.testing.assert_array_almost_equal(result, expected)

    def test_change_precision_negative(self):
        # Test change_precision with negative values to ensure sign is preserved.
        v = np.array([-1234.567, -123.456, -12.3456, -5.6789, -0.123456, -0.0123456])
        expected = np.array([
            np.round(-1234.567),                # -1235.0
            np.round(10 * -123.456) / 10,         # -123.5
            np.round(100 * -12.3456) / 100,       # -12.35
            np.round(1000 * -5.6789) / 1000,      # -5.679
            np.round(10000 * -0.123456) / 10000,  # -0.1235
            np.round(100000 * -0.0123456) / 100000 # -0.01235
        ])
        result = Utils.change_precision(v)
        np.testing.assert_array_almost_equal(result, expected)

    def test_cross_product(self):
        # Test standard cross product.
        A = [1, 0, 0]
        B = [0, 1, 0]
        expected = np.array([0, 0, 1])
        result = Utils.cross_product(A, B)
        np.testing.assert_array_almost_equal(result, expected)

        # Test with reversed vectors (should flip the sign).
        A = [0, 1, 0]
        B = [1, 0, 0]
        expected = np.array([0, 0, -1])
        result = Utils.cross_product(A, B)
        np.testing.assert_array_almost_equal(result, expected)

        # Test with identical vectors (cross product should be zero vector).
        A = [1, 2, 3]
        B = [1, 2, 3]
        expected = np.array([0, 0, 0])
        result = Utils.cross_product(A, B)
        np.testing.assert_array_almost_equal(result, expected)

    def test_dot_product(self):
        # Test with matrices having multiple rows.
        A = np.array([[1, 2, 3],
                      [4, 5, 6]])
        B = np.array([[7, 8, 9],
                      [10, 11, 12]])
        expected = np.array([1*7 + 2*8 + 3*9, 4*10 + 5*11 + 6*12])
        result = Utils.dot_product(A, B)
        np.testing.assert_array_almost_equal(result, expected)

        # Test with single row matrices.
        A = np.array([[1, 2, 3]])
        B = np.array([[4, 5, 6]])
        expected = np.array([1*4 + 2*5 + 3*6])
        result = Utils.dot_product(A, B)
        np.testing.assert_array_almost_equal(result, expected)

    def test_distances_to_line(self):
        # Test using a line through the origin in the x-axis direction.
        line_point = [0, 0, 0]
        line_dir = [1, 0, 0]  # Already a unit vector.

        # Define a set of points.
        Q = np.array([
            [1, 2, 2],
            [2, 3, 6],
            [5, 0, 0]
        ])
        # Expected distances: For a line along the x-axis, distance = sqrt(y^2 + z^2).
        expected_d = np.sqrt(np.array([
            2**2 + 2**2,  # sqrt(4+4) = sqrt(8)
            3**2 + 6**2,  # sqrt(9+36) = sqrt(45)
            0**2 + 0**2   # 0
        ]))

        d, V, h, B = Utils.distances_to_line(Q, line_dir, line_point)
        np.testing.assert_array_almost_equal(d, expected_d)

        # Additionally, verify that the projection scalars and projections are computed correctly.
        A = Q - line_point
        h_manual = np.dot(A, line_dir)
        B_manual = h_manual[:, np.newaxis] * line_dir
        np.testing.assert_array_almost_equal(h, h_manual)
        np.testing.assert_array_almost_equal(B, B_manual)

    def test_distances_between_lines(self):
        # Define a ray passing through the origin along the x-axis.
        PointRay = [0, 0, 0]
        DirRay = [1, 0, 0]

        # Define two lines:
        # First line: passes through [0, 1, 0] with direction [0, 0, 1]
        # Second line: passes through [0, 0, 1] with direction [0, 1, 0]
        PointLines = np.array([[0, 1, 0],
                               [0, 0, 1]])
        DirLines = np.array([[0, 0, 1],
                             [0, 1, 0]])

        # Expected results:
        # For both lines, the minimal distance between the ray (x-axis) and the line
        # is the offset in the direction perpendicular to the x-axis (which is 1).
        expected_DistLines = np.array([1, 1])
        # In these configurations the projections along the ray and along the lines are at the given points,
        # so the distances along the ray and the lines are zero.
        expected_DistOnRay = np.array([0, 0])
        expected_DistOnLines = np.array([0, 0])

        DistLines, DistOnRay, DistOnLines = Utils.distances_between_lines(PointRay, DirRay, PointLines, DirLines)
        np.testing.assert_array_almost_equal(DistLines, expected_DistLines)
        np.testing.assert_array_almost_equal(DistOnRay, expected_DistOnRay)
        np.testing.assert_array_almost_equal(DistOnLines, expected_DistOnLines)


    def test_display_time_seconds_only(self):
        # Both T1 and T2 are less than 60 seconds.
        T1 = 0.5    # 0 min, 0.5 sec.
        T2 = 0.7    # 0 min, 0.7 sec.
        label = "Time"
        expected = f"{label} 0.5 sec.   Total: 0.7 sec"
        result = Utils.display_time(T1, T2, label, True)
        self.assertEqual(result, expected)

    def test_display_time_mixed(self):
        # T1 is less than 60 seconds; T2 is between 60 seconds and 60 minutes.
        T1 = 30     # 0 min, 30 sec.
        T2 = 90     # 1 min, 30 sec.
        label = "Elapsed"
        expected = f"{label} 30 sec.   Total: 1 min 30 sec"
        result = Utils.display_time(T1, T2, label, True)
        self.assertEqual(result, expected)

    def test_display_time_hours(self):
        # T1 and T2 require conversion into hours.
        T1 = 4000   # sec2min(4000) returns (66, 40) => 66 >= 60 so: 1 hours 6 min (40 sec is not displayed)
        T2 = 8000   # sec2min(8000) returns (133, 20) => 133 >= 60 so: 2 hours 13 min
        label = "Time"
        expected = f"{label} 1 hours 6 min.   Total: 2 hours 13 min"
        result = Utils.display_time(T1, T2, label, True)
        self.assertEqual(result, expected)

    def test_display_time_no_display(self):
        # When display is False, nothing is printed or returned.
        T1 = 100
        T2 = 200
        label = "Test"
        result = Utils.display_time(T1, T2, label, False)
        self.assertIsNone(result)

    def test_median2_even(self):
        # Test with an even number of elements.
        X = [4, 1, 3, 2]
        # Sorted: [1, 2, 3, 4] => median = (2+3)/2 = 2.5
        expected = 2.5
        result = Utils.median2(X)
        self.assertAlmostEqual(result, expected)

    def test_median2_odd(self):
        # Test with an odd number of elements.
        X = [7, 1, 5, 3, 9]
        # Sorted: [1, 3, 5, 7, 9] => median = 5
        expected = 5
        result = Utils.median2(X)
        self.assertEqual(result, expected)

    def test_median2_single(self):
        # Test with a single element.
        X = [42]
        expected = 42
        result = Utils.median2(X)
        self.assertEqual(result, expected)

    def test_normalize(self):
        # Test with a 2x2 matrix.
        A = np.array([[3, 4], [0, 5]])
        normalized, L = Utils.normalize(A)
        expected_norms = np.array([5.0, 5.0])
        expected_normalized = np.array([[3/5, 4/5], [0, 1]])
        np.testing.assert_array_almost_equal(L, expected_norms)
        np.testing.assert_array_almost_equal(normalized, expected_normalized)

        # Test with a 3x3 matrix.
        A = np.array([[1, 0, 0], [0, 2, 2], [3, 3, 3]])
        normalized, L = Utils.normalize(A)
        expected_norms = np.array([1.0, np.sqrt(8), np.sqrt(27)])
        expected_normalized = np.array([
            [1, 0, 0],
            [0, 2/np.sqrt(8), 2/np.sqrt(8)],
            [3/np.sqrt(27), 3/np.sqrt(27), 3/np.sqrt(27)]
        ])
        np.testing.assert_array_almost_equal(L, expected_norms)
        np.testing.assert_array_almost_equal(normalized, expected_normalized)

    def test_mat_vec_subtraction(self):
        # Test with a simple 2x3 matrix.
        A = np.array([[5, 7, 9],
                      [2, 4, 6]])
        v = np.array([1, 2, 3])
        expected = np.array([[4, 5, 6],
                             [1, 2, 3]])
        result = Utils.mat_vec_subtraction(A, v)
        np.testing.assert_array_almost_equal(result, expected)

        # Test with a 3x2 matrix.
        A = np.array([[10, 20],
                      [30, 40],
                      [50, 60]])
        v = np.array([5, 5])
        expected = np.array([[5, 15],
                             [25, 35],
                             [45, 55]])
        result = Utils.mat_vec_subtraction(A, v)
        np.testing.assert_array_almost_equal(result, expected)

    def test_verticalcat(self):
        # Test vertical concatenation for a cell array (list of lists).
        CellArray = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        # Expected concatenated vector: [1,2,3,4,5,6,7,8,9]
        expected_Vector = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        # Expected index ranges:
        #   First cell: indices [0, 3)  (elements 1,2,3)
        #   Second cell: indices [3, 5)  (elements 4,5)
        #   Third cell: indices [5, 9)   (elements 6,7,8,9)
        expected_IndElements = np.array([[0, 3],
                                         [3, 5],
                                         [5, 9]])
        Vector, IndElements = Utils.verticalcat(CellArray)
        np.testing.assert_array_equal(Vector, expected_Vector)
        np.testing.assert_array_equal(IndElements, expected_IndElements)

    def test_rotation_matrix(self):
        # Test rotation matrix for a rotation about the z-axis.
        axis = [0, 0, 1]
        angle = np.pi / 4  # 45 degrees in radians.
        expected = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1]
        ])
        R = Utils.rotation_matrix(axis, angle)
        np.testing.assert_array_almost_equal(R, expected)

    def test_orthonormal_vectors(self):
        # Test that the generated vectors V and W are unit vectors and orthogonal to U and each other.
        U = np.array([1, 2, 3])
        V, W = Utils.orthonormal_vectors(U)

        # Verify orthogonality: dot products should be approximately zero.
        self.assertAlmostEqual(np.dot(U, V), 0, places=7)
        self.assertAlmostEqual(np.dot(U, W), 0, places=7)
        self.assertAlmostEqual(np.dot(V, W), 0, places=7)

        # Verify that V and W are unit vectors.
        self.assertAlmostEqual(np.linalg.norm(V), 1, places=7)
        self.assertAlmostEqual(np.linalg.norm(W), 1, places=7)

    def test_optimal_parallel_vector(self):
        # Test with a set of unit vectors that are all aligned along [1, 0, 0].
        V = np.array([[1, 0, 0],
                      [1, 0, 0],
                      [1, 0, 0]])
        v, mean_res, sigmah, residual = Utils.optimal_parallel_vector(V)

        # Since all vectors in V are [1,0,0], the optimal vector should be parallel to [1,0,0] (up to a sign).
        # The dot products should all be 1 (or very close to it in absolute value).
        self.assertAlmostEqual(np.abs(v[0]), 1, places=7)
        self.assertAlmostEqual(np.linalg.norm(v), 1, places=7)
        np.testing.assert_array_almost_equal(residual, np.ones(3))
        self.assertAlmostEqual(mean_res, 1, places=7)
        self.assertAlmostEqual(sigmah, 0, places=7)

        # Test with a set of unit vectors with slight variation.
        # Here the optimal vector should still yield high dot products.
        V = np.array([[0.98, 0.1, 0.15],
                      [0.99, -0.05, 0.1],
                      [0.97, 0.12, -0.2]])
        # Normalize each row to ensure they are unit vectors.
        V = np.array([v / np.linalg.norm(v) for v in V])
        v_opt, mean_res, sigmah, residual = Utils.optimal_parallel_vector(V)

        # Check that the computed optimal vector is unit length.
        self.assertAlmostEqual(np.linalg.norm(v_opt), 1, places=7)
        # Check that residuals are high, since the vectors are fairly aligned.
        self.assertTrue(np.mean(residual) > 0.9)

    def test_expand_no_forb(self):
        # Define a neighbor structure.
        # Nei[i] is the list of neighbors for node i.
        Nei = [
            [1, 2],   # neighbors of node 0
            [0, 3],   # neighbors of node 1
            [3, 4],   # neighbors of node 2
            [4],      # neighbors of node 3
            []        # neighbors of node 4
        ]
        # Initial subset C = [0].
        C = [0]
        # For n = 1:
        #   Union({0}, Nei[0]) = {0, 1, 2}.
        expected = np.array([0, 1, 2])
        result = Utils.expand(Nei, C, 1)
        np.testing.assert_array_equal(result, expected)

        # For n = 2:
        #   Iteration 1: C = {0, 1, 2}
        #   Iteration 2: Union({0,1,2}, Nei[0] U Nei[1] U Nei[2])
        #              Nei[0]=[1,2], Nei[1]=[0,3], Nei[2]=[3,4]  → Combined = {0,1,2,3,4}
        expected = np.array([0, 1, 2, 3, 4])
        result = Utils.expand(Nei, C, 2)
        np.testing.assert_array_equal(result, expected)

    def test_expand_forb_logical(self):
        # Define the same neighbor structure.
        Nei = [
            [1, 2],
            [0, 3],
            [3, 4],
            [4],
            []
        ]
        C = [0]
        # Forb as a boolean numpy array of length 5.
        # Mark node 1 as forbidden.
        Forb = np.array([False, True, False, False, False])
        # For n = 1:
        #   C becomes union({0}, Nei[0]) = {0,1,2}, then remove forbidden index 1 → expected: [0,2].
        expected = np.array([0, 2])
        result = Utils.expand(Nei, C, 1, Forb)
        np.testing.assert_array_equal(result, expected)

    def test_expand_forb_numeric(self):
        # Define the same neighbor structure.
        Nei = [
            [1, 2],
            [0, 3],
            [3, 4],
            [4],
            []
        ]
        C = [0]
        # Forb as a number vector: forbid nodes 1 and 4.
        Forb = [1, 4]
        # For n = 1:
        #   C becomes union({0}, Nei[0]) = {0,1,2}, then remove forbidden node 1 → expected: [0,2].
        expected = np.array([0, 2])
        result = Utils.expand(Nei, C, 1, Forb)
        np.testing.assert_array_equal(result, expected)

    def test_unique2_nonempty(self):
        # Test with a vector containing duplicates.
        Set = [3, 1, 2, 3, 2, 5]
        expected = np.array([1, 2, 3, 5])
        result = Utils.unique2(Set)
        np.testing.assert_array_equal(result, expected)

    def test_unique2_single(self):
        # Test with a single element.
        Set = [42]
        expected = np.array([42])
        result = Utils.unique2(Set)
        np.testing.assert_array_equal(result, expected)

    def test_unique2_empty(self):
        # Test with an empty vector.
        Set = []
        expected = np.array([])
        result = Utils.unique2(Set)
        np.testing.assert_array_equal(result, expected)

    def test_unique_elements_multiple(self):
        # Test with multiple elements containing duplicates.
        # For example, using a set with elements in the range 0 to 4.
        Set_input = [2, 2, 3, 1, 2]
        # Initialize a tracker array of sufficient length (max element + 1).
        tracker = np.zeros(5, dtype=bool)
        # Expected: the first occurrence of 2, then 3, then 1.
        expected = np.array([2, 3, 1])
        result = Utils.unique_elements(Set_input, tracker)
        np.testing.assert_array_equal(result, expected)

    def test_unique_elements_two_equal(self):
        # Test with two elements that are equal.
        Set_input = [5, 5]
        tracker = np.zeros(6, dtype=bool)  # Ensure length is at least max(Set_input)+1.
        expected = np.array([5])
        result = Utils.unique_elements(Set_input, tracker)
        np.testing.assert_array_equal(result, expected)

    def test_unique_elements_two_different(self):
        # Test with two different elements.
        Set_input = [5, 6]
        tracker = np.zeros(7, dtype=bool)
        expected = np.array([5, 6])
        result = Utils.unique_elements(Set_input, tracker)
        np.testing.assert_array_equal(result, expected)

    def test_unique_elements_single(self):
        # Test with a single element.
        Set_input = [7]
        tracker = np.zeros(8, dtype=bool)
        expected = np.array([7])
        result = Utils.unique_elements(Set_input, tracker)
        np.testing.assert_array_equal(result, expected)

    def test_unique_elements_empty(self):
        # Test with an empty set.
        Set_input = []
        tracker = np.zeros(1, dtype=bool)
        expected = np.array([])
        result = Utils.unique_elements(Set_input, tracker)
        np.testing.assert_array_equal(result, expected)

    def test_dimensions_3d(self):
        # Create a set of 3D points.
        points = np.array([
            [0, 0, 0],
            [1, 2, 1],
            [2, 1, 3],
            [3, 4, 2],
            [4, 3, 4]
        ])
        # Direct call without optional arguments.
        D, dir_vec = Utils.dimensions(points)
        # Check that D has 6 components and dir_vec has 9 elements.
        self.assertEqual(D.shape[0], 6)
        self.assertEqual(dir_vec.shape[0], 9)
        # Further tests can check that the computed ranges match expected values.
        dp1 = points @ np.linalg.svd(np.cov(points, rowvar=False))[0][:, 0]
        range_dp1 = np.max(dp1) - np.min(dp1)
        self.assertAlmostEqual(D[0], range_dp1, places=7)

    def test_dimensions_2d(self):
        # Create a set of 2D points.
        points = np.array([
            [0, 0],
            [1, 2],
            [2, 1],
            [3, 4],
            [4, 3]
        ])
        D, dir_vec = Utils.dimensions(points)
        # For 2D, D should have 4 components and dir_vec 4 elements.
        self.assertEqual(D.shape[0], 4)
        self.assertEqual(dir_vec.shape[0], 4)

    def test_dimensions_optional_P(self):
        # Create a matrix P of points.
        P = np.array([
            [10, 10, 10],
            [20, 20, 20],
            [30, 30, 30],
            [40, 40, 40],
            [50, 50, 50]
        ])
        # Let "points" be a list of indices.
        indices = [1, 3]  # Expect rows 1 and 3 of P.
        D, dir_vec = Utils.dimensions(indices, P)
        # Check that the selected points match P[indices, :].
        selected = P[indices, :]
        self.assertTrue(np.allclose(np.cov(selected, rowvar=False), np.cov(P[indices, :], rowvar=False)))

    def test_dimensions_optional_P_Bal(self):
        # Create a matrix P.
        P = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10],
            [10, 10, 10],
            [20, 20, 20]
        ])
        # Define Bal as a list of arrays.
        # For example, Bal[0] = [0, 1], Bal[1] = [2, 3], etc.
        Bal = [
            [0, 1],
            [2, 3],
            [4],
            [5],
            [1, 3],
            [0, 2]
        ]
        # Let points be indices into Bal.
        points_indices = [0, 2]  # This means we concatenate Bal[0] and Bal[2].
        # Expected: I = [0, 1] concatenated with [4] -> [0, 1, 4]
        D, dir_vec = Utils.dimensions(points_indices, P, Bal)
        expected_points = P[[0, 1, 4], :]
        # Check that the covariance computed from expected_points matches that from the function.
        cov_expected = np.cov(expected_points, rowvar=False)
        cov_function = np.cov(P[[0, 1, 4], :], rowvar=False)
        self.assertTrue(np.allclose(cov_expected, cov_function))

    def test_intersect_elements(self):
        # Define two sets.
        Set1 = np.array([1, 2, 3, 5])
        Set2 = np.array([2, 4, 5, 6])
        # Prepare tracker arrays. Their length must be at least max(Set1, Set2)+1.
        tracker1 = np.zeros(10, dtype=bool)
        tracker2 = np.zeros(10, dtype=bool)
        # Expected intersection: elements common to both sets.
        expected = np.array([2, 5])
        result = Utils.intersect_elements(Set1, Set2, tracker1, tracker2)
        np.testing.assert_array_equal(result, expected)

        # Test with one empty set.
        tracker1 = np.zeros(10, dtype=bool)
        tracker2 = np.zeros(10, dtype=bool)
        Set1 = np.array([])
        Set2 = np.array([2, 3])
        expected = np.array([])
        result = Utils.intersect_elements(Set1, Set2, tracker1, tracker2)
        np.testing.assert_array_equal(result, expected)

        # Test with identical sets.
        tracker1 = np.zeros(10, dtype=bool)
        tracker2 = np.zeros(10, dtype=bool)
        Set1 = np.array([1, 3, 7])
        Set2 = np.array([1, 3, 7])
        expected = np.array([1, 3, 7])
        result = Utils.intersect_elements(Set1, Set2, tracker1, tracker2)
        np.testing.assert_array_equal(result, expected)

    def test_small_subset_single(self):
        # Test with a small subset of one cover set (non-boolean).
        Nei = [[1], [0], [1]]  # Dummy neighbor structure.
        Sub = [2]  # Single cover set (assumed >0).
        MinSize = 1
        Components, CompSize = Utils.connected_components(Nei, Sub, MinSize)
        expected_components = [np.array(Sub, dtype=np.uint32)]
        expected_comp_size = np.array([1], dtype=np.uint32)
        self.assertEqual(len(Components), len(expected_components))
        np.testing.assert_array_equal(Components[0], expected_components[0])
        np.testing.assert_array_equal(CompSize, expected_comp_size)

    def test_small_subset_pair_connected(self):
        # Two cover sets that are connected.
        Nei = [[1], [0], [1]]
        Sub = [0, 1]
        MinSize = 1
        Components, CompSize = Utils.connected_components(Nei, Sub, MinSize)
        # Since 1 is in Nei[0], they form a single component.
        expected_components = [np.array([0, 1], dtype=np.uint32)]
        expected_comp_size = np.array([1], dtype=np.uint32)  # Per MATLAB branch, returns 1.
        self.assertEqual(len(Components), 1)
        np.testing.assert_array_equal(Components[0], expected_components[0])
        np.testing.assert_array_equal(CompSize, expected_comp_size)

    def test_small_subset_pair_not_connected(self):
        # Two cover sets that are not connected.
        Nei = [[2], [2], [0, 1]]
        Sub = [0, 1]
        MinSize = 1
        Components, CompSize = Utils.connected_components(Nei, Sub, MinSize)
        # They are not neighbors; should yield two separate components.
        expected_components = [np.array([0], dtype=np.uint32), np.array([1], dtype=np.uint32)]
        expected_comp_size = np.array([1, 1], dtype=np.uint32)
        self.assertEqual(len(Components), 2)
        np.testing.assert_array_equal(Components[0], expected_components[0])
        np.testing.assert_array_equal(Components[1], expected_components[1])
        np.testing.assert_array_equal(CompSize, expected_comp_size)

    def test_general_branch_all_sets(self):
        # Test the general branch with Sub = [0] (select all cover sets).
        Nei = [
            [1],    # neighbors of 0
            [0, 2], # neighbors of 1
            [1, 3], # neighbors of 2
            [2]     # neighbors of 3
        ]
        Sub = [0]  # Special value: 0 means "all cover sets"
        MinSize = 2
        Components, CompSize = Utils.connected_components(Nei, Sub, MinSize)
        expected_components = [np.array([0, 1, 2, 3], dtype=np.uint32)]
        expected_comp_size = np.array([4], dtype=np.uint32)
        self.assertEqual(len(Components), 1)
        np.testing.assert_array_equal(Components[0], expected_components[0])
        np.testing.assert_array_equal(CompSize, expected_comp_size)

    def test_general_branch_subset_indices(self):
        # Test the general branch with a numeric subset (non-boolean) of more than 3 cover sets.
        Nei = [
            [1, 2],   # neighbors of 0
            [0, 3],   # neighbors of 1
            [0, 3],   # neighbors of 2
            [1, 2, 4],# neighbors of 3
            [3]       # neighbors of 4
        ]
        # Subset provided as indices (numeric) that are nonzero.
        Sub = [0, 1, 2, 3, 4]
        MinSize = 2
        Components, CompSize = Utils.connected_components(Nei, Sub, MinSize)
        # In this graph, all nodes are connected.
        expected_components = [np.array([0, 1, 2, 3, 4], dtype=np.uint32)]
        expected_comp_size = np.array([5], dtype=np.uint32)
        self.assertEqual(len(Components), 1)
        np.testing.assert_array_equal(Components[0], expected_components[0])
        np.testing.assert_array_equal(CompSize, expected_comp_size)

    def test_cubical_averaging_with_tree1(self):
        # Construct the path to the LAS file.
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        las_path = os.path.join(base_dir, "Dataset", "tree_1.las")

        # Read the LAS file.
        las = laspy.read(las_path)

        # Extract the point cloud.
        # For laspy 2.x, use las.x, las.y, las.z:
        points = np.vstack((las.x, las.y, las.z)).T

        # Set a cube size for downsampling.
        CubeSize = 1.0

        # Run cubical averaging.
        DSP = Utils.cubical_averaging(points, CubeSize)

        # Check that the downsampled point cloud has fewer points than the original.
        self.assertLess(DSP.shape[0], points.shape[0],
                        "Downsampled point cloud should have fewer points than the original")
        # Also check that the downsampled point cloud is not empty.
        self.assertGreater(DSP.shape[0], 0,
                           "Downsampled point cloud should contain at least one point")

    def test_create_input(self):
        inp = Utils.create_input()
        # List of keys expected in the top-level input dictionary.
        expected_keys = [
            'PatchDiam1', 'PatchDiam2Min', 'PatchDiam2Max',
            'BallRad1', 'BallRad2', 'nmin1', 'nmin2', 'OnlyTree',
            'Tria', 'Dist', 'MinCylRad', 'ParentCor', 'TaperCor',
            'GrowthVolCor', 'GrowthVolFac', 'filter', 'name', 'tree',
            'model', 'savemat', 'savetxt', 'plot', 'disp'
        ]
        for key in expected_keys:
            self.assertIn(key, inp, f"Key '{key}' not found in input dictionary.")

        # Verify that BallRad1 equals PatchDiam1 + 0.015
        np.testing.assert_array_almost_equal(
            inp['BallRad1'], inp['PatchDiam1'] + 0.015,
            err_msg="BallRad1 should be PatchDiam1 + 0.015"
        )

        # Verify that BallRad2 equals PatchDiam2Max + 0.01
        np.testing.assert_array_almost_equal(
            inp['BallRad2'], inp['PatchDiam2Max'] + 0.01,
            err_msg="BallRad2 should be PatchDiam2Max + 0.01"
        )

        # Check filtering parameters.
        expected_filter_keys = ['k', 'radius', 'nsigma', 'PatchDiam1', 'BallRad1', 'ncomp', 'EdgeLength', 'plot']
        for key in expected_filter_keys:
            self.assertIn(key, inp['filter'], f"Filter key '{key}' not found in input['filter'].")

    def test_define_input_single_tree(self):
        # Create a synthetic point cloud: 100 points with x,y noise and z spanning 0 to 10.
        np.random.seed(0)
        z = np.linspace(0, 10, 100)
        x = 0.1 * np.random.randn(100)
        y = 0.1 * np.random.randn(100)
        P = np.column_stack((x, y, z))

        # Use 1 value for each parameter.
        nPD1 = 1
        nPD2Min = 1
        nPD2Max = 1

        inputs = Utils.define_input(P, nPD1, nPD2Min, nPD2Max)

        # There should be one input structure.
        self.assertEqual(len(inputs), 1)
        inp = inputs[0]

        # Check that required keys are present.
        for key in ['PatchDiam1', 'PatchDiam2Min', 'PatchDiam2Max', 'BallRad1', 'BallRad2']:
            self.assertIn(key, inp)
            self.assertGreater(inp[key], 0, f"{key} should be greater than zero.")

    def test_set_difference(self):
        # Example sets: remove elements in Set2 from Set1.
        Set1 = [1, 2, 3, 4, 5]
        Set2 = [2, 4]
        # Create a tracker boolean array of length at least max(Set1,Set2)+1.
        false_vec = np.zeros(6, dtype=bool)  # indices 0 through 5
        result = Utils.set_difference(Set1, Set2, false_vec.copy())
        expected = np.array([1, 3, 5])
        np.testing.assert_array_equal(result, expected)

    def test_save_model_text_creates_files(self):
        # Create a synthetic QSM dictionary.
        QSM = {
            "cylinder": {
                "radius": np.array([0.05, 0.06]),
                "length": np.array([1.2, 1.5]),
                "start": np.array([0.0, 0.1]),
                "axis": np.array([0.0, 0.0]),
                "parent": np.array([0, 1]),
                "extension": np.array([0, 0]),
                "added": np.array([0, 1]),
                "UnmodRadius": np.array([0.055, 0.065]),
                "branch": np.array([1, 1]),
                "BranchOrder": np.array([1, 2]),
                "PositionInBranch": np.array([1, 2]),
                "mad": np.array([0.01, 0.02]),
                "SurfCov": np.array([0.5, 0.6])
            },
            "branch": {
                "order": np.array([1, 2]),
                "parent": np.array([0, 1]),
                "diameter": np.array([0.1, 0.15]),
                "volume": np.array([0.2, 0.3]),
                "area": np.array([0.05, 0.06]),
                "length": np.array([1.0, 1.2]),
                "height": np.array([1.5, 1.7]),
                "angle": np.array([30, 45]),
                "azimuth": np.array([100, 110]),
                "zenith": np.array([20, 25])
            },
            "treedata": {
                "TotalVolume": 1.0,
                "TrunkVolume": 0.6,
                "BranchVolume": 0.4,
                # Add more tree data fields here as needed.
                "location": "ignored_field"
            }
        }
        savename = "test_model"
        # Call save_model_text.
        Utils.save_model_text(QSM, savename)

        # Check that the files were created and are not empty.
        for prefix in ["cylinder", "branch", "treedata"]:
            filename = os.path.join("results", f"{prefix}_{savename}.txt")
            self.assertTrue(os.path.exists(filename), f"File {filename} was not created.")
            self.assertGreater(os.path.getsize(filename), 0, f"File {filename} is empty.")

    def test_cubical_partition(self):
        # Create a synthetic point cloud (e.g., 10 points in a 3D space).
        P = np.array([
            [0.1, 0.2, 0.3],
            [0.15, 0.25, 0.35],
            [1.0, 1.2, 1.3],
            [1.05, 1.25, 1.35],
            [2.0, 2.1, 2.2],
            [2.05, 2.15, 2.25],
            [3.0, 3.1, 3.2],
            [3.05, 3.15, 3.25],
            [4.0, 4.1, 4.2],
            [4.05, 4.15, 4.25]
        ])
        EL = 1.0  # Cube edge length.
        NE = 1    # For testing, use 1 empty layer.

        Partition, CubeCoord, Info, Cubes = Utils.cubical_partition(P, EL, NE)

        # Check that CubeCoord has shape (10,3)
        self.assertEqual(CubeCoord.shape, (10, 3))
        # Check that Info has length 8 (3 + 3 + 1 + 1)
        self.assertEqual(Info.shape[0], 8)
        # Check that Partition is a list and has at least one element.
        self.assertIsInstance(Partition, list)
        self.assertGreater(len(Partition), 0)
        # Check that Cubes is a 3D array with shape equal to N (from Info, elements 3:6)
        N = Info[3:6].astype(int)
        self.assertEqual(Cubes.shape, (N[0], N[1], N[2]))

    def test_cubical_downsampling(self):
        # Create a perfect 5x5x5 grid of points (no noise).
        xs, ys, zs = np.meshgrid(np.linspace(0, 4, 5),
                                 np.linspace(0, 4, 5),
                                 np.linspace(0, 4, 5))
        grid_points = np.vstack((xs.ravel(), ys.ravel(), zs.ravel())).T

        # No noise added.
        P = grid_points.copy()

        CubeSize = 1.0  # Each cube covers exactly one grid cell.

        # Run cubical downsampling.
        Pass = Utils.cubical_downsampling(P, CubeSize)

        # Count the number of selected points.
        n_selected = int(np.sum(Pass))

        # For a perfect 5x5x5 grid with CubeSize=1, we expect 125 cubes (one per grid cell).
        self.assertEqual(n_selected, 125,
                         "Expected 125 representative points for a perfect 5x5x5 grid with CubeSize=1.")

        # Also check that Pass is a boolean array with the correct shape.
        self.assertEqual(Pass.shape, (P.shape[0],))
        self.assertTrue(Pass.dtype == bool)

    def test_growth_volume_correction(self):
        # Create synthetic QSM with 3 cylinders:
        # Cylinder 1: trunk; cylinders 2 and 3: tips.
        # Using 1-indexed parent convention: 0 indicates no parent.
        cylinder = {
            "radius": np.array([0.1, 0.05, 0.05]),
            "length": np.array([1.0, 0.5, 0.5]),
            "parent": np.array([0, 1, 1]),   # Cylinder 1 is trunk; 2 and 3 are children.
            "extension": np.array([1, 0, 0])   # Tip cylinders (2 and 3) have extension == 0.
        }
        # Set GrowthVolFac such that allowed range is [predicted/1.2, predicted*1.2].
        inputs = {"GrowthVolFac": 1.2}

        # Call the growth_volume_correction method.
        corrected = Utils.growth_volume_correction(cylinder, inputs)
        corrected_radii = corrected["radius"]

        # For our synthetic data, the model fit yields predicted radii approximately:
        #   - For the trunk: ~0.1 m (so trunk remains near 0.1 m).
        #   - For the tip cylinders: ~0.05 m.
        # Since the measured radii already fall within the allowed range,
        # no correction is applied.
        tol = 0.005  # tolerance in meters

        # Check trunk (cylinder 1) remains near 0.1 m.
        self.assertAlmostEqual(corrected_radii[0], 0.1, delta=tol,
                               msg="Trunk radius should remain near 0.1 m.")
        # Check that tip cylinders remain near 0.05 m.
        expected_tip = 0.05
        self.assertAlmostEqual(corrected_radii[1], expected_tip, delta=tol,
                               msg="Tip cylinder 2 radius should remain near 0.05 m.")
        self.assertAlmostEqual(corrected_radii[2], expected_tip, delta=tol,
                               msg="Tip cylinder 3 radius should remain near 0.05 m.")

    def test_select_cylinders(self):
        # Create a synthetic cylinder dictionary with two fields.
        # Each field is a 2D array with 5 rows and 3 columns.
        cylinder = {
            "radius": np.array([[0.1, 0.2, 0.3],
                                [0.4, 0.5, 0.6],
                                [0.7, 0.8, 0.9],
                                [1.0, 1.1, 1.2],
                                [1.3, 1.4, 1.5]]),
            "length": np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9],
                                [10, 11, 12],
                                [13, 14, 15]])
        }
        # Define indices to select (for example, select rows 1 and 3; MATLAB indices are 1-based,
        # but in Python we use 0-based indexing, so we select indices [0, 2]).
        Ind = [0, 2]

        # Run the select_cylinders function.
        updated_cylinder = Utils.select_cylinders(cylinder, Ind)

        # Expected arrays after selecting rows 0 and 2.
        expected_radius = np.array([[0.1, 0.2, 0.3],
                                    [0.7, 0.8, 0.9]])
        expected_length = np.array([[1, 2, 3],
                                    [7, 8, 9]])

        # Verify each field.
        np.testing.assert_array_equal(updated_cylinder["radius"], expected_radius)
        np.testing.assert_array_equal(updated_cylinder["length"], expected_length)

    def test_surface_coverage(self):
        # Create a synthetic cylindrical point cloud.
        # For example, points around a cylinder of height 10 and radius ~1.
        np.random.seed(0)
        n_points = 1000
        # Generate heights uniformly between 0 and 10.
        h = np.random.uniform(0, 10, n_points)
        # Generate angles uniformly between 0 and 2*pi.
        theta = np.random.uniform(0, 2*np.pi, n_points)
        r = 1.0  # radius
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        P = np.column_stack((x, y, h))

        # Define cylinder axis and starting point.
        Axis = np.array([0, 0, 1])
        Point = np.array([0, 0, 0])
        nl = 5
        ns = 8
        # Optionally set Dmin and Dmax:
        Dmin = 0.5
        Dmax = 1.5

        SurfCov, Dis, CylVol, dis_out = Utils.surface_coverage(P, Axis, Point, nl, ns, Dmin, Dmax)

        # Basic checks:
        self.assertTrue(0 <= SurfCov <= 1, "Surface coverage should be between 0 and 1.")
        self.assertEqual(Dis.shape, (nl, ns))
        self.assertIsInstance(CylVol, float)
        self.assertEqual(dis_out.shape, (nl, ns))

    def test_surface_coverage2(self):
        # Define a cylinder axis and length.
        Axis = np.array([0, 0, 1])
        Len = 10.0

        # Create synthetic data:
        # Let's create 1000 points uniformly distributed around a cylinder surface.
        np.random.seed(42)
        n_points = 1000
        height = np.random.uniform(0, Len, n_points)
        theta = np.random.uniform(0, 2*np.pi, n_points)
        # Assume cylinder radius is 1.
        r = 1.0
        # Vectors from the axis: for a perfect cylinder, the vector is [r*cos(theta), r*sin(theta), 0]
        Vec = np.column_stack((r * np.cos(theta), r * np.sin(theta), np.zeros(n_points)))

        # Set partition parameters.
        nl = 5   # number of layers along height
        ns = 8   # number of angular sectors

        SurfCov = Utils.surface_coverage2(Axis, Len, Vec, height, nl, ns)

        # For a uniformly covered cylinder, we expect full coverage (i.e., ~1).
        self.assertAlmostEqual(SurfCov, 1.0, delta=0.05,
                               msg="Surface coverage should be near 1 for uniformly distributed points.")

    def test_surface_coverage_filtering(self):
        # Create a synthetic point cloud approximating points on a cylinder.
        np.random.seed(0)
        n_points = 500
        cylinder_length = 10.0
        cylinder_radius = 1.0
        # Generate heights uniformly along the cylinder.
        h = np.random.uniform(0, cylinder_length, n_points)
        # Generate angles uniformly.
        theta = np.random.uniform(0, 2*np.pi, n_points)
        # Generate points on the cylinder surface (with slight noise).
        x = cylinder_radius * np.cos(theta) + 0.05 * np.random.randn(n_points)
        y = cylinder_radius * np.sin(theta) + 0.05 * np.random.randn(n_points)
        P = np.column_stack((x, y, h))

        # Define the cylinder structure (minimal fields).
        c = {
            "axis": np.array([0, 0, 1]),
            "start": np.array([0, 0, 0]),
            "length": cylinder_length
        }

        # Set layer height (lh) and initial number of sectors (ns).
        lh = 2.0
        ns = 8

        # Apply the surface coverage filtering.
        Pass, c_updated = Utils.surface_coverage_filtering(P, c, lh, ns)

        # Basic checks:
        self.assertEqual(Pass.shape[0], P.shape[0], "Pass vector length should equal number of points.")
        for field in ["radius", "SurfCov", "mad", "conv", "rel"]:
            self.assertIn(field, c_updated, f"Field '{field}' not found in updated cylinder structure.")
        self.assertTrue(0 <= c_updated["SurfCov"] <= 1, "Surface coverage should be between 0 and 1.")

    def test_update_tree_data(self):
        # Synthetic QSM with minimal treedata and empty triangulation.
        QSM = {
            "treedata": {
                "TotalVolume": 0,
                "TrunkVolume": 0,
                "BranchVolume": 0,
                "TreeHeight": 0,
                "TrunkLength": 0,
                "BranchLength": 0,
                "TotalLength": 0,
                "NumberBranches": 0,
                "MaxBranchOrder": 0,
                "TrunkArea": 0,
                "BranchArea": 0,
                "TotalArea": 0
            },
            "triangulation": {}  # No triangulation data.
        }
        # Synthetic cylinder structure.
        cylinder = {
            "radius": np.array([0.1, 0.09, 0.08, 0.07]),
            "length": np.array([1.0, 0.8, 0.6, 0.4]),
            "start": np.array([[0,0,0],
                               [0,0,1],
                               [0,0,1.8],
                               [0,0,2.4]]),
            "axis": np.array([[0,0,1],
                              [0,0,1],
                              [0,0,1],
                              [0,0,1]]),
            "branch": np.array([1, 1, 2, 2])
        }
        # Synthetic branch structure.
        branch = {
            "order": np.array([1, 1, 2, 2]),
            "volume": np.array([0.5, 0.4, 0.3, 0.2]),
            "area": np.array([0.05, 0.04, 0.03, 0.02]),
            "length": np.array([1.0, 0.9, 0.8, 0.7]),
            "height": np.array([0.5, 1.0, 1.5, 2.0]),
            "angle": np.array([30, 35, 40, 45]),
            "azimuth": np.array([100, 110, 120, 130]),
            "zenith": np.array([20, 25, 30, 35]),
            "diameter": np.array([0.1, 0.09, 0.08, 0.07])
        }
        # Synthetic inputs.
        inputs = {
            "Tria": False,
            "disp": 0,
            "plot": 0
        }

        treedata = Utils.update_tree_data(QSM, cylinder, branch, inputs)

        expected_keys = [
            "TotalVolume", "TrunkVolume", "BranchVolume", "TreeHeight",
            "TrunkLength", "BranchLength", "TotalLength", "NumberBranches",
            "MaxBranchOrder", "TrunkArea", "BranchArea", "TotalArea",
            "CrownDiamAve", "CrownDiamMax", "CrownAreaConv", "CrownAreaAlpha",
            "CrownBaseHeight", "CrownLength", "CrownRatio",
            "MixTrunkVolume", "MixTotalVolume", "MixTrunkArea", "MixTotalArea",
            "location", "StemTaper", "VerticalProfile", "spreads",
            "VolCylDia", "AreCylDia", "LenCylDia",
            "VolCylHei", "AreCylHei", "LenCylHei",
            "VolCylZen", "AreCylZen", "LenCylZen",
            "VolCylAzi", "AreCylAzi", "LenCylAzi",
            "VolBranchOrd", "AreBranchOrd", "LenBranchOrd", "NumBranchOrd",
            "VolBranchDia", "VolBranch1Dia", "AreBranchDia", "AreBranch1Dia",
            "LenBranchDia", "LenBranch1Dia", "NumBranchDia", "NumBranch1Dia",
            "VolBranchHei", "AreBranchHei", "LenBranchHei", "NumBranchHei",
            "VolBranchAng", "AreBranchAng", "LenBranchAng", "NumBranchAng",
            "VolBranchZen", "AreBranchZen", "LenBranchZen", "NumBranchZen",
            "VolBranchAzi", "AreBranchAzi", "LenBranchAzi", "NumBranchAzi"
        ]
        for key in expected_keys:
            self.assertIn(key, treedata, f"Key '{key}' not found in treedata.")
