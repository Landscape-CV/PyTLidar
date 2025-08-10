import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from TreeQSMSteps.point_model_distance import point_model_distance


def test_single_vertical_cylinder():
    """Perfect fit along cylinder axis"""
    np.random.seed(42)
    P = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]], dtype=np.float32)
    cylinder = {
        'radius': np.array([0.1]),
        'length': np.array([3.0]),
        'start': np.array([[0, 0, 0]]),
        'axis': np.array([[0, 0, 3]]),
        'BranchOrder': np.array([0])
    }
    result = point_model_distance(P, cylinder)
    assert result['median'] < 0.1
    assert result['max'] < 0.1


def test_horizontal_cylinder_with_offset():
    """Mixed points inside/outside cylinder"""
    np.random.seed(42)
    P = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0], [0, 1, 0]], dtype=np.float32)
    cylinder = {
        'radius': np.array([0.5]),
        'length': np.array([3.0]),
        'start': np.array([[0, 0, 0]]),
        'axis': np.array([[3, 0, 0]]),
        'BranchOrder': np.array([1])
    }
    result = point_model_distance(P, cylinder)
    assert abs(result['mean'] - 0.25) < 0.1


def test_multiple_cylinders():
    """Trunk (vertical) + branch (horizontal)"""
    np.random.seed(15)
    # Generate sample data
    trunk_points = np.column_stack([np.random.rand(100, 1) * 0.5, np.random.rand(100, 1) * 0.5,
                                    np.linspace(0, 5, 100)])
    branch_points = np.column_stack([np.linspace(0, 1, 50),
                                     np.random.rand(50, 1) * 0.6, 5 + np.random.rand(50, 1) * 0.6])
    P = np.vstack([trunk_points, branch_points])
    #print(P)

    cylinder = {
        'radius': np.array([0.5, 0.6]),
        'length': np.array([5.0, 1.0]),
        'start': np.array([[0, 0, 0], [5, 0, 0]]),
        'axis': np.array([[0, 0, 5], [1, 0, 0]]),
        'BranchOrder': np.array([0, 1])
    }

    result = point_model_distance(P, cylinder)
    print(result)
    #assert result['TrunkMedian'] < 0.1
    #assert result['BranchMedian'] < 0.15


def test_empty_point_cloud():
    """Edge case with no input points"""
    P = np.zeros((0, 3), dtype=np.float32)
    cylinder = {
        'radius': np.array([0.1]),
        'length': np.array([1.0]),
        'start': np.array([[0, 0, 0]]),
        'axis': np.array([[0, 0, 1]]),
        'BranchOrder': np.array([0])
    }
    result = point_model_distance(P, cylinder)
    assert len(result['CylDist']) == 0


def test_single_point():
    """Minimal valid input"""
    np.random.seed(42)
    P = np.array([[0.5, 0, 0]], dtype=np.float32)
    cylinder = {
        'radius': np.array([0.5]),
        'length': np.array([2.0]),
        'start': np.array([[0, 0, 0]]),
        'axis': np.array([[0, 0, 2]]),
        'BranchOrder': np.array([0])
    }
    result = point_model_distance(P, cylinder)
    assert abs(result['mean']) < 0.01

if __name__ == '__main__':
    #test_single_vertical_cylinder()
    #test_horizontal_cylinder_with_offset()
    test_multiple_cylinders()
