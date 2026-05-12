import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from TreeQSMSteps.point_model_distance import point_model_distance




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

    assert abs(result['mean'] - 0.03776242558395426)< 0.001
    assert len(result['CylDist']) ==2





if __name__ == '__main__':
    test_multiple_cylinders()
