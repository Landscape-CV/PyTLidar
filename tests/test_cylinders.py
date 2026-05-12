import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from TreeQSMSteps.cylinders import cylinders
from TreeQSMSteps.cover_sets import cover_sets
from TreeQSMSteps.tree_sets import tree_sets
from TreeQSMSteps.segments import segments




def test_case():
    """Large point sets, multiple cylinders"""
    # Generate points with radial noise (σ=0.01)
    np.random.seed(42)
    theta = np.random.rand(100) * 2 * np.pi
    #print(theta)
    z = np.repeat(np.arange(1, 11, 1).astype(float), 10)
    #print(z)
    noise = np.random.rand(1, 100).flatten()
    #print(noise)
    x = (10.0 + noise) * np.cos(theta)
    y = (10.0 + noise) * np.sin(theta)
    #print(x.shape, y.shape, z.shape)
    P = np.column_stack((x, y, z))
    #print(P)

    cover = {'ball': np.arange(100).reshape(10, 10)}
    segment = {
        'segments': np.array([np.array([[0]]), np.array([[1]]), np.array([[2], [3], [4], [5], [6], [7], [8], [9]])],dtype = object),
        'ParentSegment': np.array([0, 0, 1]),
        'ChildSegment': np.array([[1], [2], []],dtype = object)
    }

    inputs = {'MinCylRad': 0.005, 'ParentCor': False, 'TaperCor': False }
    cylinder = cylinders(P, cover, segment, inputs)
    print("\nTest 5 Results:")
    print(cylinder)
    print(f"Radii: {cylinder['radius']}")
    print(f"Lengths: {cylinder['length']}")


# Run tests

test_case()