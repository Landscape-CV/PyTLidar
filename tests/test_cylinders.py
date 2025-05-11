import numpy as np
from main_steps.cylinders import cylinders
from main_steps.cover_sets import cover_sets
from main_steps.tree_sets import tree_sets
from main_steps.segments import segments


def test_case_1():
    """Simple vertical branch"""
    P = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]])
    cover = {'ball': [np.array([0, 1, 2, 3])]}
    segment = {
        'segments': [np.array([0])],  # 0-based index
        'ParentSegment': [0],
        'ChildSegment': [[]]
    }
    inputs = {
        'MinCylRad': 0.01,
        'ParentCor': True,
        'TaperCor': True,
        'GrowthVolCor': False
    }

    cylinder = cylinders(P, cover, segment, inputs)
    print("\nTest 1 Results:")
    print(f"Number of cylinders: {len(cylinder['radius'])}")
    print(f"Radii: {cylinder['radius']}")


def test_case_2():
    """Y-shaped structure"""
    P = np.vstack([
        np.column_stack([np.zeros(10), np.zeros(10), np.linspace(0, 3, 10)]),
        np.column_stack([np.linspace(0, 1, 10), np.zeros(10), np.linspace(3, 4, 10)])
    ])
    cover = {'ball': [np.arange(10), np.arange(10, 20)]}
    segment = {
        'segments': [[0], [1]],
        'ParentSegment': [0, 0],  # Second branch connects to first
        'ChildSegment': [[1], []]
    }
    inputs = {'MinCylRad': 0.05, 'TaperCor': True}

    cylinder = cylinders(P, cover, segment, inputs)
    print("\nTest 2 Results:")
    print(f"Branch orders: {cylinder['BranchOrder']}")
    print(f"Parent relationships: {cylinder['parent']}")


def test_case_3():
    """Edge case: minimal points"""
    P = np.array([[0, 0, 0], [0.1, 0, 0.1], [-0.1, 0, 0.2]])
    cover = {'ball': [np.array([0, 1, 2])]}
    segment = {
        'segments': [np.array([0])],
        'ParentSegment': [0]
    }
    inputs = {'MinCylRad': 0.005}

    cylinder = cylinders(P, cover, segment, inputs)
    print("\nTest 3 Results:")
    print(f"Radii: {cylinder['radius']}")
    print(f"Lengths: {cylinder['length']}")


def test_case_4():
    """Large point sets, small segments"""
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
        'segments': [np.array([[0]]), np.array([[1], [2]]), np.array([[3], [4]]), np.array([[5], [6], [7], [8]]), np.array([[9]])],
        'ParentSegment': [0, 0, 1, 2, 3],
        'ChildSegment': [[1], [2], [3], [4], []]
    }
    '''
    inputs = {'BallRad1': 8, 'PatchDiam1': 8, 'nmin1': 1}

    cover = cover_sets(P, inputs)
    cover1, Base, Forb = tree_sets(P, cover, inputs)

    segment = segments(cover1, Base, Forb)
    '''

    inputs = {'MinCylRad': 0.005, 'ParentCor': False, 'TaperCor': False }
    cylinder = cylinders(P, cover, segment, inputs)
    print("\nTest 4 Results:")
    print(cylinder)
    print(f"Radii: {cylinder['radius']}")
    print(f"Lengths: {cylinder['length']}")


def test_case_5():
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
        'segments': [np.array([[0]]), np.array([[1]]), np.array([[2], [3], [4], [5], [6], [7], [8], [9]])],
        'ParentSegment': [0, 0, 1],
        'ChildSegment': [[1], [2], []]
    }

    inputs = {'MinCylRad': 0.005, 'ParentCor': False, 'TaperCor': False }
    cylinder = cylinders(P, cover, segment, inputs)
    print("\nTest 5 Results:")
    print(cylinder)
    print(f"Radii: {cylinder['radius']}")
    print(f"Lengths: {cylinder['length']}")


# Run tests
#test_case_1()
#test_case_2()
#test_case_3()
#test_case_4()
test_case_5()