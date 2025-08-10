import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from TreeQSMSteps.cover_sets import cover_sets

def test_uniform_cover_simple_grid():
    """Test uniform cover on a regular 3x3x3 grid of points"""
    # Create a 3x3x3 grid with spacing 0.5 units
    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 1, 3)
    z = np.linspace(0, 1, 3)
    X, Y, Z = np.meshgrid(x, y, z)
    P = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    #print(P)

    inputs = {
        'BallRad1': 0.6,
        'PatchDiam1': 0.4,
        'nmin1': 1
    }

    cover = cover_sets(P, inputs)

    # All 27 points should be centers (since spaced exactly PatchDiam1 apart)
    assert len(cover['center']) == 27
    # Each cover set should have 1 point (itself)
    assert all(len(ball) == 1 for ball in cover['ball'])
    # Each center should have ~6 neighbors (except edge points)
    #print(cover['neighbor'])
    # (simplified check: ensure neighbors exist)
    assert all(len(neighbors) > 0 for neighbors in cover['neighbor'])


def test_variable_cover_line():
    """Test variable cover on a line with varying RelSize"""
    # Create 10 points along a line
    P = np.zeros((10, 3))
    P[:, 0] = np.arange(10) * 0.5  # x-coordinates: 0.0, 0.5, ..., 4.5
    RelSize = np.array([255] * 5 + [32] * 5)  # First 5 large, last 5 small

    inputs = {
        'BallRad2': 2.0,
        'PatchDiam2Min': 0.3,
        'PatchDiam2Max': 1.2,
        'nmin2': 1
    }

    cover = cover_sets(P, inputs, RelSize)
    #print(cover['center'])
    centers = P[cover['center']]

    # First 5 points should have smaller cover sets (cover less points) as smaller sets are generated first
    first_half_sizes = [len(ball) for ball in cover['ball'][:5]]
    #print(first_half_sizes)
    second_half_sizes = [len(ball) for ball in cover['ball'][5:]]
    #print(second_half_sizes)
    assert np.mean(first_half_sizes) < np.mean(second_half_sizes)


def test_single_point():
    """Test edge case: point cloud with only one point"""
    P = np.array([[0.0, 0.0, 0.0]])
    inputs = {
        'BallRad1': 0.1,
        'PatchDiam1': 0.2,
        'nmin1': 1
    }

    cover = cover_sets(P, inputs)
    assert len(cover['center']) == 1
    assert len(cover['ball'][0]) == 1
    assert len(cover['neighbor'][0]) == 0  # No neighbors


def test_relsize_zero_exclusion():
    """Test points with RelSize=0 are excluded from cover sets"""
    P = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0]
    ])
    RelSize = np.array([55, 0, 55])  # Middle point excluded

    inputs = {
        'BallRad2': 1.5,
        'PatchDiam2Min': 0.5,
        'PatchDiam2Max': 1.0,
        'nmin2': 1
    }

    cover = cover_sets(P, inputs, RelSize)
    centers = cover['center']
    # print(cover['ball'])
    # Only first and third points should be centers
    assert set(centers) == {0, 2}
    # Middle point (index 1) should not belong to any cover set
    assert 1 not in np.concatenate(cover['ball'])


def test_min_points_per_ball():
    """Test balls with < nmin points are rejected"""
    # Create 4 points in a ball (nmin=5)
    P = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1]
    ])
    inputs = {
        'BallRad1': 0.2,
        'PatchDiam1': 0.3,
        'nmin1': 5  # Require 5 points per ball
    }

    cover = cover_sets(P, inputs)
    # No cover sets should be created (only 4 points available)
    #print(cover['ball'])
    assert len(cover['center']) == 0


def test_neighbor_symmetry():
    """Test neighbor relationships are symmetric"""
    """Note that if cover set generation start with P[1], this test will not work"""
    # Two points within each other's BallRad
    P = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0]
    ])
    inputs = {
        'BallRad1': 0.15,
        'PatchDiam1': 0.15,
        'nmin1': 1
    }

    cover = cover_sets(P, inputs)
    #print(cover)
    neighbors_0 = cover['neighbor'][0]
    neighbors_1 = cover['neighbor'][1]
    assert 1 in neighbors_0
    assert 0 in neighbors_1


def test_variable_cover_edge_case():
    """Test variable cover with maximum BallRad"""
    P = np.random.rand(100, 3) * 10.0  # Random points in 10x10x10 cube
    RelSize = np.full(100, 255)  # All points use maximum size

    inputs = {
        'BallRad2': 5.0,
        'PatchDiam2Min': 1.0,
        'PatchDiam2Max': 3.0,
        'nmin2': 3
    }

    cover = cover_sets(P, inputs, RelSize)
    #print(cover['ball'])
    # Ensure at least some cover sets are created
    assert len(cover['center']) > 0
    # All cover sets should have >=3 points
    # This assert doesn't work as when a cover set is created, it meets minimum point number requirement,
    # but when other cover sets are created, its points may be taken away because of closer to other centers
    #assert all(len(ball) >= 3 for ball in cover['ball'])


if __name__ == "__main__":
    #test_uniform_cover_simple_grid()
    test_variable_cover_line()
    #test_single_point()
    test_relsize_zero_exclusion()
    #test_min_points_per_ball()
    #test_neighbor_symmetry()
    test_variable_cover_edge_case()
    print("All tests passed!")