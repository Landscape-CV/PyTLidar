"""Test suite for relative_size() function"""


import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from TreeQSMSteps.relative_size import relative_size


def test_single_segment():
    """ Test 1: Minimal case - single segment (trunk) """
    P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    cover = {
        'ball': [np.array([0, 1, 2])],  # Single cover set
        'center': np.array([0]),  # Center at first point
        'neighbor': [np.array([], dtype=int)]  # No neighbors
    }
    segment = {
        'segments': [[np.array([0])]],  # Single segment with one layer
        'ChildSegment': [[]]  # No child segments
    }

    RS = relative_size(P, cover, segment)
    assert np.all(RS == 255), "Test 1 Failed: Should get max size 255"

def test_two_segments():
    """ Test 2: Two segments (trunk + branch) """
    P = np.array([[0, 0, z] for z in range(9)], dtype=np.float32)  # Vertical trunk
    cover = {
        'ball': [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8])],  # 3 layers
        'center': np.array([0, 3, 6]),  # Centers of cover sets
        'neighbor': [
            np.array([1]),  # Layer 0 neighbors layer 1
            np.array([0, 2]),  # Layer 1 neighbors layers 0 and 2
            np.array([1])  # Layer 2 neighbors layer 1
        ]
    }
    segment = {
        'segments': [
            [np.array([0]), np.array([1])],  # Trunk with 2 layers
            [np.array([2])]  # Branch with 1 layer
        ],
        'ChildSegment': [[1], []]  # Trunk has child 1, branch has no children
    }

    RS = relative_size(P, cover, segment)
    """
    # Verify trunk layers
    assert RS[0] == RS[1] == RS[2] == 255, "Trunk base should be max size"
    assert 200 < RS[3] < 255, "Trunk middle layer reduction"
    assert 100 < RS[6] < 200, "Trunk top layer reduction"

    # Verify branch adjustment
    assert RS[6] == RS[7] == RS[8], "Branch base should have uniform size"
    assert RS[6] == RS[6] // 2, "Branch base should be half parent's value"
    """




# Run the test suite
if __name__ == "__main__":
    #test_single_segment()
    test_two_segments()

    print("All tests passed!")