import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from TreeQSMSteps.branches import branches

# Assuming the branches() function from the previous Python translation is defined/imported

def test_branches():
    # Create a simple test case similar to the MATLAB example.
    cylinder = {}
    cylinder['radius'] = np.array([0.2, 0.1, 0.08], dtype=np.float32)  # in meters
    cylinder['length'] = np.array([3, 2, 1.8], dtype=np.float32)  # in meters
    cylinder['axis'] = np.array([[0, 0, 1],
                                 [1, 0, 0],
                                 [0, 1, 0]], dtype=np.float32)  # unit vectors
    cylinder['branch'] = np.array([0, 1, 2], dtype=np.int32)  # branch numbers
    cylinder['BranchOrder'] = np.array([0, 1, 1], dtype=np.int32) # branch orders(trunk is 0)
    cylinder['added'] = np.array([False, False, False])
    cylinder['parent'] = np.array([-1, 0, 0], dtype=np.int32)  # trunk has no parent (0); others attached to cylinder 1
    cylinder['start'] = np.array([[0, 0, 0],
                                  [0, 0, 3],
                                  [0, 0, 3]], dtype=np.float32)  # starting coordinates
    cylinder['extension'] = np.array([1, -1, -1], dtype=np.int32)  # no extension

    branch = branches(cylinder)

    # Print the branch structure for verification.
    print("Branch structure:")
    for key, value in branch.items():
        print(f"{key}: {value}")


if __name__ == '__main__':
    test_branches()