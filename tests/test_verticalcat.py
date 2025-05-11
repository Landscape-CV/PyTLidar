from tools.verticalcat import verticalcat
import numpy as np

# Example usage
cell_array = [np.array([1, 2]), np.array([3, 4, 5]), np.array([6])]
vector, ind_elements = verticalcat(cell_array)
print("Vector:", vector)
print("IndElements:\n", ind_elements)

# Test cases
def test_verticalcat():
    assert np.array_equal(verticalcat([np.array([1, 2]), np.array([3])])[0], np.array([1, 2, 3]))
    assert np.array_equal(verticalcat([np.array([10]), np.array([20, 30]), np.array([40, 50])])[0], np.array([10, 20, 30, 40, 50]))
    assert np.array_equal(verticalcat([np.array([]), np.array([1])])[0], np.array([1]))
    print("All tests passed!")

test_verticalcat()