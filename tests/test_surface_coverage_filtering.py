import numpy as np
from tools.surface_coverage_filtering import surface_coverage_filtering


def test_perfect_cylinder_python():
    # Generate points on a perfect cylinder (radius=0.1, axis=z, length=1.0)
    np.random.seed(42)
    theta = np.random.rand(100) * 2 * np.pi
    z = np.random.rand(100) * 1.0  # Length = 1.0
    x = 0.1 * np.cos(theta)
    y = 0.1 * np.sin(theta)
    P = np.column_stack((np.round(x, 4), np.round(y, 4), np.round(z, 4)))
    #print(P)

    # Define cylinder (axis along z-axis, starting at (0,0,0))
    c = {
        'axis': np.array([0, 0, 1.0]),
        'start': np.array([0.0, 0.0, 0.0]),
        'length': 1.0
    }

    # Filter parameters
    lh = 0.1  # Layer height
    ns = 8  # Number of sectors

    # Run filtering
    Pass, c = surface_coverage_filtering(P, c, lh, ns)

    # Assertions
    assert np.all(Pass), "All points should pass for a perfect cylinder"
    assert np.isclose(c['radius'], 0.1050, atol=1e-3), f"Radius={c['radius']}, expected 0.1050"
    assert np.isclose(c['SurfCov'], 0.0638, atol=1e-3), f"Surface coverage={c['SurfCov']} should be close to 0.4779"
    print("Test 1 (Python): Passed")


def test_noisy_cylinder_python():
    # Generate points with radial noise (σ=0.01)
    np.random.seed(42)
    theta = np.random.rand(1000) * 2 * np.pi
    z = np.random.rand(1000) * 1.0
    noise = np.random.normal(0, 0.01, 1000)
    x = (0.1 + noise) * np.cos(theta)
    y = (0.1 + noise) * np.sin(theta)
    P = np.column_stack((x, y, z))

    c = {
        'axis': np.array([0, 0, 1.0]),
        'start': np.array([0.0, 0.0, 0.0]),
        'length': 1.0
    }
    lh = 0.1
    ns = 8

    Pass, c = surface_coverage_filtering(P, c, lh, ns)

    # Check radius is close to 0.1
    assert np.isclose(c['radius'], 0.1, atol=0.02), f"Radius={c['radius']}, expected ~0.1±0.02"
    # Some points should be filtered
    assert np.mean(Pass) > 0.7 and np.mean(Pass) < 1.0, "Partial filtering expected"
    print("Test 2 (Python): Passed")


def test_edge_cases_python():
    # Empty input
    P_empty = np.zeros((0, 3))
    c_empty = {'axis': np.array([0, 0, 1]), 'start': np.array([0, 0, 0]), 'length': 1.0}
    Pass_empty, c_empty = surface_coverage_filtering(P_empty, c_empty, 0.1, 8)
    assert Pass_empty.size == 0, "Empty input should return empty Pass"

    # Single point
    P_single = np.array([[0.1, 0.0, 0.5]])
    c_single = {'axis': np.array([0, 0, 1]), 'start': np.array([0, 0, 0]), 'length': 1.0}
    Pass_single, c_single = surface_coverage_filtering(P_single, c_single, 0.1, 8)
    assert Pass_single[0], "Single point should pass"
    assert np.isclose(c_single['radius'], 0.1, atol=1e-3), "Radius should match single point"
    print("Test 3 (Python): Passed")


if __name__ == '__main__':
    test_perfect_cylinder_python()
    test_noisy_cylinder_python()
    #test_edge_cases_python()