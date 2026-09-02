import numpy as np

from PyTLidar.Utils import Utils
from PyTLidar.TreeQSMSteps.tree_sets import _csr_neighbors, _Links, _branch_components


def _random_neighbors(rng, nb, degree):
    Nei = [set() for _ in range(nb)]
    for i in range(nb):
        for j in rng.integers(0, nb, degree):
            if j != i:
                Nei[i].add(int(j))
                Nei[int(j)].add(i)
    return np.array([np.array(sorted(n), dtype=int) for n in Nei], dtype=object)


def _same(a, b):
    ca, sa = a
    cb, sb = b
    if len(ca) != len(cb):
        return False
    if len(ca) == 0:
        return True
    return all(np.array_equal(x, y) and x.dtype == y.dtype for x, y in zip(ca, cb)) and np.array_equal(sa, sb)


def test_components_match_the_reference_with_and_without_links():
    rng = np.random.Generator(np.random.Philox(5))
    for trial in range(20):
        nb = int(rng.integers(5, 300))
        Nei = _random_neighbors(rng, nb, int(rng.integers(1, 4)))
        Fal = np.zeros(nb, dtype=bool)
        Sub = rng.random(nb) < rng.uniform(0.2, 0.9)
        indptr, indices = _csr_neighbors(Nei)
        links = _Links()
        ref = Utils.connected_components_array(Nei, Sub.copy(), 1, Fal)
        assert _same(ref, _branch_components(indptr, indices, links, Sub))
        # append a few links the way define_main_branches does, then compare again
        for _ in range(int(rng.integers(1, 6))):
            I, J = rng.integers(0, nb, 2)
            Nei[I] = np.append(Nei[I], J)
            Nei[J] = np.append(Nei[J], I)
            links.add(I, J)
            links.add(J, I)
        ref = Utils.connected_components_array(Nei, Sub.copy(), 1, Fal)
        assert _same(ref, _branch_components(indptr, indices, links, Sub))


def test_empty_subset():
    Nei = np.array([np.array([1]), np.array([0])], dtype=object)
    indptr, indices = _csr_neighbors(Nei)
    comps, sizes = _branch_components(indptr, indices, _Links(), np.zeros(2, dtype=bool))
    assert comps == [] and sizes == 0
