import numpy as np
from nlgc.stat import fdr_control
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

def test_fdr_control_basic():
    alpha = 0.05

    d = np.array([
        [0.0, 40.0, 1.0],
        [0.5, 0.0, 25.0],
        [0.2, 0.1, 0.0],
    ])

    k = np.full_like(d, 9, dtype=int)

    J = fdr_control(d, k, alpha)

    assert J[0, 1] > 0
    assert J[1, 2] > 0

    assert J[0, 2] == 0
    assert J[1, 0] == 0
    assert J[2, 0] == 0
    assert J[2, 1] == 0

    assert np.allclose(np.diag(J), 0)


def test_fdr_control_null():
    rng = np.random.default_rng(0)

    n = 20
    k = 9

    d = rng.chisquare(df=k, size=(n, n))
    np.fill_diagonal(d, 0)

    J = fdr_control(d, k, alpha=0.05)

    assert np.count_nonzero(J) == 0


def test_fdr_control_detects_large_effect():
    n = 10
    k = 9

    d = np.random.chisquare(k, size=(n, n))
    np.fill_diagonal(d, 0)

    # inject one huge effect.
    d[2, 5] = 120.0

    J = fdr_control(d, k, alpha=0.05)

    assert J[2, 5] > 0
    assert J[2, 5] == J.max()


def test_fdr_control_matches_statsmodels():
    alpha = 0.05
    n = 8
    k = 9

    rng = np.random.default_rng(0)

    d = rng.chisquare(k, size=(n, n))
    np.fill_diagonal(d, 0)

    strong_links = {
        (0, 1): 120,
        (1, 2): 80,
        (2, 3): 60,
        (3, 4): 40,
        (4, 5): 30,
    }

    for (i, j), dev in strong_links.items():
        d[i, j] = dev

    pvals = chi2.sf(d.ravel(), k)

    reject_ref, _, _, _ = multipletests(
        pvals,
        alpha=alpha,
        method="fdr_by",
    )

    reject_ref = reject_ref.reshape(n, n)

    # nlgc approach
    J = fdr_control(d, k, alpha)

    reject = J > 0

    np.testing.assert_array_equal(reject, reject_ref)

    for i, j in strong_links:
        assert reject[i, j]

    # J should preserve ordering of effect size.
    Jvals = [J[idx] for idx in strong_links.keys()]

    assert np.all(np.diff(Jvals) < 0)

    plt.imshow(J)
    plt.show()


if __name__ == '__main__':
    print("running test fdr basic")
    test_fdr_control_basic()
    print("pass\n\n")

    print("running test fdr null")
    test_fdr_control_null()
    print("pass\n\n")

    print("running test fdr large detection")
    test_fdr_control_detects_large_effect()
    print("pass\n\n")

    print("running test fdr BY p-value rejection matches statsmodels")
    test_fdr_control_matches_statsmodels()
    print("pass\n\n")