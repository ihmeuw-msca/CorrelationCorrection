import numpy as np
import pytest
from numpy.testing import assert_approx_equal, assert_array_equal

import correlation_correction.methods.gl_methods as gl


@pytest.fixture
def data():
    np.random.seed(123)
    return (
        np.array(
            [
                np.random.normal(0.7, 0.1),
                np.random.normal(1.5, 0.1),
                np.random.normal(2, 0.1),
            ]
        ),
        np.random.randint(low=150, high=700, size=4),
        np.random.randint(low=200, high=1100),
    )


def test_sum_cases(data):
    A, _, a0, _ = gl.convex_gl(*data)
    assert (np.sum(A) + a0) == data[
        -1
    ], "Sum of pseudo-cases does not match actual sum of cases."


def test_subjects(data):
    A, B, a0, b0 = gl.convex_gl(*data)
    A_n = np.insert(A, 0, a0)
    B_n = np.insert(B, 0, b0)
    assert_array_equal(
        (A_n + B_n),
        data[1],
        err_msg="Pseudo-subjects do not match actual number of subjects.",
    )


def test_match_ratios(data):
    A, B, a0, b0 = gl.convex_gl(*data)
    L = np.log((A * b0) / (B * a0))
    assert_approx_equal(
        np.sum(data[0]),
        np.sum(L),
        significant=3,
        err_msg="Ratios from pseudo-counts fail to match adjusted ratios.",
    )
