import numpy as np
import pytest
from numpy.testing import assert_approx_equal, assert_array_equal

import correlation_correction.methods.gl_methods as gl


# Return a dictionary instead of a tuple for data where keys explain what everything means.
@pytest.fixture
def data_OR():
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


@pytest.fixture
def data_RR():
    np.random.seed(123)
    return (
        np.array(
            [
                np.random.normal(0.7, 0.1),
                np.random.normal(1.5, 0.1),
                np.random.normal(2, 0.1),
            ]
        ),
        np.random.randint(low=150, high=70000, size=4),
        np.random.randint(low=200, high=1100),
    )


def test_sum_cases_OR(data_OR):
    A, _, a0, _ = gl.convex_gl(*data_OR)
    assert (np.sum(A) + a0) == data_OR[
        -1
    ], "Sum of pseudo-cases does not match actual sum of cases."


def test_subjects_OR(data_OR):
    A, B, a0, b0 = gl.convex_gl(*data_OR)
    # Change here to np.hstack
    A_n = np.insert(A, 0, a0)
    B_n = np.insert(B, 0, b0)
    assert_array_equal(
        (A_n + B_n),
        data_OR[1],
        err_msg="Pseudo-subjects do not match actual number of subjects.",
    )


def test_match_ratios_OR(data_OR):
    A, B, a0, b0 = gl.convex_gl(*data_OR)
    L = np.log((A * b0) / (B * a0))
    assert_approx_equal(
        np.sum(data_OR[0]),
        np.sum(L),
        significant=3,
        err_msg="Ratios from pseudo-counts fail to match adjusted ratios.",
    )


def test_sum_cases_RR(data_RR):
    A, _, a0, _ = gl.convex_gl(*data_RR)
    assert (np.sum(A) + a0) == data_RR[
        -1
    ], "Sum of pseudo-cases does not match actual sum of cases."


def test_subjects_RR(data_RR):
    A, B, a0, b0 = gl.convex_gl(*data_RR)
    A_n = np.insert(A, 0, a0)
    B_n = np.insert(B, 0, b0)
    assert_array_equal(
        (A_n + B_n),
        data_RR[1],
        err_msg="Pseudo-subjects do not match actual number of subjects.",
    )


def test_match_ratios_RR(data_RR):
    A, B, a0, b0 = gl.convex_gl(*data_RR)
    L = np.log((A * b0) / (B * a0))
    assert_approx_equal(
        np.sum(data_RR[0]),
        np.sum(L),
        significant=3,
        err_msg="Ratios from pseudo-counts fail to match adjusted ratios.",
    )
