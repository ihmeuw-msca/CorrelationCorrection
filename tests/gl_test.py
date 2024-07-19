import numpy as np
import pytest
from numpy.testing import assert_approx_equal, assert_array_equal

import correlation_correction.methods.gl_methods as gl


# Return a dictionary instead of a tuple for data where keys explain what everything means.
@pytest.fixture
def data_OR():
    np.random.seed(123)
    return {
        "log odds-L": np.array(
            [
                np.random.normal(0.7, 0.1),
                np.random.normal(1.5, 0.1),
                np.random.normal(2, 0.1),
            ]
        ),
        "N": np.random.randint(low=150, high=700, size=4),
        "M": np.random.randint(low=200, high=1100),
    }


@pytest.fixture
def data_RR():
    np.random.seed(123)
    return {
        "log relative risk-L": np.array(
            [
                np.random.normal(0.7, 0.1),
                np.random.normal(1.5, 0.1),
                np.random.normal(2, 0.1),
            ]
        ),
        "N": np.random.randint(low=150, high=70000, size=4),
        "M": np.random.randint(low=200, high=1100),
    }


def test_sum_cases_OR(data_OR):
    dc = gl.convex_gl(*(tuple(data_OR.values())))
    assert (
        dc.A_sum() == data_OR["M"]
    ), "Sum of pseudo-cases does not match actual sum of cases."


def test_subjects_OR(data_OR):
    dc = gl.convex_gl(*(tuple(data_OR.values())))
    A_n = dc.data[:, 0]
    B_n = dc.data[:, 1]
    assert_array_equal(
        (A_n + B_n),
        data_OR["N"],
        err_msg="Pseudo-subjects do not match actual number of subjects.",
    )


def test_match_ratios_OR(data_OR):
    dc = gl.convex_gl(*(tuple(data_OR.values())))
    L = dc.log_ratio()
    assert_approx_equal(
        np.sum(data_OR["log odds-L"]),
        np.sum(L),
        significant=3,
        err_msg="Ratios from pseudo-counts fail to match adjusted ratios.",
    )


def test_sum_cases_RR(data_RR):
    dc = gl.convex_gl(*(tuple(data_RR.values())))
    assert (
        dc.A_sum() == data_RR["M"]
    ), "Sum of pseudo-cases does not match actual sum of cases."


def test_subjects_RR(data_RR):
    dc = gl.convex_gl(*(tuple(data_RR.values())))
    A_n = dc.data[:, 0]
    B_n = dc.data[:, 1]
    assert_array_equal(
        (A_n + B_n),
        data_RR["N"],
        err_msg="Pseudo-subjects do not match actual number of subjects.",
    )


def test_match_ratios_RR(data_RR):
    dc = gl.convex_gl(*(tuple(data_RR.values())))
    L = dc.log_ratio()
    assert_approx_equal(
        np.sum(data_RR["log relative risk-L"]),
        np.sum(L),
        significant=3,
        err_msg="Ratios from pseudo-counts fail to match adjusted ratios.",
    )
