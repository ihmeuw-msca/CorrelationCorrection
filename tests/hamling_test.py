import numpy as np
import pytest
from numpy.testing import assert_approx_equal

import correlation_correction.methods.hamling_methods as ham


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
        np.random.uniform(low=0, high=1),
        np.random.uniform(low=0, high=3),
        np.random.randn(3) ** 2,
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
        np.random.uniform(low=0, high=1),
        np.random.uniform(low=0, high=1000),
        np.random.randn(3) ** 2,
    )


def test_positive_A_OR(data_OR):
    A, _, a0, _ = ham.hamling(*data_OR)
    A_n = np.insert(A, 0, a0)
    assert np.all(A_n > 0), "Not all the elements in A are greater than 0."


def test_positive_B_OR(data_OR):
    _, B, _, b0 = ham.hamling(*data_OR)
    B_n = np.insert(B, 0, b0)
    assert np.all(B_n > 0), "Not all the elements in B are greater than 0."


def test_match_OR(data_OR):
    A, B, a0, b0 = ham.hamling(*data_OR)
    L = np.log((A * b0) / (B * a0))
    assert_approx_equal(
        np.sum(data_OR[0]),
        np.sum(L),
        significant=3,
        err_msg="ORs from pseudo-counts fail to match adjusted ratios.",
    )


def test_positive_A_RR(data_RR):
    A, _, a0, _ = ham.hamling(*data_RR)
    A_n = np.insert(A, 0, a0)
    assert np.all(A_n > 0), "Not all the elements in A are greater than 0."


def test_positive_B_RR(data_RR):
    _, B, _, b0 = ham.hamling(*data_RR)
    B_n = np.insert(B, 0, b0)
    assert np.all(B_n > 0), "Not all the elements in B are greater than 0."


def test_match_RR(data_RR):
    A, B, a0, b0 = ham.hamling(*data_RR)
    L = np.log((A * b0) / (B * a0))
    assert_approx_equal(
        np.sum(data_RR[0]),
        np.sum(L),
        significant=3,
        err_msg="ORs from pseudo-counts fail to match adjusted ratios.",
    )
