import numpy as np
import pytest
from numpy.testing import assert_approx_equal

import correlation_correction.methods.hamling_methods as ham


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
        "p": np.random.uniform(low=0, high=1),
        "z": np.random.uniform(low=0, high=3),
        "v": np.random.randn(3) ** 2,
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
        "p": np.random.uniform(low=0, high=1),
        "z": np.random.uniform(low=0, high=1000),
        "v": np.random.randn(3) ** 2,
    }


def test_positive_A_OR(data_OR):
    A, _, a0, _ = ham.hamling(*(tuple(data_OR.values())))
    A_n = np.hstack((a0, A))
    assert np.all(A_n > 0), "Not all the elements in A are greater than 0."


def test_positive_B_OR(data_OR):
    _, B, _, b0 = ham.hamling(*(tuple(data_OR.values())))
    B_n = np.hstack((b0, B))
    assert np.all(B_n > 0), "Not all the elements in B are greater than 0."


def test_match_p_OR(data_OR):
    _, B, _, b0 = ham.hamling(*(tuple(data_OR.values())))
    p = b0 / (b0 + np.sum(B))
    assert_approx_equal(
        data_OR["p"],
        p,
        significant=3,
        err_msg="p from pseudo-counts fail to match given p.",
    )


def test_match_z_OR(data_OR):
    A, B, a0, b0 = ham.hamling(*(tuple(data_OR.values())))
    z = (np.sum(B) + b0) / (np.sum(A) + a0)
    assert_approx_equal(
        data_OR["z"],
        z,
        significant=3,
        err_msg="z from pseudo-counts fail to match given z.",
    )


def test_match_OR(data_OR):
    A, B, a0, b0 = ham.hamling(*(tuple(data_OR.values())))
    L = np.log((A * b0) / (B * a0))
    assert_approx_equal(
        np.sum(data_OR["log odds-L"]),
        np.sum(L),
        significant=3,
        err_msg="ORs from pseudo-counts fail to match adjusted ratios.",
    )


def test_positive_A_RR(data_RR):
    A, _, a0, _ = ham.hamling(*(tuple(data_RR.values())))
    A_n = np.hstack((a0, A))
    assert np.all(A_n > 0), "Not all the elements in A are greater than 0."


def test_positive_B_RR(data_RR):
    _, B, _, b0 = ham.hamling(*(tuple(data_RR.values())))
    B_n = np.hstack((b0, B))
    assert np.all(B_n > 0), "Not all the elements in B are greater than 0."


def test_match_p_RR(data_RR):
    _, B, _, b0 = ham.hamling(*(tuple(data_RR.values())))
    p = b0 / (b0 + np.sum(B))
    assert_approx_equal(
        data_RR["p"],
        p,
        significant=3,
        err_msg="p from pseudo-counts fail to match given p.",
    )


def test_match_z_RR(data_RR):
    A, B, a0, b0 = ham.hamling(*(tuple(data_RR.values())))
    z = (np.sum(B) + b0) / (np.sum(A) + a0)
    assert_approx_equal(
        data_RR["z"],
        z,
        significant=3,
        err_msg="z from pseudo-counts fail to match given z.",
    )


def test_match_RR(data_RR):
    A, B, a0, b0 = ham.hamling(*(tuple(data_RR.values())))
    L = np.log((A * b0) / (B * a0))
    assert_approx_equal(
        np.sum(data_RR["log relative risk-L"]),
        np.sum(L),
        significant=3,
        err_msg="ORs from pseudo-counts fail to match adjusted ratios.",
    )
