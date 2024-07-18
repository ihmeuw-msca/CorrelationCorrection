import numpy as np
import pytest
from numpy.testing import assert_array_equal

from correlation_correction.regressions import _create_covariance_matrix


@pytest.fixture
def data():
    np.random.seed(123)
    return {
        "A": np.random.randint(low=30, high=300, size=3),
        "B": np.random.randint(low=30, high=300, size=3),
        "a0": np.random.randint(low=30, high=300),
        "b0": np.random.randint(low=30, high=300),
        "v": np.random.randn(3) ** 2,
    }


@pytest.fixture
def cov_matrix(data):
    return _create_covariance_matrix(*(tuple(data.values())))


def test_symmetry(cov_matrix):
    assert_array_equal(
        cov_matrix, cov_matrix.T, err_msg="Covariance matrix is not symmetric."
    )


def test_positive_definite(cov_matrix):
    assert np.all(
        np.linalg.eigvals(cov_matrix) >= 0
    ), "Covariance matrix is not positive semi-definite."
