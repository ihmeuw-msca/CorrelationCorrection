import numpy as np
import pytest

from correlation_correction.regressions import _create_covariance_matrix


@pytest.fixture
def data():
    np.random.seed(123)
    return (
        np.random.randint(low=30, high=300, size=3),
        np.random.randint(low=30, high=300, size=3),
        np.random.randint(low=30, high=300),
        np.random.randint(low=30, high=300),
        np.random.randn(size=3) ** 2,
    )


@pytest.fixture
def cov_matrix():
    return _create_covariance_matrix(*data())


def symmetry_test(cov_matrix):
    assert cov_matrix() == cov_matrix().T
