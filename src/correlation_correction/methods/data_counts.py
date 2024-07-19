from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from correlation_correction.regressions import _create_covariance_matrix


@dataclass
class Counts:
    """
    Data class to store outputs of GL and Hamling methods. Both GL and Hamling return 1d arrays for A, B, and
    floats for a0, b0. When initialized, an object of the Counts class will produce a 2d array of the form
        np.array([
        [a0, b0],
        [a1, b1],
        ... ,
        [an, bn]
        ])
    where n is the number of exposure levels. After initilization, there are property functions that follow that will
    return the values for A, B, a0, b0 when called. Last, the cov method simply returns the covariance matrix given the
    counts and simply requires one argument: the reported variances.
    """

    def __init__(self, A: NDArray, B: NDArray, a0: float, b0: float):
        self.n = A.shape[0]

        A_n = np.hstack((a0, A))
        B_n = np.hstack((b0, B))

        self.data = np.vstack((A_n, B_n)).T

    @property
    def A(self) -> NDArray:
        return self.data[1:, 0]

    @property
    def B(self) -> NDArray:
        return self.data[1:, 1]

    @property
    def a0(self) -> float:
        return self.data[0, 0]

    @property
    def b0(self) -> float:
        return self.data[0, 1]

    @property
    def A_sum(self) -> float:
        return np.sum(self.A) + self.a0

    @property
    def B_sum(self) -> float:
        return np.sum(self.B) + self.b0

    @property
    def log_ratio(self) -> float:
        return np.log((self.A * self.b0) / (self.B * self.a0))

    @property
    def cov(self, v: NDArray) -> NDArray:
        r"""Function that will take in necessary parameters to run gl method and returns the desired covariance matrix, calling
        _create_covariance_matrix function.

        Parameters
        ----------
        v
            The nx1 vector of reported variances. Must be variances not standard errors.

        Returns
        -------
        np.array
            Covariance matrix to then be used in GLS to create estimates

        """
        return _create_covariance_matrix(self.A, self.B, self.a0, self.b0, v)
