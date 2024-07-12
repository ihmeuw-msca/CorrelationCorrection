from typing import Union

import numpy as np
from numpy.typing import NDArray

from .methods.gl_methods import convex_gl, gl
from .methods.hamling_methods import hamling

float_or_int = Union[np.float64, int]


def _create_covariance_matrix(
    Ax: NDArray, Bx: NDArray, a0x: np.float64, b0x: np.float64, v: NDArray
) -> NDArray:
    r"""Function that creates the covariance matrix from estimated correlations via process explained in GL.
    Take Ax, Bx, a0x, and b0x to be pseudo-counts. Then estimated standard errors from pseudo-counts are:
       s = sqrt(1/Ax + 1/Bx + 1/a0x + 1/b0x).
    From this, we calculate the correlations by:
       rxz = (1/a0x + 1/b0x)/(sx*sz)
    for exposure levels x, z. The covariance between exposures x, z are then given by
       Cxz = rxz(vx*vz)^{1/2}
    for the variance elements of v vx and vz. This constructs the covariance matrix.
    Parameters
    ----------
    Ax
       The nx1 vector of non-reference pseudo-cases.
    Bx
       The nx1 vector of non-reference pseudo-noncases.
    a0x
       The float number of reference pseudo-cases.
    b0x
       The float number of reference pseudo-noncases.
    v
       The nx1 vector of reported variances. Must be variances, not standard errors.

    Returns
    -------
    NDArray
       Covariance matrix to then be used in GLS to create estimates

    """
    n = Ax.shape[0]
    # Try using actual (reported) variances for s instead of constructing s. Hamling is the same here bc it matches the variances.
    s = np.sqrt(1 / Ax + 1 / Bx + 1 / a0x + 1 / b0x)
    # s = np.sqrt(v)
    r = ((1 / np.outer(s, s)) * (1 / a0x + 1 / b0x))[np.triu_indices(n, k=1)]
    c = r * np.sqrt((np.outer(v, v))[np.triu_indices(n, k=1)])
    triu_indices = np.triu_indices(n, k=1)
    C2 = np.zeros((n, n))
    C2[triu_indices] = c
    C = C2 + C2.T
    C += np.diag(v)
    return C


def covariance_matrix_convex_gl(
    L: NDArray,
    N: NDArray,
    M1: float_or_int,
    v: NDArray,
    constraints=None,
    A_const=False,
    N_const=False,
    M1_const=False,
    OR=True,
) -> NDArray:
    r"""Function that will take in necessary parameters to run convex_gl method and returns the desired covariance matrix, calling
    _create_covariance_matrix function.

    Parameters
    ----------
    L
        The nx1 vector of LOG ORs or RRs for each exposure level.
    N
        The (n+1)x1 vector of subjects for each exposure level.
    M1
        The integer of total number of cases in the study.
    v
        The nx1 vector of reported variances. Must be variances not standard errors.
    constraints
        If list nonempty, enforces constraints on optimization problem. See notes for list requirements.
    A_const
        Boolean variable that optimizes over possible A if False, holds constant if True.
    N_const
        Boolean variable that optimizes over possible N if False, holds constant if True.
    M1_const
        Boolean variable that optimizes over possible M1 if False, holds constant if True.
    OR
        Boolean variable that performs GL convex optimization for OR if True, RR if False.

    Returns
    -------
    np.array
        Covariance matrix to then be used in GLS to create estimates

    Notes
    -------
    Note that every "constraints" argument must have elements defined as lambda functions of the form:
            lambda A,N,M1: cp.sum(A) == 115
    (as an example). This is because A, N, M1 are not being defined until the function is called.

    """
    Ax, Bx, a0x, b0x = convex_gl(
        L, N, M1, constraints, A_const, N_const, M1_const, OR
    )
    return _create_covariance_matrix(Ax, Bx, a0x, b0x, v)


def covariance_matrix_gl(
    L: NDArray,
    A0: NDArray,
    N: NDArray,
    M1: float_or_int,
    v: NDArray,
    OR: bool = True,
    i_ret: bool = False,
) -> NDArray:
    r"""Function that will take in necessary parameters to run gl method and returns the desired covariance matrix, calling
    _create_covariance_matrix function.

    Parameters
    ----------
    L
        The nx1 vector of LOG ORs or RRs for each exposure level.
    A0
        The nx1 vector of reported cases or null expected value cases. Serves as initial guess for rootfinding procedure.
    N
        The (n+1)x1 vector of subjects for each exposure level.
    M1
        The integer of total number of cases in the study.
    v
        The nx1 vector of reported variances. Must be variances not standard errors.
    OR
        Boolean variable that performs GL convex optimization for OR if True, RR if False.
    i_ret
        Boolean variable that returns number of iterations to converge if True, doesn't return if False.

    Returns
    -------
    np.array
        Covariance matrix to then be used in GLS to create estimates

    """
    Ax, Bx, a0x, b0x = gl(L, A0, N, M1, OR, i_ret)
    return _create_covariance_matrix(Ax, Bx, a0x, b0x, v)


def covariance_matrix_hamling(
    L: NDArray,
    p0: np.float64,
    z0: np.float64,
    v: NDArray,
    x_feas: NDArray | None = None,
    OR: bool = True,
) -> NDArray:
    r"""Function that will take in necessary parameters to run hamling method and returns the desired covariance matrix, calling
    _create_covariance_matrix function.

    Parameters
    ----------
    L
        The nx1 vector of LOG ORs or RRs for each exposure level.
    p0
        The float of reference non-cases to total non-cases (for every exposure, including reference).
    z0
        The float of total non-cases to total cases (for every exposure, including reference).
    x_feas
        The 2x1 vector that serves as the intial guess in Hamling optimization procedure.
    v
        The nx1 vector of reported variances. Must be variances, not standard errors.
    OR
        Boolean variable that performs GL convex optimization for OR if True, RR if False.

    Returns
    -------
    np.array
        Covariance matrix to then be used in GLS to create estimates


    """
    if x_feas is None:
        x_feas = np.array([10 / np.min(v), 10 / np.min(v)])

    Ax, Bx, a0x, b0x = hamling(L, p0, z0, v, x_feas, OR)
    return _create_covariance_matrix(Ax, Bx, a0x, b0x, v)


def gls_reg(
    C: NDArray, L: NDArray, x: NDArray
) -> tuple[np.float64, np.float64]:
    r"""Performs generalized least squares using the covariance matrix C generated by GL or Hamling methods:
        (x^\top C^{-1} x)^{-1}(x^\top C^{-1} L)

    Parameters
    ----------
        C
            The nxn covariance matrix generated by the GL or Hamling methods. This function is robust to the method used.
        L
            The nx1 vector of LOG ORs or RRs for each exposure level.
        x
            The nx1 vector of exposure levels.

    Returns
    -------
        tuple[np.float64, np.float64]
            The tuple of the corrected point (slope) estimate and the corrected variance estimate.
    """
    Cinv = np.linalg.inv(C)
    vb_star = 1 / (np.dot(x, np.dot(Cinv, x)))
    b_star = vb_star * (np.dot(x, np.dot(Cinv, L)))
    return b_star, vb_star


def wls_reg(
    L: NDArray, x: NDArray, v: NDArray
) -> tuple[np.float64, np.float64]:
    r"""Performs weighted least squares using the diagonal covariance matrix from reported variances of the form:
        (x^\top V x)^{-1}(x^\top V x)
    where V is defined to be the diagonal matrix whose diagonal elements V_{i,i} = 1 / v[i]

    Parameters
    ----------
        L
            The nx1 vector of LOG ORs or RRs for each exposure level.
        x
            The nx1 vector of exposure levels.
        v
            The nx1 vector of reported variances. Must be variances, not standard errors.

    Returns
    -------
        tuple[np.float64, np.float64]
            The tuple of the uncorrected point (slope) estimate and the uncorrected variance estimate.
    """
    vb = 1 / (np.dot(x, np.dot(np.linalg.inv(np.diag(v)), x)))
    b = vb * (np.dot(x, np.dot(np.linalg.inv(np.diag(v)), L)))
    return b, vb
