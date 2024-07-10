import numpy as np

from .GLMethods import GL, convex_GL
from .HamMethods import ham_vanilla


def covariance_matrix(L, A0, N, M1, p0, z0, x_feas, v, method, OR=True):
    # def covariance_matrix_gl(L, A0, N, M1, v, method, OR=True):
    # TODO: Split up covariance matrix for GL and Hamling methods
    r"""Function that creates the covariance matrix from estimated correlations via process explained in GL.
    Take Ax, Bx, a0x, and b0x to be pseudo-counts. Then estimated standard errors from pseudo-counts are:
        s = sqrt(1/Ax + 1/Bx + 1/a0x + 1/b0x).
    From this, we calculate the correlations by:
        rxz = (1/a0x + 1/b0x)/(sx*sz)
    for exposure levels x, z. The covariance between exposures x, z are then given by
        Cxz = rxz(vx*vz)^{1/2}
    for the variance elements of v vx and vz. This constructs the covariance matrix.
    Will take in necessary parameters to run either GL or Hamling-based methods and returns the desired covariance matrix.

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
    p0
        The float of reference non-cases to total non-cases (for every exposure, including reference).
    z0
        The float of total non-cases to total cases (for every exposure, including reference).
    x_feas
        The 2x1 vector that serves as the intial guess in Hamling optimization procedure.
    v
        The nx1 vector of reported variances. Must be variances, not standard errors.
    method
        String parameter that accepts only one of the following for each function call:
            1. "convex_GL" to run method convex_GL (convex optimization GL).
            2. "GL" to run method GL (rootfinding GL).
            3. "ham_vanilla" to run Hamling method
    OR
        Boolean variable that performs GL convex optimization for OR if True, RR if False.

    Returns
    -------
    np.array
        Covariance matrix to then be used in GLS to create estimates

    """

    # Generating Ax, Bx, a0x, b0x depending on selected method.
    if method == "convex_GL":
        Ax, Bx, a0x, b0x = convex_GL(L, N, M1, OR=OR)
    if method == "GL":
        Ax, Bx, a0x, b0x = GL(L, A0, N, M1, OR=OR)
    if method == "ham_vanilla":
        Ax, Bx, a0x, b0x = ham_vanilla(L, p0, z0, v, x_feas, OR=OR)

    # Calculates the covariance matrix according to math shown in description.
    n = Ax.shape[0]
    # Try using actual (reported) variances for s instead of constructing s. Hamling is the same here bc it matches the variances.
    # s = np.sqrt(1 / Ax + 1 / Bx + 1 / a0x + 1 / b0x)
    s = np.sqrt(v)
    r = ((1 / np.outer(s, s)) * (1 / a0x + 1 / b0x))[np.triu_indices(n, k=1)]
    # r = ((1 / np.outer(v, v)) * (1 / a0x + 1 / b0x))[np.triu_indices(n, k=1)]
    c = r * np.sqrt((np.outer(v, v))[np.triu_indices(n, k=1)])
    triu_indices = np.triu_indices(n, k=1)
    C2 = np.zeros((n, n))
    C2[triu_indices] = c
    C = C2 + C2.T
    C += np.diag(v)
    return C


def trend_est(Ax, Bx, a0x, b0x, v, x, L, unadj=False):
    """Performs adjusted correlation on meta-analysis according to Greenland and Longnecker"""
    # Calculating adjusted point and variance estimates
    C = covariance_matrix(Ax, Bx, a0x, b0x, v)
    Cinv = np.linalg.inv(C)
    vb_star = 1 / (np.dot(x, np.dot(Cinv, x)))
    b_star = vb_star * (np.dot(x, np.dot(Cinv, L)))

    # Calculating unadjusted point and variance estimates if desired
    if unadj:
        vb = 1 / (np.dot(x, np.dot(np.linalg.inv(np.diag(v)), x)))
        b = vb * (np.dot(x, np.dot(np.linalg.inv(np.diag(v)), L)))
        return b_star, vb_star, b, vb

    return b_star, vb_star
