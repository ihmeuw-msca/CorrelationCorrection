import cvxpy as cp
import numpy as np
import scipy
from numpy.typing import NDArray


def convex_gl(
    L: NDArray,
    N: NDArray,
    M1: NDArray,
    constraints=None,
    A_const=False,
    N_const=False,
    M1_const=False,
    OR=True,
) -> tuple[NDArray, NDArray, np.float64, np.float64]:
    r"""Function that will solve the convex optimization problem
    G(A) = -L^\top A + (a_0(A)log(a_0(A)) - a_0(A)) + \sum_{i=1}^{n}(B_i(A)log(B_i(A)) - B_i(A)) +
            \sum_{i=1}^{n}(A_ilog(A_i) - A_i) + (b_0(A)log(b_0(A)) - b_0(A))

    using Disciplined Convex Programming via cvxpy. No initialization required.

    Parameters
    ----------
    L
        The nx1 vector of LOG ORs or RRs for each exposure level.
    N
        The (n+1)x1 vector of subjects for each exposure level.
    M1
        The integer of total number of cases in the study.
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
    tuple -> np.array, np.array, np.float64, np.float64
        Pseudo-counts for cases, pseudo-counts for non-cases, reference pseudo-cases, reference pseudo-non-cases

    Notes
    -------
    Note that every "constraints" argument must have elements defined as lambda functions of the form:
            lambda A,N,M1: cp.sum(A) == 115
    (as an example). This is because A, N, M1 are not being defined until the function is called.

    """

    # Get number of exposure levels
    n = L.shape[0]

    # Checking if we optimize over N or not. If yes, create as cvxpy variable.
    if ~N_const:
        N = N.copy()
    else:
        N = cp.Variable(shape=(n + 1))

    # Checking if we optimize over M1 or not. If yes, create as cvxpy variable.
    if ~M1_const:
        M1 = M1.copy()
    else:
        M1 = cp.variable()

    # Checking if we want to pass in any constraints or not.
    if constraints is None:
        constraints = []

    # Create L, A, and constraints (if they exist) for cvxpy.
    L = L.copy()
    A = cp.Variable(shape=n)
    constraints_eval = [c(A, N, M1) for c in constraints]
    # Construct objective function
    if OR:
        obj = cp.Minimize(
            cp.scalar_product(-L, A)
            - cp.entr(M1 - cp.sum(A))
            - (M1 - cp.sum(A))
            + cp.sum(-cp.entr(N[1:] - A) - (N[1:] - A))
            + cp.sum(-cp.entr(A) - A)
            - cp.entr(N[0] - M1 + cp.sum(A))
            - (N[0] - M1 + cp.sum(A))
        )
    else:
        log_N = np.log(N[1:])
        log_n0 = np.log(N[0])
        obj = cp.Minimize(
            cp.scalar_product(A, (-L - log_N + log_n0))
            + cp.sum(-cp.entr(A) - A)
            - cp.entr(M1 - cp.sum(A))
            - M1
            + cp.sum(A)
        )

    # Solves constrained optimization problem and returns desired values (held constant if desired)
    if len(constraints) > 0:
        problem = cp.Problem(obj, constraints_eval)
        problem.solve(solver=cp.ECOS)
        if A_const:
            return A
        if A_const and N_const:
            return A, N
        if A_const and M1_const:
            return A, M1
        return A, N, M1

    # Solves unconstrained optimization problem
    problem = cp.Problem(obj)
    problem.solve(solver=cp.CLARABEL)

    # For unconstrained optimization, returns OR and RR values
    if OR:
        A_cvx = A.value
        B_cvx = N[1:] - A_cvx
        a0_cvx = M1 - np.sum(A_cvx)
        b0_cvx = N[0] - M1 + np.sum(A_cvx)
    else:
        A_cvx = A.value
        B_cvx = N[1:]
        a0_cvx = M1 - np.sum(A_cvx)
        b0_cvx = N[0]
    return A_cvx, B_cvx, a0_cvx, b0_cvx


def gl(
    L: NDArray, A0: NDArray, N: NDArray, M1: NDArray, OR=True, i_ret=False
) -> tuple[NDArray, NDArray, np.float64, np.float64]:
    r"""Function that will solve solve the rootfinding problem of the gradient function
    g(A) = -L - log(a_0(A))1 - log(B(A)) + log(A) + log(b_0(A))
    according to Greenland and Longnecker via Newton's method.

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
    OR
        Boolean variable that performs GL convex optimization for OR if True, RR if False.
    i_ret
        Boolean variable that returns number of iterations to converge if True, doesn't return if False.

    Returns
    -------
    tuple -> np.array, np.array, np.float64, np.float64, (int)
        Pseudo-counts for cases, pseudo-counts for non-cases, reference pseudo-cases, reference pseudo-non-cases, (iterations to converge)

    """

    # Creating necessary copies and initializing data to be used in procedure.
    n = L.shape[0]
    N = N.copy()
    M1 = M1
    A = A0.copy()
    L = L.copy()
    diff = 1
    i = 1

    # Performing Newton's method according to GL. Will construct the correct pseudo-A after the while loop
    while diff > 1e-6:
        A1 = A
        Aplus = A.sum()
        a0 = M1 - Aplus
        if OR:
            b0 = N[0] - a0
            B = N[1:] - A
            if np.any(B <= 0):
                print("There is an element of B < 0")
            c0 = 1 / a0 + 1 / b0
            c = 1 / A + 1 / B
        else:
            b0 = N[0]
            B = N[1:]
            c0 = 1 / a0
            c = 1 / A

        # Gradient step in Newton's Method and get ∆A
        e = L + np.log(a0) + np.log(B) - np.log(A) - np.log(b0)

        # Create Hessian matrix
        H = np.ones((n, n)) * c0
        H += np.diag(c)

        # Update A according to Newton's method
        A += scipy.linalg.solve(H, e, assume_a="pos")
        i += 1
        diff = np.linalg.norm(A1 - A)

    # Get other counts from A
    B = N[1:] - A
    a0 = M1 - A.sum()
    b0 = N[0] - a0

    # Return with or without iteration numbers
    if ~i_ret:
        return A, a0, B, b0, i
    else:
        return A, B, a0, b0
