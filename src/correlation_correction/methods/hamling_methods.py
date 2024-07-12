import numpy as np
import scipy
from numpy.typing import NDArray


def hamling(
    L: NDArray,
    p0: np.float64,
    z0: np.float64,
    v: NDArray,
    x_feas: NDArray = None,
    OR: bool = True,
) -> tuple[NDArray, NDArray, np.float64, np.float64]:
    r"""Function that performs Hamling's method. Finds a0, b0 values to minimize the squared residual summed error:
            (p0-p1)^2/p0 + (z0-z1)^2/z0 .
    Uses equations defined directly in the Hamling paper. We introduce an initialization x_feas that always converges.

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
    tuple -> np.array, np.array, np.float64, np.float64
        Pseudo-counts for cases, pseudo-counts for non-cases, reference pseudo-cases, reference pseudo-non-cases

    Notes
    -------
    Here we introduce p0 and z0 as parameters of the function. In the future, we could directly calculate p0,z0 inside this function.

    """

    # Initialize x_feas vecotr
    if x_feas is None:
        x_feas = np.array([10 / np.min(v), 10 / np.min(v)])

    # Defining function to optimize using scipy (off-the-shelf) minimization
    def f_LogLik(C_Val, p0, z0, Lx, v):
        a0x = C_Val[0]
        b0x = C_Val[1]

        if OR:
            Vextra = v - 1 / a0x - 1 / b0x
            Est_A = (1 + (a0x / b0x) * np.exp(Lx)) / (Vextra)
            Est_B = (1 + b0x / (a0x * np.exp(Lx))) / (Vextra)
        else:
            Vextra = v - 1 / a0x + 1 / b0x
            Est_A = (1 - np.exp(Lx) * a0x / b0x) / (Vextra)
            Est_B = (b0x / (np.exp(Lx) * a0x) - 1) / (Vextra)

        SumA = a0x + np.sum(Est_A)
        SumB = b0x + np.sum(Est_B)

        p1 = b0x / SumB
        F1 = ((p1 - p0) / p0) ** 2
        z1 = SumB / SumA
        F2 = ((z1 - z0) / z0) ** 2

        return F1 + F2

    # Perform minimization
    a0_b0_res = scipy.optimize.minimize(
        f_LogLik, x_feas, args=(p0, z0, L, v), options={"disp": False}
    )

    # Get estimates for A and B
    a0_fit, b0_fit = a0_b0_res.x[0], a0_b0_res.x[1]
    if OR:
        denom = v - 1 / a0_fit - 1 / b0_fit
        A_num = 1 + (a0_fit / b0_fit) * np.exp(L)
        B_num = 1 + (b0_fit / (a0_fit * np.exp(L)))
        A_fit = A_num / denom
        B_fit = B_num / denom
    else:
        denom = v - 1 / a0_fit + 1 / b0_fit
        A_num = 1 - np.exp(L) * a0_fit / b0_fit
        B_num = b0_fit / (np.exp(L) * a0_fit) - 1
        A_fit = A_num / denom
        B_fit = B_num / denom

    # Return a0, b0 in that order
    return A_fit, B_fit, a0_fit, b0_fit


# def ham_solved(b0,B,M1,Lx,v,a0,OR=True):

#     # Construct true p and z values
#     B0sum = np.sum(B) + b0

#     p = b0 / B0sum
#     z = B0sum / M1

#         # Objective function adjustments
#     # def a0(B_plus, A_plus, z, p):
#     #     # Prevent division by zero by ensuring the denominator is never zero
#     #     return max(1 / (z * (1 - p)) * B_plus - A_plus, 1e-8)

#     # def b0(B_plus, p):
#     #     # Similarly, ensure b0 is never zero to avoid division by zero
#     #     return max(p / (1 - p) * B_plus, 1e-8)

#     # Constsruct objective to minimize
#     def objective(C_val,v,Lx,z,p):
#         a0x = C_val[0]
#         b0x = C_val[1]
#         B_plus = ((1-p)/p)*b0x
#         A_plus = (1/(z*p))*b0x - a0x

#         # Ensure denominators are always positive
#         if OR:
#             denom = v  - 1/a0x - 1/b0x
#             Est_A = (B_plus - np.sum((1 + b0x / (a0x * np.exp(Lx))) / denom))**2
#             Est_B = (A_plus - np.sum((1 + a0x * np.exp(Lx)  / b0x ) / denom))**2
#         else:
#             denom = v - 1/a0x + 1/b0x
#             Est_A = (A_plus - np.sum((1 - np.exp(Lx)*a0x/b0x)/denom))**2
#             Est_B = (B_plus - np.sum((b0x/(np.exp(Lx)*a0x) - 1)/denom))**2

#         return Est_A + Est_B
#       # Constsruct objective to minimize
#     # def objective(C_val,v,Lx,z,p):
#     #     A_plus = C_val[0]
#     #     B_plus = C_val[1]
#     #     a0x = a0(B_plus, A_plus, z, p)
#     #     b0x = b0(B_plus,p)

#     #     # Ensure denominators are always positive
#     #     if OR:
#     #         denom = v  - 1/a0x - 1/b0x
#     #         Est_A = (B_plus - np.sum((1 + b0x / (a0x * np.exp(Lx))) / denom))**2
#     #         Est_B = (A_plus - np.sum((1 + a0x * np.exp(Lx)  / b0x ) / denom))**2
#     #     else:
#     #         denom = v - 1/a0x + 1/b0x
#     #         Est_A = (A_plus - np.sum((1 - np.exp(Lx)*a0x/b0x)/denom))**2
#     #         Est_B = (B_plus - np.sum((b0x/(np.exp(Lx)*a0x) - 1)/denom))**2

#     #     return Est_A + Est_B

#     # Perform minimization
#     # x_feas = np.array([10/np.min(v),10/np.min(v)])
#     # constraints = [
#     #     {'type': 'ineq', 'fun': lambda x: x[0] - 1},  # Ensuring a_0 >= 1
#     #     {'type': 'ineq', 'fun': lambda x: x[1] - 1},  # Ensuring b_0 >= 1
#     #     {"type": "ineq", "fun": lambda x: (1 + (x[0]/x[1])*np.exp(Lx))/(v - 1/x[0] - 1/x[1]) - np.ones(4)}, # Ensuring elements of A >= 1
#     #     {"type": "ineq", "fun": lambda x: (1 + (x[1]/(x[0]*np.exp(Lx))))/(v - 1/x[0] - 1/x[1]) - np.ones(4)} # Ensuring elements of B >= 1
#     # ]
#     def constraint_A(x):
#         vector = (1 + (x[0]/x[1])*np.exp(Lx))/(v - 1/x[0] - 1/x[1])
#         return vector
#     def constraint_B(x):
#         vector = (1 + (x[1]/(x[0]*np.exp(Lx))))/(v - 1/x[0] - 1/x[1])
#         return vector
#     def nonlinear_constraint_A(x):
#         return constraint_A(x) - 1
#     def nonlinear_constraint_B(x):
#         return constraint_B(x) - 1
#     def nonlinear_constraint_a0(x):
#         return x[0] - 1
#     def nonlinear_constraint_b0(x):
#         return x[1] - 1
#     vec_constraint_A = scipy.optimize.NonlinearConstraint(nonlinear_constraint_A,0,np.inf)
#     vec_constraint_B = scipy.optimize.NonlinearConstraint(nonlinear_constraint_B,0,np.inf)
#     vec_constraint_a0 = scipy.optimize.NonlinearConstraint(nonlinear_constraint_a0,0,np.inf)
#     vec_constraint_b0 = scipy.optimize.NonlinearConstraint(nonlinear_constraint_b0,0,np.inf)
#     constraints = [vec_constraint_A,vec_constraint_B,vec_constraint_a0,vec_constraint_b0]
#     x_feas = np.array([a0,b0])
#     a0_b0_res = scipy.optimize.minimize(objective,x_feas,args=(v, Lx, z, p),method="trust-constr",constraints=constraints,options={"disp":False})
#     # print("Optimization results:",a0_b0_res)
#     # Get estimates for A and B
#     a0_fit, b0_fit = a0_b0_res.x[0], a0_b0_res.x[1]
#     if OR:
#         denom = v - 1/a0_fit - 1/b0_fit
#         A_num = 1 + (a0_fit/b0_fit)*np.exp(Lx)
#         B_num = 1 + (b0_fit/(a0_fit*np.exp(Lx)))
#         A_fit = A_num / denom
#         B_fit = B_num / denom
#     else:
#         denom = v - 1/a0_fit + 1/b0_fit
#         A_num = 1 - np.exp(Lx)*a0_fit/b0_fit
#         B_num = b0_fit/(np.exp(Lx)*a0_fit) - 1
#         A_fit = A_num / denom
#         B_fit = B_num / denom
#     # A_B_plus_res = scipy.optimize.minimize(objective,[M1,B0sum],args=(v,Lx,z,p))

#     # # Get estimates for A and B
#     # Aplus_fit, Bplus_fit = A_B_plus_res.x[0], A_B_plus_res.x[1]
#     # if OR:
#     #     a0_fit = a0(Bplus_fit,Aplus_fit,z,p)
#     #     b0_fit = b0(Bplus_fit,p)
#     #     denom = v - 1/a0_fit - 1/b0_fit
#     #     A_num = 1 + (a0_fit/b0_fit)*np.exp(Lx)
#     #     B_num = 1 + (b0_fit/(a0_fit*np.exp(Lx)))
#     #     A_fit = A_num / denom
#         # B_fit = B_num / denom

#     return A_fit, B_fit, a0_fit, b0_fit
