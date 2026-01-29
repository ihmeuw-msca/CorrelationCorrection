import scipy
import numpy as np

def ham_vanilla(L,p0,z0,v,x_feas,OR=True):
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
    
    # Defining function to optimize using scipy (off-the-shelf) minimization
    def f_LogLik(C_Val,p0,z0,Lx,v):
        a0x = C_Val[0]
        b0x = C_Val[1]

        if OR:
            Vextra = v - 1/a0x - 1/b0x
            Est_A = (1 + (a0x/b0x)*np.exp(Lx))/(Vextra)
            Est_B = (1 + b0x/(a0x*np.exp(Lx)))/(Vextra)
        else:
            Vextra = v - 1/a0x + 1/b0x
            Est_A = (1 - np.exp(Lx)*a0x/b0x)/(Vextra)
            Est_B = (b0x/(np.exp(Lx)*a0x) - 1)/(Vextra)

        SumA = a0x + np.sum(Est_A)
        SumB = b0x + np.sum(Est_B)

        p1 = b0x / SumB
        F1 = ((p1-p0)/p0)**2
        z1 = SumB/SumA
        F2 = ((z1-z0)/z0)**2

        return F1 + F2

    # Perform minimization
    a0_b0_res = scipy.optimize.minimize(f_LogLik,x_feas,args=(p0,z0,L,v),options={"disp":False})

    # Get estimates for A and B
    a0_fit, b0_fit = a0_b0_res.x[0], a0_b0_res.x[1]
    if OR:
        denom = v - 1/a0_fit - 1/b0_fit
        A_num = 1 + (a0_fit/b0_fit)*np.exp(L)
        B_num = 1 + (b0_fit/(a0_fit*np.exp(L)))
        A_fit = A_num / denom
        B_fit = B_num / denom
    else:
        denom = v - 1/a0_fit + 1/b0_fit
        A_num = (1 - np.exp(L)*a0_fit/b0_fit)
        B_num = (b0_fit/(np.exp(L)*a0_fit) - 1)
        A_fit = A_num / denom
        B_fit = B_num / denom

    # Return a0, b0 in that order
    return A_fit, B_fit, a0_fit, b0_fit
