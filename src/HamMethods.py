import scipy
import numpy as np

def ham_vanilla(b0,B,M1,Lx,v,a0):

    # Construct true p and z values
    B0sum = np.sum(B) + b0

    p0 = b0 / B0sum
    z0 = B0sum / M1  

    # Defining function to optimize using scipy (off-the-shelf) minimization
    def f_LogLik(C_Val,p0,z0,Lx,v):
        a0x = C_Val[0]
        b0x = C_Val[1]

        Vextra = v - 1/a0x - 1/b0x
        Est_A = (1 + (a0x/b0x)*np.exp(Lx))/(Vextra)
        Est_B = (1 + b0x/(a0x*np.exp(Lx)))/(Vextra)
        
        SumA = a0x + np.sum(Est_A)
        SumB = b0x + np.sum(Est_B)

        p1 = b0x / SumB
        F1 = ((p1-p0)/p0)**2
        z1 = SumB/SumA
        F2 = ((z1-z0)/z0)**2

        return F1 + F2

    # Perform minimization
    a0_b0_res = scipy.optimize.minimize(f_LogLik,[a0,b0],args=(p0,z0,Lx,v))

    # Get estimates for A and B
    a0_fit, b0_fit = a0_b0_res.x[0], a0_b0_res.x[1]
    denom = v - 1/a0_fit - 1/b0_fit
    A_num = 1 + (a0_fit/b0_fit)*np.exp(Lx)
    B_num = 1 + (b0_fit/(a0_fit*np.exp(Lx)))
    A_fit = A_num / denom
    B_fit = B_num / denom

    # Return a0, b0 in that order
    return A_fit, B_fit, a0_fit, b0_fit

def ham_solved(b0,B,M1,Lx,v,a0):

    # Construct true p and z values
    B0sum = np.sum(B) + b0

    p = b0 / B0sum
    z = B0sum / M1

    # Constsruct objective to minimize
    def objective(C_val,v,Lx,z,p):
        a0x = C_val[0]
        b0x = C_val[1]
        B_plus = ((1-p)/p)*b0x
        A_plus = (1/(z*p))*b0x - a0x
        
        # Ensure denominators are always positive
        v_denominators = v  - 1 / a0x - 1 / b0x

        term1 = (B_plus - np.sum((1 + b0x / (a0x * np.exp(Lx))) / v_denominators))**2
        term2 = (A_plus - np.sum((1 + a0x * np.exp(Lx)  / b0x ) / v_denominators))**2

        return term1 + term2    

    # Perform minimization
    a0_b0_res = scipy.optimize.minimize(objective,[a0,b0],args=(v, Lx, z, p))

    # Get estimates for A and B
    a0_fit, b0_fit = a0_b0_res.x[0], a0_b0_res.x[1]
    denom = v - 1/a0_fit - 1/b0_fit
    A_num = 1 + (a0_fit/b0_fit)*np.exp(Lx)
    B_num = 1 + (b0_fit/(a0_fit*np.exp(Lx)))
    A_fit = A_num / denom
    B_fit = B_num / denom

    return A_fit, B_fit, a0_fit, b0_fit
