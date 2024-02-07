import numpy as np

def covariance_matrix(it_fit, v, cvx=False, GLL=False, GL=False):
    """Constructs covariance matrix according to Greenland and Longnecker
    """
    if cvx:
        it_fit.convexProgram()
        Ax = it_fit.A_cvx
        Bx = it_fit.B_cvx
        a0x = it_fit.a0_cvx
        b0x = it_fit.b0_cvx
    if GLL:
        it_fit.GL_linesearch()
        Ax = it_fit.A_GLL
        Bx = it_fit.B_GLL
        a0x = it_fit.a0_GLL
        b0x = it_fit.b0_GLL
    if GL:
        it_fit.GL()
        Ax = it_fit.A_GL
        Bx = it_fit.B_GL
        a0x = it_fit.a0_GL
        b0x = it_fit.b0_GL
    n = Ax.shape[0]
    s = np.sqrt(1/Ax + 1/Bx + 1/a0x + 1/b0x)
    r = ((1/np.outer(s,s))*(1/a0x + 1/b0x))[np.triu_indices(n,k=1)]
    c = r*np.sqrt((np.outer(v,v))[np.triu_indices(n,k=1)])
    C1 = np.diag(v)
    triu_indices = np.triu_indices(n,k=1)
    tril_indices = np.tril_indices(n,k=-1)
    C2 = C1.copy()
    C2[triu_indices] = c
    C = C2.copy()
    C[tril_indices] = c
    return C

def trend_est(it_fit, v, x, cvx=False, GLL=False, GL=False, unadj=False):
    """Performs adjusted correlation on meta-analysis according to Greenland and Longnecker
    """
    # Calculating adjusted point and variance estimates
    C = covariance_matrix(it_fit,v,cvx=cvx,GLL=GLL,GL=GL)
    Cinv = np.linalg.inv(C)
    vb_star = 1/(np.dot(x,np.dot(Cinv,x)))
    b_star = vb_star*(np.dot(x,np.dot(Cinv,it_fit.L)))

    # Calculating unadjusted point and variance estimates if desired
    if unadj:
        vb = 1/(np.dot(x,np.dot(np.linalg.inv(np.diag(v)),x)))
        b = vb*(np.dot(x,np.dot(np.linalg.inv(np.diag(v)),it_fit.L)))
        return b_star, vb_star, b, vb
    
    return b_star, vb_star