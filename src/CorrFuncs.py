import numpy as np

def covariance_matrix(Ax,Bx,a0x,b0x,v):
    """Constructs covariance matrix according to Greenland and Longnecker
    """
    n = Ax.shape[0]
    s = np.sqrt(1/Ax + 1/Bx + 1/a0x + 1/b0x)
    r = ((1/np.outer(s,s))*(1/a0x + 1/b0x))[np.triu_indices(n,k=1)]
    c = r*np.sqrt((np.outer(v,v))[np.triu_indices(n,k=1)])
    triu_indices = np.triu_indices(n,k=1)
    C2 = np.zeros((n,n))
    C2[triu_indices] = c
    C = C2 + C2.T
    C += np.diag(v)
    return C

def trend_est(Ax,Bx,a0x,b0x,v,x,L,unadj=False):
    """Performs adjusted correlation on meta-analysis according to Greenland and Longnecker
    """
    # Calculating adjusted point and variance estimates
    C = covariance_matrix(Ax,Bx,a0x,b0x,v)
    Cinv = np.linalg.inv(C)
    vb_star = 1/(np.dot(x,np.dot(Cinv,x)))
    b_star = vb_star*(np.dot(x,np.dot(Cinv,L)))

    # Calculating unadjusted point and variance estimates if desired
    if unadj:
        vb = 1/(np.dot(x,np.dot(np.linalg.inv(np.diag(v)),x)))
        b = vb*(np.dot(x,np.dot(np.linalg.inv(np.diag(v)),L)))
        return b_star, vb_star, b, vb
    
    return b_star, vb_star