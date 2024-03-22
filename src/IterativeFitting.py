import scipy
import numpy as np
import cvxpy as cp

class IterativeFitting:
    # TODO: Change class name to GLMethods
    def __init__(self, L, A0, N, M1):
        self.N = N
        self.M1 = M1
        self.L = L
        self.A0 = A0

        self.A_cvx = None
        self.A_GLL = None
        self.A_GL = None

        self.B_cvx = None
        self.B_GLL = None
        self.B_GL = None

        self.a0_cvx = None
        self.a0_GLL = None
        self.a0_GL = None

        self.b0_cvx = None
        self.b0_GLL = None
        self.b0_GL = None

        self.n = L.shape[0]

    def convexProgram(self, constraints=[], A_const = False, N_const=False, M1_const=False, OR=True):
        """Solves the convex minimization problem of the paper.
        Note that every "constraints" argument must have elements defined as lambda functions of the form:
            lambda A,N,M1: cp.sum(A) == 115 
        as an example. This is because A, N, M1 is not being defined until the function is being called.
        """
        if ~N_const:
            N = self.N.copy()
        else:
            N = cp.Variable(shape=(self.n + 1))
        if ~M1_const:
            M1 = self.M1
        else:
            M1 = cp.Variable()
        L = self.L.copy()
        A = cp.Variable(shape=self.n)
        constraints_eval = [c(A,N,M1) for c in constraints]
        if OR:
            obj = cp.Minimize(
                cp.scalar_product(-L, A) -
                cp.entr(M1 - cp.sum(A)) - (M1 - cp.sum(A)) +
                cp.sum(-cp.entr(N[1:] - A) - (N[1:] - A)) +
                cp.sum(-cp.entr(A) - A) -
                cp.entr(N[0] - M1 + cp.sum(A)) - (N[0] - M1 + cp.sum(A)) 
            )
        else:
            log_N = np.log(N[1:])
            log_n0 = np.log(N[0])
            obj = cp.Minimize(
                cp.scalar_product(A,(-L - log_N + log_n0)) + 
                cp.sum(-cp.entr(A) - A) - 
                cp.entr(M1 - cp.sum(A)) - M1 + cp.sum(A)
            )
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
        problem = cp.Problem(obj)
        problem.solve(solver=cp.ECOS)
        if OR:
            self.A_cvx = A.value
            self.B_cvx = N[1:] - self.A_cvx
            self.a0_cvx = M1 - np.sum(self.A_cvx)
            self.b0_cvx = N[0] - M1 + np.sum(self.A_cvx)
        else:
            self.A_cvx = A.value
            self.B_cvx = N[1:]
            self.a0_cvx = M1 - np.sum(self.A_cvx)
            self.b0_cvx = N[0]
        return self.A_cvx, self.B_cvx, self.a0_cvx, self.b0_cvx

    def GL_linesearch(self):
        """This implements our function with the line search over root-finding Newton's.
        Performs a line search.
        """
        N = self.N.copy()
        M1 = self.M1
        A = self.A0.copy()
        L = self.L.copy()
        diff = 1
        i = 1
        while diff > 1e-6:
            A1 = A
            Aplus = A.sum()
            a0 = M1 - Aplus
            b0 = N[0] - a0
            B = N[1:] - A
            c0 = 1/a0 + 1/b0
            c = 1/A + 1/B

            # Gradient step in Newton's Method and get ∆A
            e = L + np.log(a0) + np.log(B) - np.log(A) - np.log(b0)
            H = np.ones((self.n,self.n))*c0
            H += np.diag(c)
            dA = scipy.linalg.solve(H,e,assume_a="pos")
            dA_pos = np.where(dA > 0, dA, np.nan)
            dA_neg = np.where(dA < 0, dA, np.nan)

            # Taking minimum over positive change values
            A_pos = np.where(dA>0, A, np.nan)
            N_pos = np.where(dA>0, (N[1:]), np.nan)
            pre_alpha1 = (N_pos - A_pos)/(dA_pos)
            pre_alpha1 = np.nan_to_num(pre_alpha1, nan=1e10)
            alpha1 = np.min(pre_alpha1) # take smallest element from here

            # Taking minimum over negative change values
            A_neg = np.where(dA<0, A, np.nan)
            pre_alpha2 = -A_neg / dA_neg
            pre_alpha2 = np.nan_to_num(pre_alpha2, nan=1e10)
            alpha2 = np.min(pre_alpha2)

            # Choosing the smallest alpha (step size)
            alpha = 0.99*np.min(np.array([alpha1,alpha2,1]))
            # alpha = 0.5

            # Update A
            A += alpha*dA
            i += 1
            diff = np.linalg.norm(A1 - A)
        B = N[1:] - A
        a0 = M1 - A.sum()
        b0 = N[0] - a0
        self.A_GLL = A
        self.B_GLL = B
        self.a0_GLL = a0
        self.b0_GLL = b0
        return A, a0, B, b0, i

    def GL(self,OR=True):
        """Standard GL method.
        """
        N = self.N.copy()
        M1 = self.M1
        A = self.A0.copy()
        L = self.L.copy()
        diff = 1
        i = 1
        while diff > 1e-6:
            A1 = A
            Aplus = A.sum()
            a0 = M1 - Aplus
            if OR:
                b0 = N[0] - a0
                B = N[1:] - A
                c0 = 1/a0 + 1/b0
                c = 1/A + 1/B
            else:
                b0 = N[0]
                B = N[1:]
                c0 = 1/a0
                c = 1/A

            # Gradient step in Newton's Method and get ∆A
            e = L + np.log(a0) + np.log(B) - np.log(A) - np.log(b0)
            H = np.ones((self.n,self.n))*c0
            H += np.diag(c)
            A += scipy.linalg.solve(H,e,assume_a="pos")
            i += 1
            diff = np.linalg.norm(A1 - A)
        B = N[1:] - A
        a0 = M1 - A.sum()
        b0 = N[0] - a0
        self.A_GL = A
        self.B_GL = B
        self.a0_GL = a0
        self.b0_GL = b0
        return A, a0, B, b0, i