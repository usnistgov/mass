"""
mass.utilities

J. Fowler, NIST

March 24, 2011
"""

import numpy

class ToeplitzSolver(object):
    """
    Solve a Toeplitz matrix for one or more vectors.

    A Toeplitz matrix is an NxN square matrix where T_ij = R_(i-j) for some 
    vector R_k  with k=-(N-1),-(N-2),...-1,0,1,2,...(N-1).
    A symmetric Toeplitz matrix has R_k = R_(-k).

    Initialize the object with the R vector.
    """

    def __init__(self, R, symmetric=True):
        """
        The meaning of <R> depends on <symmetric>.  In both cases, it represents
        the values in an NxN Toeplitz matrix.
        
        When <symmetric> is True, <R> is of length N and equals both the 0-row and
        the 0-column of the matrix.
        
        When <symmetric> is False, <R> is of length (2N-1), matrix T_i is represented by 
        R[i+N-1].  Thus R[N-1] is the main diagonal, R[0] is the upper right value of T,
        and R[2*N-2] is the lower left value of T. 
        """
        
        self.symmetric=symmetric

        if symmetric:
            self.n = len(R)
        else:
            # R needs to be of length 2n-1 for integer n
            assert len(R) %2 == 1
            self.n = (len(R)+1)/2
            
        # Be very careful with self.R, because it's stored as a copy of the input R.
        # For symmetric matrices, T_(0,0) and T_(1,0) are R[0] and R[1].
        # For non-symmetric, they are R[n-1] and R[n].
        self.R = numpy.array(R).astype(float)


    def __call__(self, y):
        """Return the solution x for Tx=y"""
        if self.symmetric: return self.solve_symmetric(y)
        n = self.n
        assert len(y) == n

        zeros = lambda n: numpy.zeros(n, dtype=numpy.float)
        x = zeros(n)
        g = zeros(n)
        h = zeros(n)
        xh_denom = zeros(n)

        R0 = self.R[n-1]
        x[0] = y[0]/R0
        g[0] = self.R[n-2]/R0
        h[0] = self.R[n]/R0

        for K in range(1, n): # i = m+1
            # Steps b, c, and d (exit test)
            xh_denom[K] = (self.R[n:K+n]*g[:K]).sum() - R0
            x[K] = ((self.R[K+n-1:n-1:-1]*x[:K]).sum()-y[K])/xh_denom[K]
            x[:K] -= x[K]*g[K-1::-1]
            if K==n-1:  return x

            # Step e
            g_denom = (self.R[n-K-1:n-1]*h[K-1::-1]).sum() - R0
            h[K] = ((self.R[n+K-1:n-1:-1]*h[:K]).sum()-self.R[K+n])/xh_denom[K]
            g[K] = ((self.R[n-K-1:n-1]*g[:K]).sum()-self.R[n-K-2])/g_denom
            
            # Step f (careful not to clobber the prev iteration of g)
            gsave = g[:K].copy()
            g[:K] -= g[K]*h[K-1::-1]
            h[:K] -= h[K]*gsave[K-1::-1]


    def solve_symmetric(self, y):
        """Return the solution x when Tx=y for a symmetric Toeplitz matrix T."""
        n = self.n
        assert len(y) == n
        assert self.symmetric

        zeros = lambda n: numpy.zeros(n, dtype=numpy.float)
        x = zeros(n)
        g = zeros(n)

        R = self.R.copy()
        R0 = R[0]
        x[0] = y[0]/R0
        g[0] = R[1]/R0

        for K in range(1, n): # K = M+1
            # Steps b, c, and d (the exist test)
            x_denom = (R[1:K+1]*g[:K]).sum() - R0
            x[K] = ((R[K:0:-1]*x[:K]).sum()-y[K])/x_denom
            x[:K] -= x[K]*g[K-1::-1]
            if K==n-1:  return x

            # Step e
            g_denom = (R[1:K+1]*g[:K]).sum() - R0
            g[K] = ((R[K:0:-1]*g[:K]).sum()-R[K+1])/g_denom

            # Step f
            g[:K] -= g[K]*g[K-1::-1]
