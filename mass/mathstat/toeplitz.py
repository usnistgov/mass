'''
toeplitz - General-purpose solver of Toeplitz matrices

Created on Nov 7, 2011

@author: fowlerj
'''

import numpy as np

__all__ = ['ToeplitzSolver']


class ToeplitzSolver(object):
    """Solve a Toeplitz matrix for one or more vectors.

    A Toeplitz matrix is an NxN square matrix where T_ij = R_(i-j) for some
    vector R_k  with k=-(N-1),-(N-2),...-1,0,1,2,...(N-1).
    A symmetric Toeplitz matrix has R_k = R_(-k).

    Initialize the object with the R vector. Careful!  Notice that R is to be specified
    differently depending on the choice of symmetric vs asymmetric matrix.

    Typical usage for a symmetric Toeplitz matrix:
    ac = compute_autocorrelation(...) # ac[0] is 0-lag, ac[1] is lag-1, etc..
    rhs_vect = compute_rhs_vector(...)  # rhs_vect and ac should have same length
    ts = ToeplitzSolver(ac, symmetric=True)
    solution_vect = ts(rhs_vect)

    The solver uses Levinson's algorithm, as explained in Numerical Recipes, 3rd
    Edition section 2.8.2.  For my exact notation, see Joe Fowler's NIST lab book 2
    pages 148-151 (March 30, 2011).

    Timing results from my 4-core 2010-era Mac show that the calculation (as implemented
    on March 31, 2011; with changes, your results may vary) could solve a symmetric
    N=3000 system once in 0.25 seconds, N=5000 in 0.50 seconds, N=8192 in 1.0 seconds,
    N=10k in 1.4 seconds, and N=20k in 4.6 seconds.  Additional solutions to the same
    matrix should take between 0.5 and 0.6 times as long, since the one-time precomputation
    step is approximately as long as the per-solution computations.
    """

    def __init__(self, R, symmetric=True):
        """Initialize a Toeplitz matrix solver.

        Args:
            R: The values in an NxN Toeplitz matrix. The meaning of depends on symmetric.
            symmetric (bool): How to interpret R.

        When <symmetric> is True, <R> is of length N and gives both the 0-row and
        the 0-column of the matrix.

        When <symmetric> is False, <R> is of length (2N-1), and matrix T_ij is represented by
        R[i-j+N-1].  Thus R[N-1] is the main diagonal, R[0] is the upper right value of T,
        and R[2*N-2] is the lower left value of T.
        """

        # Whether this Toeplitz matrix is symmetric.  Governs how we compute solutions.
        self.symmetric = symmetric

        # The dimension of the square matrix T.
        self.n = len(R)
        if not symmetric:
            # R needs to be of length 2n-1 for integer n
            assert len(R) % 2 == 1
            self.n = (len(R)+1) // 2

        # Be very careful with self.R, because it's stored as a copy of the input R.
        # For symmetric matrices, T_(0,0) and T_(1,0) are R[0] and R[1].
        # For non-symmetric, they are R[n-1] and R[n].
        # The non-redundant elements of the Toeplitz matrix.  This will be the top
        # row if symmetric, or otherwise the first column (bottom to top) appended to the rest of
        # the top row.
        self.R = np.array(R).astype(float)

        # It would be good to have a precomputation step for asymmetric matrices, too,
        # but I don't need it now and don't want to spend the time!
        if symmetric:
            self.__precompute_symmetric()

    def mult(self, x):
        """Return y=Tx"""
        assert self.symmetric
        N = len(x)
        assert N == self.n
        y = np.zeros_like(x)
        y[0] = np.dot(self.R, x)
        for i in range(1, N):
            y[i] = np.dot(self.R[:-i], x[i:])
            y[i] += np.dot(self.R[1:1+i], x[i-1::-1])
        return y

    def __call__(self, y):
        """Return the solution x for Tx=y"""
        if self.symmetric:
            return self.__solve_symmetric(y)
        return self.__solve_asymmetric(y)

    def __solve_asymmetric(self, y):
        """Return the solution x when Tx=y for an asymmetric Toeplitz matrix T."""
        n = self.n
        assert len(y) == n

        x = np.zeros(n, dtype=float)
        g = np.zeros(n, dtype=float)
        h = np.zeros(n, dtype=float)
        xh_denom = np.zeros(n, dtype=float)

        R0 = self.R[n-1]
        x[0] = y[0]/R0
        g[0] = self.R[n-2]/R0
        h[0] = self.R[n]/R0

        for K in range(1, n):  # i = m+1
            # Steps b, c, and d (exit test)
            xh_denom[K] = (self.R[n:K+n]*g[:K]).sum() - R0
            x[K] = ((self.R[K+n-1:n-1:-1]*x[:K]).sum()-y[K])/xh_denom[K]
            x[:K] -= x[K]*g[K-1::-1]
            if K == n-1:
                return x

            # Step e
            g_denom = (self.R[n-K-1:n-1]*h[K-1::-1]).sum() - R0
            h[K] = ((self.R[n+K-1:n-1:-1]*h[:K]).sum()-self.R[K+n])/xh_denom[K]
            g[K] = ((self.R[n-K-1:n-1]*g[:K]).sum()-self.R[n-K-2])/g_denom

            # Step f (careful not to clobber the prev iteration of g)
            gsave = g[:K].copy()
            g[:K] -= g[K]*h[K-1::-1]
            h[:K] -= h[K]*gsave[K-1::-1]

    def __precompute_symmetric(self):
        """Precompute some data so that the solve_symmetric method can be done in
        roughly half the time per solve."""
        n = self.n
        assert self.symmetric

        g = np.zeros(n, dtype=float)
        # The constant denominator of the x_g computation
        self.xg_denom = np.zeros(n, dtype=float)
        # The constant leading value g[K] for each iteration K
        self.gK_leading = np.zeros(n, dtype=float)

        R = self.R.copy()
        R0 = R[0]
        g[0] = R[1]/R0

        for K in range(1, n):  # K = M+1
            self.xg_denom[K] = (R[1:K+1]*g[:K]).sum() - R0
            if K == n-1:
                return
            g[K] = ((R[K:0:-1]*g[:K]).sum()-R[K+1])/self.xg_denom[K]
            self.gK_leading[K] = g[K]
            g[:K] -= g[K]*g[K-1::-1]

    def __solve_symmetric(self, y):
        """Return the solution x when Tx=y for a symmetric Toeplitz matrix T."""
        n = self.n
        assert len(y) == n
        assert self.symmetric

        x = np.zeros(n, dtype=float)
        g = np.zeros(n, dtype=float)

        R = self.R.copy()
        R0 = R[0]
        x[0] = y[0]/R0
        g[0] = R[1]/R0

        for K in range(1, n):  # K = M+1
            # Steps b, c, and d (the exit test)
            x[K] = ((R[K:0:-1]*x[:K]).sum()-y[K])/self.xg_denom[K]
            x[:K] -= x[K]*g[K-1::-1]
            if K == n-1:
                return x

            # Steps e and f
            g[K] = self.gK_leading[K]
            g[:K] -= g[K]*g[K-1::-1]
