"""
Unit tests for MASS utilities.  
(So far, only the Toeplitz solver)

Also contains a speed test object TestToeplitzSpeed

J. Fowler, NIST

March 30, 2011
"""

import utilities
import numpy
import scipy.linalg
import time
import unittest

class TestToeplitzSolverSmallSymmetric(unittest.TestCase):
    "Test ToeplitzSolver on a 5x5 symmetric matrix"
    def setUp(self):
        self.autocorr=numpy.array((6.,4.,2.,1.,0.))
        self.n=len(self.autocorr)
        self.solver = utilities.ToeplitzSolver(self.autocorr, symmetric=True)
        self.R = scipy.linalg.toeplitz(self.autocorr)

    def test_all_unit_vectors(self):
        for i in range(self.n):
            x_in = numpy.zeros(self.n,dtype=numpy.float)
            x_in[i] = 1.0
            y = numpy.dot(self.R, x_in)
            x_out = self.solver(y)
            big_dif = numpy.abs(x_out-x_in).max()
            self.assertAlmostEqual(0, big_dif, 12)

    def test_arb_vectors(self):
        for _i in range(self.n):
            x_in = 5*numpy.random.standard_normal(self.n)
            y = numpy.dot(self.R, x_in)
            x_out = self.solver(y)
            big_dif = numpy.abs(x_out-x_in).max()
            self.assertAlmostEqual(0, big_dif, 12)

class TestToeplitzSolverSmallAsymmetric(unittest.TestCase):
    "Test ToeplitzSolver on a 5x5 non-symmetric matrix"
    def setUp(self):
        self.autocorr=numpy.array((-1,-2,0,3,6.,4.,2.,1.,0.))
        self.n=(len(self.autocorr)+1)/2
        self.solver = utilities.ToeplitzSolver(self.autocorr, symmetric=False)
        self.R = scipy.linalg.toeplitz(self.autocorr[self.n-1:], self.autocorr[self.n-1::-1])

    def test_all_unit_vectors(self):
        for i in range(self.n):
            x_in = numpy.zeros(self.n,dtype=numpy.float)
            x_in[i] = 1.0
            y = numpy.dot(self.R, x_in)
            x_out = self.solver(y)
            big_dif = numpy.abs(x_out-x_in).max()
            self.assertAlmostEqual(0, big_dif, 12)

    def test_arb_vectors(self):
        for _i in range(self.n):
            x_in = 5*numpy.random.standard_normal(self.n)
            y = numpy.dot(self.R, x_in)
            x_out = self.solver(y)
            big_dif = numpy.abs(x_out-x_in).max()
            self.assertAlmostEqual(0, big_dif, 12)

class TestToeplitzSolver_32(unittest.TestCase):
    "Test ToeplitzSolver on a 32x32 symmetric matrix"
    def setUp(self):
        self.n = 32
        t = numpy.arange(self.n)
        pi = numpy.pi
        T = 1.0*self.n
        self.autocorr = numpy.sin(pi*t/T)/(pi*t/T)
        self.autocorr[0] = 1
        self.autocorr[:5]*=numpy.arange(5,.5,-1)
        self.solver = utilities.ToeplitzSolver(self.autocorr, symmetric=True)
        self.R = scipy.linalg.toeplitz(self.autocorr)

    def test_all_unit_vectors(self):
        for i in range(self.n):
            x_in = numpy.zeros(self.n,dtype=numpy.float)
            x_in[i] = 1.0
            y = numpy.dot(self.R, x_in)
            x_out = self.solver(y)
            big_dif = numpy.abs(x_out-x_in).max()
            self.assertAlmostEqual(0, big_dif, 10, msg='Unit vector trial i=%2d gives x_out=%s'%(i,x_out))

class TestToeplitzSolver_6144(unittest.TestCase):
    "Test ToeplitzSolver on a 6144x6144 symmetric matrix"
    def setUp(self):
        self.n = 6144
        t = numpy.arange(self.n)
        self.autocorr = 1.0+3.2*numpy.exp(-t/100.)
        self.autocorr[0] = 9
        self.solver = utilities.ToeplitzSolver(self.autocorr, symmetric=True)
        self.R = scipy.linalg.toeplitz(self.autocorr)

    def test_some_unit_vectors(self):
        for i in (0,1024,2048,3210,6143):#4096,6000,8191):
            x_in = numpy.zeros(self.n,dtype=numpy.float)
            x_in[i] = 1.0
            y = numpy.dot(self.R, x_in)
            x_out = self.solver(y)
            big_dif = numpy.abs(x_out-x_in).max()
            self.assertAlmostEqual(0, big_dif, 10, msg='Unit vector trial i=%2d gives x_out=%s'%(i,x_out))
    
    def test_arb_vectors(self):
        for _i in range(5):
            x_in = 5*numpy.random.standard_normal(self.n)
            y = numpy.dot(self.R, x_in)
            x_out = self.solver(y)
            big_dif = numpy.abs(x_out-x_in).max()
            self.assertAlmostEqual(0, big_dif, 10, msg='Random vector trial gives rms diff=%sf'%(x_out-x_in).std())


class TestToeplitzSpeed(object):
    def __init__(self, maxsize=8192):
        self.sizes=numpy.hstack((100,200,numpy.arange(500,5500,500),6144,8192,10000,15000,20000,30000,50000, 100000))
#        self.sizes=numpy.array((4000,5000,6144,8192,10000,20000))
        t = numpy.arange(100000)
        self.autocorr = 1.0+3.2*numpy.exp(-t/100.)
        self.autocorr[0] = 9
#        self.solver = utilities.ToeplitzSolver(self.autocorr, symmetric=True)

        self.ts_time=numpy.zeros(len(self.sizes), dtype=numpy.float)
        self.build_time = numpy.zeros_like(self.ts_time)
        self.mult_time = numpy.zeros_like(self.ts_time)
        self.solve_time = numpy.zeros_like(self.ts_time)
        self.lu_time = numpy.zeros_like(self.ts_time)
        for i,s in enumerate(self.sizes):
            times = self.test(s, maxsize)
            if s == 10000:
                times = [times[0], 32.2,6.2, 58, 24.3]
            self.ts_time[i], self.build_time[i], self.mult_time[i], self.solve_time[i], self.lu_time[i] = times
            
    def test(self, size, maxsize=8192):
#        long_times={20000:4.433825,
#                    30000:9.065444,
#                    50000:22.867852,
#                    100000:93.42273998}
        long_times={50000:23.770, 100000:96.738}
        if size in long_times:
            return [long_times[size]] + 4*[numpy.NAN]
        elif size>150000:
            return 5*[numpy.NaN]
        
        ac = self.autocorr[:size]
        v = numpy.random.standard_normal(size)
        nv=-v

        t0 = time.time()
        solver = utilities.ToeplitzSolver(ac, symmetric=True)
#        x = solver(nv)  # If you want to solve two...
        x = solver(v)
        dt = [time.time()-t0]
        
        if size<=maxsize:
            # dt[1] = creating R time
            t0 = time.time()
            R = scipy.linalg.toeplitz(ac)
            dt.append(time.time()-t0)
            
            # dt[2] = R * vector time
            t0 = time.time()
            v2 = numpy.dot(R, x)
            dt.append(time.time()-t0)
            
            # dt[3] = solve(R,v) time
            t0 = time.time()
            x2 = numpy.linalg.solve(R, v)
            dt.append(time.time()-t0) 
            
            t0 = time.time()
            lu_piv = scipy.linalg.lu_factor(R)
            x3 = scipy.linalg.lu_solve(lu_piv, v, overwrite_b=False)
            dt.append(time.time()-t0) 
            print 'rms rhs diff: %.3g, solution diff: %.3g %.3g'%((v-v2).std(), (x-x2).std(), (x-x3).std())
            
        else:
            dt.extend(4*[numpy.NaN])                
        print size, ['%6.3f'%t for t in dt]
        return dt

    def plot(self):
        import pylab
        pylab.clf()
        pylab.plot(self.sizes, self.ts_time, label='Toeplitz solver')
        pylab.plot(self.sizes, self.build_time, label='Matrix build')
        pylab.plot(self.sizes, self.mult_time, label='Matrix-vector mult')
        pylab.plot(self.sizes, self.solve_time, label='Matrix solve')
        pylab.plot(self.sizes, self.lu_time, label='LU solve')
        pylab.legend(loc='upper left')
        pylab.xlabel("Vector size")
        pylab.ylabel("Time (sec)")
        pylab.grid()
        pylab.loglog()
        
        
if __name__ == '__main__':
    unittest.main()