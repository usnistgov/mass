'''
Test framework for MaximumLikelihoodGaussianFitter and MaximumLikelihoodHistogramFitter

Created on Jan 13, 2012

@author: fowlerj
'''
import unittest
import numpy
#from mass.mathstat.fitting import  MaximumLikelihoodGaussianFitter #, MaximumLikelihoodHistogramFitter
import mass

class Test_ratio_weighted_averages(unittest.TestCase):
    """Run a test with a known, constant histogram"""
    
    def setUp(self):
        self.nobs = numpy.array([0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,
                            0,  0,  1,  1,  0,  2,  0,  2,  1,  1,  3,  2,  2,  6,  2,  5,  3,
                            6,  5, 15, 17,  9, 18, 12,  9, 17, 17, 14, 11, 22, 28, 21, 16, 14,
                           19, 16, 14, 24, 16,  7, 15,  8,  5, 15, 12, 13,  6,  8,  6,  6,  7,
                            7,  4,  2,  3,  5,  2,  1,  1,  1,  1,  0,  0,  1,  2,  0,  0,  0,
                            0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])

        self.x = numpy.arange(-1.98, 2.0, .04)
        params = [1.09, 0, 17.7, 0.0, 0.0]
        self.fitter = mass.mathstat.fitting.MaximumLikelihoodGaussianFitter(self.x, self.nobs, params, TOL=1e-6)
    
    def test_weighted_averages_nobg(self):
        """Wt avg (data/model)=1 for weight = model when the model has no background."""
        weightavg = lambda a,w: (a*w).sum()/w.sum()
        self.fitter.hold(3, 0.0)
        self.fitter.hold(4, 0.0)
        self.fitter.fit()
#        print self.fitter.params
        
        self.assertAlmostEqual(self.fitter.params[0], 1.08841365, 4)
        self.assertAlmostEqual(self.fitter.params[1], -.00963243161, 4)
        self.assertAlmostEqual(self.fitter.params[2], 17.7115968, 4)
        
        y = self.fitter.theory_function(self.fitter.params, self.x)
        ratio = self.nobs/y
        self.assertAlmostEqual( weightavg(ratio, y), 1.0, 7)
        self.assertAlmostEqual( self.fitter.chisq, 94.4462749877, 4)

    def test_weighted_averages_constbg(self):
        """Wt avg (data/model)=1 for weight = {constant, model}when the model has a constant background."""
        weightavg = lambda a,w: (a*w).sum()/w.sum()
        self.fitter.free(3)
        self.fitter.hold(4, 0.0)
        self.fitter.fit()
#        print self.fitter.params
        y = self.fitter.theory_function(self.fitter.params, self.x)
        ratio = self.nobs/y
        self.assertAlmostEqual( ratio.mean(), 1.0, 7)
        self.assertAlmostEqual( weightavg(ratio, y), 1.0, 7)
        self.assertAlmostEqual( self.fitter.chisq, 89.0739977032, 4)

    def test_weighted_averages_slopedbg(self):
        """Wt avg (data/model)=1 for weight = {constant, model, x} when the model has a linear background."""
        weightavg = lambda a,w: (a*w).sum()/w.sum()
        self.fitter.free(3)
        self.fitter.free(4)
        self.fitter.fit()
        self.fitter.params[3]=0
        self.fitter.TOL=1e-8
        self.fitter.fit()
#        print self.fitter.params
        y = self.fitter.theory_function(self.fitter.params, self.x)
        ratio = self.nobs/y
        self.assertAlmostEqual(ratio.mean(), 1, 6)
        self.assertAlmostEqual( weightavg(ratio, self.x), 1.0, 0)
        self.assertAlmostEqual( weightavg(ratio, y), 1.0, 2)
        self.assertAlmostEqual( self.fitter.chisq, 88.1947709122, 4)

class Test_gaussian(unittest.TestCase):
    """Simulate some Gaussian data, fit the histograms, and make sure that the results are
    consistent with the expectation at the 2-sigma level."""

    def generate_data(self, N, fwhm=1.0, ctr=0.0, nbins=100, use_bg=0):
        x = numpy.arange(.5,nbins)*4.0/nbins-2.0
        
        n_signal = numpy.random.poisson(N)
        n_bg = numpy.random.poisson(use_bg)
        
        data = numpy.random.standard_normal(size=n_signal)*fwhm/numpy.sqrt(8*numpy.log(2))
        if use_bg>0:
            data = numpy.hstack((data, numpy.random.uniform(size=n_bg)*4.0-2.0))
        nobs, _bins = numpy.histogram(data, numpy.arange(nbins+1)*4.0/nbins-2.0)
        self.sum = nobs.sum()
        self.mean = (x*nobs).sum()/nobs.sum()
        self.var = (x*x*nobs).sum()/nobs.sum() - self.mean**2
        
        params = (fwhm, self.mean, N/fwhm*4.0/nbins/1.06, use_bg*1.0/nbins)
        self.fitter = mass.mathstat.fitting.MaximumLikelihoodGaussianFitter(x, nobs, params, TOL=1e-8)
        if use_bg <= 0:
            self.fitter.hold(3,0.0)
        self.fitter.hold(4,0.0)

    def run_several_fits(self, N=1000, nfits=10, fwhm=1.0, ctr=0.0, nbins=100, use_bg=10):
        correct_params = (fwhm, ctr, .037535932*N, 0,0)
        sigma_errors = numpy.zeros((5,nfits), dtype=numpy.float)
        for i in range(nfits):
            self.generate_data(N, fwhm, ctr, nbins, use_bg)
            try:
#                print self.fitter.params
                params, covar = self.fitter.fit()
#                print params
                
            except numpy.linalg.LinAlgError:
                continue
            if numpy.any(numpy.isnan(params)): continue
            if numpy.any(numpy.isnan(covar)): continue
                        
            sigma_errors[:,i] = (params-correct_params)/covar.diagonal()**0.5
#            if i==1:
#                import pylab
#                pylab.clf()
#                pylab.ioff()
#                mass.plot_as_stepped_hist(pylab.gca(), self.fitter.x, self.fitter.nobs)
#                pylab.plot(self.fitter.x, self.fitter.theory_function(params, self.fitter.x))
#                pylab.draw()
#                print self.sum, self.mean, self.var
#                print params
#                import time
#                try:
#                    while True:
#                        time.sleep(.01)
#                except KeyboardInterrupt:
#                    pass

        sigma_errors[4,:] = 0
        maxparam=4
        if use_bg <=0:
            sigma_errors[3,:] = 0
            maxparam=3

        print sigma_errors.std(axis=1)
        self.assertTrue( numpy.all(sigma_errors[:maxparam].std(axis=1) < 1+2/nfits**0.5))
        
    def test_30fits_with_bg(self):
        "Run 30 fits with nonzero background and verify consistent parameters"
        fwhm = 1.0
        ctr = 0.0
        nbins = 100
        nfits = 30
        use_bg=100
        self.run_several_fits(1000, nfits, fwhm, ctr, nbins, use_bg)

    def test_30fits_small_bg(self):
        "Run 30 fits with one background event and verify consistent parameters"
        fwhm = 1.0
        ctr = 0.0
        nbins = 100
        nfits = 30
        use_bg=1
        self.run_several_fits(1000, nfits, fwhm, ctr, nbins, use_bg)

    def test_fits_zero_bg(self):
        "Run 50 fits with one background event and verify consistent parameters"
        fwhm = 1.0
        ctr = 0.0
        nbins = 100
        nfits = 50
        use_bg=0
        self.run_several_fits(1000, nfits, fwhm, ctr, nbins, use_bg)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()