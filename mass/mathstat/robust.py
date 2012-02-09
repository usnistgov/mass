'''
mass.mathstat.robust

Functions from the field of robust statistics.

Created on Feb 9, 2012

@author: fowlerj
'''

__all__ = ['shorth_range',]

import numpy #, scipy.stats



def shorth_range(x, normalize=False, sort_inplace=False, location=False):
    """
    Return the Shortest Half (shorth) Range, a robust estimator of dispersion.
    
    The Shortest Half of a data set {x} means that closed interval [a,b] where (1) a and b are both 
    elements of the data set, (2) at least half of the elements are in the closed interval, and 
    (3) which minimizes the length of the closed interval (b-a).  The shorth range is (b-a).
    See mass.mathstat.robust.shorth_information for further explanation and references in the literature.
    
    x            - The data set under study.  Must be a sequence of values.
    normalize    - If False (default), then return the actual range b-a.  If True, then the range will be
                   divided by 1.xxxx, which normalizes the range to be a consistent estimator of the parameter 
                   sigma in the case of an exact Gaussian distribution.  (A small correction of order 1/N is
                   applied, too, which mostly corrects for bias at modest values of the sample size N.)
    sort_inplace - Permit this function to reorder the data set <x>.  If False (default), then x will be 
                   copied and the copy will be sorted.  (Note that if <x> is not a numpy.ndarray, an error 
                   will be raised if <sort_inplace> is True.)
    location     - Whether to return two location estimators in addition to the dispersion estimator.  Default: False.
    
    Returns:

    shorth range   if <location> evaluates to False, or otherwise the tuple (shorth range, shorth mean, shorth center).
    In this, shorth mean is the mean of all samples in the closed range [a,b], and shorth center is (a+b)/2.
    Beware that both of these location estimators have the undesirable property that their asymptotic standard
    deviation improves only as N^(-1/3) rather than the more usual N^(-1/2).  So it is not a very good idea to
    use them as location estimators.
    """
    
    n = len(x)            # Number of data values
    nhalves=int((n+1)/2)  # Number of minimal intervals containing at least half the data 
    nobs=1+int(n/2)       # Number of data values in each minimal interval

    if not sort_inplace:
        x = numpy.array(x)
    elif not isinstance(x, numpy.ndarray):
        raise ValueError("sort_inplace cannot be True unless the data set x is a numpy.ndarray.")
    x.sort()

    range_each_half = x[n-nhalves:n]-x[0:nhalves]
    idxa = range_each_half.argmin()
    a, b = x[idxa], x[idxa+nobs-1]
    shorth_range = b-a
    
    if normalize:
        shorth_range = shorth_range/(2*.674480)   # The asymptotic expectation for normal data is sigma*2*0.674480
        
        # The small-n corrections depend on n mod 4.  See [citation]
        if n%4==0:
            shorth_range *= (n+1.0)/n
        elif n%4==1:
            shorth_range *= (n+1.0)/(n-1.0)
        elif n%4==2:
            shorth_range *= (n+1.0)/n
        else:
            shorth_range *= (n+1.0)/(n-1.0)
    
    if location:
        return shorth_range, x[idxa:idxa+nobs].mean(), 0.5*(a+b)
    return shorth_range



shorth_information = """
The shorth ("shortest half") is a useful concept from robust statistics, leading to simple robust estimators
for both the dispersion and the location of a unimodal distribution.

....
"""