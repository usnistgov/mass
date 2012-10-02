'''
mass.mathstat.robust

Functions from the field of robust statistics.

Location estimators:
bisquare_weighted_mean - Mean with weights given by the bisquare rho function.
huber_weighted_mean    - Mean with weights given by Huber's rho function.
trimean                - Tukey's trimean, the average of the median and the midhinge.
shorth_range           - Primarily a dispersion estimator, but location=True gives a (poor) location.

Dispersion estimators:
median_abs_dev - Median absolute deviation from the median.
shorth_range   - Length of the shortest closed interval containing at least half the data.  
Qscale         - Normalized Rousseeuw & Croux Q statistic, from the 25%ile of all 2-point distances.

Utility functions:
high_median    - Weighted median

Recommendations:
For location, suggest the bisquare_weighted_mean with k=3.9*sigma, if you can make any reasonable
guess as to the Gaussian-like width sigma.  If not, trimean is a good second choice, though less
efficient.

For dispersion, the Qscale is very efficient for nearly Gaussian data.  The median_abs_dev is 
the most robust though less efficient.  If Qscale doesn't work, then short_range is a good
second choice.

Created on Feb 9, 2012

@author: fowlerj
'''

__all__ = ['bisquare_weighted_mean', 'huber_weighted_mean', 'trimean', 
           'median_abs_dev', 'shorth_range','high_median', 'Qscale']

import numpy, scipy.stats

try:
    from mass.mathstat import _robust
except ImportError:
    from mass.mathstat.utilities import MissingLibrary
    _robust = MissingLibrary("_robust.so")


def bisquare_weighted_mean(x, k, center=None, tol=None):
    """Return the bisquare weighted mean of the data <x> with a k-value of <k>.
    A sensible choice of <k> is 3 to 5 times the rms width or 1.3 to 2 times the
    full width at half max of a peak.  For strictly Gaussian data, the choices of 
    k= 3.14, 3.88, and 4.68 times sigma will be 80%, 90%, and 95% efficient.

    <center> is used as an initial guess at the weighted mean.
    If <center> is None, then the data median will be used.

    The answer is found iteratively, revised until it changes by less than <tol>.  If
    <tol> is None (the default), then <tol> will use 1e-5 times the median absolute
    deviation of <x> about its median.

    Data values a distance of more than <k> from the weighted mean are given no
    weight."""
    
    if center is None:
        center = numpy.median(x)
    if tol is None:
        tol = 1e-5*median_abs_dev(x, normalize=True)

    for _iteration in xrange(100):
        weights = (1-((x-center)/k)**2.0)**2.0
        weights[numpy.abs(x-center)>k] = 0.0
        newcenter = (weights*x).sum()/weights.sum()
        if abs(newcenter - center)<tol:
            return newcenter
        center = newcenter
    raise RuntimeError("bisquare_weighted_mean used too many iterations.\n"+
                       "Consider using higher <tol> or better <center>, or change to trimean(x).")


def huber_weighted_mean(x, k, center=None, tol=None):
    """Return Huber's weighted mean of the data <x> with a k-value of <k>.
    A sensible choice of <k> is 1 to 1.5 times the rms width or 0.4 to 0.6 times the
    full width at half max of a peak.  For strictly Gaussian data, the choices of 
    k=1.0 and 1.4 sigma give ...

    <center> is used as an initial guess at the weighted mean.
    If <center> is None, then the data median will be used.

    The answer is found iteratively, revised until it changes by less than <tol>.  If
    <tol> is None (the default), then <tol> will use 1e-5 times the median absolute
    deviation of <x> about its median.

    Data values a distance of more than <k> from the weighted mean are given no
    weight."""
    
    if center is None:
        center = numpy.median(x)
    if tol is None:
        tol = 1e-5*median_abs_dev(x, normalize=True)

    for _iteration in xrange(100):
        weights = numpy.asarray((1.0*k)/numpy.abs(x-center))
        weights[weights > 1.0] = 1.0
        newcenter = (weights*x).sum()/weights.sum()
        if abs(newcenter - center)<tol:
            return newcenter
        center = newcenter
    raise RuntimeError("huber_weighted_mean used too many iterations.\n"+
                       "Consider using higher <tol> or better <center>, or change to trimean(x).")


def trimean(x):
    """Return Tukey's trimean for a data set <x>, a measure of its central tendency
    ("location" or "center").

    If (q1,q2,q3) are the quartiles (i.e., the 25%ile, median, and 75 %ile),
    the trimean is (q1+q3)/4 + q2/2. """
    q1,q2,q3 = [scipy.stats.scoreatpercentile(x, per) for per in (25,50,75)]
    trimean = 0.25*(q1+q3) + 0.5*q2
    return trimean


def median_abs_dev(x, normalize=False):
    """
    Median absolute deviation (from the median) of data set <x>.
    
    If normalize is True (default:False), then return MAD/0.675, which scaling makes the statistic
    consistent with the standard deviation for an asymptotically large sample of Gaussian deviates.
    """
    mad = numpy.median(numpy.abs(numpy.asarray(x)-numpy.median(x)))
    if normalize:
        return mad/0.674480  # Half of the normal distribution has abs(x-mu) < 0.674480*sigma
    return mad



def shorth_range(x, normalize=False, sort_inplace=False, location=False):
    """
    Return the Shortest Half (shorth) Range, a robust estimator of dispersion.
    
    The Shortest Half of a data set {x} means that closed interval [a,b] where (1) a and b are both 
    elements of the data set, (2) at least half of the elements are in the closed interval, and 
    (3) which minimizes the length of the closed interval (b-a).  The shorth range is (b-a).
    See mass.mathstat.robust.shorth_information for further explanation and references in the literature.
    
    x            - The data set under study.  Must be a sequence of values.
    normalize    - If False (default), then return the actual range b-a.  If True, then the range will be
                   divided by 1.348960, which normalizes the range to be a consistent estimator of the
                   parameter sigma in the case of an exact Gaussian distribution.  (A small correction of 
                   order 1/N is applied, too, which mostly corrects for bias at modest values of the sample
                   size N.)
    sort_inplace - Permit this function to reorder the data set <x>.  If False (default), then x will be 
                   copied and the copy will be sorted.  (Note that if <x> is not a numpy.ndarray, an error 
                   will be raised if <sort_inplace> is True.)
    location     - Whether to return two location estimators in addition to the dispersion estimator.  
                   Default: False.
    
    Returns:
    --------

    shorth range   if <location> evaluates to False; otherwise returns:
    (shorth range, shorth mean, shorth center)
    
    In this, shorth mean is the mean of all samples in the closed range [a,b], and
    shorth center = (a+b)/2.  Beware that both of these location estimators have the
    undesirable property that their asymptotic standard deviation improves only as
    N^(-1/3) rather than the more usual N^(-1/2).  So it is not a very good idea to
    use them as location estimators.  They are really only included here for testing
    just how useless they are.
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


def high_median(x, weights=None, return_index=False):
    """Compute the weighted high median of data set x with weights <weights>.
    
    It returns the smallest x[j] such that the sum of all weights for data
    with x[i] <= x[j] is strictly greater than half the total weight.
    
    If return_index is True, then the chosen index is returned also as (highmed, index).
    """
    x = numpy.asarray(x)
    sort_idx = x.argsort()  # now x[sort_idx] is sorted
    n = len(x)
    if weights is None:
        weights = numpy.ones(n, dtype=numpy.float)
    else:
        weights = numpy.asarray(weights, dtype=numpy.float)

    # If possible, use the Cython version _robust._high_median, though it only speeds up
    # by 15 to 20%.
    try:
        ri = _robust._high_median(sort_idx, weights, n)
    except ImportError:
        ri = _high_median_python_subroutine(sort_idx, weights, n)
        
    if return_index:
        return x[ri], ri
    return x[ri]



def Qscale(x, sort_inplace=False):
    """Compute the robust estimator of scale Q of Rousseeuw and Croux using only O(n log n)
    memory and computations.  (A naive implementation is O(n^2) in both.)
    
    <x>   The data set, an unsorted sequence of values.
    <sort_inplace> Whether it is okay for the function to reorder the set <x>.
                   If True, <x> must be a numpy.ndarray (or ValueError is raised).
    
    Q is defined as d_n * 2.2219 * {|xi-xj|; i<j}_k, where 
    
        {a}_k means the kth order-statistic of the set {a},
        this set is that of the distances between all (n 2) possible pairs of data in {x}
        n=# of observations in set {x},
        k = (n choose 2)/4,
        2.2219 makes Q consistent for sigma in normal distributions as n-->infinity,
        and d_n is a correction factor to the 2.2219 when n is not large.
    
    This function does apply the correction factors to make Q consistent with sigma for a 
    Gaussian distribution.
    
    Technique from C. Croux & P. Rousseeuw in Comp. Stat Vol 1 (1992) ed. Dodge & Whittaker,
    Heidelberg: Physica-Verlag pages 411-428.  Available at 
    ftp://ftp.win.ua.ac.be/pub/preprints/92/Timeff92.pdf
    
    The estimator is further studied in Rousseeuw & Croux, J Am. Stat. Assoc 88 (1993), pp 1273-1283.
    """
    
    if not sort_inplace:
        x = numpy.array(x)
    elif not isinstance(x, numpy.ndarray):
        raise ValueError("sort_inplace cannot be True unless the data set x is a numpy.ndarray.")
    x.sort()
    n = len(x)
    if n<2:
        raise ValueError("Data set <x> must contain at least 2 values!")
    h = n/2 + 1
    target_k = h*(h-1)/2 -1 # -1 so that order count can start at 0 instead of conventional 1,2,3...
    
    # Compute the n-dependent prefactor to make Q consistent with sigma of a Gaussian.
    prefactor = 2.2219
    if n <= 9:
        prefactor *= [0, 0, 0.399, 0.994, 0.512, 0.844, 0.611, 0.857, 0.669, 0.872][n]
    else:
        if n%2 == 1:
            prefactor *= n/(n+1.4)
        else:
            prefactor *= n/(n+3.8)

    # Now down to business finding the 25%ile of |xi - xj| for i<j (or equivalently, for i != j)
    # Imagine the upper triangle of the matrix Aij = xj - xi (upper means j>i).
    # If the set is sorted such that xi <= x(i+1) for any i, then the upper triangle of Aij contains
    # exactly those distances from which we need the k=n/4 order statistic.
    
    # For small lists, too many boundary problems arise.  Just do it the slow way:
    if n<=5:
        dist = numpy.hstack([x[j]-x[:j] for j in range(1,n)])
        assert len(dist) == (n*(n-1))/2
        dist.sort()
        return dist[target_k] * prefactor
    
    try:
        q, npasses = _robust._Qscale_subroutine(x, n, target_k)
    except ImportError:
        q, npasses = _Qscale_subroutine_python(x, n, target_k)
        
    if npasses > n:
        raise RuntimeError("Qscale tried %d distances, which is too many"%npasses)
    return q * prefactor

####################################################################################
# The next two subroutines are the Python equivalent of the much faster routines in the
# Cython file robust.pyx, and they are used only if Cython can't be built.
# _high_median_python_subroutine
# _Qscale_subroutine_python
####################################################################################


def _high_median_python_subroutine(sort_idx, weights, n):
    """Compute the weighted high median of data set with weights <weights>.
    Instead of sending the data set x, send the order statistics <sort_idx> over
    the data.
    
    USED ONLY if Cython fails to import _robust.so
    
    It returns the smallest j such that the sum of all weights for data
    with x[i] <= x[j] is strictly greater than half the total weight.
    """
    total_weight = weights.sum()
    
    imin, imax = 0, n  # The possible range of j will always be the half-open interval [imin,imax)
    left_weight = right_weight = 0 # Total weight in (...,imin) and in [imax,...)
    itrial = n/2
    while imax-imin > 1:
        partial_left_weight = weights[sort_idx[imin:itrial]].sum() # from [imin,itrial)
        partial_right_weight = weights[sort_idx[itrial+1:imax]].sum()  # from (itrial,imax)
        
        if left_weight + partial_left_weight > 0.5*total_weight: # j < itrial
            right_weight += partial_right_weight
            imax = itrial
        elif right_weight + partial_right_weight >= 0.5*total_weight: # j > itrial
            left_weight += partial_left_weight
            imin = itrial
        else: # j == itrial
            return sort_idx[itrial]
        itrial = (imin+imax)/2
    
    return sort_idx[itrial]


def _Qscale_subroutine_python(x, n, target_k):
    """This subroutine does most of the computation in Qscale(), and it should not normally
    be used.  Instead, the Cython routine _robust.Qscale_subroutine is preferred (it runs some
    12x faster).  This is here, though, for non-Cython installations to use."""

    # Now down to business finding the 25%ile of |xi - xj| for i<j (or equivalently, for i != j)
    # Imagine the upper triangle of the matrix Aij = xj - xi (upper means j>i).
    # If the set is sorted such that xi <= x(i+1) for any i, then the upper triangle of Aij contains
    # exactly those distances from which we need the k=n/4 order statistic.
    
    # Keep track of which candidates on each ROW are still in the running.
    # These limits are of length (n-1) because the lowest row has no upper-triangle elements. 
    # These mean that element A_ij = xj-xi is in the running if and only if  left(i) <= j <= right(i).
    left = numpy.arange(1, n, dtype=numpy.int)
    right = numpy.zeros(n-1, dtype=numpy.int) + (n-1)
    row_bias = x[numpy.arange(n-1)]
    trial_column = numpy.zeros(n-1, dtype=numpy.int)

    def choose_trial_val(left, right, x):
        """Choose a trial val as the weighted median of the medians of the remaining candidates in
        each row, where the weights are the number of candidates remaining in each row."""
        weights = right+1-left
        none_valid = left>right
        weights[none_valid] = 0
        ci = (left+right)/2
        ci[ci>=n] = n-1
        row_median = x[ci]-x[:n-1]
                
        trial_val, chosen_row =  high_median(row_median, weights=weights, return_index=True)
        chosen_col = ci[chosen_row]
        return trial_val, chosen_row, chosen_col

    for counter in xrange(n+10):
        trial_distance, trial_i, trial_j = choose_trial_val(left, right, x)
        per_row_value = trial_distance + row_bias
        
        # In each row i, find the highest index trial_column such that x[tc]-x[i] < trial_distance
        # If such an index is out of the candidate range, then let it be left-1 or right. 
        # We must be extremely careful in this loop, because rounding errors can make 
        # (trial_distance + x[i]) != x[j] even when trial distance was defined as (x[j]-x[i])
        # So instead we have the choose_trial_val tell us the (i,j) pair being considered
        # and test for it.
        
        # trial_column tracks the column in each row which is the highest column strictly less than
        # the trial distance.
        for i, trial_val in enumerate(per_row_value):
            if i==trial_i:
                trial_column[i] = trial_j-1
                continue
            
            ia, ib = left[i], right[i]
            if ia >= n:
                trial_column[i] = n-1
                continue
            if x[ia] >= trial_val:
                trial_column[i] = ia-1
                continue
            if x[ib] <= trial_val:
                trial_column[i] = ib
                continue
            while ib-ia>1:
                ip = (ib+ia)/2
                if x[ip] < trial_val:
                    ia = ip
                elif x[ip] > trial_val:
                    ib = ip
                else:
                    ia = ip-1
                    ib = ip
                    continue
            trial_column[i] = ia
        candidates_below_trial_dist = trial_column.sum() - ((n-2)*(n-1))/2
        
        if candidates_below_trial_dist == target_k:
#            print "Returning after %d passes"%counter
            return trial_distance, counter
        elif candidates_below_trial_dist > target_k:
            right = trial_column.copy()
            right[right>=n] = n-1
        else:
            left = trial_column+1
            left[trial_i] += 1
    raise RuntimeError("Qscale tried %d distances, which is too many"%counter)


