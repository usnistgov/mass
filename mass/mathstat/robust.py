'''
mass.mathstat.robust

Functions from the field of robust statistics.

Created on Feb 9, 2012

@author: fowlerj
'''

__all__ = ['shorth_range','high_median', 'Qscale']

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
        weights = numpy.ones(n)
    else:
        weights = numpy.asarray(weights)
    total_weight = weights.sum()
    
    imin, imax = 0, n  # The possible range of j will always be the half-open interval [imin,imax)
    left_weight = right_weight = 0 # Total weight in (...,imin) and in [imax,...)
    itrial = n/2
    while imax-imin > 1:
        partial_left_weight = weights[sort_idx[imin:itrial]].sum() # from [imin,itrial)
        partial_right_weight = weights[sort_idx[itrial+1:imax]].sum()  # from (itrial,imax)
#        trial_weight = weights[sort_idx][itrial]
#        print "(%d, %d, %d) weights [%d, %d <%d> %d, %d]"%(imin, itrial, imax, left_weight, partial_left_weight, trial_weight, partial_right_weight, right_weight)
        
        if left_weight + partial_left_weight > 0.5*total_weight: # j < itrial
            right_weight += partial_right_weight
            imax = itrial
        elif right_weight + partial_right_weight >= 0.5*total_weight: # j > itrial
            left_weight += partial_left_weight
            imin = itrial
        else: # j == itrial
            if return_index:
                ri = sort_idx[itrial]
                return x[ri], ri
            return x[sort_idx[itrial]]
        itrial = (imin+imax)/2
    
    if return_index:
        ri = sort_idx[itrial]
        return x[ri], ri
    return x[sort_idx[itrial]]


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
    Heidelberg: Physica-Verlag pages 411-428.  Available at ftp://ftp.win.ua.ac.be/pub/preprints/92/Timeff92.pdf
    
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
        
        w = []
        rm = []
        ci = []
#        print left, right, 'xxxxxx'
        for i,(l,r) in enumerate(zip(left, right)):
            if l>r:
                w.append(0)
                rm.append(0) 
                ci.append(0)
                continue
            w.append(r+1-l)
            ctr_index = (l+r)/2
            if ctr_index >= n:
                ctr_index = n-1
            rm.append(x[ctr_index] - x[i])
            ci.append(ctr_index)
        weights = numpy.array(w)
        row_median = numpy.array(rm)

#        print "Have to choose from", row_median, weights
        trial_val, chosen_row =  high_median(row_median, weights=weights, return_index=True)
        chosen_col = ci[chosen_row]
        return trial_val, chosen_row, chosen_col

#    print "Data: ", x
#    dist = numpy.array([x-x[j] for j in range(n)])
#    dist[dist<0] = 0
#    print dist

    for _counter in range((n*(n-1))/2):
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
        
#        print 'Iter %3d: %2d candidates below trial distance of %f (ij=%d,%d)'%(_counter, candidates_below_trial_dist, trial_distance, trial_i, trial_j
#                                                                                ), trial_column, trial_column-numpy.arange(n-1)
        if candidates_below_trial_dist == target_k:
            return prefactor * trial_distance
        elif candidates_below_trial_dist > target_k:
            right = trial_column.copy()
            right[right>=n] = n-1
        else:
            left = trial_column+1
            left[trial_i] += 1
    raise RuntimeError("Qscale tried %d distances, which is too many"%_counter)