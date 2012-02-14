"""
robust.pyx - Routines for robust statistics that are written in Cython for speed.

Joe Fowler, NIST Boulder Labs

February 13-14, 2012
"""


import numpy
cimport cython, numpy
ITYPE=numpy.int
UTYPE=numpy.uint
FTYPE=numpy.float
ctypedef numpy.int_t ITYPE_t
ctypedef numpy.uint_t UTYPE_t
ctypedef numpy.float_t FTYPE_t


def hello_world():
    """This function simply proves that the Cython module robust.pyx exists."""
    print "The robust.pyx test is working!"
    
    
@cython.boundscheck(False)
def _high_median(numpy.ndarray[ITYPE_t, ndim=1] sort_idx, 
                numpy.ndarray[FTYPE_t, ndim=1] weights,
                int n):
    """
    Compute the weighted high median of data set with weights <weights>.
    Instead of sending the data set x, send the order statistics <sort_idx> over
    the data.
    
    It returns the smallest j such that the sum of all weights for data
    with x[i] <= x[j] is strictly greater than half the total weight.
    
    If return_index is True, then the chosen index is returned also as (highmed, index).
    """
    cdef FTYPE_t total_weight, half_weight
    cdef int i
    
    total_weight = 0.0
    for i in range(n):
        total_weight += weights[i]
    half_weight = 0.5*total_weight
    
    cdef int imin, imax, itrial
    cdef FTYPE_t left_weight, right_weight
    cdef FTYPE_t trial_left_weight, trial_right_weight

    imin, imax = 0, n  # The possible range of j will always be the half-open interval [imin,imax)
    left_weight = right_weight = 0 # Total weight in (...,imin) and in [imax,...)
    itrial = n/2
    while imax-imin > 1:
#        trial_left_weight = weights[sort_idx[imin:itrial]].sum() # from [imin,itrial)
#        trial_right_weight = weights[sort_idx[itrial+1:imax]].sum()  # from (itrial,imax)
        
        trial_left_weight = 0
        trial_right_weight = 0
        for i in range(imin, itrial):
            trial_left_weight += weights[sort_idx[i]]
        for i in range(itrial+1, imax):
            trial_right_weight += weights[sort_idx[i]]
        
        if left_weight + trial_left_weight > half_weight: # j < itrial
            right_weight += trial_right_weight
            imax = itrial
        elif right_weight + trial_right_weight >= half_weight: # j > itrial
            left_weight += trial_left_weight
            imin = itrial
        else: # j == itrial
            break
        itrial = (imin+imax)/2

    return sort_idx[itrial]



@cython.boundscheck(False)
def _choose_trial_val(numpy.ndarray[ITYPE_t, ndim=1] left,
                      numpy.ndarray[ITYPE_t, ndim=1] right, 
                      numpy.ndarray[FTYPE_t, ndim=1] x, 
                      int n):
    """Choose a trial val as the weighted median of the medians of the remaining candidates in
    each row, where the weights are the number of candidates remaining in each row."""

    cdef int i, chosen_row, chosen_col, ctr_index
    cdef float trial_val
    cdef numpy.ndarray[FTYPE_t, ndim=1] weights = numpy.zeros(n-1, dtype=FTYPE)
    cdef numpy.ndarray[FTYPE_t, ndim=1] row_median = numpy.zeros(n-1, dtype=FTYPE)
    
    for i in range(n-1):
        weights[i] = right[i]+1-left[i]
        if left[i]>right[i]:
            weights[i] = 0
        ctr_index = (left[i]+right[i])/2
        if ctr_index>=n:
            ctr_index = n-1
        row_median[i] = x[ctr_index]-x[i]
    
    cdef numpy.ndarray[ITYPE_t, ndim=1] row_sort_idx
    row_sort_idx = numpy.argsort(row_median)
    chosen_row =  _high_median(row_sort_idx, weights, n-1)
    trial_val = row_median[chosen_row]
    chosen_col = (left[chosen_row]+right[chosen_row])/2
    if chosen_col >= n:
        chosen_col = n-1
    return trial_val, chosen_row, chosen_col



@cython.boundscheck(False) # turn off bounds-checking for entire function
def _Qscale_subroutine(numpy.ndarray[FTYPE_t, ndim=1] x,
                      unsigned int n,
                      unsigned int target_k):
    
    cdef unsigned int i, trial_q_row=0, trial_q_col=0
    cdef int counter=0
    cdef FTYPE_t trial_distance=0.0#, trial_val=0.0
    cdef unsigned int candidates_below_trial_dist
    
    # Keep track of which candidates on each ROW are still in the running.
    # These limits are of length (n-1) because the lowest row has no upper-triangle elements. 
    # These mean that element A_ij = xj-xi is in the running if and only if  left(i) <= j <= right(i).
    cdef numpy.ndarray[ITYPE_t, ndim=1] trial_column = numpy.zeros(n-1, dtype=ITYPE)
    cdef numpy.ndarray[ITYPE_t, ndim=1] left = numpy.arange(1, n, dtype=ITYPE)
    cdef numpy.ndarray[ITYPE_t, ndim=1] right = numpy.zeros(n-1, dtype=ITYPE)
    for i in range(n-1):
        right[i] = n-1
    cdef numpy.ndarray[FTYPE_t, ndim=1] per_row_value = numpy.zeros(n-1, dtype=numpy.float)

    # In this loop, we close in on the Q that we seek by doing a bisection search of each row,
    # with left and right representing the smallest and largest column #s still in the running.
    for counter in xrange(n+10):
        trial_distance, trial_q_row, trial_q_col = _choose_trial_val(left, right, x, n)
#        for i in range(n-1):
#            per_row_value[i] = trial_distance + x[i]
        per_row_value = x[:n-1] + trial_distance
        
        # In each row i, find the highest index trial_column such that x[tc]-x[i] < trial_distance
        # If such an index is out of the candidate range, then let it be left-1 or right. 
        # We must be extremely careful in this loop, because rounding errors can make 
        # (trial_distance + x[i]) != x[j] even when trial distance was defined as (x[j]-x[i])
        # So instead we have the choose_trial_val tell us the (i,j) pair being considered
        # and test for it.
        
        # trial_column tracks the column in each row which is the highest column strictly less than
        # the trial distance.
        for i, trial_val in enumerate(per_row_value):
            
            # Test for if this is the row containing the trial Q-value.  If so, column is known.
            # Use this to avoid comparing exact equality of floats on the trial Q's (row,col).
            if i==trial_q_row:
                trial_column[i] = trial_q_col-1
                continue
            
            ia = left[i]
            ib = right[i]
            if ia>ib or x[ia] >= trial_val:
                trial_column[i] = ia-1
                continue
            if x[ib] <= trial_val:
                trial_column[i] = ib
                continue
            while ib-ia>1:
                imiddle = (ib+ia)/2
                if x[imiddle] < trial_val:
                    ia = imiddle
                elif x[imiddle] > trial_val:
                    ib = imiddle
                else:
                    ia = imiddle-1
                    ib = imiddle
                    break
            trial_column[i] = ia
        candidates_below_trial_dist = trial_column.sum() - ((n-2)*(n-1))/2

        
#        print 'Iter %3d: %2d cand < tri_dist %f (ij=%d,%d)'%(_counter, candidates_below_trial_dist, trial_distance, trial_q_row, trial_q_col
#                                                                                ), trial_column, trial_column-numpy.arange(n-1)
        if candidates_below_trial_dist == target_k:
            return trial_distance, counter
        elif candidates_below_trial_dist > target_k:
            for i in range(n-1):
                right[i] = trial_column[i]
                if right[i] >= n: 
                    right[i] = n-1
        else:
            for i in range(n-1):
                left[i] = trial_column[i] + 1
            left[trial_q_row] += 1
    return trial_distance, counter
