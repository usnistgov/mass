'''
test_robust.py

Test any functions in mass.mathstat.robust

Created on Feb 9, 2012

@author: fowlerj
'''
import unittest
import numpy
from mass.mathstat.robust import shorth_range #@UnresolvedImport

class Test_Shorth(unittest.TestCase):
    """Test the function shorth_range, which computes the range of the shortest half"""

    def testUnnormalized(self):
        """Verify that you get actual shorth when normalize=False for odd and even size lists."""
        r = shorth_range([1,4,6,8,11], normalize=False)
        self.assertEqual(r, 4, msg="Did not find shortest half range in length-5 list")

        x = numpy.array([1,4.6,6,8,11])
        r, shr_mean, shr_ctr = shorth_range(x, normalize=False, location=True)
        self.assertEqual(r, x[3]-x[1], msg="Did not find shortest half range in length-5 list")
        self.assertEqual(shr_mean, x[1:4].mean(), msg="Did not find shortest half mean in length-5 list")
        self.assertEqual(shr_ctr, 0.5*(x[1]+x[3]), msg="Did not find shortest half center in length-5 list")
        
        r = shorth_range([2,4,6,8,11,15], normalize=False)
        self.assertEqual(r, 6, msg="Did not find shortest half range in length-6 list")
        
        x = numpy.array([1, 4.6, 6, 8, 11, 100])
        r, shr_mean, shr_ctr = shorth_range(x, normalize=False, location=True)
        self.assertEqual(r, x[4]-x[1], msg="Did not find shortest half range in length-6 list")
        self.assertEqual(shr_mean, x[1:5].mean(), msg="Did not find shortest half mean in length-6 list")
        self.assertEqual(shr_ctr, 0.5*(x[1]+x[4]), msg="Did not find shortest half center in length-6 list")
        
       

    def testSortInplace(self):
        """Verify behavior of the sort_inplace argument"""
        x = [7,1,2,3,4,5,6]
        y = numpy.array(x)
        _ignore = shorth_range(x, sort_inplace=False) 
        _ignore = shorth_range(y, sort_inplace=False) 
        self.assertEqual(x[0], 7, msg="shorth_range has reordered a list")
        self.assertEqual(y[0], 7, msg="shorth_range has reordered a ndarray")

        _ignore = shorth_range(y, sort_inplace=True) 
        self.assertEqual(y[0], 1, msg="shorth_range has not sorted a ndarray in place when requested")
        # Skip these two tests, as interface does not promise what will happen to a non-ndarray
        # when sort_inplace is True.
#        _ignore = shorth_range(x, sort_inplace=True) 
#        self.assertEqual(x[0], 7, msg="shorth_range has reordered a list")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()