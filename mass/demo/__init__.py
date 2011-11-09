'''
Demonstration scripts for learning how to use Mass.

See mass.demo.helptxt for more info.

Created on Nov 9, 2011

@author: fowlerj
'''

__all__ = []

import os
this_dir, this_file= os.path.split(__file__)

demo_files=['demo.py']

fullpath = lambda f: os.path.join(this_dir, f) 
print fullpath('yomama')

helptxt = """
This package (mass.demo) consist of several demonstration scripts.
To run one as an IPython demo, you say:

from IPython.lib.demo import Demo
massdemo = Demo('%s')
massdemo() # Each call to the Demo object moves forward 1 more block in the demo.
massdemo()

The full list of available demos is:
%s
""" % (fullpath(demo_files[0]), demo_files)

print helptxt