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

helptxt = """
This package (mass.demo) consist of several demonstration scripts.
To run one as an IPython demo, you say:

# On Joe's Mac:

from IPython.lib.demo import Demo
massdemo = Demo('%s')
massdemo()
massdemo() # Each call to the Demo object moves forward 1 more block in the demo.
           # The basic demo.py has close to a dozen blocks to step through.

# On Linux (at least on Horton 11/14/11):
from IPython.demo import Demo
massdemo = Demo('%s')
massdemo()

The full list of available demos is:
%s
""" % (fullpath(demo_files[0]), demo_files)

print helptxt