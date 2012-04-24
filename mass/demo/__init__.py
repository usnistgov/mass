'''
Demonstration scripts for learning how to use Mass.

See mass.demo.helptxt for more info.

Created on Nov 9, 2011

@author: fowlerj
'''

try:
    from IPython.lib.demo import Demo
except ImportError:
    from IPython.demo import Demo

__all__ = []

import os
this_dir, this_file= os.path.split(__file__)

demo_files=['demo.py', 'fitting_demo.py']

fullpath = lambda f: os.path.join(this_dir, f) 

demos = {}
for f in demo_files:
    demos[f] = Demo(fullpath(f)) 

helptxt = """
This package (mass.demo) consist of several demonstration scripts.
To run one as an IPython demo, you say:

massdemo = mass.demo.demos['%s']
massdemo()
massdemo() # Each call to the Demo object moves forward 1 more block in the demo.
#      The basic demo.py has close to a dozen blocks to step through.
#      If you want to start over, in the middle or after completing the demo:
massdemo.reset()

The full list of available demos is:
%s
""" % (demo_files[0], demo_files)

print helptxt
