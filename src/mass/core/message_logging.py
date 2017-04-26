"""
Simple logging functions, to help us control how verbose MASS is at a terminal
or when being tested.
"""

import logging

mylogger = logging.getLogger("mass")
mylogger.setLevel(logging.INFO)

# create console handler and a formatter
ch = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')

# add formatter to ch and ch to mylogger
ch.setFormatter(formatter)
mylogger.addHandler(ch)
