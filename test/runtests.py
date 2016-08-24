import unittest
import matplotlib
matplotlib.use("svg") #set to common backend so will run on semphora ci with fewer dependencies

import mass
import sys, imp
from os import path

# add paths to folders that contain tests
# add new folders containing test files here
sys.path.insert(0,path.expanduser("~/mass/mass/calibration"))

# import test files as modules
# add new test filenames here
module_names = ["test_fits", "test_calibration", "test_algorithms"]
modules = [imp.load_module(name,*imp.find_module(name)) for name in module_names]

# load up all tests into a suite
suite = unittest.TestSuite()
for module in modules:
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))

runner=unittest.TextTestRunner()
result = runner.run(suite)

# zero exit code means success, all others mean failure
if len(result.errors)>0:
    sys.exit(1)
else:
    sys.exit(0)
