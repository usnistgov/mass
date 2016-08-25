import unittest
import matplotlib
matplotlib.use("svg") #set to common backend so will run on semphora ci with fewer dependencies

import mass
import sys, imp
import os
import os.path as path
import re

# search mass and all subdirs for files matching "test_*.py"
module_dirs = set()
module_paths = set()
rootdir = os.path.dirname(os.path.realpath(__file__))
for dirpath, dirnames, filenames in os.walk(path.expanduser(rootdir)):
    for filename in filenames:
        if re.match("test_.+\.py\Z",filename) and not "build" in dirpath:
            module_dirs.add(dirpath)
            filepath = path.join(dirpath, filename)
            module_paths.add(filepath)
# add path to folders containing one or more matching files to path
for module_dir in module_dirs:
    sys.path.insert(0,module_dir)
# import modules from those  files
modules = []
for module_path in module_paths:
    module_name = path.splitext(path.split(module_path)[-1])[0]
    modules.append(imp.load_module(module_name,*imp.find_module(module_name)))
# print out the modules found
print os.path.realpath(__file__)
print "found the following modules to test:"
print modules

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
