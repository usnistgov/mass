## \mainpage Overview of Mass
#
# \section summary Summary
#
# This summarizes some things.
# 
# \section install Installation
# 
# \section starting Getting started
# 
# Here is how we get started.
# 
# \section requirements Requirements
# 
# There are some requirements.


## \package  mass
#
# \brief Microcalorimeter Analysis Software Suite
# 
# Python tools to analyze microcalorimeter data offline.

"""
Mass: a Microcalorimeter Analysis Software Suite

Python tools to analyze microcalorimeter data offline.

For a demonstration of some capabilities:
>>> import mass.demo
>>> print mass.demo.helptxt # and then follow the directions


Joe Fowler, NIST Boulder Labs.  November 2010--
"""

from _version import __version__, __version_info__
from core import *


def reload_all():
    """Mass is constantly under development.  If you want to reload every module in the package
    hierarchy, then do a mass.reload().
    
    WARNING: your TESGroup or CDMGroup will need to be fixed via: data=data.copy(), or else its
    methods will still be the methods of the old code.
    """
    print("We are reloading MASS.")

    import imp
    import os
    import pkgutil

    # Use pkgutil to walk the package tree, but then reverse order to go depth-first.  
    modnames = [name for _importer, name, _ispkg in pkgutil.walk_packages(__path__, "mass.")]
    modnames.reverse()
    
    for modname in modnames:
        if modname.endswith("demo"):
            continue    
        print("Reloading %s..." % modname)
        module_path = "/".join(modname.split(".")[1:-1])
        module_path = os.path.join(__path__[0], module_path)
        try:
            x, y, z = imp.find_module(modname.split(".")[-1], [module_path])
            imp.load_module(modname, x, y, z)
        except Exception as ex:
            print("Error on reloading", modname)
            print(ex)
