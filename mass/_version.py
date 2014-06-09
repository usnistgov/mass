"""
The mass versioning information, which can be parsed by setup.py (without
importing) and found by the user as mass.__version__

For more, see discussion at
http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package

Joe Fowler, NIST
9 June 2014 
"""

__version__ = '0.3.0'
__version_info__ = tuple([ int(num) for num in __version__.split('.')])
