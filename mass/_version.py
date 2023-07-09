"""
The mass versioning information, which can be parsed by setup.py (without
importing) and found by the user as mass.__version__

For more, see discussion at
http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package

Joe Fowler, NIST
"""

__version__ = '0.7.11'
__version_info__ = tuple([int(num) for num in __version__.split('.')])
