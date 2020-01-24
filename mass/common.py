"""
common.py

Stand-along functions that can be used throughout MASS.
"""

import six


def isstr(s):
    """Is s a string or a bytes type? Make python 2 and 3 compatible."""
    return isinstance(s, (six.string_types, bytes))
