import numpy as np
from mass.off.util import logical_and

def test_logic_and_and_logic_or():
    a = np.array([True, True, True])
    b = np.array([True, True, False])
    c = np.array([True, False, True])
    assert all(logical_and(a,b) == b)
    assert all(logical_and(a,c) == c)
    assert all(logical_and(a,b,c) == np.array([True, False, False]))

