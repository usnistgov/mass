import uncertainties
from uncertainties import unumpy as unp
import numpy as np

def is_uncertain_scalar(x):
    return hasattr(x, "nominal_value")

def ensure_uncertain(x):
    """if give a scalar, returns a ufloat
    if given a numpy array of scalars, return a uarray
    if given a ufloat or uarray return it unchanged
    default uncertainty will be 100%, so people will know not to take it seriously until they've 
    put it in manually
    """
    if isinstance(x, np.ndarray):
        if is_uncertain_scalar(x[0]):
            return x
        else:
            return unp.uarray(x, x)
    elif isinstance(x, float):
        return uncertainties.ufloat(x, x)
    elif is_uncertain_scalar(x):
        return x
    else:
        raise Exception(f"{x} of type {type(x)} not supported")

def with_fractional_uncertainty(x, fractional_uncertainty):
    if isinstance(x, float):
        return uncertainties.ufloat(x, x*fractional_uncertainty)
    elif isinstance(x, np.ndarray):
        return unp.uarray(x, x*fractional_uncertainty)
    else:
        raise Exception(f"{x} of type {type(x)} not supported")