import unittest as ut
import numpy as np

import mass
import mass.materials


found_xraylib = "xraylib" in mass.materials.__dict__

def test_dict():
    if not found_xraylib:
        return
    expected_models = ("EBIT 2018", "Horton 2018", "RAVEN1 2019")
    for k in expected_models:
        assert k in mass.materials.filterstack_models
    for k, model in mass.materials.filterstack_models.items():
        assert isinstance(model, mass.materials.FilterStack), f"mass.filterstack_models['{k}'] not a mass.FilterStack"

def test_filter():
    """Make sure we can compute composite filter QE and that results are reasonable."""
    if not found_xraylib:
        return
    m = mass.materials.filterstack_models["Horton 2018"]
    e = np.linspace(4000, 9000, 6)
    qe = m(e)
    assert np.any(qe > 0.21), "Horton filter model QE < 21% everywhere"
    assert np.all(qe < 0.25), "Horton filter model QE > 25% somewhere"


