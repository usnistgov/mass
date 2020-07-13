import unittest as ut
import numpy as np

import mass


class TestFilterModels(ut.TestCase):
    def test_dict(self):
        expected_models = ("EBIT 2018", "Horton 2018", "RAVEN1 2019")
        for k in expected_models:
            self.assertTrue(k in mass.filterstack_models)
        for k, model in mass.filterstack_models.items():
            self.assertTrue(isinstance(model, mass.FilterStack),
                            msg="mass.filterstack_models['{}'] not a mass.FilterStack".format(k))

    def test_filter(self):
        """Make sure we can compute composite filter QE and that results are reasonable."""
        m = mass.filterstack_models["Horton 2018"]
        e = np.linspace(4000, 9000, 6)
        qe = m(e)
        self.assertTrue(np.any(qe > 0.21), msg="Horton filter model QE < 21% everywhere")
        self.assertTrue(~np.all(qe > 0.21), msg="Horton filter model QE > 21% everywhere")
        self.assertTrue(np.all(qe < 0.25), msg="Horton filter model QE > 25% somewhere")


if __name__ == '__main__':
    ut.main()
