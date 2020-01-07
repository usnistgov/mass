import unittest as ut


import logging
LOG = logging.getLogger("mass")


class TestDemos(ut.TestCase):
    def test_fitting_demo(self):
        import fitting_demo
        self.assertIsNotNone(fitting_demo)

    def test_fitting_fluorescence_demo(self):
        import fitting_fluorescence
        self.assertIsNotNone(fitting_fluorescence)

    # fails with NameError: global name 'get_ipython' is not defined
    # def test_intro_demo(self):
    #     import intro
    #
    # # fails to import soureroot
    # def test_cuts_demo(self):
    #     import cuts

    # commented out because make_reference_microcal_files_available()
    # doesnt exist
    # def test_full_analysis_example(self):
    #     make_reference_microcal_files_available()
    #     import full_analysis_example


if __name__ == '__main__':
    ut.main()
