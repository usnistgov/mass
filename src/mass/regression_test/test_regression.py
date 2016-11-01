import numpy as np
import pylab as pl
import glob, os
import mass
import unittest as ut
import numpy.testing as nt
from os import path

ljhdir = os.path.dirname(os.path.realpath(__file__))

def process_file(prefix, cuts, do_filter=True):

    channels=(1,)
    pulse_files=[path.join(ljhdir,"%s_chan%d.ljh"%(prefix, c)) for c in channels]
    noise_files=[path.join(ljhdir,"%s_chan%d.noi"%(prefix, c)) for c in channels]
    pulse_files = path.join(ljhdir,"%s_chan*.ljh"%prefix)
    noise_files = path.join(ljhdir,"%s_chan*.noi"%prefix)

    # Start from clean slate by removing any hdf5 files
    for fl in glob.glob(path.join(ljhdir,"%s_mass.hdf5" % prefix)):
        os.remove(fl)
    for fl in glob.glob(path.join(ljhdir,"%s_noise_mass.hdf5" % prefix)):
        os.remove(fl)

    data = mass.TESGroup(pulse_files, noise_files)
    data.summarize_data(peak_time_microsec=600.0, forceNew=True)

    for ds in data:
        ds.clear_cuts()
        ds.apply_cuts(cuts)

    data.compute_noise_spectra()
    data.avg_pulses_auto_masks()
    if do_filter:
        data.compute_filters(f_3db=10000)
        data.summarize_filters(std_energy=600)
        data.filter_data()
        data.drift_correct(forceNew=True)

        data.filter_data(forceNew=True, use_cython=True)

    return data


class TestSummaries(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        cuts = mass.core.controller.AnalysisControl(
            pulse_average=(0.0, None),
            pretrigger_rms=(None, 70),
            pretrigger_mean_departure_from_median=(-50, 50),
            peak_value=(0.0, None),
            postpeak_deriv=(0, 30),
            rise_time_ms=(None, 0.2),
            peak_time_ms=(None, 0.2)
        )
        cls.data = process_file("regress", cuts)
        cls.d = np.load(path.join(ljhdir,"regress_ds0.npz"))

    def test_summaries(self):
        nt.assert_allclose(self.data.datasets[0].p_peak_index, self.d['p_peak_index'])
        nt.assert_allclose(self.data.datasets[0].p_peak_time, self.d['p_peak_time'])
        nt.assert_allclose(self.data.datasets[0].p_peak_value, self.d['p_peak_value'])
        nt.assert_allclose(self.data.datasets[0].p_postpeak_deriv, self.d['p_postpeak_deriv'])
        nt.assert_allclose(self.data.datasets[0].p_pretrig_mean, self.d['p_pretrig_mean'])
        nt.assert_allclose(self.data.datasets[0].p_pretrig_rms, self.d['p_pretrig_rms'])
        nt.assert_allclose(self.data.datasets[0].p_pulse_average, self.d['p_pulse_average'])
        nt.assert_allclose(self.data.datasets[0].p_pulse_rms, self.d['p_pulse_rms'])
        nt.assert_allclose(self.data.datasets[0].p_rise_time, self.d['p_rise_time'])

    def test_cuts(self):
        nt.assert_equal(self.data.datasets[0].good(), self.d['good'])
        nt.assert_equal(self.data.datasets[0].bad(), self.d['bad'])

    def test_post_filter(self):
        nt.assert_allclose(self.data.datasets[0].p_filt_value, self.d['p_filt_value'])
        nt.assert_allclose(self.data.datasets[0].p_filt_value_dc, self.d['p_filt_value_dc'])


class TestFilters(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        cuts = mass.core.controller.AnalysisControl(
            pulse_average=(0.0, None),
            pretrigger_rms=(None, 70),
            pretrigger_mean_departure_from_median=(-50, 50),
            peak_value=(0.0, None),
            postpeak_deriv=(0, 30),
            rise_time_ms=(None, 0.2),
            peak_time_ms=(None, 0.2)
        )
        cls.data = process_file("regress", cuts, do_filter=False)

    def filter_summaries(self, newstyle):
        """Make sure that filters either old-style or new-style have a predicted resolution,
        whether the filters are created fresh or are loaded from HDF5."""
        for ds in self.data:
            ds._use_new_filters = newstyle
        self.data.compute_filters(f_3db=10000, forceNew=True)
        for ds in self.data:
            f = ds.filter
            self.assertIn("noconst", f.variances)
            self.assertIn("noconst", f.predicted_v_over_dv)
            self.assertAlmostEqual(f.variances["noconst"], 8.8e-7, delta=3e-8)
            expected = 449.4 if newstyle else 456.7
            self.assertAlmostEqual(f.predicted_v_over_dv["noconst"], expected, delta=0.1)

    def test_vdv_oldfilters(self):
        self.filter_summaries(False)

    def test_vdv_newfilters(self):
        self.filter_summaries(True)

if __name__ == '__main__':
    ut.main()
