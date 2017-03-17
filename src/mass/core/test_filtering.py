import numpy as np
import pylab as pl
import glob, os
import mass
import unittest as ut
import mass.core.channel_group as mcg
from mass.core.files import *

import numpy as np
import pylab as pl
import glob, os
import mass
import unittest as ut
import numpy.testing as nt

ljhdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","regression_test")

def process_file(prefix, cuts, do_filter=True):

    channels=(1,)
    pulse_files=[os.path.join(ljhdir,"%s_chan%d.ljh"%(prefix, c)) for c in channels]
    noise_files=[os.path.join(ljhdir,"%s_chan%d.noi"%(prefix, c)) for c in channels]
    pulse_files = os.path.join(ljhdir,"%s_chan*.ljh"%prefix)
    noise_files = os.path.join(ljhdir,"%s_chan*.noi"%prefix)

    # Start from clean slate by removing any hdf5 files
    for fl in glob.glob(os.path.join(ljhdir,"%s_mass.hdf5" % prefix)):
        os.remove(fl)
    for fl in glob.glob(os.path.join(ljhdir,"%s_noise_mass.hdf5" % prefix)):
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


class TestFilters(ut.TestCase):
    def setUp(self):
        cuts = mass.core.controller.AnalysisControl(
            pulse_average=(0.0, None),
            pretrigger_rms=(None, 70),
            pretrigger_mean_departure_from_median=(-50, 50),
            peak_value=(0.0, None),
            postpeak_deriv=(0, 30),
            rise_time_ms=(None, 0.2),
            peak_time_ms=(None, 0.2)
        )
        self.data = process_file("regress", cuts, do_filter=False)

    def tearDown(self):
        self.data.hdf5_file.close()
        self.data.hdf5_noisefile.close()

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
        """Make sure old filters have a v/dv"""
        self.filter_summaries(newstyle=False)

    def test_vdv_newfilters(self):
        """Make sure new filters have a v/dv"""
        self.filter_summaries(newstyle=True)


    def filter_reload(self, newstyle):
        self.filter_summaries(newstyle=newstyle)
        ds = self.data.channel[1]
        self.assertEqual(newstyle, ds._use_new_filters)
        filter1 = ds.filter

        pf = ds.filename
        nf = ds.noise_records.filename
        data2 = mass.TESGroup(pf, nf)
        ds = data2.channel[1]
        filter2 = ds.filter
        self.assertEqual(type(filter1), type(filter2))
        self.assertEqual(newstyle, ds._use_new_filters)
        if newstyle:
            for ds in self.data:
                self.assertIsNotNone(ds.filter.filt_aterms)
        data2.hdf5_file.close()
        data2.hdf5_noisefile.close()

    def test_filter_reload_new(self):
        """Make sure we can create new filters and reload them"""
        self.filter_reload(True)

    def test_filter_reload_old(self):
        """Make sure we can create old filters and reload them"""
        self.filter_reload(False)

    def test_filter_notmanypulses(self):
        """Be sure we can filter only a small # of pulses. See issue #87"""

        # Temporarily cut all pulses but the first 40. Try to build a filter.
        self.data.register_boolean_cut_fields("temporary")
        ds = self.data.channel[1]
        c = np.ones(ds.nPulses, dtype=np.bool)
        c[:40] = False
        ds.cuts.cut("temporary", c)
        ds.compute_newfilter(f_3db=5000)
        f = ds.filter.filt_noconst
        self.assertFalse(np.any(np.isnan(f)))

        # Now un-do the temporary cut and re-build the filter
        c = np.zeros(ds.nPulses, dtype=np.bool)
        ds.cuts.cut("temporary", c)
        ds.compute_newfilter(f_3db=5000)
        self.assertFalse(np.any(np.isnan(f)))

if __name__ == '__main__':
    ut.main()
