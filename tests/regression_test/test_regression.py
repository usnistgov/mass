import glob
import os
from os import path

import numpy as np
import numpy.testing as nt

import mass


ljhdir = os.path.dirname(os.path.realpath(__file__))


def process_file(prefix, cuts, do_filter=True):
    channels = (1,)
    pulse_files = [path.join(ljhdir, "%s_chan%d.ljh" % (prefix, c)) for c in channels]
    noise_files = [path.join(ljhdir, "%s_noise_chan%d.ljh" % (prefix, c)) for c in channels]
    pulse_files = path.join(ljhdir, "%s_chan*.ljh" % prefix)
    noise_files = path.join(ljhdir, "%s_noise_chan*.ljh" % prefix)

    # Start from clean slate by removing any hdf5 files
    for fl in glob.glob(path.join(ljhdir, "%s_mass.hdf5" % prefix)):
        os.remove(fl)
    for fl in glob.glob(path.join(ljhdir, "%s_noise_mass.hdf5" % prefix)):
        os.remove(fl)

    data = mass.TESGroup(pulse_files, noise_files)
    data.summarize_data(forceNew=True)

    for ds in data:
        ds.clear_cuts()
        ds.apply_cuts(cuts)

    data.compute_noise()
    data.avg_pulses_auto_masks()
    if do_filter:
        data.compute_ats_filter(f_3db=10000)
        data.summarize_filters(std_energy=600)
        data.filter_data()
        data.drift_correct(forceNew=True)
        data.filter_data(forceNew=True)

    return data


class TestSummaries:
    @classmethod
    def setup_class(cls):
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
        cls.d = np.load(path.join(ljhdir, "regress_ds0.npz"))

    @classmethod
    def teardown_class(cls):
        cls.data.hdf5_file.close()
        cls.data.hdf5_noisefile.close()

    def test_ext_trigger_reading(self):
        _ = self.data.external_trigger_subframe_count
        nt.assert_equal(self.data.subframe_divisions, 64)
        ds = self.data.datasets[0]
        nt.assert_equal(ds.subframe_divisions, 64)

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

        # test_cuts
        nt.assert_equal(self.data.datasets[0].good(), self.d['good'])
        nt.assert_equal(self.data.datasets[0].bad(), self.d['bad'])

        # test_post_filter
        nt.assert_allclose(self.data.datasets[0].p_filt_value, self.d['p_filt_value'], rtol=1e-6)
        nt.assert_allclose(self.data.datasets[0].p_filt_value_dc,
                           self.d['p_filt_value_dc'], rtol=1e-6)

        # test_peak_time:
        """Be sure that peak_time_microsec=89.0 comes out to the same answers"""
        ds = self.data.datasets[0]
        ppd1 = ds.p_postpeak_deriv[:]
        ds.summarize_data(forceNew=True, peak_time_microsec=89.0)
        ppd2 = ds.p_postpeak_deriv[:]
        ds.summarize_data(forceNew=True, peak_time_microsec=None)
        nt.assert_allclose(ppd1, ppd2)


class TestStoredFilters:
    """Make sure we can read filters stored by MASS v0.7.0"""

    def test_filters_in_old_hdf5_files(self):
        fname = f"{ljhdir}/regress_mass_v0_7_0.hdf5"
        # The following will error if cannot read pre-v0.7.1 filters.
        data = mass.TESGroupHDF5(fname, read_only=True)
        assert isinstance(data, mass.core.channel_group_hdf5_only.TESGroupHDF5)
