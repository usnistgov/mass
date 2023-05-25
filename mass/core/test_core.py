import h5py
import numpy as np
import os
import os.path
import shutil
import tempfile
import unittest as ut
import pytest

import mass
from mass.core.ljh_modify import LJHFile, ljh_copy_traces, ljh_append_traces, ljh_truncate
import mass.off

import logging
LOG = logging.getLogger("mass")


class TestFiles(ut.TestCase):

    def test_ljh_copy_and_append_traces(self):
        """Test copying and appending traces to LJH files."""
        src_name = os.path.join('mass', 'regression_test', 'regress_chan1.ljh')
        src = LJHFile.open(src_name)
        with tempfile.NamedTemporaryFile(suffix="_chan1.ljh") as destfile:
            dest_name = destfile.name
            source_traces = [20]
            ljh_copy_traces(src_name, dest_name, source_traces, overwrite=True)
            dest = LJHFile.open(dest_name)
            for i, st in enumerate(source_traces):
                self.assertTrue(np.all(src.read_trace(st) == dest.read_trace(i)))

            source_traces = [0, 30, 20, 10]
            ljh_copy_traces(src_name, dest_name, source_traces, overwrite=True)
            dest = LJHFile.open(dest_name)
            for i, st in enumerate(source_traces):
                self.assertTrue(np.all(src.read_trace(st) == dest.read_trace(i)))

            source_traces.append(5)
            ljh_append_traces(src_name, dest_name, [5])
            dest = LJHFile.open(dest_name)
            for i, st in enumerate(source_traces):
                self.assertTrue(np.all(src.read_trace(st) == dest.read_trace(i)))

            new_traces = [15, 25, 3]
            source_traces.extend(new_traces)
            ljh_append_traces(src_name, dest_name, new_traces)
            dest = LJHFile.open(dest_name)
            for i, st in enumerate(source_traces):
                self.assertTrue(np.all(src.read_trace(st) == dest.read_trace(i)))

    def test_ljh_truncate_wrong_format(self):
        # First a file using LJH format 2.1.0 - should raise an exception
        src_name = os.path.join('mass', 'regression_test', 'regress_chan1.ljh')
        with tempfile.NamedTemporaryFile(suffix="_chan1.ljh") as destfile:
            dest_name = destfile.name

            def func():
                ljh_truncate(src_name, dest_name, n_pulses=100, segmentsize=2054*500)
            self.assertRaises(Exception, func)

    def run_test_ljh_truncate_timestamp(self, src_name, n_pulses_expected, timestamp, segmentsize):
        with tempfile.NamedTemporaryFile(suffix="_chan1.ljh") as destfile:
            dest_name = destfile.name
            ljh_truncate(src_name, dest_name, timestamp=timestamp, segmentsize=segmentsize)

            src = LJHFile.open(src_name)
            dest = LJHFile.open(dest_name)
            self.assertEqual(n_pulses_expected, dest.nPulses)
            for k in range(n_pulses_expected):
                self.assertTrue(np.all(src.read_trace(k) == dest.read_trace(k)))
                self.assertEqual(src.rowcount[k], dest.rowcount[k])
                self.assertAlmostEqual(src.datatimes_float[k], dest.datatimes_float[k], 5)

    def run_test_ljh_truncate_n_pulses(self, src_name, n_pulses, segmentsize):
        # Tests with a file with 1230 pulses, each 1016 bytes long
        with tempfile.NamedTemporaryFile(suffix="_chan1.ljh") as destfile:
            dest_name = destfile.name
            ljh_truncate(src_name, dest_name, n_pulses=n_pulses, segmentsize=segmentsize)

            src = LJHFile.open(src_name)
            dest = LJHFile.open(dest_name)
            self.assertEqual(n_pulses, dest.nPulses)
            for k in range(n_pulses):
                self.assertTrue(np.all(src.read_trace(k) == dest.read_trace(k)))
                self.assertEqual(src.rowcount[k], dest.rowcount[k])
                self.assertAlmostEqual(src.datatimes_float[k], dest.datatimes_float[k], 5)

    def test_ljh_truncate_n_pulses(self):
        # Want to make sure that we didn't screw something up with the
        # segmentation, so try various lengths
        # Tests with a file with 1230 pulses, each 1016 bytes long
        src_name = os.path.join('mass', 'regression_test', 'regress_chan3.ljh')
        self.run_test_ljh_truncate_n_pulses(src_name, 1000, None)
        self.run_test_ljh_truncate_n_pulses(src_name, 0, None)
        self.run_test_ljh_truncate_n_pulses(src_name, 1, None)
        self.run_test_ljh_truncate_n_pulses(src_name, 100, 1016*2000)
        self.run_test_ljh_truncate_n_pulses(src_name, 49, 1016*50)
        self.run_test_ljh_truncate_n_pulses(src_name, 50, 1016*50)
        self.run_test_ljh_truncate_n_pulses(src_name, 51, 1016*50)
        self.run_test_ljh_truncate_n_pulses(src_name, 75, 1016*50)
        self.run_test_ljh_truncate_n_pulses(src_name, 334, 1016*50)

    def test_ljh_truncate_timestamp(self):
        # Want to make sure that we didn't screw something up with the
        # segmentation, so try various lengths
        # Tests with a file with 1230 pulses, each 1016 bytes long
        src_name = os.path.join('mass', 'regression_test', 'regress_chan3.ljh')
        self.run_test_ljh_truncate_timestamp(src_name, 1000, 1510871067891481/1e6, None)
        self.run_test_ljh_truncate_timestamp(src_name,  100, 1510871020202899/1e6, 1016*2000)
        self.run_test_ljh_truncate_timestamp(src_name,   49, 1510871016889751/1e6, 1016*50)
        self.run_test_ljh_truncate_timestamp(src_name,   50, 1510871016919543/1e6, 1016*50)
        self.run_test_ljh_truncate_timestamp(src_name,   51, 1510871017096192/1e6, 1016*50)
        self.run_test_ljh_truncate_timestamp(src_name,   75, 1510871018591985/1e6, 1016*50)
        self.run_test_ljh_truncate_timestamp(src_name,  334, 1510871031629499/1e6, 1016*50)

    def test_ljh_dastard_other_reading(self):
        "Make sure we read DASTARD vs non-DASTARD LJH files correctly"
        src_name1 = os.path.join('mass', 'regression_test', 'regress_chan1.ljh')
        src_name2 = os.path.join('mass', 'regression_test', 'regress_dastard_chan1.ljh')
        data1 = mass.TESGroup(src_name1)
        data2 = mass.TESGroup(src_name2)
        for d in (data1, data2):
            d.summarize_data()
            d.read_segment(0)
        ds1 = data1.channel[1]
        ds2 = data2.channel[1]
        self.assertTrue(b"MATTER" in ds1.pulse_records.datafile.client)
        self.assertTrue(b"DASTARD" in ds2.pulse_records.datafile.client)
        self.assertEqual(int(ds1.pulse_records.datafile.header_dict[b"Presamples"]), 512)
        self.assertEqual(int(ds2.pulse_records.datafile.header_dict[b"Presamples"]), 515)
        self.assertEqual(515, ds1.nPresamples)
        self.assertEqual(515, ds2.nPresamples)
        v1 = ds1.data[0]
        v2 = ds2.data[0]
        self.assertTrue((v1 == v2).all())
        self.assertEqual(ds1.p_pretrig_mean[0], ds2.p_pretrig_mean[0])
        self.assertEqual(ds1.p_pretrig_rms[0], ds2.p_pretrig_rms[0])
        self.assertEqual(ds1.p_pulse_average[0], ds2.p_pulse_average[0])


class TestTESGroup(ut.TestCase):
    """Basic tests of the TESGroup object."""

    def __init__(self, methodName="runTest"):
        super().__init__(methodName=methodName)
        self.__files_to_clean_up__ = []

    def __del__(self):
        for f in self.__files_to_clean_up__:
            os.unlink(f)

    def clean_up_later(self, filename):
        self.__files_to_clean_up__.append(filename)

    def load_data(self, hdf5_filename=None, hdf5_noisefilename=None, skip_noise=False,
                  experimentStateFile=None):
        src_name = ['mass/regression_test/regress_chan1.ljh']
        noi_name = ['mass/regression_test/regress_noise_chan1.ljh']
        if skip_noise:
            noi_name = None
        if hdf5_filename is None:
            hdf5_file = tempfile.NamedTemporaryFile(suffix='_mass.hdf5', delete=False)
            hdf5_filename = hdf5_file.name
            self.clean_up_later(hdf5_filename)
        if hdf5_noisefilename is None:
            hdf5_noisefile = tempfile.NamedTemporaryFile(suffix='_mass_noise.hdf5', delete=False)
            hdf5_noisefilename = hdf5_noisefile.name
            self.clean_up_later(hdf5_noisefilename)
        return mass.TESGroup(src_name, noi_name, hdf5_filename=hdf5_filename,
                             hdf5_noisefilename=hdf5_noisefilename,
                             experimentStateFile=experimentStateFile)

    def test_experiment_state(self):
        # First test with the default experimentStateFile
        # It should have only the trivial START state, hence all 300 records
        # will pass ds.good(state="START")
        data = self.load_data()
        data.summarize_data()
        # The following fails until issue 225 is fixed.
        self.assertEqual(data.n_good_channels(), 1)
        ds = data.channel[1]
        self.assertEqual(len(ds.good(state="START")), 300)

        # Next test with an experimentStateFile that has a nontrivial state "funstate".
        # In this case, START should not be a valid state, but funstate will have Only
        # 252 of the 300 records valid because of their timestamps.
        esf = "mass/regression_test/regress_experiment_state_v2.txt"
        data = self.load_data(experimentStateFile=esf)
        data.summarize_data()
        self.assertEqual(data.n_good_channels(), 1)
        ds = data.channel[1]
        with self.assertRaises(ValueError):
            ds.good(state="START")
        nfun = np.sum(ds.good(state="funstate"))
        self.assertEqual(nfun, 252)

    def test_nonoise_data(self):
        """Test behavior of a TESGroup without noise data."""
        data = self.load_data(skip_noise=True)
        with self.assertRaises(AttributeError):
            data.channel[1].noise_records
        with self.assertRaises(Exception):
            data.compute_noise()
        data.summarize_data()
        data.avg_pulses_auto_masks()
        with self.assertRaises(Exception):
            data.compute_ats_filter()
        data.assume_white_noise()
        data.compute_ats_filter()

    def test_all_channels_bad(self):
        """Make sure it isn't an error to load a data set where all channels are marked bad"""
        data = self.load_data()
        data.set_chan_bad(1, "testing all channels bad")
        hdf5_filename = data.hdf5_file.filename
        hdf5_noisefilename = data.hdf5_noisefile.filename
        del data

        data2 = self.load_data(hdf5_filename=hdf5_filename, hdf5_noisefilename=hdf5_noisefilename)
        self.assertNotIn(1, data2.good_channels)
        data2.set_chan_good(1)
        LOG.info("Testing printing of a TESGroup")
        LOG.info(data2)

    def test_save_hdf5_calibration_storage(self):
        "calibrate a dataset, make sure it saves to hdf5"
        data = self.load_data()
        data.summarize_data()
        data.calibrate("p_pulse_rms", [10000.])
        data.calibrate("p_pulse_rms", [10000.], name_ext="abc")
        ds = data.first_good_dataset

        data2 = self.load_data(hdf5_filename=data.hdf5_file.filename,
                               hdf5_noisefilename=data.hdf5_noisefile.filename)
        ds2 = data2.first_good_dataset
        self.assertTrue(all([k in ds.calibration.keys() for k in ds2.calibration.keys()]))
        self.assertEqual(len(ds.calibration.keys()), 2)

        # These 2 checks test issue #102.
        self.assertIsNotNone(ds2.peak_samplenumber)
        self.assertEqual(ds2.peak_samplenumber, ds.peak_samplenumber)

    def test_make_auto_cuts(self):
        """Make sure that non-trivial auto-cuts are generated and reloadable from file."""
        data = self.load_data()
        ds = data.first_good_dataset
        data.summarize_data()
        data.auto_cuts(forceNew=True, clearCuts=True)
        ngood = ds.cuts.good().sum()
        self.assertLess(ngood, ds.nPulses)
        self.assertGreater(ngood, 0)

        data2 = self.load_data(hdf5_filename=data.hdf5_file.filename,
                               hdf5_noisefilename=data.hdf5_noisefile.filename)
        for ds in data2:
            self.assertGreater(ds.saved_auto_cuts.cuts_prm["postpeak_deriv"][1], 0.)
            self.assertGreater(ds.saved_auto_cuts.cuts_prm["pretrigger_rms"][1], 0.)

    def test_auto_cuts_after_others(self):
        """Make sure that non-trivial auto-cuts are generated even if other cuts are made first.
        Tests for issue 147 being fixed."""
        data = self.load_data()
        ds = data.first_good_dataset
        data.summarize_data()
        ds.clear_cuts()
        arbcut = np.zeros(ds.nPulses, dtype=bool)
        arbcut[::30] = True
        ds.cuts.cut("postpeak_deriv", arbcut)
        cuts = ds.auto_cuts(forceNew=False, clearCuts=False)
        self.assertIsNotNone(cuts, msg="auto_cuts not run after other cuts (issue 147)")
        ngood = ds.good().sum()
        self.assertLess(ngood, ds.nPulses-arbcut.sum())
        self.assertGreater(ngood, 0)

    def test_plot_filters(self):
        "Check that issue 105 is fixed: data.plot_filters() doesn't fail on 1 channel."
        data = self.load_data()
        data.set_chan_good(1)
        data.summarize_data()
        data.avg_pulses_auto_masks()
        with self.assertWarns(DeprecationWarning):
            data.compute_noise_spectra()
        data.compute_noise()
        data.compute_5lag_filter()  # not enough pulses for ats filters
        data.plot_filters()

    @pytest.mark.filterwarnings("ignore:divide by zero encountered")
    def test_time_drift_correct(self):
        "Check that time_drift_correct at least runs w/o error"
        data = self.load_data()
        data.summarize_data()
        data.auto_cuts(forceNew=True, clearCuts=True)
        data.avg_pulses_auto_masks()
        data.compute_noise()
        data.compute_filters()
        data.filter_data()
        data.drift_correct()
        data.phase_correct()
        data.time_drift_correct()

    def test_invert_data(self):
        data = self.load_data()
        ds = data.channel[1]
        _ = ds.read_segment(0)
        raw = ds.data
        rawinv = 0xffff - raw

        ds.clear_cache()
        ds.invert_data = True
        _ = ds.read_segment(0)
        raw2 = ds.data
        self.assertTrue(np.all(rawinv == raw2))

    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    def test_issue156(self):
        "Make sure phase_correct works when there are too few valid bins of pulse height"
        data = self.load_data()
        ds = data.channel[1]
        ds.clear_cuts()
        ds.p_filt_value_dc[:150] = np.linspace(1, 6000.0, 150)
        ds.p_filt_value_dc[150:] = 5898.8
        ds.p_filt_phase[:] = np.random.standard_normal(ds.nPulses)
        NBINS = 10
        for lowestbin in range(5, 10):
            data.set_chan_good(1)
            dc = ds.p_filt_value_dc[:]
            top = 6000.0
            bin = np.digitize(dc, np.linspace(0, top, 1+NBINS))-1
            ds.p_filt_value_dc[np.logical_or(bin >= NBINS, bin < lowestbin)] = 5898.8
            data.phase_correct(method2017=True, forceNew=True, save_to_hdf5=False)
            if ds.channum not in data.good_channels:
                raise ValueError("Failed issue156 test with %d valid bins (lowestbin=%d)" %
                                 (NBINS-lowestbin, lowestbin))

    def test_noncontinuous_noise(self):
        "Test for issue 157: failure when noise_is_continuous=False"
        src_name = 'mass/regression_test/regress_chan1.ljh'
        noi_name = 'mass/regression_test/regress_noise_chan1.ljh'
        data = mass.TESGroup(src_name, noi_name, noise_is_continuous=False)
        ds = data.channel[1]
        ds.compute_noise()

    def test_pulse_model_and_ljh2off(self):
        np.random.seed(0)
        data = self.load_data()
        data.compute_noise()
        data.summarize_data()
        data.auto_cuts()
        data.compute_ats_filter(shift1=False)
        data.filter_data()
        ds = data.datasets[0]
        n_basis = 5
        hdf5_filename = data.pulse_model_to_hdf5(replace_output=True, n_basis=n_basis)
        with tempfile.TemporaryDirectory() as output_dir:
            max_channels = 100
            n_ignore_presamples = 0
            ljh_filenames, off_filenames = mass.ljh2off.ljh2off_loop(
                ds.filename, hdf5_filename, output_dir, max_channels,
                n_ignore_presamples, require_experiment_state=False)
            off = mass.off.off.OffFile(off_filenames[0])
            self.assertTrue(np.allclose(off._mmap_with_coefs["coefs"][:, 2], ds.p_filt_value[:]))

            x, y = off.recordXY(0)

            with h5py.File(hdf5_filename, "r") as h5:
                group = h5["1"]
                pulse_model = mass.PulseModel.fromHDF5(group)
            self.assertEqual(pulse_model.projectors.shape, (n_basis, ds.nSamples))
            self.assertEqual(pulse_model.basis.shape, pulse_model.projectors.shape[::-1])
            mpc = pulse_model.projectors.dot(ds.read_trace(0))
            self.assertTrue(np.allclose(off._mmap_with_coefs["coefs"][0, :], mpc))

            should_be_identity = np.matmul(pulse_model.projectors, pulse_model.basis)
            wrongness = np.abs(should_be_identity-np.identity(n_basis))
            # ideally we could set this lower, like 1e-9, but the linear algebra needs more work
            self.assertTrue(np.amax(wrongness) < 4e-2)
            pulse_model.plot()

            # test multi_ljh2off_loop with multiple ljhfiles
            basename, channum = mass.ljh_util.ljh_basename_channum(ds.filename)
            N = len(off)
            prefix = os.path.split(basename)[1]
            offbase = f"{output_dir}/{prefix}"
            ljh_filename_lists, off_filenames_multi = mass.ljh2off.multi_ljh2off_loop(
                [basename]*2, hdf5_filename, offbase, max_channels,
                n_ignore_presamples, require_experiment_state=False
            )
            self.assertEqual(ds.filename, ljh_filename_lists[0][0])
            off_multi = mass.off.off.OffFile(off_filenames_multi[0])
            self.assertEqual(2*N, len(off_multi))
            self.assertEqual(off[7], off_multi[7])
            self.assertEqual(off[7], off_multi[N+7])
            self.assertNotEqual(off[7], off_multi[N+6])

    def test_projectors_script(self):
        import mass.core.projectors_script

        class Args():
            def __init__(self):
                self.pulse_path = os.path.join('mass', 'regression_test', 'regress_chan1.ljh')
                self.noise_path = os.path.join('mass', 'regression_test', 'regress_noise_chan1.ljh')
                self.output_path = os.path.join(
                    'mass', 'regression_test', 'projectors_script_test.hdf5')
                self.replace_output = True
                self.max_channels = 4
                self.n_ignore_presamples = 2
                self.n_sigma_pt_rms = 8
                self.n_sigma_max_deriv = 8
                self.n_basis = 5
                self.maximum_n_pulses = 4000
                self.silent = False
                self.mass_hdf5_path = os.path.join(
                    'mass', 'regression_test', 'projectors_script_test_mass.hdf5')
                self.mass_hdf5_noise_path = None
                self.invert_data = False
                self.dont_optimize_dp_dt = True
                self.extra_n_basis_5lag = 1
                self.noise_weight_basis = True
                self.f_3db_ats = None
                self.f_3db_5lag = None

        mass.core.projectors_script.main(Args())

    def test_expt_state_files(self):
        """Check that experiment state files are loaded and turned into categorical cuts
        with category "state" if the file exists."""
        def make_data(have_esf):
            src_name = 'mass/regression_test/regress_chan1.ljh'
            dir = tempfile.TemporaryDirectory()
            src_name = shutil.copy(src_name, dir.name)
            hdf5_filename = os.path.join(dir.name, "blah_mass.hdf5")
            if have_esf:
                contents = """# unix time in nanoseconds, state label
10476435385280, START
10476891776960, A
10491427707840, B
"""
                esfname = "{}/regress_experiment_state.txt".format(dir.name)
                with open(esfname, "w") as fp:
                    fp.write(contents)
            return mass.TESGroup([src_name], hdf5_filename=hdf5_filename), dir

        for have_esf in (False, True):
            data, dir = make_data(have_esf)
            # data.summarize_data()
            ds = data.channel[1]
            ds.good()
            if have_esf:
                ds.good(state="A")
                ds.good(state="B")
                ds.good(state="uncategorized")
                with self.assertRaises(ValueError):
                    ds.good(state="a state not listed in the file")
            else:
                with self.assertRaises(ValueError):
                    ds.good(state="A")
            dir.cleanup()


class TestTESHDF5Only(ut.TestCase):
    """Basic tests of the TESGroup object when we use the HDF5-only variant."""

    def test_basic_hdf5_only(self):
        """Make sure it mass can open a mass generated file in HDF5 Only mode."""
        src_name = 'mass/regression_test/regress_chan1.ljh'
        noi_name = 'mass/regression_test/regress_noise_chan1.ljh'
        hdf5_file = tempfile.NamedTemporaryFile(suffix='_mass.hdf5')
        hdf5_noisefile = tempfile.NamedTemporaryFile(suffix='_mass_noise.hdf5')
        mass.TESGroup([src_name], [noi_name], hdf5_filename=hdf5_file.name,
                      hdf5_noisefilename=hdf5_noisefile.name)

        data2 = mass.TESGroupHDF5(hdf5_file.name)
        LOG.info("Testing printing of a TESGroupHDF5")
        LOG.info(data2)

    def test_ordering_hdf5only(self):
        src_name = "mass/regression_test/regress_chan1.ljh"
        with tempfile.TemporaryDirectory() as dir:
            dest_name = "%s/temporary_chan%d.ljh"
            chan1_dest = dest_name % (dir, 1)
            shutil.copy(src_name, chan1_dest)
            cnums = (1, 3, 5, 11, 13, 15)
            for c in cnums[1:]:
                os.link(chan1_dest, dest_name % (dir, c))

            data1 = mass.TESGroup("%s/temporary_chan*.ljh" % dir)
            # Make sure the usual TESGroup is in the right order
            for i, ds in enumerate(data1):
                self.assertEqual(ds.channum, cnums[i])
            fname = data1.hdf5_file.filename
            del data1

            # Make sure the usual TESGroup is in the right order
            data = mass.TESGroupHDF5(fname)
            for i, ds in enumerate(data):
                self.assertEqual(ds.channum, cnums[i])


if __name__ == '__main__':
    ut.main()
