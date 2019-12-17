import tempfile
import os.path

import numpy as np
import os
import shutil
import unittest as ut

import mass
from mass.core.ljh_modify import LJHFile, ljh_copy_traces, ljh_append_traces, ljh_truncate
import mass.off

import logging
LOG = logging.getLogger("mass")


class TestFiles(ut.TestCase):

    def test_ljh_copy_and_append_traces(self):
        """Test copying and appending traces to LJH files."""
        src_name = os.path.join('src', 'mass', 'regression_test', 'regress_chan1.ljh')
        dest_name = os.path.join(tempfile.gettempdir(), 'foo_chan1.ljh')
        src = LJHFile(src_name)

        source_traces = [20]
        ljh_copy_traces(src_name, dest_name, source_traces, overwrite=True)
        dest = LJHFile(dest_name)
        for i, st in enumerate(source_traces):
            self.assertTrue(np.all(src.read_trace(st) == dest.read_trace(i)))

        source_traces = [0, 30, 20, 10]
        ljh_copy_traces(src_name, dest_name, source_traces, overwrite=True)
        dest = LJHFile(dest_name)
        for i, st in enumerate(source_traces):
            self.assertTrue(np.all(src.read_trace(st) == dest.read_trace(i)))

        source_traces.append(5)
        ljh_append_traces(src_name, dest_name, [5])
        dest = LJHFile(dest_name)
        for i, st in enumerate(source_traces):
            self.assertTrue(np.all(src.read_trace(st) == dest.read_trace(i)))

        new_traces = [15, 25, 3]
        source_traces.extend(new_traces)
        ljh_append_traces(src_name, dest_name, new_traces)
        dest = LJHFile(dest_name)
        for i, st in enumerate(source_traces):
            self.assertTrue(np.all(src.read_trace(st) == dest.read_trace(i)))

    def test_ljh_truncate_wrong_format(self):
        # First a file using LJH format 2.1.0 - should raise an exception
        src_name = os.path.join('src', 'mass', 'regression_test', 'regress_chan1.ljh')
        dest_name = os.path.join(tempfile.mkdtemp(), 'foo_chan1.ljh')

        def func():
            ljh_truncate(src_name, dest_name, n_pulses=100, segmentsize=2054*500)
        self.assertRaises(Exception, func)

    def run_test_ljh_truncate_timestamp(self, src_name, n_pulses_expected, timestamp, segmentsize):
        dest_name = os.path.join(tempfile.mkdtemp(), 'foo_chan3.ljh')
        ljh_truncate(src_name, dest_name, timestamp=timestamp, segmentsize=segmentsize)

        src = LJHFile(src_name)
        dest = LJHFile(dest_name)
        self.assertEquals(n_pulses_expected, dest.nPulses)
        for k in range(n_pulses_expected):
            self.assertTrue(np.all(src.read_trace(k) == dest.read_trace(k)))
            self.assertEqual(src.datatimes_float[k], dest.datatimes_float[k])
            self.assertEqual(src.rowcount[k], dest.rowcount[k])

    def run_test_ljh_truncate_n_pulses(self, src_name, n_pulses, segmentsize):
        # Tests with a file with 1230 pulses, each 1016 bytes long
        dest_name = os.path.join(tempfile.mkdtemp(), 'foo_chan3.ljh')
        ljh_truncate(src_name, dest_name, n_pulses=n_pulses, segmentsize=segmentsize)

        src = LJHFile(src_name)
        dest = LJHFile(dest_name)
        self.assertEqual(n_pulses, dest.nPulses)
        for k in range(n_pulses):
            self.assertTrue(np.all(src.read_trace(k) == dest.read_trace(k)))
            self.assertEqual(src.datatimes_float[k], dest.datatimes_float[k])
            self.assertEqual(src.rowcount[k], dest.rowcount[k])

    def test_ljh_truncate_n_pulses(self):
        # Want to make sure that we didn't screw something up with the
        # segmentation, so try various lengths
        # Tests with a file with 1230 pulses, each 1016 bytes long
        src_name = os.path.join('src', 'mass', 'regression_test', 'regress_chan3.ljh')
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
        src_name = os.path.join('src', 'mass', 'regression_test', 'regress_chan3.ljh')
        self.run_test_ljh_truncate_timestamp(src_name, 1000, 1510871067891481/1e6, None)
        self.run_test_ljh_truncate_timestamp(src_name,  100, 1510871020202899/1e6, 1016*2000)
        self.run_test_ljh_truncate_timestamp(src_name,   49, 1510871016889751/1e6, 1016*50)
        self.run_test_ljh_truncate_timestamp(src_name,   50, 1510871016919543/1e6, 1016*50)
        self.run_test_ljh_truncate_timestamp(src_name,   51, 1510871017096192/1e6, 1016*50)
        self.run_test_ljh_truncate_timestamp(src_name,   75, 1510871018591985/1e6, 1016*50)
        self.run_test_ljh_truncate_timestamp(src_name,  334, 1510871031629499/1e6, 1016*50)

    def test_ljh_dastard_other_reading(self):
        "Make sure we read DASTARD vs non-DASTARD LJH files correctly"
        src_name1 = os.path.join('src', 'mass', 'regression_test', 'regress_chan1.ljh')
        src_name2 = os.path.join('src', 'mass', 'regression_test', 'regress_dastard_chan1.ljh')
        data1 = mass.TESGroup(src_name1)
        data2 = mass.TESGroup(src_name2)
        for d in (data1, data2):
            d.summarize_data()
            d.read_segment(0)
        ds1 = data1.channel[1]
        ds2 = data2.channel[1]
        self.assertTrue("MATTER" in ds1.pulse_records.datafile.client)
        self.assertTrue("DASTARD" in ds2.pulse_records.datafile.client)
        self.assertTrue(b"Presamples: 512\r\n" in ds1.pulse_records.datafile.header_lines)
        self.assertTrue(b"Presamples: 515\n" in ds2.pulse_records.datafile.header_lines)
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

    def load_data(self, clear_hdf5=True):
        src_name = 'src/mass/regression_test/regress_chan1.ljh'
        noi_name = 'src/mass/regression_test/regress_chan1.noi'
        if clear_hdf5:
            for name in ['src/mass/regression_test/regress_mass.hdf5',
                         'src/mass/regression_test/regress_noise_mass.hdf5']:
                if os.path.isfile(name):
                    os.remove(name)
        return mass.TESGroup([src_name], [noi_name])

    def test_all_channels_bad(self):
        """Make sure it isn't an error to load a data set where all channels are marked bad"""
        data = self.load_data()
        data.set_chan_bad(1, "testing all channels bad")
        data.hdf5_file.close()
        data.hdf5_noisefile.close()
        del data

        try:
            data = self.load_data(clear_hdf5=False)
        except Exception:
            self.fail("Opening a file with all channels bad raises and Exception.")
        self.assertNotIn(1, data.good_channels)
        data.set_chan_good(1)
        LOG.info("Testing printing of a TESGroup")
        LOG.info(data)

    def test_save_hdf5_calibration_storage(self):
        "calibrate a dataset, make sure it saves to hdf5"
        data = self.load_data()
        data.summarize_data()
        data.calibrate("p_pulse_rms", [10000.])
        data.calibrate("p_pulse_rms", [10000.], name_ext="abc")
        ds = data.first_good_dataset

        data2 = self.load_data(clear_hdf5=False)
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

        data2 = self.load_data(clear_hdf5=False)
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
        arbcut = np.zeros(ds.nPulses, dtype=np.bool)
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
        data.compute_noise_spectra()
        data.compute_5lag_filter() # not enough pulses for ats filters
        data.plot_filters()

    def test_time_drift_correct(self):
        "Check that time_drift_correct at least runs w/o error"
        data = self.load_data()
        data.summarize_data()
        data.auto_cuts(forceNew=True, clearCuts=True)
        data.avg_pulses_auto_masks()
        data.compute_noise_spectra()
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
        src_name = 'src/mass/regression_test/regress_chan1.ljh'
        noi_name = 'src/mass/regression_test/regress_chan1.noi'
        data = mass.TESGroup(src_name, noi_name, noise_is_continuous=False)
        ds = data.channel[1]
        ds.compute_noise_spectra()

    def test_projectors_and_ljh2off(self):
        data = self.load_data()
        data.compute_noise_spectra()
        data.summarize_data()
        data.compute_ats_filter(shift1=False)
        data.filter_data()
        ds = data.datasets[0]
        n_basis = 5
        hdf5_filename = data.projectors_to_hdf5(replace_output=True, n_basis=n_basis)
        output_dir = tempfile.mkdtemp()
        max_channels = 100
        n_ignore_presamples = 0
        ljh_filenames, off_filenames = mass.ljh2off.ljh2off_loop(ds.filename, hdf5_filename, output_dir, max_channels, 
        n_ignore_presamples, require_experiment_state=False)
        off = mass.off.off.OffFile(off_filenames[0])
        self.assertTrue(np.allclose(off["coefs"][:, 2], ds.p_filt_value[:]))

        # x,y=off.recordXY(0)

        # with h5py.File(hdf5_filename,"r") as h5:
        #     projectors = h5["1/svdbasis/projectors"][()]
        #     basis = h5["1/svdbasis/basis"][()]
        # self.assertEqual(projectors.shape, (ds.nSamples, n_basis))
        # self.assertEqual(basis.shape, projectors.shape[::-1])
        # mpc = np.matmul(ds.read_trace(0), projectors)
        # self.assertTrue(np.allclose(off["coefs"][0, :], mpc))
        # import h5py
        # import pylab as plt
        # also need to remove matplotlib.use("svg") from runtests.py and run only this file to avoid lots of plots
        # are the projectors orthogonal? NO :(
        # print "projectors.T * projectors"
        # print np.matmul(projectors.T, projectors)
        # print "basis * basis.T"
        # print np.matmul(basis, basis.T)
        # print "basis*projectors"
        # print np.matmul(basis, projectors) # should this be the identity matrix? or just very close to it?

        # plt.figure()
        # plt.plot(basis.T)
        # plt.title("basis.T")
        # plt.legend(["mean", "deriv", "pulse", "svd1","svd2"])
        # plt.figure()
        # plt.plot(y, label="from off")
        # plt.plot(ds.read_trace(1), label="from ljh")
        # plt.legend()
        # plt.figure()
        # plt.plot(projectors)
        # plt.legend(["mean", "deriv", "pulse", "svd1","svd2"])
        # plt.title("projectors")

        # plt.show()
        # plt.pause(20)




class TestTESHDF5Only(ut.TestCase):
    """Basic tests of the TESGroup object when we use the HDF5-only variant."""

    def test_all_channels_bad(self):
        """Make sure it mass can open a mass generated file in HDF5 Only mode."""
        src_name = 'src/mass/regression_test/regress_chan1.ljh'
        noi_name = 'src/mass/regression_test/regress_chan1.noi'
        for name in ['src/mass/regression_test/regress_mass.hdf5',
                     'src/mass/regression_test/regress_noise_mass.hdf5']:
            if os.path.isfile(name):
                os.remove(name)
        data = mass.TESGroup([src_name], [noi_name])
        h5filename = data.hdf5_file.filename
        data.hdf5_file.close()
        data.hdf5_noisefile.close()
        del data

        data2 = mass.TESGroupHDF5(h5filename)
        LOG.info("Testing printing of a TESGroupHDF5")
        LOG.info(data2)

    def test_ordering_hdf5only(self):
        src_name = "src/mass/regression_test/regress_chan1.ljh"
        dir = tempfile.mkdtemp()
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
