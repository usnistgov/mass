import h5py
import numpy as np
import os
import os.path
import shutil
import tempfile
import pytest

import logging

import mass
from mass.core.ljh_modify import LJHFile, ljh_copy_traces, ljh_append_traces, ljh_truncate
import mass.off
import mass.core.projectors_script

LOG = logging.getLogger("mass")

# ruff: noqa: PLR0914


class TestFiles:

    @staticmethod
    def test_ljh_copy_and_append_traces():
        """Test copying and appending traces to LJH files."""
        src_name = os.path.join('tests', 'regression_test', 'regress_chan1.ljh')
        src = LJHFile.open(src_name)
        with tempfile.NamedTemporaryFile(suffix="_chan1.ljh") as destfile:
            dest_name = destfile.name
            destfile.close()
            source_traces = [20]
            ljh_copy_traces(src_name, dest_name, source_traces, overwrite=True)
            dest = LJHFile.open(dest_name)
            for i, st in enumerate(source_traces):
                assert np.all(src.read_trace(st) == dest.read_trace(i))

            source_traces = [0, 30, 20, 10]
            ljh_copy_traces(src_name, dest_name, source_traces, overwrite=True)
            dest = LJHFile.open(dest_name)
            for i, st in enumerate(source_traces):
                assert np.all(src.read_trace(st) == dest.read_trace(i))

            source_traces.append(5)
            ljh_append_traces(src_name, dest_name, [5])
            dest = LJHFile.open(dest_name)
            for i, st in enumerate(source_traces):
                assert np.all(src.read_trace(st) == dest.read_trace(i))

            new_traces = [15, 25, 3]
            source_traces.extend(new_traces)
            ljh_append_traces(src_name, dest_name, new_traces)
            dest = LJHFile.open(dest_name)
            for i, st in enumerate(source_traces):
                assert np.all(src.read_trace(st) == dest.read_trace(i))

    @staticmethod
    def test_ljh_truncate_wrong_format():
        # First a file using LJH format 2.1.0 - should raise an exception
        src_name = os.path.join('tests', 'regression_test', 'regress_chan1.ljh')
        with tempfile.NamedTemporaryFile(suffix="_chan1.ljh") as destfile:
            dest_name = destfile.name
            destfile.close()

            def func():
                ljh_truncate(src_name, dest_name, n_pulses=100, segmentsize=2054 * 500)
            pytest.raises(Exception, func)

    @staticmethod
    def run_test_ljh_truncate_timestamp(src_name, n_pulses_expected, timestamp, segmentsize):
        with tempfile.NamedTemporaryFile(suffix="_chan1.ljh") as destfile:
            dest_name = destfile.name
            destfile.close()
            ljh_truncate(src_name, dest_name, timestamp=timestamp, segmentsize=segmentsize)

            src = LJHFile.open(src_name)
            dest = LJHFile.open(dest_name)
            assert n_pulses_expected == dest.nPulses
            for k in range(n_pulses_expected):
                assert np.all(src.read_trace(k) == dest.read_trace(k))
                assert src.subframecount[k] == dest.subframecount[k]
                assert src.datatimes_float[k] == pytest.approx(dest.datatimes_float[k], abs=1e-5)

    @staticmethod
    def run_test_ljh_truncate_n_pulses(src_name, n_pulses, segmentsize):
        # Tests with a file with 1230 pulses, each 1016 bytes long
        with tempfile.NamedTemporaryFile(suffix="_chan1.ljh") as destfile:
            dest_name = destfile.name
            destfile.close()
            ljh_truncate(src_name, dest_name, n_pulses=n_pulses, segmentsize=segmentsize)

            src = LJHFile.open(src_name)
            dest = LJHFile.open(dest_name)
            assert n_pulses == dest.nPulses
            for k in range(n_pulses):
                assert np.all(src.read_trace(k) == dest.read_trace(k))
                assert src.subframecount[k] == dest.subframecount[k]
                assert src.datatimes_float[k] == pytest.approx(dest.datatimes_float[k], abs=1e-5)

    def test_ljh_truncate_n_pulses(self):
        # Want to make sure that we didn't screw something up with the
        # segmentation, so try various lengths
        # Tests with a file with 1230 pulses, each 1016 bytes long
        src_name = os.path.join('tests', 'regression_test', 'regress_chan3.ljh')
        self.run_test_ljh_truncate_n_pulses(src_name, 1000, None)
        self.run_test_ljh_truncate_n_pulses(src_name, 0, None)
        self.run_test_ljh_truncate_n_pulses(src_name, 1, None)
        self.run_test_ljh_truncate_n_pulses(src_name, 100, 1016 * 2000)
        self.run_test_ljh_truncate_n_pulses(src_name, 49, 1016 * 50)
        self.run_test_ljh_truncate_n_pulses(src_name, 50, 1016 * 50)
        self.run_test_ljh_truncate_n_pulses(src_name, 51, 1016 * 50)
        self.run_test_ljh_truncate_n_pulses(src_name, 75, 1016 * 50)
        self.run_test_ljh_truncate_n_pulses(src_name, 334, 1016 * 50)

    def test_ljh_truncate_timestamp(self):
        # Want to make sure that we didn't screw something up with the
        # segmentation, so try various lengths
        # Tests with a file with 1230 pulses, each 1016 bytes long
        src_name = os.path.join('tests', 'regression_test', 'regress_chan3.ljh')
        self.run_test_ljh_truncate_timestamp(src_name, 1000, 1510871067891481 / 1e6, None)
        self.run_test_ljh_truncate_timestamp(src_name, 100, 1510871020202899 / 1e6, 1016 * 2000)
        self.run_test_ljh_truncate_timestamp(src_name, 49, 1510871016889751 / 1e6, 1016 * 50)
        self.run_test_ljh_truncate_timestamp(src_name, 50, 1510871016919543 / 1e6, 1016 * 50)
        self.run_test_ljh_truncate_timestamp(src_name, 51, 1510871017096192 / 1e6, 1016 * 50)
        self.run_test_ljh_truncate_timestamp(src_name, 75, 1510871018591985 / 1e6, 1016 * 50)
        self.run_test_ljh_truncate_timestamp(src_name, 334, 1510871031629499 / 1e6, 1016 * 50)

    @staticmethod
    def test_ljh_dastard_other_reading():
        "Make sure we read DASTARD vs non-DASTARD LJH files correctly"
        src_name1 = os.path.join('tests', 'regression_test', 'regress_chan1.ljh')
        src_name2 = os.path.join('tests', 'regression_test', 'regress_dastard_chan1.ljh')
        data1 = mass.TESGroup(src_name1)
        data2 = mass.TESGroup(src_name2)
        for d in (data1, data2):
            d.summarize_data()
        ds1 = data1.channel[1]
        ds2 = data2.channel[1]
        assert b"MATTER" in ds1.pulse_records.datafile.client
        assert b"DASTARD" in ds2.pulse_records.datafile.client
        assert int(ds1.pulse_records.datafile.header_dict[b"Presamples"]) == 512
        assert int(ds2.pulse_records.datafile.header_dict[b"Presamples"]) == 515
        assert 515 == ds1.nPresamples  # b/c LJHFile2_1 adds +3 to what's in the file
        assert 515 == ds2.nPresamples
        v1 = ds1.data[0]
        v2 = ds2.data[0]
        assert (v1 == v2).all()
        assert ds1.p_pretrig_mean[0] == ds2.p_pretrig_mean[0]
        assert ds1.p_pretrig_rms[0] == ds2.p_pretrig_rms[0]
        assert ds1.p_pulse_average[0] == ds2.p_pulse_average[0]

    @staticmethod
    def test_ragged_size_file():
        "Make sure we can open a file that was truncated during a pulse record."
        mass.LJHFile.open("tests/regression_test/phase_correct_test_data_4k_pulses_chan1.ljh")

    @staticmethod
    def test_peak_time_property():
        "Check that a peak during pretrigger is handled properly (issue 259)"
        # A clever trick to get pulses that peak during the pretrigger period: use noise records
        src_name1 = os.path.join('tests', 'regression_test', 'regress_noise_chan1.ljh')
        data = mass.TESGroup(src_name1)
        data.summarize_data()
        ds = data.channel[1]
        # Find all records where the peak is in the pretrigger period.
        peak_in_pretrig = ds.p_peak_index[:] < ds.nPresamples
        assert peak_in_pretrig.sum() > 0
        assert np.all(ds.p_peak_time[peak_in_pretrig] < 0)


def test_ljh_file_rownum():
    "Check for bug (issue 268) where LJH row number is read incorrectly."
    src_name = os.path.join('tests', 'ljh_files', '20230626', '0001', '20230626_run0001_chan4109.ljh')
    ljh = mass.LJHFile.open(src_name)
    assert ljh.row_number == 13
    assert ljh.col_number == 0
    assert ljh.number_of_rows == 33
    assert ljh.number_of_columns == 1


class TestTESGroup:
    """Basic tests of the TESGroup object."""

    @staticmethod
    def load_data(hdf5_filename=None, hdf5_noisefilename=None, skip_noise=False,
                  experimentStateFile=None, hdf5dir=None):
        if hdf5_filename is None or hdf5_noisefilename is None:
            assert hdf5dir is not None
        if hdf5dir is None:
            hdf5dir = tempfile.mkdtemp()
        src_name = ['tests/regression_test/regress_chan1.ljh']
        noi_name = ['tests/regression_test/regress_noise_chan1.ljh']
        if skip_noise:
            noi_name = None
        if hdf5_filename is None:
            hdf5_filename = hdf5dir / "data_mass.hdf5"
        if hdf5_noisefilename is None:
            hdf5_noisefilename = hdf5dir / "data_noise.hdf5"
        return mass.TESGroup(src_name, noi_name, hdf5_filename=hdf5_filename,
                             hdf5_noisefilename=hdf5_noisefilename,
                             experimentStateFile=experimentStateFile)

    def test_readonly_view(self, tmp_path):
        """Make sure summarize_data() runs with a readonly memory view and small 'segments'.
        Check both cython and non-cython."""
        data = self.load_data(hdf5dir=tmp_path)
        ds = data.channel[1]

        # Make segments be short enough that even this small test file contains > 1 of them.
        ds.pulse_records.set_segment_size(512 * 1024)
        assert ds.pulse_records.pulses_per_seg < ds.nPulses

        # Summarize with Cython
        ds.p_pretrig_mean[:] = 0.0
        ds.summarize_data(forceNew=True, use_cython=True)
        assert np.all(ds.p_pretrig_mean[:] > 0)
        assert np.all(ds.p_pulse_rms[:] > 0)
        results_cython = {k: ds.__dict__[k][:] for k in ds.__dict__ if k.startswith("p_")}

        # Summarize with pure Python
        ds.p_pretrig_mean[:] = 0.0
        ds.summarize_data(forceNew=True, use_cython=False)
        assert np.all(ds.p_pretrig_mean[:] > 0)
        assert np.all(ds.p_pulse_rms[:] > 0)
        results_python = {k: ds.__dict__[k][:] for k in ds.__dict__ if k.startswith("p_")}

        # Be sure the Cython and Python results are pretty close
        for k in results_cython:
            # print(k)
            # print(results_cython[k])
            # print(results_cython[k]/results_python[k])
            assert results_cython[k] == pytest.approx(results_python[k], rel=0.003)

    def test_experiment_state(self, tmp_path_factory):
        # First test with the default experimentStateFile
        # It should have only the trivial START state, hence all 300 records
        # will pass ds.good(state="START")
        data = self.load_data(hdf5dir=tmp_path_factory.mktemp("1"))
        data.summarize_data()
        # The following fails until issue 225 is fixed.
        assert data.n_good_channels() == 1
        ds = data.channel[1]
        assert len(ds.good(state="START")) == 300

        # Next test with an experimentStateFile that has a nontrivial state "funstate".
        # In this case, START should not be a valid state, but funstate will have Only
        # 252 of the 300 records valid because of their timestamps.
        esf = "tests/regression_test/regress_experiment_state_v2.txt"

        data = self.load_data(experimentStateFile=esf, hdf5dir=tmp_path_factory.mktemp("2"))
        data.summarize_data()
        assert data.n_good_channels() == 1
        ds = data.channel[1]
        with pytest.raises(ValueError):
            ds.good(state="START")
        nfun = np.sum(ds.good(state="funstate"))
        assert nfun == 252

    def test_nonoise_data(self, tmp_path):
        """Test behavior of a TESGroup without noise data."""
        data = self.load_data(skip_noise=True, hdf5dir=tmp_path)
        with pytest.raises(AttributeError):
            data.channel[1].noise_records
        with pytest.raises(Exception):
            data.compute_noise()
        data.summarize_data()
        data.avg_pulses_auto_masks()
        with pytest.raises(Exception):
            data.compute_ats_filter()
        data.assume_white_noise()
        data.compute_ats_filter()

    @staticmethod
    def test_noise_only():
        """Test behavior of a TESGroup without pulse data."""
        pattern = "tests/regression_test/regress_noise_*.ljh"
        data = mass.TESGroup(pattern, noise_only=True)
        data.compute_noise()

    def test_all_channels_bad(self, tmp_path):
        """Make sure it isn't an error to load a data set where all channels are marked bad"""
        data = self.load_data(hdf5dir=tmp_path)
        data.set_chan_bad(1, "testing all channels bad")
        hdf5_filename = data.hdf5_file.filename
        hdf5_noisefilename = data.hdf5_noisefile.filename
        del data

        data2 = self.load_data(hdf5_filename=hdf5_filename, hdf5_noisefilename=hdf5_noisefilename)
        assert 1 not in data2.good_channels
        data2.set_chan_good(1)
        LOG.info("Testing printing of a TESGroup")
        LOG.info(data2)

    def test_save_hdf5_calibration_storage(self, tmp_path):
        "calibrate a dataset, make sure it saves to hdf5"
        data = self.load_data(hdf5dir=tmp_path)
        data.summarize_data()
        data.calibrate("p_pulse_rms", [10000.])
        data.calibrate("p_pulse_rms", [10000.], name_ext="abc")
        ds = data.first_good_dataset

        data2 = self.load_data(hdf5_filename=data.hdf5_file.filename,
                               hdf5_noisefilename=data.hdf5_noisefile.filename)
        ds2 = data2.first_good_dataset
        assert all([k in ds.calibration.keys() for k in ds2.calibration.keys()])
        assert len(ds.calibration.keys()) == 2

        # These 2 checks test issue #102.
        assert ds2.peak_samplenumber is not None
        assert ds2.peak_samplenumber == ds.peak_samplenumber

    def test_make_auto_cuts(self, tmp_path):
        """Make sure that non-trivial auto-cuts are generated and reloadable from file."""
        data = self.load_data(hdf5dir=tmp_path)
        ds = data.first_good_dataset
        data.summarize_data()
        data.auto_cuts(forceNew=True, clearCuts=True)
        ngood = ds.cuts.good().sum()
        assert ngood < ds.nPulses
        assert ngood > 0

        data2 = self.load_data(hdf5_filename=data.hdf5_file.filename,
                               hdf5_noisefilename=data.hdf5_noisefile.filename)
        for ds in data2:
            assert ds.saved_auto_cuts.cuts_prm["postpeak_deriv"][1] > 0.
            assert ds.saved_auto_cuts.cuts_prm["pretrigger_rms"][1] > 0.

    def test_auto_cuts_after_others(self, tmp_path):
        """Make sure that non-trivial auto-cuts are generated even if other cuts are made first.
        Tests for issue 147 being fixed."""
        data = self.load_data(hdf5dir=tmp_path)
        ds = data.first_good_dataset
        data.summarize_data()
        ds.clear_cuts()
        arbcut = np.zeros(ds.nPulses, dtype=bool)
        arbcut[::30] = True
        ds.cuts.cut("postpeak_deriv", arbcut)
        cuts = ds.auto_cuts(forceNew=False, clearCuts=False)
        assert cuts is not None, "auto_cuts not run after other cuts (issue 147)"
        ngood = ds.good().sum()
        assert ngood < ds.nPulses - arbcut.sum()
        assert ngood > 0

    def test_plot_filters(self, tmp_path):
        "Check that issue 105 is fixed: data.plot_filters() doesn't fail on 1 channel."
        data = self.load_data(hdf5dir=tmp_path)
        data.set_chan_good(1)
        data.summarize_data()
        data.avg_pulses_auto_masks()
        with pytest.warns(DeprecationWarning):
            data.compute_noise_spectra()
        data.compute_noise()
        data.compute_5lag_filter()  # not enough pulses for ats filters
        data.plot_filters()

    @pytest.mark.filterwarnings("ignore:divide by zero encountered")
    def test_time_drift_correct(self, tmp_path):
        "Check that time_drift_correct at least runs w/o error"
        data = self.load_data(hdf5dir=tmp_path)
        data.summarize_data()
        data.auto_cuts(forceNew=True, clearCuts=True)
        data.avg_pulses_auto_masks()
        data.compute_noise()
        data.compute_5lag_filter()
        data.filter_data()
        data.drift_correct()
        data.phase_correct()
        data.time_drift_correct()

    @pytest.mark.xfail
    def test_invert_data(self, tmp_path):
        data = self.load_data(hdf5dir=tmp_path)
        ds = data.channel[1]
        raw = ds.data
        rawinv = 0xffff - raw

        ds.invert_data = True
        raw2 = ds.data
        assert np.all(rawinv == raw2)

    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    def test_issue156(self, tmp_path):
        "Make sure phase_correct works when there are too few valid bins of pulse height"
        data = self.load_data(hdf5dir=tmp_path)
        ds = data.channel[1]
        ds.clear_cuts()
        ds.p_filt_value_dc[:150] = np.linspace(1, 6000.0, 150)
        ds.p_filt_value_dc[150:] = 5898.8
        rng = np.random.default_rng(9823)
        ds.p_filt_phase[:] = rng.standard_normal(ds.nPulses)
        NBINS = 10
        for lowestbin in range(5, 10):
            data.set_chan_good(1)
            dc = ds.p_filt_value_dc[:]
            top = 6000.0
            bin = np.digitize(dc, np.linspace(0, top, 1 + NBINS)) - 1
            ds.p_filt_value_dc[np.logical_or(bin >= NBINS, bin < lowestbin)] = 5898.8
            data.phase_correct(method2017=True, forceNew=True, save_to_hdf5=False)
            if ds.channum not in data.good_channels:
                raise ValueError("Failed issue156 test with %d valid bins (lowestbin=%d)" %
                                 (NBINS - lowestbin, lowestbin))

    @staticmethod
    def test_noncontinuous_noise():
        "Test for issue 157: failure when noise_is_continuous=False"
        src_name = 'tests/regression_test/regress_chan1.ljh'
        noi_name = 'tests/regression_test/regress_noise_chan1.ljh'
        data = mass.TESGroup(src_name, noi_name, noise_is_continuous=False)
        ds = data.channel[1]
        ds.compute_noise()

    def test_pulse_model_and_ljh2off(self, tmp_path):
        data = self.load_data(hdf5dir=tmp_path)

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
            _ljh_filenames, off_filenames = mass.ljh2off.ljh2off_loop(
                ds.filename, hdf5_filename, output_dir, max_channels,
                n_ignore_presamples, require_experiment_state=False)
            off = mass.off.off.OffFile(off_filenames[0])
            assert np.allclose(off._mmap_with_coefs["coefs"][:, 2], ds.p_filt_value[:])

            _x, _y = off.recordXY(0)

            with h5py.File(hdf5_filename, "r") as h5:
                group = h5["1"]
                pulse_model = mass.PulseModel.fromHDF5(group)
            assert pulse_model.projectors.shape == (n_basis, ds.nSamples)
            assert pulse_model.basis.shape == pulse_model.projectors.shape[::-1]
            mpc = pulse_model.projectors.dot(ds.read_trace(0))
            assert np.allclose(off._mmap_with_coefs["coefs"][0, :], mpc)

            should_be_identity = np.matmul(pulse_model.projectors, pulse_model.basis)
            wrongness = np.abs(should_be_identity - np.identity(n_basis))
            # ideally we could set this lower, like 1e-9, but the linear algebra needs more work
            print(wrongness)
            print(np.amax(wrongness))
            assert np.amax(wrongness) < 0.16
            pulse_model.plot()

        with tempfile.TemporaryDirectory() as output_dir:
            # test multi_ljh2off_loop with multiple ljhfiles
            basename, _channum = mass.ljh_util.ljh_basename_channum(ds.filename)
            N = len(off)
            prefix = os.path.split(basename)[1]
            offbase = os.path.join(output_dir, prefix)
            ljh_filename_lists, off_filenames_multi = mass.ljh2off.multi_ljh2off_loop(
                [basename] * 2, hdf5_filename, offbase, max_channels,
                n_ignore_presamples)
            assert ds.filename == ljh_filename_lists[0][0]
            off_multi = mass.off.off.OffFile(off_filenames_multi[0])
            assert 2 * N == len(off_multi)
            assert off[7] == off_multi[7]
            assert off[7] == off_multi[N + 7]
            assert off[7] != off_multi[N + 6]

    def test_ljh_records_to_off(self, tmp_path):
        """Be sure ljh_records_to_off works with ljh files of 2 or more segments."""
        data = self.load_data(hdf5dir=tmp_path)
        data.compute_noise()
        data.summarize_data()
        data.auto_cuts()
        data.compute_ats_filter(shift1=False)
        data.filter_data()

        # Reduce the segment size, so we test that this works with LJH files having
        # 2 or more segments. Here choose 3 segments
        bsize = np.max([ds.pulse_records.datafile.binary_size for ds in data])
        segsize = (bsize + 3 * 4096) // 3
        segsize -= segsize % 4096

        ljhfile = LJHFile.open(data.channel[1].filename)
        ljhfile.set_segment_size(segsize)
        with tempfile.NamedTemporaryFile(suffix='_dummy.off') as f:
            n_ignore_presamples = 0
            nbasis = 4
            projectors = np.zeros((nbasis, data.nSamples), dtype=float)
            basis = projectors.T
            off_version = "0.3.0"
            dtype = mass.off.off.recordDtype(off_version, nbasis, descriptive_coefs_names=False)
            mass.ljh2off.ljh_records_to_off(ljhfile, f, projectors, basis, n_ignore_presamples, dtype)

    @staticmethod
    def test_projectors_script(tmp_path):

        class Args:
            def __init__(self):
                self.pulse_path = os.path.join('tests', 'regression_test', 'regress_chan1.ljh')
                self.noise_path = os.path.join('tests', 'regression_test', 'regress_noise_chan1.ljh')
                self.output_path = tmp_path / 'projectors_script_test.hdf5'
                self.replace_output = True
                self.max_channels = 4
                self.n_ignore_presamples = 2
                self.n_sigma_pt_rms = 8
                self.n_sigma_max_deriv = 8
                self.n_basis = 5
                self.maximum_n_pulses = 4000
                self.silent = False
                self.mass_hdf5_path = tmp_path / 'projectors_script_test_mass.hdf5'
                self.mass_hdf5_noise_path = tmp_path / 'projectors_script_test_mass_noise.hdf5'
                self.invert_data = False
                self.dont_optimize_dp_dt = True
                self.extra_n_basis_5lag = 1
                self.noise_weight_basis = True
                self.f_3db_ats = None
                self.f_3db_5lag = None

        mass.core.projectors_script.main(Args())

    @staticmethod
    def test_expt_state_files():
        """Check that experiment state files are loaded and turned into categorical cuts
        with category "state" if the file exists."""
        def make_data(have_esf, dirname):
            src_name = 'tests/regression_test/regress_chan1.ljh'
            src_name = shutil.copy(src_name, dirname)
            hdf5_filename = os.path.join(dirname, "blah_mass.hdf5")
            if have_esf:
                contents = """# unix time in nanoseconds, state label
10476435385280, START
10476891776960, A
10491427707840, B
"""
                esfname = f"{dirname}/regress_experiment_state.txt"
                with open(esfname, "w", encoding="utf-8") as fp:
                    fp.write(contents)
            return mass.TESGroup([src_name], hdf5_filename=hdf5_filename)

        for have_esf in (False, True):
            with tempfile.TemporaryDirectory() as dirname:
                data = make_data(have_esf, dirname)
                # data.summarize_data()
                ds = data.channel[1]
                ds.good()
                if have_esf:
                    ds.good(state="A")
                    ds.good(state="B")
                    ds.good(state="uncategorized")
                    with pytest.raises(ValueError):
                        ds.good(state="a state not listed in the file")
                else:
                    with pytest.raises(ValueError):
                        ds.good(state="A")


class TestTESHDF5Only:
    """Basic tests of the TESGroup object when we use the HDF5-only variant."""

    @staticmethod
    def test_basic_hdf5_only():
        """Make sure it mass can open a mass generated file in HDF5 Only mode."""
        src_name = 'tests/regression_test/regress_chan1.ljh'
        noi_name = 'tests/regression_test/regress_noise_chan1.ljh'
        hdf5_file = tempfile.NamedTemporaryFile(suffix='_mass.hdf5')
        hdf5_file.close()
        hdf5_noisefile = tempfile.NamedTemporaryFile(suffix='_mass_noise.hdf5')
        hdf5_noisefile.close()
        mass.TESGroup([src_name], [noi_name], hdf5_filename=hdf5_file.name,
                      hdf5_noisefilename=hdf5_noisefile.name)

        data2 = mass.TESGroupHDF5(hdf5_file.name)
        LOG.info("Testing printing of a TESGroupHDF5")
        LOG.info(data2)

    @staticmethod
    def test_ordering_hdf5only():
        src_name = "tests/regression_test/regress_chan1.ljh"
        with tempfile.TemporaryDirectory() as dirname:
            dest_name = "%s/temporary_chan%d.ljh"
            chan1_dest = dest_name % (dirname, 1)
            shutil.copy(src_name, chan1_dest)
            cnums = (1, 3, 5, 11, 13, 15)
            for c in cnums[1:]:
                os.link(chan1_dest, dest_name % (dirname, c))

            data1 = mass.TESGroup(f"{dirname}/temporary_chan*.ljh")
            # Make sure the usual TESGroup is in the right order
            for i, ds in enumerate(data1):
                assert ds.channum == cnums[i]
            fname = data1.hdf5_file.filename
            del data1

            # Make sure the usual TESGroup is in the right order
            data = mass.TESGroupHDF5(fname)
            for i, ds in enumerate(data):
                assert ds.channum == cnums[i]
