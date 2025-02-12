import h5py
import numpy as np
import glob
import os
import pytest

import mass


ljhdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "regression_test")


def process_file(prefix, cuts, do_filter=True):
    """Returns a TESGroup given the file prefix; preloads a set of cuts."""
    pulse_files = os.path.join(ljhdir, f"{prefix}_chan*.ljh")
    noise_files = os.path.join(ljhdir, f"{prefix}_noise_chan*.ljh")

    # Start from clean slate by removing any hdf5 files
    for fl in glob.glob(os.path.join(ljhdir, f"{prefix}_mass.hdf5")):
        os.remove(fl)
    for fl in glob.glob(os.path.join(ljhdir, f"{prefix}_noise_mass.hdf5")):
        os.remove(fl)

    data = mass.TESGroup(pulse_files, noise_files)
    data.summarize_data(forceNew=True)

    for ds in data:
        ds.clear_cuts()
        ds.apply_cuts(cuts)

    data.compute_noise()
    data.avg_pulses_auto_masks()
    if do_filter:
        data.compute_5lag_filter(f_3db=10000)
        data.summarize_filters(std_energy=600)
        data.filter_data()
        data.drift_correct(forceNew=True)
        data.filter_data(forceNew=True, use_cython=True)

    return data


class TestFilters:
    """Test optimal filtering."""

    def setup_method(self):
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

    def teardown_method(self):
        self.data.hdf5_file.close()
        self.data.hdf5_noisefile.close()

    def filter_summaries(self, filter_type):
        """Make sure that filters either old-style or new-style have a predicted resolution,
        whether the filters are created fresh or are loaded from HDF5."""
        if filter_type == "5lag":
            self.data.compute_5lag_filter(f_3db=10000, forceNew=True)
        elif filter_type == "ats":
            self.data.compute_ats_filter(f_3db=10000, forceNew=True)
        else:
            raise ValueError(f"filter_type='{filter_type}', should be 'ats' or '5lag'")
        self.verify_filters(self.data, filter_type)

    @staticmethod
    def verify_filters(data, filter_type):
        "Check that the filters contain what we expect"
        expected_vdv = {"ats": 490, "5lag": 457}[filter_type]
        expected_var = {"ats": 135.7, "5lag": 155.1}[filter_type]
        for ds in data:
            f = ds.filter
            assert f.variance == pytest.approx(expected_var, abs=0.2)
            assert f.predicted_v_over_dv == pytest.approx(expected_vdv, abs=0.5)

    def test_vdv_5lag_filters(self):
        """Make sure old filters have a v/dv"""
        self.filter_summaries(filter_type="5lag")

    def test_vdv_ats_filters(self):
        """Make sure new filters have a v/dv"""
        self.filter_summaries(filter_type="ats")

    def filter_reload(self, filter_type):
        """Make sure filters can be reloaded, whether new or old-style."""
        self.filter_summaries(filter_type=filter_type)
        ds = self.data.channel[1]
        assert filter_type == ds._filter_type
        filter1 = ds.filter

        pf = ds.filename
        nf = ds.noise_records.filename
        data2 = mass.TESGroup(pf, nf)
        self.verify_filters(data2, filter_type)
        if filter_type == "5lag":
            data2.compute_5lag_filter(f_3db=10000, forceNew=True)
        elif filter_type == "ats":
            data2.compute_ats_filter(f_3db=10000, forceNew=True)
        else:
            raise ValueError(f"filter_type='{filter_type}', should be 'ats' or '5lag'")
        self.verify_filters(data2, filter_type)
        ds = data2.channel[1]
        filter2 = ds.filter
        assert filter_type == ds._filter_type
        assert filter_type == filter2._filter_type
        assert type(filter1) is type(filter2)
        if filter_type == "ats":
            for ds in self.data:
                assert ds.filter.dt_values is not None
        data2.hdf5_file.close()
        data2.hdf5_noisefile.close()

    def test_filter_reload_new(self):
        """Make sure we can create new filters and reload them"""
        self.filter_reload(filter_type="ats")

    def test_filter_reload_old(self):
        """Make sure we can create old filters and reload them"""
        self.filter_reload(filter_type="5lag")

    def test_filter_notmanypulses(self):
        """Be sure we can filter only a small # of pulses. See issue #87"""

        # Temporarily cut all pulses but the first 40. Try to build a filter.
        self.data.register_boolean_cut_fields("temporary")
        ds = self.data.channel[1]
        c = np.ones(ds.nPulses, dtype=bool)
        c[:40] = False
        ds.cuts.cut("temporary", c)
        ds.compute_ats_filter(f_3db=5000)
        f = ds.filter.values
        assert not np.any(np.isnan(f))

        # Now un-do the temporary cut and re-build the filter
        c = np.zeros(ds.nPulses, dtype=bool)
        ds.cuts.cut("temporary", c)
        ds.compute_ats_filter(f_3db=5000)
        assert not np.any(np.isnan(f))

    def test_masked_filter(self):
        """Test that zero-weighting samples from the beginning and end works."""
        ds = self.data.channel[1]
        ds.compute_ats_filter(f_3db=5000)
        # Use a limited number of pulses, cutting first sample
        NPulses = 50
        d = np.array(ds.data[:NPulses, 1:])
        assert ds.filter.values is not None

        # Test that filters actually have zero weight where they are supposed to.
        PREMAX, POSTMAX = 50, 200
        for pre in [0, PREMAX // 2, PREMAX]:
            for post in [0, POSTMAX // 2, POSTMAX]:
                ds.compute_ats_filter(f_3db=5000, cut_pre=pre, cut_post=post, forceNew=True)
                f = ds.filter.values
                d2 = np.array(d)
                if pre > 0:
                    assert np.all(f[:pre] == 0)
                    d2[:, :pre] = np.random.default_rng().standard_normal((NPulses, pre))
                if post > 0:
                    assert np.all(f[-post:] == 0)
                    d2[:, -post:] = np.random.default_rng().standard_normal((NPulses, post))
                resultsA = np.dot(d, f)
                resultsB = np.dot(d2, f)
                assert np.allclose(resultsA, resultsB)

        # Test that filters are the same whether made from short or long pulse models,
        # at least after they are forced to be the same size.
        N, n_pre = ds.nSamples, ds.nPresamples
        dt = ds.timebase

        pulse = np.zeros((N, 2), dtype=float)
        pulse[:, 0] = ds.average_pulse[:]
        pulse[1:, 1] = np.diff(ds.average_pulse)
        noise = np.exp(-np.arange(N) * .01)

        for cut_pre in (0, n_pre // 10, n_pre // 4):
            for cut_post in (0, (N - n_pre) // 10, (N - n_pre) // 4):
                thispulse = pulse[cut_pre:N - cut_post]
                filterS = ATSF(thispulse, n_pre - cut_pre, noise, sample_time_sec=dt)
                fS = filterS.values

                filterL = ATSF(pulse, n_pre, noise, sample_time_sec=dt,
                               cut_pre=cut_pre, cut_post=cut_post)
                fL = filterL.values[cut_pre:N - cut_post]
                assert np.allclose(fS, fL)


def ATSF(pulse, npre, noise, sample_time_sec, peak=0.0, f_3db=None, cut_pre=0, cut_post=0):
    assert pulse.shape[1] == 2
    maker = mass.FilterMaker(pulse[:, 0], npre, noise, dt_model=pulse[:, 1],
                             sample_time_sec=sample_time_sec, peak=peak)
    return maker.compute_ats(f_3db=f_3db, cut_pre=cut_pre, cut_post=cut_post)


@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_dc_insensitive():
    """When f_3db or fmax applied, filter should not become DC-sensitive.
    Tests for issue #176."""
    nSamples = 100
    nPresamples = 50
    nPost = nSamples - nPresamples

    # Some fake data, fake noise, and a fake noise spectrum
    pulse_like = np.append(np.zeros(nPresamples), np.linspace(nPost - 1, 0, nPost))
    deriv_like = np.append(np.zeros(nPresamples), -np.ones(nPost))

    fake_noise = np.random.default_rng().standard_normal(nSamples)
    fake_noise[0] = 10.0
    dt = 6.72e-6

    nPSD = 1 + (nSamples // 2)
    fPSD = np.linspace(0, 0.5, nPSD)
    PSD = 1 + 10 / (1 + (fPSD / 0.1)**2)

    maker_no_psd = mass.FilterMaker(
        pulse_like, nPresamples, fake_noise, dt_model=deriv_like,
        sample_time_sec=dt, peak=np.max(pulse_like))
    with pytest.raises(ValueError):
        maker_no_psd.compute_fourier()  # impossible with no PSD

    maker = mass.FilterMaker(
        pulse_like, nPresamples, fake_noise, dt_model=deriv_like, noise_psd=PSD,
        sample_time_sec=dt, peak=np.max(pulse_like))
    for computer in (maker.compute_ats, maker.compute_5lag, maker.compute_fourier):
        filter_to_test = computer(f_3db=None, fmax=None)
        std = np.median(np.abs(filter_to_test.values))
        mean = filter_to_test.values.mean()
        assert mean < 1e-10 * std, f"{filter_to_test} failed DC test w/o lowpass"

        filter_to_test = computer(f_3db=1e4, fmax=None)
        mean = filter_to_test.values.mean()
        assert mean < 1e-10 * std, f"{filter_to_test} failed DC test w/ f_3db"

        filter_to_test = computer(f_3db=None, fmax=1e4)
        mean = filter_to_test.values.mean()
        assert mean < 1e-10 * std, f"{filter_to_test} failed DC test w/ fmax"


def test_long_filter(tmp_path):
    """Be sure we can save and restore a long filter. See issue #208."""
    outfile = tmp_path / "test.hdf5"
    with h5py.File(outfile, "w") as h:
        g = h.require_group("blah")
        pulserec = {
            "nSamples": 20000,
            "nPresamples": 5000,
            "nPulses": 5,
            "timebase": 10e-6,
            "channum": 1,
            "timestamp_offset": 0,
            "subframe_divisions": 1,
            "subframe_offset": 0,
        }
        ds = mass.MicrocalDataSet(pulserec, hdf5_group=g)
        nc = np.zeros(20000, dtype=float)
        nc[0:3] = [1, .3, .1]
        ds.noise_autocorr = nc
        ds.average_pulse = np.ones(ds.nSamples, dtype=float)
        ds.average_pulse[:ds.nPresamples] = 0.0
        # ds.compute_ats_filter(f_3db=5000)
        aterms = np.zeros_like(ds.average_pulse)
        aterms[ds.nPresamples + 1] = 1.0
        model = np.vstack([ds.average_pulse, aterms]).T
        modelpeak = np.max(ds.average_pulse)

        f = ATSF(model, ds.nPresamples, ds.noise_autocorr,
                 sample_time_sec=ds.timebase, peak=modelpeak, f_3db=5000)
        ds.filter = f
        ds._filter_to_hdf5()


def test_no_concrete_baseFilter():
    "Make sure that mass.Filter is an abstract base class, can't be instantiated directly."
    with pytest.raises(TypeError):
        _ = mass.Filter(np.zeros(100), 100.0, 100.0, 100.0)


class TestWhitener:
    """Test ToeplitzWhitener."""

    @staticmethod
    def test_trivial():
        """Be sure that the trivial whitener does nothing."""
        w = mass.ToeplitzWhitener([1.0], [1.0])  # the trivial whitener
        r = np.random.default_rng().standard_normal(100)
        assert np.allclose(r, w(r))
        assert np.allclose(r, w.solveW(r))
        assert np.allclose(r, w.applyWT(r))
        assert np.allclose(r, w.solveWT(r))

    @staticmethod
    def test_reversible():
        """Use a nontrivial whitener, and make sure that inverse operations are inverses."""
        w = mass.ToeplitzWhitener([1.0, -1.7, 0.72], [1.0, .95])
        r = np.random.default_rng().standard_normal(100)
        assert np.allclose(r, w.solveW(w(r)))
        assert np.allclose(r, w(w.solveW(r)))
        assert np.allclose(r, w.solveWT(w.applyWT(r)))
        assert np.allclose(r, w.applyWT(w.solveWT(r)))

        # Also check that w isn't trivial
        assert not np.allclose(r, w(r))
        assert not np.allclose(r, w.solveW(r))
        assert not np.allclose(r, w.applyWT(r))
        assert not np.allclose(r, w.solveWT(r))

        # Check that no operations applied twice cancel out.
        assert not np.allclose(r, w(w(r)))
        assert not np.allclose(r, w.solveW(w.solveW(r)))
        assert not np.allclose(r, w.applyWT(w.applyWT(r)))
        assert not np.allclose(r, w.solveWT(w.solveWT(r)))

    @staticmethod
    def test_causal():
        """Make sure that the whitener and its inverse are causal,
        and that WT and its inverse anti-causal."""
        w = mass.ToeplitzWhitener([1.0, -1.7, 0.72], [1.0, .95])
        Nzero = 100
        z = np.zeros(Nzero, dtype=float)
        r = np.hstack([z, np.random.default_rng().standard_normal(100), z])

        # Applying and solving W are causal operations.
        wr = w(r)
        wir = w.solveW(r)
        assert np.all(r[:Nzero] == 0)
        assert np.all(wr[:Nzero] == 0)
        assert np.all(wir[:Nzero] == 0)
        assert not np.all(wr[Nzero:] == 0)
        assert not np.all(wir[Nzero:] == 0)

        # Applying and solving WT are anti-causal operations.
        wtr = w.applyWT(r)
        wtir = w.solveWT(r)
        assert np.all(r[-Nzero:] == 0)
        assert np.all(wtr[-Nzero:] == 0)
        assert np.all(wtir[-Nzero:] == 0)
        assert not np.all(wtr[:-Nzero] == 0)
        assert not np.all(wtir[:-Nzero] == 0)
