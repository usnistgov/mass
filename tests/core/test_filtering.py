import h5py
import numpy as np
import glob
import os
import tempfile
import unittest as ut
import pytest

import mass


ljhdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "regression_test")


def process_file(prefix, cuts, do_filter=True):
    """Returns a TESGroup given the file prefix; preloads a set of cuts."""
    channels = (1,)
    pulse_files = [os.path.join(ljhdir, "%s_chan%d.ljh" % (prefix, c)) for c in channels]
    noise_files = [os.path.join(ljhdir, "%s_noise_chan%d.ljh" % (prefix, c)) for c in channels]
    pulse_files = os.path.join(ljhdir, "%s_chan*.ljh" % prefix)
    noise_files = os.path.join(ljhdir, "%s_noise_chan*.ljh" % prefix)

    # Start from clean slate by removing any hdf5 files
    for fl in glob.glob(os.path.join(ljhdir, "%s_mass.hdf5" % prefix)):
        os.remove(fl)
    for fl in glob.glob(os.path.join(ljhdir, "%s_noise_mass.hdf5" % prefix)):
        os.remove(fl)

    data = mass.TESGroup(pulse_files, noise_files)
    data.summarize_data(forceNew=True)

    for ds in data:
        ds.clear_cuts()
        ds.apply_cuts(cuts)

    data.compute_noise()
    data.avg_pulses_auto_masks()
    if do_filter:
        data.compute_filters(f_3db=10000)
        data.summarize_filters(std_energy=600)
        data.filter_data()
        data.drift_correct(forceNew=True)

        data.filter_data(forceNew=True, use_cython=True)

    return data


class TestFilters(ut.TestCase):
    """Test optimal filtering."""

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

    def filter_summaries(self, filter_type):
        """Make sure that filters either old-style or new-style have a predicted resolution,
        whether the filters are created fresh or are loaded from HDF5."""
        self.data.compute_filters(f_3db=10000, forceNew=True, filter_type=filter_type)
        self.verify_filters(self.data, filter_type)

    def verify_filters(self, data, filter_type):
        "Check that the filters contain what we expect"
        expected = {"ats": 461.57, "5lag": 456.7}[filter_type]
        for ds in data:
            f = ds.filter
            self.assertIn("noconst", f.variances)
            self.assertIn("noconst", f.predicted_v_over_dv)
            self.assertAlmostEqual(f.variances["noconst"], 8.46e-7, delta=3e-8)
            self.assertAlmostEqual(f.predicted_v_over_dv["noconst"], expected, delta=0.1)

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
        self.assertEqual(filter_type, ds._filter_type)
        filter1 = ds.filter

        pf = ds.filename
        nf = ds.noise_records.filename
        data2 = mass.TESGroup(pf, nf)
        self.verify_filters(data2, filter_type)
        data2.compute_filters()
        self.verify_filters(data2, filter_type)
        ds = data2.channel[1]
        filter2 = ds.filter
        self.assertEqual(filter_type, ds._filter_type)
        self.assertEqual(type(filter1), type(filter2))
        if filter_type == "ats":
            for ds in self.data:
                self.assertIsNotNone(ds.filter.filt_aterms)
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
        f = ds.filter.filt_noconst
        self.assertFalse(np.any(np.isnan(f)))

        # Now un-do the temporary cut and re-build the filter
        c = np.zeros(ds.nPulses, dtype=bool)
        ds.cuts.cut("temporary", c)
        ds.compute_ats_filter(f_3db=5000)
        self.assertFalse(np.any(np.isnan(f)))

    def test_long_filter(self):
        """Be sure we can save and restore a long filter. See issue #208."""
        outfile = tempfile.TemporaryFile(suffix=".hdf5")
        with h5py.File(outfile, "w") as h:
            g = h.require_group("blah")
            pulserec = {
                "nSamples": 20000,
                "nPresamples": 5000,
                "nPulses": 5,
                "timebase": 10e-6,
                "channum": 1,
                "timestamp_offset": 0,
            }
            ds = mass.MicrocalDataSet(pulserec, hdf5_group=g)
            nc = np.zeros(20000, dtype=float)
            nc[0:3] = [1, .3, .1]
            ds.noise_autocorr = nc
            ds.average_pulse = np.ones(ds.nSamples, dtype=float)
            ds.average_pulse[:ds.nPresamples] = 0.0
            # ds.compute_ats_filter(f_3db=5000)
            aterms = np.zeros_like(ds.average_pulse)
            aterms[ds.nPresamples+1] = 1.0
            model = np.vstack([ds.average_pulse, aterms]).T
            modelpeak = np.max(ds.average_pulse)

            f = mass.core.optimal_filtering.ArrivalTimeSafeFilter(
                model, ds.nPresamples, ds.noise_autocorr,
                sample_time=ds.timebase, peak=modelpeak)
            f.compute(f_3db=5000)
            ds.filter = f
            ds._filter_type = "ats"
            ds._filter_to_hdf5()

    def test_masked_filter(self):
        """Test that zero-weighting samples from the beginning and end works."""
        ds = self.data.channel[1]
        ds.compute_ats_filter(f_3db=5000)
        NP = 50
        d = np.array(ds.data[:NP, 1:])  # NP pulses, cutting first sample
        self.assertIsNotNone(ds.filter.filt_noconst)

        # Test that filters actually have zero weight where they are supposed to.
        PREMAX, POSTMAX = 50, 200
        for pre in [0, PREMAX//2, PREMAX]:
            for post in [0, POSTMAX//2, POSTMAX]:
                ds.filter.compute(f_3db=5000, cut_pre=pre, cut_post=post)
                f = ds.filter.filt_noconst
                resultsA = np.dot(d, f)

                d2 = np.array(d)
                if pre > 0:
                    d2[:, :pre] = np.random.standard_normal((NP, pre))
                if post > 0:
                    d2[:, -post:] = np.random.standard_normal((NP, post))
                resultsB = np.dot(d2, f)
                self.assertTrue(np.allclose(resultsA, resultsB))

        # Test that filters are the same whether made from short or long pulse models,
        # at least after they are forced to be the same size.
        N, n_pre = ds.nSamples, ds.nPresamples
        dt = ds.timebase

        pulse = np.zeros((N, 1), dtype=float)
        pulse[:, 0] = ds.average_pulse[:]
        noise = np.exp(-np.arange(N)*.01)
        filterL = mass.ArrivalTimeSafeFilter(pulse, n_pre, noise_autocorr=noise, sample_time=dt)

        for cut_pre in (0, n_pre//10, n_pre//4):
            for cut_post in (0, (N-n_pre)//10, (N-n_pre)//4):
                thispulse = pulse[cut_pre:N-cut_post]
                filterS = mass.ArrivalTimeSafeFilter(thispulse, n_pre-cut_pre,
                                                     noise_autocorr=noise, sample_time=dt)
                filterS.compute()
                fS = filterS.filt_noconst

                filterL.compute(cut_pre=cut_pre, cut_post=cut_post)
                fL = filterL.filt_noconst[cut_pre:N-cut_post]
                self.assertTrue(np.allclose(fS, fL))

    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    def test_dc_insensitive(self):
        """When f_3db or fmax applied, filter should not become DC-sensitive.
        Tests for issue #176."""
        nSamples = 100
        nPresamples = 50
        nPost = nSamples-nPresamples

        # Some fake data
        pulse_like = np.append(np.zeros(nPresamples), np.linspace(nPost-1, 0, nPost))
        deriv_like = np.append(np.zeros(nPresamples), -np.ones(nPost))
        model = np.column_stack((pulse_like, deriv_like))

        fake_noise = np.random.randn(nSamples)
        fake_noise[0] = 10.0
        whitener = None
        dt = 6.72e-6

        fnew = mass.ArrivalTimeSafeFilter(
            model, nPresamples, fake_noise, whitener, dt, np.max(pulse_like))
        fold = mass.Filter(pulse_like, nPresamples, noise_autocorr=fake_noise, sample_time=dt)
        fnew.name = "AT Safe filter"
        fold.name = "Classic filter"
        for test_filter in (fold, fnew):
            test_filter.compute(f_3db=None, fmax=None)
            std = np.median(np.abs(test_filter.filt_noconst))
            mean = test_filter.filt_noconst.mean()
            self.assertLess(
                mean, 1e-10*std, msg="{} failed DC test w/o lowpass".format(test_filter.name))

            test_filter.compute(f_3db=1e4, fmax=None)
            mean = test_filter.filt_noconst.mean()
            self.assertLess(
                mean, 1e-10*std, msg="{} failed DC test w/ f_3db".format(test_filter.name))

            test_filter.compute(f_3db=None, fmax=1e4)
            mean = test_filter.filt_noconst.mean()
            self.assertLess(
                mean, 1e-10*std, msg="{} failed DC test w/ fmax".format(test_filter.name))


class TestWhitener(ut.TestCase):
    """Test ToeplitzWhitener."""

    def test_trivial(self):
        """Be sure that the trivial whitener does nothing."""
        w = mass.ToeplitzWhitener([1.0], [1.0])  # the trivial whitener
        r = np.random.standard_normal(100)
        self.assertTrue(np.allclose(r, w(r)))
        self.assertTrue(np.allclose(r, w.solveW(r)))
        self.assertTrue(np.allclose(r, w.applyWT(r)))
        self.assertTrue(np.allclose(r, w.solveWT(r)))

    def test_reversible(self):
        """Use a nontrivial whitener, and make sure that inverse operations are inverses."""
        w = mass.ToeplitzWhitener([1.0, -1.7, 0.72], [1.0, .95])
        r = np.random.standard_normal(100)
        self.assertTrue(np.allclose(r, w.solveW(w(r))))
        self.assertTrue(np.allclose(r, w(w.solveW(r))))
        self.assertTrue(np.allclose(r, w.solveWT(w.applyWT(r))))
        self.assertTrue(np.allclose(r, w.applyWT(w.solveWT(r))))

        # Also check that w isn't trivial
        self.assertFalse(np.allclose(r, w(r)))
        self.assertFalse(np.allclose(r, w.solveW(r)))
        self.assertFalse(np.allclose(r, w.applyWT(r)))
        self.assertFalse(np.allclose(r, w.solveWT(r)))

        # Check that no operations applied twice cancel out.
        self.assertFalse(np.allclose(r, w(w(r))))
        self.assertFalse(np.allclose(r, w.solveW(w.solveW(r))))
        self.assertFalse(np.allclose(r, w.applyWT(w.applyWT(r))))
        self.assertFalse(np.allclose(r, w.solveWT(w.solveWT(r))))

    def test_causal(self):
        """Make sure that the whitener and its inverse are causal,
        and that WT and its inverse anti-causal."""
        w = mass.ToeplitzWhitener([1.0, -1.7, 0.72], [1.0, .95])
        Nzero = 100
        z = np.zeros(Nzero, dtype=float)
        r = np.hstack([z, np.random.standard_normal(100), z])

        # Applying and solving W are causal operations.
        wr = w(r)
        wir = w.solveW(r)
        self.assertTrue(np.all(r[:Nzero] == 0))
        self.assertTrue(np.all(wr[:Nzero] == 0))
        self.assertTrue(np.all(wir[:Nzero] == 0))
        self.assertFalse(np.all(wr[Nzero:] == 0))
        self.assertFalse(np.all(wir[Nzero:] == 0))

        # Applying and solving WT are anti-causal operations.
        wtr = w.applyWT(r)
        wtir = w.solveWT(r)
        self.assertTrue(np.all(r[-Nzero:] == 0))
        self.assertTrue(np.all(wtr[-Nzero:] == 0))
        self.assertTrue(np.all(wtir[-Nzero:] == 0))
        self.assertFalse(np.all(wtr[:-Nzero] == 0))
        self.assertFalse(np.all(wtir[:-Nzero] == 0))


if __name__ == '__main__':
    ut.main()