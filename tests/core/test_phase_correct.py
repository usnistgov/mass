import mass
import numpy as np
import unittest as ut
import os

import pylab as plt


class TestPhaseCorrect(ut.TestCase):
    def load_data(self, clear_hdf5=True):
        name = 'tests/regression_test/phase_correct_test_data_4k_pulses_mass.hdf5'
        if os.path.isfile(name):
            os.remove(name)
        new_src_name = 'tests/regression_test/phase_correct_test_data_4k_pulses_chan1.ljh'
        data = mass.TESGroup([new_src_name], [new_src_name])
        ds = data.channel[1]
        return ds

    def test_phase_correct_through_microcaldataset(self, plot=False):
        # the final fit resolutions are quite sensitive to this, easily varying from 3 to 5 eV
        np.random.seed(1231)
        energies = np.arange(4000)
        ph_peaks = []
        line_names = ["MnKAlpha", "FeKAlpha", "CuKAlpha", "CrKAlpha"]
        for i, name in enumerate(line_names):
            spect = mass.spectra[name]
            energies[i*1000:(i+1)*1000] = spect.rvs(size=1000,
                                                    instrument_gaussian_fwhm=3)
            ph_peaks.append(spect.nominal_peak_energy)
        phase = np.linspace(-0.6, 0.6, len(energies))
        np.random.shuffle(energies)
        np.random.shuffle(phase)
        ph = energies+phase*10  # this pushes the resolution up to roughly 10 eV

        ds = self.load_data()
        self.assertEqual(ds.nPulses, len(energies))
        ds.p_filt_value_dc[:] = ph[:]
        ds.p_filt_value[:] = ph[:]
        ds.p_filt_phase[:] = phase[:]
        ds.phase_correct(ph_peaks=ph_peaks)
        # there seems to be some sort of rounding issue of np.float32 vs hdf5 storage as np.float32
        # such that I don't get exactly the same value for this case, so loop with approximat comparison
        # I'm a bit disturbed and confused here, but just going with it for now
        for (a, b) in zip(ds.phaseCorrector(phase, ph), ds.p_filt_value_phc[:]):
            self.assertAlmostEqual(a, b, 3)

        if plot:
            plt.figure()
            plt.plot(ds.p_filt_phase, ds.p_filt_value_dc, ".", label="dc")
            plt.plot(ds.p_filt_phase, ds.p_filt_value_phc, ".", label="phc")
            plt.legend()

        resolutions = []
        for name in line_names:
            line = mass.spectra[name]
            model = line.model()
            bin_edges = np.arange(-100, 100)+line.peak_energy
            bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
            counts, _ = np.histogram(ds.p_filt_value_phc, bin_edges)
            params = model.guess(counts, bin_centers=bin_centers)
            params["dph_de"].set(1.0, vary=False)
            result = model.fit(counts, params, bin_centers=bin_centers)
            resolutions.append(result.best_values["fwhm"])
            if plot:
                result.plotm()
        self.assertLessEqual(resolutions[0], 4.3)
        self.assertLessEqual(resolutions[1], 4.4)
        self.assertLessEqual(resolutions[2], 5.0)
        self.assertLessEqual(resolutions[3], 4.1)

        # load from hdf5
        phaseCorrectorLoaded = mass.core.phase_correct.PhaseCorrector.fromHDF5(ds.hdf5_group)
        self.assertTrue(all(ds.phaseCorrector(phase, ph) == phaseCorrectorLoaded(phase, ph)))
        self.assertEqual(ds.phaseCorrector.indicatorName, phaseCorrectorLoaded.indicatorName)
        print(ds.phaseCorrector.uncorrectedName)
        print(phaseCorrectorLoaded.uncorrectedName)
        self.assertEqual(ds.phaseCorrector.uncorrectedName, phaseCorrectorLoaded.uncorrectedName)

    def test_phase_correct(self, plot=False):
        # the final fit resolutions are quite sensitive to this, easily varying from 3 to 5 eV
        np.random.seed(1231)
        energies = np.arange(4000)
        ph_peaks = []
        line_names = ["MnKAlpha", "FeKAlpha", "CuKAlpha", "CrKAlpha"]
        for i, name in enumerate(line_names):
            spect = mass.spectra[name]
            energies[i*1000:(i+1)*1000] = spect.rvs(size=1000, instrument_gaussian_fwhm=3)
            ph_peaks.append(spect.nominal_peak_energy)
        phase = np.linspace(-0.6, 0.6, len(energies))
        np.random.shuffle(energies)
        np.random.shuffle(phase)
        ph = energies+phase*10  # this pushes the resolution up to roughly 10 eV

        phaseCorrector = mass.core.phase_correct.phase_correct(phase, ph, ph_peaks=ph_peaks)
        corrected = phaseCorrector(phase, ph)

        resolutions = []
        for name in line_names:
            line = mass.spectra[name]
            model = line.model()
            bin_edges = np.arange(-100, 100)+line.peak_energy
            bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
            counts, _ = np.histogram(corrected, bin_edges)
            params = model.guess(counts, bin_centers=bin_centers)
            params["dph_de"].set(1.0, vary=False)
            result = model.fit(counts, params, bin_centers=bin_centers)
            resolutions.append(result.best_values["fwhm"])
            if plot:
                result.plotm()
        self.assertLessEqual(resolutions[0], 4.3)
        self.assertLessEqual(resolutions[1], 4.4)
        self.assertLessEqual(resolutions[2], 5.0)
        self.assertLessEqual(resolutions[3], 4.1)


if __name__ == '__main__':
    ut.main()
