import pytest
import mass
import numpy as np
import os

import pylab as plt


class TestPhaseCorrect:
    def load_data(self, clear_hdf5=True):
        name = 'tests/regression_test/phase_correct_test_data_4k_pulses_mass.hdf5'
        if os.path.isfile(name):
            os.remove(name)
        new_src_name = 'tests/regression_test/phase_correct_test_data_4k_pulses_chan1.ljh'
        data = mass.TESGroup([new_src_name], [new_src_name])
        ds = data.channel[1]
        return ds

    def test_phase_correct_through_microcaldataset(self, plot=False):
        ds = self.load_data()

        # the final fit resolutions are quite sensitive to this, easily varying from 3 to 5 eV
        rng = np.random.default_rng(2233)
        energies = np.arange(ds.nPulses)
        ph_peaks = []
        line_names = ["MnKAlpha", "FeKAlpha", "CuKAlpha", "CrKAlpha"]
        for i, name in enumerate(line_names):
            spect = mass.spectra[name]
            n = 1000
            if i*1000+n > ds.nPulses:
                n = ds.nPulses-i*1000
            energies[i*1000:i*1000+n] = spect.rvs(size=n, rng=rng,
                                                  instrument_gaussian_fwhm=3)
            ph_peaks.append(spect.nominal_peak_energy)
        phase = np.linspace(-0.6, 0.6, len(energies))
        rng.shuffle(energies)
        rng.shuffle(phase)
        ph = energies+phase*10  # this pushes the resolution up to roughly 10 eV

        assert ds.nPulses == len(energies)
        ds.p_filt_value_dc[:] = ph[:]
        ds.p_filt_value[:] = ph[:]
        ds.p_filt_phase[:] = phase[:]
        ds.phase_correct(ph_peaks=ph_peaks)
        # there seems to be some sort of rounding issue of np.float32 vs hdf5 storage as np.float32
        # such that I don't get exactly the same value for this case, so loop with approximat comparison
        # I'm a bit disturbed and confused here, but just going with it for now
        for (a, b) in zip(ds.phaseCorrector(phase, ph), ds.p_filt_value_phc[:]):
            assert a == pytest.approx(b, abs=0.01)

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
        assert resolutions[0] <= 3.5
        assert resolutions[1] <= 3.9
        assert resolutions[2] <= 4.0
        assert resolutions[3] <= 4.2

        # load from hdf5
        phaseCorrectorLoaded = mass.core.phase_correct.PhaseCorrector.fromHDF5(ds.hdf5_group)
        assert all(ds.phaseCorrector(phase, ph) == phaseCorrectorLoaded(phase, ph))
        assert ds.phaseCorrector.indicatorName == phaseCorrectorLoaded.indicatorName
        print(ds.phaseCorrector.uncorrectedName)
        print(phaseCorrectorLoaded.uncorrectedName)
        assert ds.phaseCorrector.uncorrectedName == phaseCorrectorLoaded.uncorrectedName

    def test_phase_correct(self, plot=False):
        # the final fit resolutions are quite sensitive to this, easily varying from 3 to 5 eV
        rng = np.random.default_rng(5632)
        energies = np.arange(4000)
        ph_peaks = []
        line_names = ["MnKAlpha", "FeKAlpha", "CuKAlpha", "CrKAlpha"]
        for i, name in enumerate(line_names):
            spect = mass.spectra[name]
            energies[i*1000:(i+1)*1000] = spect.rvs(size=1000, rng=rng,
                                                    instrument_gaussian_fwhm=3)
            ph_peaks.append(spect.nominal_peak_energy)
        phase = np.linspace(-0.6, 0.6, len(energies))
        rng.shuffle(energies)
        rng.shuffle(phase)
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
        print(resolutions)
        assert resolutions[0] <= 4.5
        assert resolutions[1] <= 4.4
        assert resolutions[2] <= 4.0
        assert resolutions[3] <= 4.4


def fix_screwed_up_LJH_file():
    """File tests/regression_test/phase_correct_test_data_4k_pulses_chan1.ljh

    This file has been screwed up since Galen added it to the repository.
    It seems like the first 300 records are LJH 2.1 data, but the remaining 3683
    records are LJH 2.2 data. Oops! This function should be run ONCE ONLY and
    used to update the file into an LJH 2.2-compatible form.

    Run on June 10, 2023. Function left here to document what was done.
    """
    infile = "tests/regression_test/phase_correct_test_data_4k_pulses_chan1.ljh"
    outfile = "tests/regression_test/REPAIRED_chan1.ljh"
    ljh = mass.LJHFile.open(infile)
    header_length = ljh.header_size
    binary_length = ljh.pulse_size_bytes
    with open(infile, "rb") as fin:
        header = fin.read(header_length)
        newheader = header.replace(b"Save File Format Version: 2.1.0",
                                   b"Save File Format Version: 2.2.0")
        newheader = newheader.replace(b"Dummy: 0\r\n", b"")
        with open(outfile, "wb") as fout:
            fout.write(newheader)
            padding = b"deadbeef.."
            for recnum in range(300):
                data = fin.read(binary_length)
                fout.write(padding)
                fout.write(data)
            while True:
                data = fin.read(binary_length+10)
                if len(data) < binary_length+10:
                    break
                fout.write(data)
