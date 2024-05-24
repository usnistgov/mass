import mass
import os
import numpy as np
import h5py
import tempfile
import pytest

import pylab as plt
plt.ion()


d = os.path.dirname(os.path.realpath(__file__))

pulse_files = os.path.join(d, "data_for_test", "nsls2_2024_pulses", "20240405_run0001_chan*.ljh")
noise_files = os.path.join(d, "data_for_test", "nsls2_2024_pulses", "20240405_run0000_chan*.ljh")
off_filenames = mass.off.getOffFileListFromOneFile(os.path.join(d, "data_for_test", 
"nsls2_2024_pulses", "20240405_run0002_chan1.off"))

data0 = mass.TESGroup(pulse_files, noise_files, overwrite_hdf5_file=True)
for ds0 in data0:
    ds0.invert_data = True
ds0 = data0.channel[1]
ds0.summarize_data(use_cython=False,forceNew=True)
ppderiv1 = ds0.p_postpeak_deriv[:]
ds0.summarize_data(use_cython=True, forceNew=True)
ppderiv2 = ds0.p_postpeak_deriv[:]
assert np.allclose(ppderiv1, ppderiv2)
ds0.auto_cuts(nsigma_pt_rms=5000, nsigma_max_deriv=20, forceNew=True)
ds0.plot_traces(np.arange(30))



temp_d = tempfile.mkdtemp()
model_hdf5_path = os.path.join(temp_d, "model.hdf5")
model_mass_hdf5_path = os.path.join(temp_d, "model_mass.hdf5")
model_mass_noise_hdf5_path = os.path.join(temp_d, "model_mass_noise.hdf5")

# The projector creation process uses a random algorithm for svds, this ensures we get the same answer each time
mass.mathstat.utilities.rng = np.random.default_rng(200)
with h5py.File(model_hdf5_path, "w") as h5:
    mass.make_projectors(
        pulse_files=pulse_files,
        noise_files=noise_files,
        h5=h5,
        n_sigma_pt_rms=1000,  # we want tails of previous pulses in our basis
        n_sigma_max_deriv=10,
        n_basis=5,
        maximum_n_pulses=5000,
        mass_hdf5_path=model_mass_hdf5_path,
        mass_hdf5_noise_path=model_mass_noise_hdf5_path,
        f_3db_5lag=25000,
        invert_data=True,
        optimize_dp_dt=False,  # seems to work better for gamma data
        extra_n_basis_5lag=0,  # mostly for testing--might help you make a more efficient basis
                                # for gamma rays, but doesn't seem neccesary
        noise_weight_basis=True)  # only for testing, may not even work right to set to False


data: mass.off.ChannelGroup = mass.off.ChannelGroup(off_filenames)
data.setDefaultBinsize(10)  # set the default bin size in eV for fits
for ds in data.values():
    ds.recipes.add("pretriggerMeanCorrected", lambda pretriggerMean: pretriggerMean % 2**12)
data.add5LagRecipes(model_hdf5_path)
with h5py.File(model_hdf5_path, "r") as h5:
    models = {int(ch): mass.pulse_model.PulseModel.fromHDF5(h5[ch]) for ch in h5.keys()}
model = models[1]
model.plot()
data.learnDriftCorrection()
data.learnDriftCorrection(indicatorName="pretriggerMeanCorrected", uncorrectedName="filtValue5Lag",
                            correctedName="filtValue5LagDC")
data.learnDriftCorrection(indicatorName="filtPhase", uncorrectedName="filtValue5LagDC",
                            correctedName="filtValue5LagDCPC")
ds = data[1]
ds.plotHist(np.arange(0,40000,10),"filtValue5LagDCPC")

# ds.calibrationPlanInit("filtValue5LagDCPC")
# ds.calibrationPlanAddPoint(4369, 'ErKAlpha1')
# ds.calibrationPlanAddPoint(7230, 'Ho166m_80')
# ds.calibrationPlanAddPoint(10930, 'Co57_122')
# ds.calibrationPlanAddPoint(16450, 'Ho166m_184')
# ds.learnResidualStdDevCut(n_sigma_equiv=15, plot=True, setDefault=True)
# _ = ds.calibrateFollowingPlan("filtValue5LagDCPC", calibratedName="energy",
#                                 dlo=400, dhi=400, overwriteRecipe=True)

# # here we save the recipes, then load them and see that they give the same answer
# recipe_book_path = os.path.join(temp_d, "recipe_books.pkl")
# data.saveRecipeBooks(recipe_book_path)
# data2: mass.off.ChannelGroup = mass.off.ChannelGroup(off_filenames)
# with pytest.raises(Exception):
#     # before loading the recipes trying to access energy raises an error because it hasn't been defined
#     data2[3].energy  # this shouldn't exist yet
# data2.loadRecipeBooks(recipe_book_path)
# # after loading the recipes
# assert all(data[3].energy == data2[3].energy)
# assert all(data[3].pretriggerMeanCorrected == data2[3].pretriggerMeanCorrected)
# assert all(data[3].filtValue5LagDCPC == data2[3].filtValue5LagDCPC)
# assert all(data[3].cutResidualStdDev == data2[3].cutResidualStdDev)
# assert data[3].recipes["energy"].f.uncalibratedName == data2[3].recipes["energy"].f.uncalibratedName
# assert data[3].recipes["energy"].f._names == data2[3].recipes["energy"].f._names

# data3 = mass.off.ChannelGroup(off_filenames)
# data3[3].learnCalibrationPlanFromEnergiesAndPeaks("filtValue",
#     states=None, ph_fwhm=100, line_names=['ErKAlpha1', 'Ho166m_80', 'Co57_122', 'Ho166m_184'],
#     maxacc=0.1)
# assert data3[3].calibrationPlan.uncalibratedVals[0] == pytest.approx(4333.8, rel=1e-3)
