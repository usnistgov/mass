import mass
import os
import numpy as np
import h5py
import tempfile
import pytest

import pylab as plt
# plt.ion()
# plt.close("all")


def test_nsls2_2024():
    # tests with some nsls2 data from 2024
    # tests these features which may be needed for nsls2 processing upgrades
    # 1. projector creation and 5lag filtering with off files works with inverted data ljh files
    # 2. learnCalibrationPlanFromEnergiesAndPeaks with a polynomial fit
    # 3. recipe book saving and loading and calculating from info in dastard summary zmq messages
    d = os.path.dirname(os.path.realpath(__file__))
    temp_d = tempfile.mkdtemp()

    pulse_files = os.path.join(d, "data_for_test", "nsls2_2024_pulses", "20240405_run0001_chan*.ljh")
    noise_files = os.path.join(d, "data_for_test", "nsls2_2024_pulses", "20240405_run0000_chan*.ljh")
    off_filenames = mass.off.getOffFileListFromOneFile(os.path.join(d, "data_for_test", 
    "nsls2_2024_pulses", "20240405_run0002_chan1.off"))

    data0 = mass.TESGroup(pulse_files, noise_files, 
                        hdf5_filename=os.path.join(temp_d, "mass.hdf5"), 
                        hdf5_noisefilename=os.path.join(temp_d, "mass_noise.hdf5"),
                        invert_data=True)
    ds0 = data0.channel[1]
    ds0.summarize_data(use_cython=False,forceNew=True)
    ppderiv1 = ds0.p_postpeak_deriv[:]
    ds0.summarize_data(use_cython=True, forceNew=True)
    ppderiv2 = ds0.p_postpeak_deriv[:]
    assert np.allclose(ppderiv1, ppderiv2)
    ds0.auto_cuts(nsigma_pt_rms=1000, nsigma_max_deriv=10, forceNew=True)
    ds0.plot_traces(np.arange(30))



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
            f_3db_5lag=15000,
            invert_data=True,
            optimize_dp_dt=False,  # seems to work better for gamma data
            extra_n_basis_5lag=0,  # mostly for testing--might help you make a more efficient basis
                                    # for gamma rays, but doesn't seem neccesary
            noise_weight_basis=True)  # only for testing, may not even work right to set to False


    data: mass.off.ChannelGroup = mass.off.ChannelGroup(off_filenames)
    data.setDefaultBinsize(0.25)  # set the default bin size in eV for fits
    data.add5LagRecipes(model_hdf5_path)
    with h5py.File(model_hdf5_path, "r") as h5:
        models = {int(ch): mass.pulse_model.PulseModel.fromHDF5(h5[ch]) for ch in h5.keys()}
    model = models[1]
    model.plot()
    data.learnDriftCorrection(indicatorName="pretriggerMean", uncorrectedName="filtValue5Lag",
                                correctedName="filtValue5LagDC")
    data.learnDriftCorrection(indicatorName="filtPhase", uncorrectedName="filtValue5LagDC",
                                correctedName="filtValue5LagDCPC")
    data.learnDriftCorrection(indicatorName="pretriggerMean", uncorrectedName="filtValue",
                                correctedName="filtValueDC")
    data.learnDriftCorrection(indicatorName="filtPhase", uncorrectedName="filtValueDC",
                                correctedName="filtValueDCPC")

    ds = data[1]
    ds.plotHist(np.arange(0,16000,10),"filtValue5LagDCPC",
                states=["CAL0","SCAN1","CAL2"],coAddStates=False)


    # mass.line_models.VALIDATE_BIN_SIZE = False
    # ds.calibrationPlanInit("filtValueDCPC")
    # ds.calibrationPlanAddPoint(5183, 'CKAlpha', states="CAL0")
    # ds.calibrationPlanAddPoint(6917, 'NKAlpha', states="CAL0")
    # ds.calibrationPlanAddPoint(8615, 'OKAlpha', states="CAL0")
    # ds.calibrationPlanAddPoint(10391, 'FeLAlpha', states="CAL0")
    # ds.calibrationPlanAddPoint(11364, 'NiLAlpha', states="CAL0")
    # ds.calibrationPlanAddPoint(11695, 'CuLAlpha', states="CAL0")

    # data.alignToReferenceChannel(ds,"filtValueDCPC", np.arange(0,40000,10),states=["CAL0"])

    data.learnCalibrationPlanFromEnergiesAndPeaks(attr="filtValueDC",
        states=["CAL0"],
        ph_fwhm=50,
        line_names=["CKAlpha","NKAlpha","OKAlpha", "FeLAlpha", "NiLAlpha", "CuLAlpha"],
        maxacc=1e4
        )
    data.learnPhaseCorrection(uncorrectedName="filtValueDC",
                            correctedName="filtValueDCPCSpline",
                            states=["CAL0", "CAL2"])
    data.calibrateFollowingPlan("filtValueDCPCSpline", overwriteRecipe=True,
                            dlo=20,dhi=20, calibratedName="energyPCSpline")

    data.learnCalibrationPlanFromEnergiesAndPeaks(attr="filtValue5LagDCPC",
        states=["CAL0"],
        ph_fwhm=50,
        line_names=["CKAlpha","NKAlpha","OKAlpha", "FeLAlpha", "NiLAlpha", "CuLAlpha"],
        maxacc=1e4
        )
    data.calibrateFollowingPlan("filtValue5LagDCPC", overwriteRecipe=True,
                            dlo=20,dhi=20)
    ds.diagnoseCalibration("energy")

    ds.plotHist(np.arange(0,1000,0.25),"energy",
                states=["CAL0","SCAN1","CAL2"],coAddStates=False)
    ds.linefit(750, states="SCAN1")

    ds.plotAvsB("peakX5Lag","energyRough",states="SCAN1")
    plt.ylim(737,745)
    plt.xlim(-0.3, 0.8)
    plt.grid(True)
    ds.plotAvsB("peakX5Lag","energy",states="SCAN1")
    plt.ylim(746,754)
    plt.xlim(-0.3, 0.8)
    plt.grid(True)
    ds.plotAvsB("peakX5Lag","energyPCSpline",states="SCAN1")
    plt.ylim(746,754)
    plt.xlim(-0.3, 0.8)
    plt.grid(True)

    ds.linefit(750, states="SCAN1")
    ds.linefit(750, attr="energyPCSpline", states="SCAN1")
    data[4].linefit(750, states="SCAN1")
    data[4].linefit(750, attr="energyPCSpline", states="SCAN1")

    recipe_book_path = os.path.join(temp_d,"recipebook")
    data.saveRecipeBooks(recipe_book_path)

    def loadRecipeBooks(filename):
        import dill
        with open(filename, "rb") as f:
            d = dill.load(f)
        # maps channel name to recipeBook
        return d

    recipe_books = loadRecipeBooks(recipe_book_path)

    # make some messages like we would recieve from the dastard zmq socket
    first_pulse_off_entry_with_coefs = ds.offFile._mmap_with_coefs[0:1]
    dtype = ds.offFile._dtype_non_descriptive
    dtype_dastard1 = np.dtype([('channelNumber','<u2'),('headerVersion','<u2'),('recordPreSamples', '<i4'), 
                            ('recordSamples', '<i4'), ('pretriggerMean', '<f4'), ('peakValue', '<f4'), 
                            ('pulseRMS', '<f4'), ('pulseAverage', '<f4'), ('residualStdDev', '<f4'),
                            ('unixnano', '<i8'), ('framecount', '<i8')])
    dtype_dastard2 = np.dtype([('coefs', '<f4', (5,))])
    msg2 = np.zeros(1, dtype=dtype_dastard2)
    msg2["coefs"] = first_pulse_off_entry_with_coefs["coefs"]
    msg1 = np.zeros(1, dtype=dtype_dastard1)
    msg1["headerVersion"]=0
    msg1["channelNumber"]=1
    msg1["recordPreSamples"]=first_pulse_off_entry_with_coefs["recordPreSamples"]
    msg1["recordSamples"]=first_pulse_off_entry_with_coefs["recordSamples"]
    msg1["pretriggerMean"]=first_pulse_off_entry_with_coefs["pretriggerMean"]
    msg1["peakValue"]=ds0.p_peak_value[0]
    msg1["pulseRMS"]=ds0.p_pulse_rms[0]
    msg1["pulseAverage"]=ds0.p_pulse_average[0]
    msg1["residualStdDev"]=first_pulse_off_entry_with_coefs["residualStdDev"]
    msg1["unixnano"]=first_pulse_off_entry_with_coefs["unixnano"]
    msg1["framecount"]=first_pulse_off_entry_with_coefs["framecount"]

    def messages2offentry(msg1, msg2):
        target_dtype = ds.offFile.dtype
        offentry = np.zeros(1, dtype=ds.offFile._dtype_non_descriptive)
        for key in target_dtype.fields.keys():
            try:
                offentry[key]=msg1[key]
            except:
                pass
        offentry["coefs"] = msg2["coefs"]
        return offentry.view(target_dtype)

    off_entry = messages2offentry(msg1, msg2)
    # pretriggerDelta doesn't exist in messages, so assign it so we can check equality
    off_entry["pretriggerDelta"]=first_pulse_off_entry_with_coefs["pretriggerDelta"]
    assert all(off_entry == ds.offFile[0])

    ch = msg1["channelNumber"][0]
    recipe_book = recipe_books[ch]
    a = recipe_book.craft("energy", off_entry)
    assert all(a==ds.getAttr("energy",slice(0,1)))
    b = recipe_book.craft("energyPCSpline", off_entry)
    assert all(b==ds.getAttr("energyPCSpline",slice(0,1)))
    plt.close("all")