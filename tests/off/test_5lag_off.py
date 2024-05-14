import mass
import os
import numpy as np
import h5py
import tempfile
import pytest

def test_off_5lag_with_saving_and_loading_recipes():
    d = os.path.dirname(os.path.realpath(__file__))

    pulse_files = [os.path.join(d,"data_for_test","20181018_144520","20181018_144520_chan3.ljh")]
    noise_files = [os.path.join(d,"data_for_test","20181018_144325","20181018_144325_chan3.noi")]

    temp_d = tempfile.mkdtemp()
    model_hdf5_path = os.path.join(temp_d,"model.hdf5")
    model_mass_hdf5_path = os.path.join(temp_d, "model_mass.hdf5")
    model_mass_noise_hdf5_path = os.path.join(temp_d, "model_mass_noise.hdf5")

    # The projector creation process uses a random algorithm for svds, this ensures we get the same answer each time
    mass.mathstat.utilities.rng = np.random.default_rng(200)
    with h5py.File(model_hdf5_path,"w") as h5:
        mass.make_projectors(pulse_files=pulse_files,
            noise_files=noise_files,
            h5=h5,
            n_sigma_pt_rms=1000, # we want tails of previous pulses in our basis
            n_sigma_max_deriv=10,
            n_basis=5,
            maximum_n_pulses=5000,
            mass_hdf5_path=model_mass_hdf5_path,
            mass_hdf5_noise_path=model_mass_noise_hdf5_path,
            invert_data=False,
            optimize_dp_dt=False, # seems to work better for gamma data
            extra_n_basis_5lag=0, # mostly for testing, might help you make a more efficient basis for gamma rays, but doesn't seem neccesary
            noise_weight_basis=True) # only for testing, may not even work right to set to False

    output_dir = os.path.join(temp_d, "20181018_144520_off")
    os.mkdir(output_dir)
    r = mass.ljh2off.ljh2off_loop(ljhpath = pulse_files[0],
        h5_path = model_hdf5_path,
        output_dir = output_dir,
        max_channels = 240,
        n_ignore_presamples = 0,
        require_experiment_state=False,
        show_progress=True)
    ljh_filenames, off_filenames = r

    # write a dummy experiment state file, since the data didn't come with one
    with open(os.path.join(output_dir, "20181018_144520_experiment_state.txt"),"w") as f:
        f.write("# yo yo\n")
        f.write("0, START\n")

    data: mass.off.ChannelGroup = mass.off.ChannelGroup(off_filenames)
    data.setDefaultBinsize(10) # set the default bin size in eV for fits
    for ds in data.values():
        ds.recipes.add("pretriggerMeanCorrected", lambda pretriggerMean: pretriggerMean%2**12)
    data.add5LagRecipes(model_hdf5_path)
    data.learnDriftCorrection()
    data.learnDriftCorrection(indicatorName="pretriggerMeanCorrected", uncorrectedName="filtValue5Lag", 
                            correctedName="filtValue5LagDC") 
    data.learnDriftCorrection(indicatorName="filtPhase", uncorrectedName="filtValue5LagDC", 
                            correctedName="filtValue5LagDCPC")
    mass.STANDARD_FEATURES['Ho166m_80'] = 80.574e3
    mass.STANDARD_FEATURES['Co57_122'] = 122.06065e3
    mass.STANDARD_FEATURES['Ho166m_184'] = 184.4113e3
    ds = data[3]
    ds.calibrationPlanInit("filtValue5LagDCPC")
    ds.calibrationPlanAddPoint(4369, 'ErKAlpha1')
    ds.calibrationPlanAddPoint(7230, 'Ho166m_80')
    ds.calibrationPlanAddPoint(10930, 'Co57_122')
    ds.calibrationPlanAddPoint(16450, 'Ho166m_184')
    ds.learnResidualStdDevCut(n_sigma_equiv=15, plot=True, setDefault=True)
    results = ds.calibrateFollowingPlan("filtValue5LagDCPC", calibratedName="energy",
        dlo=400, dhi=400,overwriteRecipe=True)

    # here we save the recipes, then load them and see that they give the same answer
    recipe_book_path = os.path.join(temp_d, "recipe_books.pkl")
    data.saveRecipeBooks(recipe_book_path)
    data2: mass.off.ChannelGroup = mass.off.ChannelGroup(off_filenames)
    with pytest.raises(Exception):
        # before loading the recipes trying to access energy raises an error because it hasn't been defined
        data2[3].energy # this shouldn't exist yet
    data2.loadRecipeBooks(recipe_book_path)
    # after loading the recipes
    assert all(data[3].energy == data2[3].energy)
    assert all(data[3].pretriggerMeanCorrected == data2[3].pretriggerMeanCorrected)
    assert all(data[3].filtValue5LagDCPC == data2[3].filtValue5LagDCPC)
    assert all(data[3].cutResidualStdDev == data2[3].cutResidualStdDev)
    assert data[3].recipes["energy"].f.uncalibratedName == data2[3].recipes["energy"].f.uncalibratedName
    assert data[3].recipes["energy"].f._names == data2[3].recipes["energy"].f._names

    

