import mass
import os
import pylab as plt
import numpy as np

_PATH = os.path.dirname(os.path.realpath(__file__))
ljh_root = os.path.join(_PATH,"..", "ljh_files")

def load_data():
    print(ljh_root)
    pulse_str = os.path.join(ljh_root,"20230626", "0001", "20230626_run0001_chan*.ljh")
    noise_str = os.path.join(ljh_root,"20230626", "0000", "20230626_run0000_chan*.ljh")
    data = mass.TESGroup(pulse_str, noise_str, overwrite_hdf5_file=True)
    return data

def test_process1():
    data = load_data()
    data.summarize_data()
    data.auto_cuts()
    data.avg_pulses_auto_masks()
    data.compute_noise()
    data.compute_5lag_filter(f_3db=10e3)
    data.filter_data()
    data.drift_correct()
    data.phase_correct()
    data.calibrate("p_filt_value_phc", ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta"], fit_range_ev=80,
    bin_size_ev=0.5, diagnose=True, _rethrow=True)
    ds = data.channel[4102]
    ds2 = data.channel[4109]
    ds.plot_hist(np.arange(0,10000,0.5), attr="p_energy", label_lines=["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta","PdLAlpha"])
    ds.linefit("PdLAlpha", binsize=0.5)
    result = ds.linefit("MnKAlpha", binsize=0.5, has_tails=False)
    result_tail = ds.linefit("MnKAlpha", binsize=0.5, has_tails=True)
    result2 = ds2.linefit("MnKAlpha", binsize=0.5, has_tails=False)
    result_data = data.linefit("MnKAlpha", binsize=0.5, has_tails=False, category={"state":"START"})

    assert result.params["fwhm"].value < 3.6
    assert result2.params["fwhm"].value < 3.1
    assert result_data.params["fwhm"].value < 3.35

    

if __name__ == "__main__":
    test_process1()