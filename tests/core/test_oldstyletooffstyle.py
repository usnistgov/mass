import mass
import mass.off
import numpy as np
from pathlib import Path

try:
    d = Path(__file__).parent
except NameError:
    d = Path.cwd()


def load_data(path, hdf5_filename=None, hdf5_noisefilename=None, skip_noise=False,
              experimentStateFile=None):
    src_name = str((d.parent / "regression_test" / "regress_chan1.ljh").resolve())
    noi_name = str((d.parent / "regression_test" / "regress_noise_chan1.ljh").resolve())
    if skip_noise:
        noi_name = None
    if hdf5_filename is None:
        hdf5_file = path / "oldstyletooffstyle_mass.hdf5"
        hdf5_filename = str(hdf5_file)
    if hdf5_noisefilename is None:
        hdf5_noisefile = path / "oldstyletooffstyle_mass_noise.hdf5"
        hdf5_noisefilename = str(hdf5_noisefile)
    return mass.TESGroup(src_name, noi_name, hdf5_filename=hdf5_filename,
                         hdf5_noisefilename=hdf5_noisefilename,
                         experimentStateFile=experimentStateFile)


def test_oldstyletooffstyle(tmp_path):
    dataold = load_data(tmp_path)
    dataold.summarize_data()
    dataold.auto_cuts()
    dataold.compute_noise()
    dataold.compute_ats_filter()
    dataold.filter_data()
    dsold = dataold.first_good_dataset
    dsoffstyle = dsold.toOffStyle()
    dsoffstyle.hist(np.arange(0, 1000, 10), "p_filt_value")
    dsoffstyle.learnDriftCorrection("p_pretrig_mean", "p_filt_value")
    dsoffstyle.hist(np.arange(0, 1000, 10,), "p_filt_valueDC")
    dsoffstyle.plotAvsB("relTimeSec", "p_filt_valueDC")
    assert dsoffstyle.getAttr("unixnano", slice(0, 1))[0] * 1e-9 == round(dsold.p_timestamp[0])

    dataoffstyle = dataold.toOffStyle()
    dataoffstyle.hist(np.arange(0, 1000, 10), "p_filt_value")
    dataoffstyle.plotHist(np.arange(0, 1000, 10), "p_filt_value")
