import mass
import mass.off
import tempfile
import numpy as np

def load_data(hdf5_filename=None, hdf5_noisefilename=None, skip_noise=False,
    experimentStateFile=None):
    src_name = ['mass/regression_test/regress_chan1.ljh']
    noi_name = ['mass/regression_test/regress_noise_chan1.ljh']
    if skip_noise:
        noi_name = None
    if hdf5_filename is None:
        hdf5_file = tempfile.NamedTemporaryFile(suffix='_mass.hdf5', delete=False)
        hdf5_filename = hdf5_file.name
    if hdf5_noisefilename is None:
        hdf5_noisefile = tempfile.NamedTemporaryFile(suffix='_mass_noise.hdf5', delete=False)
        hdf5_noisefilename = hdf5_noisefile.name
    return mass.TESGroup(src_name, noi_name, hdf5_filename=hdf5_filename,
                            hdf5_noisefilename=hdf5_noisefilename,
                            experimentStateFile=experimentStateFile)

def test_oldstyletooffstyle():
    dataold = load_data()
    dataold.summarize_data()
    dataold.auto_cuts()
    dataold.compute_noise_spectra()
    dataold.compute_ats_filter()
    dataold.filter_data()
    dsold = dataold.first_good_dataset
    dsoffstyle = dsold.toOffStyle()
    dsoffstyle.hist(np.arange(0,1000,10), "filtValue")