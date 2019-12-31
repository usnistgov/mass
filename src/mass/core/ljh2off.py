#!/usr/bin/python
import mass
import mass.off
import os
import json
import collections
import numpy as np
import h5py
import progress.bar
import argparse

# intended for the application of converting ljh files to mass files given somep projectors and basis

_OFF_VERSION = "0.2.0"


def off_header_dict_from_ljhfile(ljhfile, projectors, basis, h5_path):
    d = collections.OrderedDict()
    d["FileFormatVersion"] = "0.2.0"
    d["FramePeriodSeconds"] = ljhfile.timebase
    d["NumberOfBases"] = projectors.shape[0]
    d["FileFormat"] = "OFF"
    d["ModelInfo"] = collections.OrderedDict()
    d["ModelInfo"]["Projectors"] = {
        "Rows": projectors.shape[0],
        "Cols": projectors.shape[1],
        "SavedAs": "row-major float64 binary data after header and before records. projectors first then basis, nbytes = rows*cols*8 for each projectors and basis"
    }
    d["ModelInfo"]["Basis"] = {
        "Rows": basis.shape[0],
        "Cols": basis.shape[1],
        "SavedAs": "row-major float64 binary data after header and before records. projectors first then basis, nbytes = rows*cols*8 for each projectors and basis"
    }
    d["ModelInfo"]["ModelFile"] = h5_path
    d["PulseFile"] = ljhfile.filename
    d["ReadoutInfo"] = {
        "ColumnsNum": ljhfile.column_number,
        "RowNum": ljhfile.row_number,
        "NumberOfColumns": ljhfile.number_of_columns,
        "NumberOfRows": ljhfile.number_of_rows
    }
    d["CreationInfo"] = {
        "SourceName": "ljh2off.py"
    }
    return d


def off_header_string_from_ljhfile(ljhfile, projectors, basis, h5_path):
    d = off_header_dict_from_ljhfile(ljhfile, projectors, basis, h5_path)
    return json.dumps(d, indent=4)+"\n"


def ljh2off(ljhpath, offpath, projectors, basis, n_ignore_presamples, h5_path, off_version=_OFF_VERSION):
    ljhfile = mass.LJHFile(ljhpath)
    nbasis = projectors.shape[0]
    dtype = mass.off.off.recordDtype(off_version, nbasis)
    with open(offpath, "wb") as f:  # opening in binary form prevents windows from messing up newlines
        f.write(off_header_string_from_ljhfile(
            ljhfile, projectors, basis, h5_path).encode('utf-8'))
        projectors.tofile(f)
        basis.tofile(f)
        n = 0
        for (i_lo, i_hi, i_segment, data) in ljhfile.iter_segments():
            records_this_seg = data.shape[0]
            n += records_this_seg
            # print("i_lo {}, i_hi {}, i_segment {}, nnow {}, nsum {}".format(i_lo, i_hi, i_segment, data.shape[0], n))
            timestamps = ljhfile.datatimes_float
            rowcounts = ljhfile.rowcount
            mpc = np.matmul(projectors, data.T)  # modeled pulse coefs
            mp = np.matmul(basis, mpc)  # modeled pulse
            residuals = mp-data.T
            residual_std_dev = np.std(residuals, axis=0)
            pretrig_mean = data[:, :ljhfile.nPresamples-n_ignore_presamples].mean(axis=1)
            offdata = np.zeros(records_this_seg, dtype)
            if True:  # load data into offdata: implementation 1
                offdata["recordSamples"] = ljhfile.nSamples
                offdata["recordPreSamples"] = ljhfile.nPresamples
                offdata["framecount"] = rowcounts//ljhfile.number_of_rows
                offdata["unixnano"] = timestamps*1e9
                offdata["pretriggerMean"] = pretrig_mean
                offdata["residualStdDev"] = residual_std_dev
                offdata["coefs"] = mpc.T
            else:  # load data into offdata: implementation 2
                for i in range(records_this_seg):
                    offdata[i] = (
                        ljhfile.nSamples, ljhfile.nPresamples, rowcounts[i]//ljhfile.number_of_rows,
                        np.int64(timestamps[i]*1e9),
                        pretrig_mean[i], residual_std_dev[i],
                        mpc[:, i])
            # write offdata to file
            offdata.tofile(f)


def ljh2off_loop(ljhpath, h5_path, output_dir, max_channels, n_ignore_presamples, require_experiment_state=True):
    pulse_model_dict = load_pulse_models(h5_path)
    basename, channum = mass.ljh_util.ljh_basename_channum(ljhpath)
    ljhdir, file_basename = os.path.split(basename)
    off_basename = os.path.join(output_dir, file_basename)
    n_channels = min(max_channels, len(pulse_model_dict))
    bar = progress.bar.Bar("processing ljh files to off files:", max=n_channels)
    off_filenames = []
    ljh_filenames = []
    for channum, pulse_model in pulse_model_dict.items():
        ljhpath = "{}_chan{}.ljh".format(basename, channum)
        offpath = '{}_chan{}.off'.format(off_basename, channum)
        if not os.path.isfile(ljhpath):
            continue
        pulse_model = pulse_model_dict[channum]
        ljh2off(ljhpath, offpath, pulse_model.projectors, pulse_model.basis, n_ignore_presamples, h5_path)
        bar.next()
        off_filenames.append(offpath)
        ljh_filenames.append(ljhpath)
        if bar.index == max_channels:
            break
    bar.finish()
    source_experiment_state_filename = "{}_experiment_state.txt".format(basename)
    sink_experiment_state_filename = "{}_experiment_state.txt".format(off_basename)
    if os.path.isfile(source_experiment_state_filename):
        if source_experiment_state_filename != sink_experiment_state_filename:
            with open(source_experiment_state_filename, "r") as f_source:
                with open(sink_experiment_state_filename, "w") as f_sink:
                    for line in f_source:
                        f_sink.write(line)
                    print("wrote experiment state file to : {}".format(
                        os.path.abspath(sink_experiment_state_filename)))
        else:
            print("not copying experiment state file {} because the source and destination are the same".format(
                source_experiment_state_filename))
    elif require_experiment_state:
        raise Exception("{} does not exist, and require_experiment_state=True".format(
            source_experiment_state_filename))

    return ljh_filenames, off_filenames


def load_pulse_models(h5_path):
    pulse_model_dict = collections.OrderedDict()
    with h5py.File(h5_path, "r") as h5:
        channel_numbers = sorted(map(int, h5.keys()))
        for channum in channel_numbers:
            pulse_model = mass.PulseModel.fromHDF5(h5["{}".format(channum)])
            pulse_model_dict[channum] = pulse_model
    return pulse_model_dict


def parse_args(fake):
    if fake:
        return FakeArgs()
    example_usage = """ python ljh2off.py data/20190924/0010/20190924_run0010_chan1.ljh data/20190923/0003/20190923_run0003_model.hdf5 test_ljh2off -m 4 -r"""
    parser = argparse.ArgumentParser(
        description="convert ljh files to off files, example:\n"+example_usage)
    parser.add_argument(
        "ljh_path", help="path a a single ljh file, other channel numbers will be found automatically")
    parser.add_argument("h5_path", help="path to a hdf5 file with projectors and bases")
    parser.add_argument(
        "output_dir", help="path to output dir (will be created if it doesn't exist)")
    parser.add_argument("-r", "--replace_output",
                        help="pass this to overwrite off files with the same path", action="store_true")
    parser.add_argument("-m", "--max_channels",
                        help="stop after processing this many channels", default=2**31, type=int)
    parser.add_argument("--n_ignore_presamples",
                        help="ignore this many presample before the rising edge when calculating pretrigger_mean", default=3, type=int)
    args = parser.parse_args()
    return args


class FakeArgs():
    def __init__(self):
        self.ljh_path = "/Users/oneilg/Documents/EBIT/data/20190924/0010/20190924_run0010_chan1.ljh"
        self.h5_path = "/Users/oneilg/Documents/EBIT/data/20190923/0003/20190923_run0003_model.hdf5"
        self.output_dir = "test_ljh2off"
        self.max_channels = 1
        self.replace_output = True
        self.n_ignore_presamples = 3

    def __repr__(self):
        return "FakeArgs: change the script to have _TEST=False to use real args, this is just for testing from within ipython"
