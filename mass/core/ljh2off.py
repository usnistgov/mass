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
import logging
import sys
LOG = logging.getLogger("mass")

# Intended for conversion of LJH files to OFF files, given some projectors and basis

_OFF_VERSION = "0.3.0"


def off_header_dict_from_ljhfile(ljhfile, projectors, basis, h5_path):
    d = collections.OrderedDict()
    d["FileFormatVersion"] = "0.3.0"
    d["FramePeriodSeconds"] = ljhfile.timebase
    d["NumberOfBases"] = projectors.shape[0]
    d["FileFormat"] = "OFF"
    d["ModelInfo"] = collections.OrderedDict()
    saved_msg = " ".join(
        ["row-major float64 binary data after header and before records.",
         "projectors first then basis, nbytes = rows*cols*8 for each projectors and basis"])
    d["ModelInfo"]["Projectors"] = {
        "Rows": projectors.shape[0],
        "Cols": projectors.shape[1],
        "SavedAs": saved_msg
    }
    d["ModelInfo"]["Basis"] = {
        "Rows": basis.shape[0],
        "Cols": basis.shape[1],
        "SavedAs": saved_msg
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
    return multi_ljh2off([ljhpath], offpath, projectors, basis, n_ignore_presamples, h5_path, off_version)


def multi_ljh2off(ljhpaths, offpath, projectors, basis, n_ignore_presamples, h5_path, off_version=_OFF_VERSION):
    ljhfile0 = mass.LJHFile.open(ljhpaths[0])
    nbasis = projectors.shape[0]
    dtype = mass.off.off.recordDtype(off_version, nbasis, descriptive_coefs_names=False)
    with open(offpath, "wb") as f:  # opening in binary form prevents windows from messing up newlines
        f.write(off_header_string_from_ljhfile(
            ljhfile0, projectors, basis, h5_path).encode('utf-8'))
        projectors.tofile(f)
        basis.tofile(f)
        n = ljh_records_to_off(ljhfile0, f, projectors, basis, n_ignore_presamples, dtype)
        assert n == ljhfile0.nPulses, "wrong number of records written"
        for ljhpath in ljhpaths[1:]:
            ljhfile = mass.LJHFile.open(ljhpath)
            n = ljh_records_to_off(ljhfile, f, projectors, basis, n_ignore_presamples, dtype)
            assert n == ljhfile.nPulses, "wrong number of records written"


def ljh_records_to_off(ljhfile, f, projectors, basis, n_ignore_presamples, dtype):
    """append to `f` off file formatted records based on the records in ljhfile.

    Returns how many records were written"""

    # To keep linear algebra sizes manageable, loop over the file in segments
    seg_bytes = 2**21
    rec_per_seg = seg_bytes // ljhfile.binary_size
    if rec_per_seg < 1:
        rec_per_seg = 1
    records_written = 0
    for idx_lo in np.arange(0, ljhfile.nPulses, rec_per_seg):
        idx_hi = min(idx_lo + rec_per_seg, ljhfile.nPulses)
        records_this_seg = idx_hi - idx_lo
        records_written += records_this_seg

        data = ljhfile.alldata[idx_lo:idx_lo+rec_per_seg]
        timestamps = ljhfile.datatimes_float[idx_lo:idx_hi]
        rowcounts = ljhfile.rowcount[idx_lo:idx_hi]
        projector_record_length = projectors.shape[1]
        data_record_length = data.shape[1]
        assert projector_record_length == data_record_length, \
            f"projectors are for records of length {projector_record_length}, but {ljhfile} has records of length {data_record_length}"
        mpc = np.matmul(projectors, data.T)  # modeled pulse coefs
        mp = np.matmul(basis, mpc)  # modeled pulse
        residuals = mp-data.T
        residual_std_dev = np.std(residuals, axis=0)
        pretrig_mean = data[:, :ljhfile.nPresamples-n_ignore_presamples].mean(axis=1)
        offdata = np.zeros(records_this_seg, dtype)
        pfit_pt_delta = np.polyfit(np.arange(ljhfile.nPresamples-n_ignore_presamples),
                                   data[:, :ljhfile.nPresamples-n_ignore_presamples].T, deg=1)
        pt_delta = np.polyval(pfit_pt_delta, ljhfile.nPresamples
                              - n_ignore_presamples-1) - np.polyval(pfit_pt_delta, 0)

        offdata["recordSamples"] = ljhfile.nSamples
        offdata["recordPreSamples"] = ljhfile.nPresamples
        offdata["framecount"] = rowcounts // ljhfile.number_of_rows
        offdata["unixnano"] = timestamps*1e9
        offdata["pretriggerMean"] = pretrig_mean
        offdata["pretriggerDelta"] = pt_delta
        offdata["residualStdDev"] = residual_std_dev
        offdata["coefs"] = mpc.T
        offdata.tofile(f)
    return records_written


def multi_ljh2off_loop(ljhbases, h5_path, off_basename, max_channels, n_ignore_presamples,
                       require_experiment_state=True,
                       show_progress=LOG.isEnabledFor(logging.WARN)):
    pulse_model_dict = load_pulse_models(h5_path)
    n_channels = min(max_channels, len(pulse_model_dict))
    if show_progress:
        bar = progress.bar.Bar("processing ljh files to off files:", max=n_channels)
    off_filenames = []
    ljh_filename_lists = []
    handled_channels = 0
    for channum, pulse_model in pulse_model_dict.items():
        ljhpaths = [f'{ljhbase}_chan{channum}.ljh' for ljhbase in ljhbases]
        offpath = f'{off_basename}_chan{channum}.off'
        if not any([os.path.isfile(ljhpath) for ljhpath in ljhpaths]):
            continue  # make sure at least one of the desired files exists
        pulse_model = pulse_model_dict[channum]
        multi_ljh2off(ljhpaths, offpath, pulse_model.projectors,
                      pulse_model.basis, n_ignore_presamples, h5_path)
        if show_progress:
            bar.next()
        off_filenames.append(offpath)
        ljh_filename_lists.append(ljhpaths)
        handled_channels += 1
        if handled_channels == max_channels:
            break
    if show_progress:
        bar.finish()
    return ljh_filename_lists, off_filenames


def ljh2off_loop(ljhpath, h5_path, output_dir, max_channels, n_ignore_presamples, require_experiment_state=True,
                 show_progress=LOG.isEnabledFor(logging.WARN)):
    basename, channum = mass.ljh_util.ljh_basename_channum(ljhpath)
    ljhdir, file_basename = os.path.split(basename)
    off_basename = os.path.join(output_dir, file_basename)
    ljh_filename_lists, off_filenames = multi_ljh2off_loop([basename], h5_path, off_basename,
                                                           max_channels, n_ignore_presamples, require_experiment_state, show_progress)
    ljh_filenames = [fname[0] for fname in ljh_filename_lists]
    for fname in ljh_filename_lists:
        assert len(fname) == 1
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
    example_usage = """ python ljh2off.py data/20190924/0010/20190924_run0010_chan1.ljh """
    example_usage += """data/20190923/0003/20190923_run0003_model.hdf5 test_ljh2off -m 4 -r"""
    parser = argparse.ArgumentParser(
        description="convert ljh files to off files, example:\n"+example_usage,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
                        help="ignore this many presample before the rising edge when calculating pretrigger_mean", default=0, type=int)
    args = parser.parse_args()
    return args


def main():
    print("starting ljh2off")
    args = mass.ljh2off.parse_args(fake=False)
    for k in sorted(vars(args).keys()):
        print("{}: {}".format(k, vars(args)[k]))
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    elif not args.replace_output:
        print("dir {} exists, pass --replace_output to write into it anyway".format(args.output_dir))
        sys.exit()
    ljh_filenames, off_filenames = mass.ljh2off.ljh2off_loop(
        args.ljh_path, args.h5_path, args.output_dir, args.max_channels, args.n_ignore_presamples)
    print("full path to first off file:")
    print(os.path.abspath(off_filenames[0]))


class FakeArgs():
    def __init__(self):
        self.ljh_path = "/Users/oneilg/Documents/EBIT/data/20190924/0010/20190924_run0010_chan1.ljh"
        self.h5_path = "/Users/oneilg/Documents/EBIT/data/20190923/0003/20190923_run0003_model.hdf5"
        self.output_dir = "test_ljh2off"
        self.max_channels = 1
        self.replace_output = True
        self.n_ignore_presamples = 3
        self.f_3db_ats = None
        self.f_3db_5lag = None

    def __repr__(self):
        return "FakeArgs: change the script to have _TEST=False to use real args, this is just for testing from within ipython"
