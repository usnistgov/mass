import mass
import os
import logging
import h5py
import sys
import argparse
LOG = logging.getLogger("mass")
LOG.setLevel(logging.DEBUG)

# these are define in core so they can be tested easily, then they'll be run via a script


def make_projectors(pulse_files, noise_files, h5, n_sigma_pt_rms, n_sigma_max_deriv,
                    n_basis, maximum_n_pulses, mass_hdf5_path):
    data = mass.TESGroup(pulse_files, noise_files, overwrite_hdf5_file=True,
                         hdf5_filename=mass_hdf5_path)
    data.summarize_data()
    data.auto_cuts(nsigma_pt_rms=n_sigma_pt_rms, nsigma_max_deriv=n_sigma_max_deriv)
    data.compute_noise_spectra()
    data.compute_ats_filter(shift1=False)
    data.pulse_model_to_hdf5(h5, n_basis=n_basis, maximum_n_pulses=maximum_n_pulses)


def parse_args(fake):
    if fake:
        return mass.ljh2off.FakeArgs()
    example_usage = """python make_projectors.py pulse_path noise_path"""
    parser = argparse.ArgumentParser(
        description="convert ljh files to off files, example:\n"+example_usage)
    parser.add_argument(
        "pulse_path", help="path a a single ljh file with pulses, other channel numbers will be found automatically")
    parser.add_argument(
        "noise_path", help="path a a single ljh file with noise records, other channel numbers will be found automatically")
    parser.add_argument("-o", "--output_path",
                        help="output filename (should be .hdf5), the default behavior will place it in the same directory as pulse_path", default=None, type=str)
    parser.add_argument("-r", "--replace_output",
                        help="pass this to overwrite off files with the same path", action="store_true", default=False)
    parser.add_argument("-m", "--max_channels",
                        help="stop after processing this many channels", default=2**31, type=int)
    parser.add_argument("--n_ignore_presamples",
                        help="ignore this many presample before the rising edge when calculating pretrigger_mean", default=3, type=int)
    parser.add_argument("--n_sigma_pt_rms", type=float, default=8,
                        help="passed to autocuts to determine pulses used to generate pulse model")
    parser.add_argument("--n_sigma_max_deriv", type=float, default=8,
                        help="passed to autocuts to determine pulses used to generate pulse model")
    parser.add_argument("-n", "--n_basis", type=int, default=5,
                        help="how many projectors to generate, must be >=3")
    parser.add_argument("--maximum_n_pulses", type=int, default=4000,
                        help="maximum number of pulses to take a truncated SVD of, will affect speed")
    parser.add_argument("--silent", action="store_true",
                        help="supress text output, mostly for testing")
    parser.add_argument("--mass_hdf5_path", default=None,
                        help="specify the path for the mass hdf5 file that is generated as a byproduct of this script")
    args = parser.parse_args()
    return args


def main(args=None):
    if args is None:
        args = parse_args(fake=False)
    if not args.silent:
        print("starting make_projectors")
    for k in sorted(vars(args).keys()):
        print("{}: {}".format(k, vars(args)[k]))
    # find files
    channums = mass.ljh_util.ljh_get_channels_both(args.pulse_path, args.noise_path)
    if not args.silent:
        print("found these {} channels with both pulse and noise files: {}".format(len(channums), channums))
    nchan = len(channums)
    if args.max_channels < nchan:
        channums = channums[:nchan]
        if not args.silent:
            print("chose first max_channels={} channels".format(args.max_channels))
    if len(channums) == 0:
        raise Exception("no channels found for files matching {} and {}".format(
            args.pulse_path, args.noise_path))
    pulse_basename, _ = mass.ljh_util.ljh_basename_channum(args.pulse_path)
    noise_basename, _ = mass.ljh_util.ljh_basename_channum(args.noise_path)
    pulse_files = [pulse_basename+"_chan{}.ljh".format(channum) for channum in channums]
    noise_files = [noise_basename+"_chan{}.ljh".format(channum) for channum in channums]
    # handle output filename
    if args.output_path is None:
        args.output_path = pulse_basename+"_model.hdf5"
    # handle replace_output
    if os.path.isfile(args.output_path) and not args.replace_output:
        print("output: {} already exists, pass --replace_output or -r to overwrite".format(args.output_path))
        print("aborting")
        sys.exit(1)
    # create output file
    with h5py.File(args.output_path, "w") as h5:
        make_projectors(pulse_files=pulse_files, noise_files=noise_files, h5=h5,
                        n_sigma_pt_rms=args.n_sigma_pt_rms, n_sigma_max_deriv=args.n_sigma_max_deriv,
                        n_basis=args.n_basis, maximum_n_pulses=args.maximum_n_pulses, mass_hdf5_path=args.mass_hdf5_path)