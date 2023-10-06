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
                    n_basis, maximum_n_pulses, mass_hdf5_path, mass_hdf5_noise_path, invert_data, optimize_dp_dt,
                    extra_n_basis_5lag, noise_weight_basis, f_3db_ats=None, f_3db_5lag=None):
    data = mass.TESGroup(pulse_files, noise_files, overwrite_hdf5_file=True,
                         hdf5_filename=mass_hdf5_path,
                         hdf5_noisefilename=mass_hdf5_noise_path)
    for ds in data:
        ds.invert_data = invert_data
    data.summarize_data()
    data.auto_cuts(nsigma_pt_rms=n_sigma_pt_rms, nsigma_max_deriv=n_sigma_max_deriv)
    data.compute_noise()
    data.compute_ats_filter(shift1=False, optimize_dp_dt=optimize_dp_dt, f_3db=f_3db_ats)
    hdf5_filename = data.pulse_model_to_hdf5(h5, n_basis=n_basis, maximum_n_pulses=maximum_n_pulses,
                                             extra_n_basis_5lag=extra_n_basis_5lag, noise_weight_basis=noise_weight_basis,
                                             f_3db_5lag=f_3db_5lag, _rethrow=True)

    return data.n_good_channels(), data.n_channels


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
                        help="output filename (should be .hdf5), the default behavior will place it in the same directory as pulse_path",
                        default=None, type=str)
    parser.add_argument("-r", "--replace_output",
                        help="pass this to overwrite off files with the same path", action="store_true", default=False)
    parser.add_argument("-m", "--max_channels",
                        help="stop after processing this many channels", default=2**31, type=int)
    parser.add_argument("--f_3db_ats",
                        help="cutoff frequency for ats filter", default=None, type=float)
    parser.add_argument("--f_3db_5lag",
                        help="cutoff frequency for 5lag filter", default=None, type=float)
    parser.add_argument("--n_ignore_presamples",
                        help="ignore this many presample before the rising edge when calculating pretrigger_mean", default=3, type=int)
    parser.add_argument("--n_sigma_pt_rms", type=float, default=10000,
                        help="passed to autocuts to determine pulses used to generate pulse model, the default large value basically disables this cut, which causes the projectors to be able to model the non-flat pretrigger region better")
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
    parser.add_argument("--mass_hdf5_noise_path", default=None,
                        help="specify the path for the mass noise hdf5 file that is generated a as a byproduct of this script")
    parser.add_argument("-i", "--invert_data", action="store_true",
                        help="pass this flag for downward going pulses (eg RAVEN)")
    parser.add_argument("--dont_optimize_dp_dt", help="simpler derivative like calculating, better for gamma data",
                        action="store_true")
    parser.add_argument("--extra_n_basis_5lag", help="use this many basis components to improve 5 lag filter representation",
                        default=0, type=int)
    parser.add_argument("--noise_weight_basis", help="pass false to not apply noise weighting when making projectors",
                        default=False, type=bool)
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
        channums = channums[:args.max_channels]
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
        n_good, n = make_projectors(pulse_files=pulse_files, noise_files=noise_files, h5=h5,
                                    n_sigma_pt_rms=args.n_sigma_pt_rms, n_sigma_max_deriv=args.n_sigma_max_deriv,
                                    n_basis=args.n_basis, maximum_n_pulses=args.maximum_n_pulses, mass_hdf5_path=args.mass_hdf5_path,
                                    mass_hdf5_noise_path=args.mass_hdf5_noise_path,
                                    invert_data=args.invert_data, optimize_dp_dt=not args.dont_optimize_dp_dt,
                                    extra_n_basis_5lag=args.extra_n_basis_5lag,
                                    f_3db_ats=args.f_3db_ats, f_3db_5lag=args.f_3db_5lag, noise_weight_basis=True)
    if not args.silent:
        if n_good == 0:
            print(f"all channels bad, could be because you need -i for inverted pulses")
        print(f"made projectors for {n_good} of {n} channels")
        print(f"written to {args.output_path}")
