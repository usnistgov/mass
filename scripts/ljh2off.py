#!/usr/bin/env python

import mass
import argparse

print("starting ljh2off")
args = mass.ljh2off.parse_args(fake=False)
for k in sorted(vars(args).keys()):
    print("{}: {}".format(k, vars(args)[k]))
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
elif not args.replace_output:
    print("dir {} exists, pass --replace_output to write into it anyway".format(args.output_dir))
    sys.exit()
ljh_filenames, off_filenames = mass.ljh2off.ljh2off_loop(args.ljh_path, args.h5_path, args.output_dir, args.max_channels, args.n_ignore_presamples)
print("full path to first off file:")
print(os.path.abspath(off_filenames[0]))
            
