# Regression Tests

There is a set of regression tests in `test_regression.py` which will fail if the output of a script is changed. This change could reflect either an improvement or worse results, but at least we will know the change happened. These tests use `regress_chan1.ljh` and `regress_noise_chan1.ljh` as input files. Any other tests that require LJH files to exist, but don't care a lot about the contents, can use these as input files as well.

## Reference Data

The `test_regression.py` compares the output of `test_regression.process_file` to stored data. The data is stored in `regress_ds0.npz`. If you think this data should be updated, you may use the following steps:
1. Checkout code that you trust to be bug-free (hah!).
2. Run `update_regress_reference_data.py`
  * You may need to copy this if the code you checked is from before this file was committed.

#### Updating the LJH and NOI files
Consider updating the ljh and noi file if you have specific data you need access to. Don't do this often, it makes the mass repository a lot bigger.

The data now in regress_chan1.* are data from the Advanced Photon Source, dated August 13, 2015, from noise file named BA and a data file named `BB_8x16_600eV_100um_chan101.ljh`
