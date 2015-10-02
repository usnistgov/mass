This directory contains the first pass at automated testing framework for the python portions of mass.

Currently we only do basic regression testing.
The idea is to compare the output of the current version of the codebase to the output of a past, "tuested" version.
Any discrepencies are assumed to be due to bugs (probably a bug that was newly introduced, but perhaps due to a bug that has been fixed).

The approach is currently very simple, and the best way to understand what is going on is to just read the code.
The basic outline is as follows.

1. Choose a representative subset of pulses, and put them into their own ljh file.
Also create a noise file.
The fuctions`ljh_copy_traces` and `ljh_append_traces` from `mass.core.files` can help with this.
A simple example is:
```
#!python
import mass.core.files as f
import numpy as np
f.ljh_copy_traces('20150828_163641_chan1.ljh', 'regress_chan1.ljh', np.arange(0,100))
f.ljh_copy_traces('20150829_110048_chan1.noi', 'regress_chan1.ljh', np.arange(0,100))
```

2. Checkout code that you trust to be bug-free (hah!).

3. Run data analysis using this "trusted" code on the ljh files, and save the results that you care about to a file. `save_test_data.py` does this for the above files. This expected data is saved to a .npz file using `numpy.savez()`.
Now you have a "trusted" analysis result that you can use in the future for regression testing.
3. We use python's `unittest` and `numpy.testing` modules for automating the testing. See code in `regression.py`.
