import pytest
import time

import mass


def python_summarize(data, use_cython):
    for _ in range(1):
        data.summarize_data(forceNew=True, use_cython=use_cython)
    ds = data.datasets[0]
    print(ds.p_pretrig_mean[:5], ds.p_pretrig_mean[-5:])


def test_python(benchmark):
    # data = mass.TESGroup("tests/off/data_for_test/20181018_144520/20181018_144520_chan*.ljh")
    data = mass.TESGroup("/Users/fowlerj/data/Gamma/20221118/0005/20221118_run0005_chan*.ljh")
    data.set_all_chan_good()

    resultPy = benchmark(python_summarize, data, False)


def test_cython(benchmark):
    # data = mass.TESGroup("tests/off/data_for_test/20181018_144520/20181018_144520_chan*.ljh")
    data = mass.TESGroup("/Users/fowlerj/data/Gamma/20221118/0005/20221118_run0005_chan*.ljh")
    data.set_all_chan_good()

    resultCy = benchmark(python_summarize, data, True)
