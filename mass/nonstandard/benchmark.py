"""
benchmark.py

This is an experiment: July 11, 2023.
Just curious how to use Numba, and also how our Cython vs Python versions of
`summarize_data()` stack up in speed.

Save this just as a starting point for future explorations.
"""

from numba import njit
# import numpy as np
import mass


def python_summarize(data):
    for _ in range(1):
        data.summarize_data(forceNew=True)
    ds = data.datasets[0]
    print(ds.p_pretrig_mean[:5], ds.p_pretrig_mean[-5:])


def test_numba(benchmark):
    data = mass.TESGroup("tests/off/data_for_test/20181018_144520/20181018_144520_chan*.ljh")
    data.set_all_chan_good()
    benchmark(python_summarize, data)


def NOT_A_test_python2(benchmark):
    data = mass.TESGroup("tests/off/data_for_test/20181018_144520/20181018_144520_chan*.ljh")
    data.set_all_chan_good()
    benchmark(summarize_v1, data)


def summarize_v1(data):
    for ds in data:
        # Don't look for retriggers before this # of samples. Use the most common
        # value of the peak index in the currently-loaded segment.
        if ds.peak_samplenumber is None:
            ds._compute_peak_samplenumber()

        ds.p_timestamp[:] = ds.times[:]
        ds.p_subframecount[:] = ds.subframecount[:]
        for idx in range(ds.nPulses):
            ptm, ptr = analyze_pretrig(ds.data[idx, ds.cut_pre:ds.nPresamples-ds.pretrigger_ignore_samples])
            ds.p_pretrig_mean[idx] = ptm
            ds.p_pretrig_rms[idx] = ptr


@njit
def analyze_pretrig(data):
    ptm = data.mean()
    ptr = data.std()
    return ptm, ptr


# def _summarize_data_segment(self, idx_range, doPretrigFit=False):
#     """Summarize one segment of the data file, loading it into cache."""
#     first, end = idx_range
#     if first >= self.nPulses:
#         return
#     if end > self.nPulses:
#         end = self.nPulses

#     # Don't look for retriggers before this # of samples. Use the most common
#     # value of the peak index in the currently-loaded segment.
#     if self.peak_samplenumber is None:
#         self._compute_peak_samplenumber()

#     seg_size = end-first

#     # Fit line to pretrigger and save the derivative and offset
#     if doPretrigFit:
#         presampleNumbers = np.arange(self.cut_pre, self.nPresamples
#                                         - self.pretrigger_ignore_samples)
#         ydata = self.data[first:end, self.cut_pre:self.nPresamples
#                             - self.pretrigger_ignore_samples].T
#         self.p_pretrig_deriv[first:end], self.p_pretrig_offset[first:end] = \
#             np.polyfit(presampleNumbers, ydata, deg=1)

#     self.p_peak_index[first:end] = self.data[first:end,
#                                                 self.cut_pre:self.nSamples-self.cut_post].argmax(axis=1)+self.cut_pre
#     self.p_peak_value[first:end] = self.data[first:end,
#                                                 self.cut_pre:self.nSamples-self.cut_post].max(axis=1)
#     self.p_min_value[first:end] = self.data[first:end,
#                                             self.cut_pre:self.nSamples-self.cut_post].min(axis=1)
#     self.p_pulse_average[first:end] = self.data[first:end,
#                                                 self.nPresamples:self.nSamples-self.cut_post].mean(axis=1)

#     # Remove the pretrigger mean from the peak value and the pulse average figures.
#     ptm = self.p_pretrig_mean[first:end]
#     self.p_pulse_average[first:end] -= ptm
#     self.p_peak_value[first:end] -= np.asarray(ptm, dtype=self.p_peak_value.dtype)
#     self.p_pulse_rms[first:end] = np.sqrt(
#         (self.data[first:end, self.nPresamples:self.nSamples-self.cut_post]**2.0).mean(axis=1)
#         - ptm*(ptm + 2*self.p_pulse_average[first:end]))

#     shift1 = (self.data[first:end, self.nPresamples-1]-ptm
#                 > 4.3*self.p_pretrig_rms[first:end])
#     self.p_shift1[first:end] = shift1

#     halfidx = (self.nPresamples+2+self.peak_samplenumber)//2
#     pkval = self.p_peak_value[first:end]
#     prompt = (self.data[first:end, self.nPresamples+2:halfidx].mean(axis=1)
#                 - ptm) / pkval
#     prompt[shift1] = (self.data[first:end, self.nPresamples+1:halfidx-1][shift1, :].mean(axis=1)
#                         - ptm[shift1]) / pkval[shift1]
#     self.p_promptness[first:end] = prompt

#     self.p_rise_time[first:end] = \
#         mass.core.analysis_algorithms.estimateRiseTime(self.data[first:end, self.cut_pre:self.nSamples-self.cut_post],
#                                                         timebase=self.timebase,
#                                                         nPretrig=self.nPresamples-self.cut_pre)

#     self.p_postpeak_deriv[first:end] = \
#         mass.core.analysis_algorithms.compute_max_deriv(self.data[first:end, self.cut_pre:self.nSamples-self.cut_post],
#                                                         ignore_leading=self.peak_samplenumber-self.cut_pre)
