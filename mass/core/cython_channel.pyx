"""
Contains the class CythonMicrocalDataSet, which subclasses MicrocalDataSet to
add a much faster, Cython version of .summarize_data_segment().
"""

import numpy as np

from libc.math cimport sqrt
cimport cython
cimport numpy as np
cimport libc.limits


@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def summarize_data_cython(
    const unsigned short[:,:] rawdata,
    double timebase,
    long peak_samplenumber,
    long pretrigger_ignore,
    long nPresamples,
    long first=0,
    long end=0,
):
    """Summarize one segment of the data file, loading it into cache."""
    cdef:
        Py_ssize_t j, k
        const unsigned short[:] pulse

        float[::1] p_pretrig_mean_array,
        float[::1] p_pretrig_rms_array,
        float[::1] p_pulse_average_array,
        float[::1] p_pulse_rms_array,
        float[::1] p_promptness_array,
        float[::1] p_rise_times_array,
        float[::1] p_postpeak_deriv_array,
        unsigned short[::1] p_peak_index_array,
        unsigned short[::1] p_peak_value_array,
        unsigned short[::1] p_min_value_array,
        unsigned short[::1] p_shift1_array,

        double pretrig_sum, pretrig_rms_sum
        double pulse_sum, pulse_rms_sum
        double promptness_sum
        double ptm
        unsigned short peak_value, peak_index, min_value
        unsigned short signal
        unsigned short nPulses, nSamples, seg_size
        unsigned short e_nPresamples, s_prompt, e_prompt

        unsigned short low_th, high_th
        unsigned short high_value
        unsigned short low_idx, high_idx

        long f0 = 2, f1 = 1, f3 = -1, f4 = -2
        long s0, s1, s2, s3, s4
        long t0, t1, t2, t3, t_max_deriv
        dict results

    nPulses = rawdata.shape[0]
    nSamples = rawdata.shape[1]

    if end <= 0:
        end = nPulses
    seg_size = end-first

    e_nPresamples = nPresamples - pretrigger_ignore

    # Buffers for a single segment calculation.
    p_pretrig_mean_array = np.empty(seg_size, dtype=np.float32)
    p_pretrig_rms_array = np.empty(seg_size, dtype=np.float32)
    p_pulse_average_array = np.empty(seg_size, dtype=np.float32)
    p_pulse_rms_array = np.empty(seg_size, dtype=np.float32)
    p_promptness_array = np.empty(seg_size, dtype=np.float32)
    p_rise_times_array = np.empty(seg_size, dtype=np.float32)
    p_postpeak_deriv_array = np.empty(seg_size, dtype=np.float32)
    p_peak_index_array = np.empty(seg_size, dtype=np.uint16)
    p_peak_value_array = np.empty(seg_size, dtype=np.uint16)
    p_min_value_array = np.empty(seg_size, dtype=np.uint16)
    p_shift1_array = np.empty(seg_size, dtype=np.uint16)
    results = {}
    results["pretrig_mean"] = p_pretrig_mean_array
    results["pretrig_rms"] = p_pretrig_rms_array
    results["pulse_average"] = p_pulse_average_array
    results["pulse_rms"] = p_pulse_rms_array
    results["promptness"] = p_promptness_array
    results["rise_times"] = p_rise_times_array
    results["postpeak_deriv"] = p_postpeak_deriv_array
    results["peak_index"] = p_peak_index_array
    results["peak_value"] = p_peak_value_array
    results["min_value"] = p_min_value_array
    results["shift1"] = p_shift1_array

    for j in range(seg_size):
        pulse = rawdata[j+first, :]
        pretrig_sum = 0.0
        pretrig_rms_sum = 0.0
        pulse_sum = 0.0
        pulse_rms_sum = 0.0
        promptness_sum = 0.0
        peak_value = 0
        peak_index = 0
        min_value = libc.limits.USHRT_MAX
        # Reset s_ and e_prompt for each pulse, b/c they can be shifted
        # for individual pulses
        s_prompt = nPresamples + 2
        e_prompt = nPresamples + 8

        # Memory access (pulse[k]) is expensive.
        # So calculate several quantities with a single memory access.
        for k in range(0, nSamples):
            signal = pulse[k]

            if signal > peak_value:
                peak_value = signal
                peak_index = k
            if signal < min_value:
                min_value = signal

            if k < e_nPresamples:
                pretrig_sum += signal
                pretrig_rms_sum += (<double>signal)**2

            if s_prompt <= k and k < e_prompt:
                promptness_sum += signal

            if k == nPresamples - 1:
                ptm = pretrig_sum / e_nPresamples
                ptrms = sqrt(pretrig_rms_sum / e_nPresamples - ptm**2)
                if signal - ptm > 4.3 * ptrms:
                    e_prompt -= 1
                    s_prompt -= 1
                    p_shift1_array[j] = <unsigned short>True
                else:
                    p_shift1_array[j] = <unsigned short>False

            if k >= nPresamples - 1:
                pulse_sum += signal
                pulse_rms_sum += (<double>signal)**2

        p_pretrig_mean_array[j] = <float>ptm
        p_pretrig_rms_array[j] = <float>ptrms
        if ptm < peak_value:
            peak_value -= <unsigned short>(ptm+0.5)
            p_promptness_array[j] = <float>((promptness_sum / 6.0 - ptm) / peak_value)
            p_peak_value_array[j] = <unsigned short>peak_value
            p_peak_index_array[j] = <unsigned short>peak_index
        else:
            # Basically a nonsense pulse: the pretrigger mean exceeds the highest post-trigger value.
            # This would normally happen only if the crate's re-lock mechanism fires during a record.
            p_promptness_array[j] = <float>0.0
            p_peak_value_array[j] = <unsigned short>0
            p_peak_index_array[j] = <unsigned short>0
        p_min_value_array[j] = min_value
        pulse_avg = pulse_sum / (nSamples - nPresamples + 1) - ptm
        p_pulse_average_array[j] = <float>pulse_avg
        p_pulse_rms_array[j] = <float>sqrt(pulse_rms_sum / (nSamples - nPresamples + 1) -
                                            ptm*pulse_avg*2 - ptm**2)

        # Estimating a rise time.
        # Beware! peak_value here has already had the pretrigger mean (ptm) subtracted!
        low_th = <unsigned short>(0.1 * peak_value + ptm)
        high_th = <unsigned short>(0.9 * peak_value + ptm)

        k = nPresamples
        low_value = high_value = pulse[k]
        while k < nSamples:
            signal = pulse[k]
            if signal > low_th:
                low_idx = k
                low_value = signal
                break
            k += 1

        high_value = low_value
        high_idx = low_idx

        while k < nSamples:
            signal = pulse[k]
            if signal > high_th:
                high_idx = k - 1
                high_value = pulse[high_idx]
                break
            k += 1

        if high_value > low_value:
            p_rise_times_array[j] = <float>(timebase * (high_idx - low_idx) *
                                            (<double>peak_value) / (high_value - low_value))
        else:
            p_rise_times_array[j] = <float> timebase

        # Calculating the postpeak_deriv with a simple kernel
        # (f0, f1, f2 = 0, f3, f4) and spike_reject on.
        s0 = pulse[peak_samplenumber]
        s1 = pulse[peak_samplenumber + 1]
        s2 = pulse[peak_samplenumber + 2]
        s3 = pulse[peak_samplenumber + 3]
        s4 = pulse[peak_samplenumber + 4]
        t0 = f4 * s0 + f3 * s1 + f1 * s3 + f0 * s4
        s0, s1, s2, s3 = s1, s2, s3, s4
        s4 = pulse[peak_samplenumber + 5]
        t1 = f4 * s0 + f3 * s1 + f1 * s3 + f0 * s4
        t_max_deriv = libc.limits.LONG_MIN

        for k in range(peak_samplenumber + 6, nSamples):
            s0, s1, s2, s3 = s1, s2, s3, s4
            s4 = pulse[k]
            t2 = f4 * s0 + f3 * s1 + f1 * s3 + f0 * s4

            t3 = t2 if t2 < t0 else t0
            if t3 > t_max_deriv:
                t_max_deriv = t3

            t0, t1 = t1, t2

        p_postpeak_deriv_array[j] = <float>(0.1 * t_max_deriv)

    return results