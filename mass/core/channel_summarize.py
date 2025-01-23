import numpy as np
from numba import njit
from numpy.typing import NDArray

# Define the dtype for the structured array
result_dtype = np.dtype([
    ('pretrig_mean', np.float32),
    ('pretrig_rms', np.float32),
    ('pulse_average', np.float32),
    ('pulse_rms', np.float32),
    ('promptness', np.float32),
    ('rise_times', np.float32),
    ('postpeak_deriv', np.float32),
    ('peak_index', np.uint16),
    ('peak_value', np.uint16),
    ('min_value', np.uint16),
    ('shift1', np.uint16)
])

# Create a type alias for the structured array
ResultArrayType = NDArray[result_dtype]


@njit
def summarize_data_numba(  # noqa: PLR0914
    rawdata: NDArray[np.uint16],
    timebase: float,
    peak_samplenumber: int,
    pretrigger_ignore: int,
    nPresamples: int,
) -> ResultArrayType:
    """Summarize one segment of the data file, loading it into cache."""
    nPulses = rawdata.shape[0]
    nSamples = rawdata.shape[1]

    e_nPresamples = nPresamples - pretrigger_ignore

    # Create the structured array for results
    results = np.zeros(nPulses, dtype=result_dtype)

    for j in range(nPulses):
        pulse = rawdata[j, :]
        pretrig_sum = 0.0
        pretrig_rms_sum = 0.0
        pulse_sum = 0.0
        pulse_rms_sum = 0.0
        promptness_sum = 0.0
        peak_value = 0
        peak_index = 0
        min_value = np.iinfo(np.uint16).max
        s_prompt = nPresamples + 2
        e_prompt = nPresamples + 8

        for k in range(nSamples):
            signal = pulse[k]

            if signal > peak_value:
                peak_value = signal
                peak_index = k
            min_value = min(signal, min_value)

            if k < e_nPresamples:
                pretrig_sum += signal
                pretrig_rms_sum += signal ** 2

            if s_prompt <= k < e_prompt:
                promptness_sum += signal

            if k == nPresamples - 1:
                ptm = pretrig_sum / e_nPresamples
                ptrms = np.sqrt(pretrig_rms_sum / e_nPresamples - ptm ** 2)
                if signal - ptm > 4.3 * ptrms:
                    e_prompt -= 1
                    s_prompt -= 1
                    results['shift1'][j] = 1
                else:
                    results['shift1'][j] = 0

            if k >= nPresamples - 1:
                pulse_sum += signal
                pulse_rms_sum += signal ** 2

        results['pretrig_mean'][j] = ptm
        results['pretrig_rms'][j] = ptrms
        if ptm < peak_value:
            peak_value -= int(ptm + 0.5)
            results['promptness'][j] = (promptness_sum / 6.0 - ptm) / peak_value
            results['peak_value'][j] = peak_value
            results['peak_index'][j] = peak_index
        else:
            results['promptness'][j] = 0.0
            results['peak_value'][j] = 0
            results['peak_index'][j] = 0
        results['min_value'][j] = min_value
        pulse_avg = pulse_sum / (nSamples - nPresamples + 1) - ptm
        results['pulse_average'][j] = pulse_avg
        results['pulse_rms'][j] = np.sqrt(pulse_rms_sum / (nSamples - nPresamples + 1) - ptm * pulse_avg * 2 - ptm ** 2)

        low_th = int(0.1 * peak_value + ptm)
        high_th = int(0.9 * peak_value + ptm)

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
            results['rise_times'][j] = timebase * (high_idx - low_idx) * peak_value / (high_value - low_value)
        else:
            results['rise_times'][j] = timebase

        f0, f1, f3, f4 = 2, 1, -1, -2
        s0, s1, s2, s3 = pulse[peak_samplenumber], pulse[peak_samplenumber +
                                                         1], pulse[peak_samplenumber + 2], pulse[peak_samplenumber + 3]
        s4 = pulse[peak_samplenumber + 4]
        t0 = f4 * s0 + f3 * s1 + f1 * s3 + f0 * s4
        s0, s1, s2, s3 = s1, s2, s3, s4
        s4 = pulse[peak_samplenumber + 5]
        t1 = f4 * s0 + f3 * s1 + f1 * s3 + f0 * s4
        t_max_deriv = np.iinfo(np.int32).min

        for k in range(peak_samplenumber + 6, nSamples):
            s0, s1, s2, s3 = s1, s2, s3, s4
            s4 = pulse[k]
            t2 = f4 * s0 + f3 * s1 + f1 * s3 + f0 * s4

            t3 = min(t2, t0)
            t_max_deriv = max(t3, t_max_deriv)

            t0, t1 = t1, t2

        results['postpeak_deriv'][j] = 0.1 * t_max_deriv

    return results
