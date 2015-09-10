cimport cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef nearest_arrivals(long long[:] pulse_timestamps, long long[:] external_trigger_timestamps):
    cdef:
        Py_ssize_t num_pulses, num_triggers
        Py_ssize_t i = 0, j = 0, t
        long long[:] delay_from_last_trigger
        long long[:] delay_until_next_trigger
        long long a = np.iinfo(np.int64).min, b, pt

    num_pulses = pulse_timestamps.shape[0]
    num_triggers = external_trigger_timestamps.shape[0]

    if num_pulses < 1:
        return np.array([],dtype=np.int64)

    delay_from_last_trigger = np.zeros_like(pulse_timestamps, dtype=np.int64)
    delay_until_next_trigger = np.zeros_like(pulse_timestamps, dtype=np.int64)

    if num_triggers > 0:
        b = external_trigger_timestamps[0]
    else:
        b = np.iinfo(np.int64).max

    while True:
        pt = pulse_timestamps[i]
        if pt < b:
            if b > pulse_timestamps[0]:
                # at this point in the code a and b are values from external_trigger_timestamps that bracket pulse_timestamp[i]
                delay_from_last_trigger[i] = pulse_timestamps[i] - a
                delay_until_next_trigger[i] = b - pulse_timestamps[i]
            else:
                # handle the case where pulses arrive before the fist external trigger
                delay_from_last_trigger[i] = np.iinfo(np.int64).max
                delay_until_next_trigger[i] = b - pulse_timestamps[i]
            i += 1
            if i >= num_pulses:
                break
        else:
            j += 1
            if j >= num_triggers:
                for t in range(i, num_pulses):
                    # handle the case where pulses arrive after the last external trigger
                    delay_from_last_trigger[t] = pulse_timestamps[t] - b
                    delay_until_next_trigger[t] = np.iinfo(np.int64).max
                break
            else:
                a, b = b, external_trigger_timestamps[j]

    return np.asarray(delay_from_last_trigger,dtype=np.int64), np.asarray(delay_until_next_trigger, dtype=np.int64)