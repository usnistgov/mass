cimport cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def nearest_arrivals(long long[:] pulse_timestamps, long long[:] external_trigger_timestamps):
    cdef:
        Py_ssize_t num_pulses, num_triggers
        Py_ssize_t i = 0, j = 0, t
        long long[:] delay_from_last_trigger
        long long[:] delay_until_next_trigger
        long long a, b, pt
        long long max_value

    num_pulses = pulse_timestamps.shape[0]
    num_triggers = external_trigger_timestamps.shape[0]

    if num_pulses < 1:
        return np.array([],dtype=np.int64)

    delay_from_last_trigger = np.zeros_like(pulse_timestamps, dtype=np.int64)
    delay_until_next_trigger = np.zeros_like(pulse_timestamps, dtype=np.int64)

    max_value = np.iinfo(np.int64).max

    if num_triggers > 1:
        a = external_trigger_timestamps[0]
        b = external_trigger_timestamps[1]
        j = 1

        # handle the case where pulses arrive before the fist external trigger
        while True:
            pt = pulse_timestamps[i]
            if pt < a:
                delay_from_last_trigger[i] = max_value
                delay_until_next_trigger[i] = a - pt
                i += 1
            else:
                break

        # at this point in the code a and b are values from external_trigger_timestamps that bracket pulse_timestamp[i]
        while True:
            pt = pulse_timestamps[i]
            if pt < b:
                delay_from_last_trigger[i] = pt - a
                delay_until_next_trigger[i] = b - pt
                i += 1
                if i >= num_pulses:
                    break
            else:
                j += 1
                if j >= num_triggers:
                    break
                else:
                    a, b = b, external_trigger_timestamps[j]

        # handle the case where pulses arrive after the last external trigger
        for t in range(i, num_pulses):
            delay_from_last_trigger[t] = pulse_timestamps[t] - b
            delay_until_next_trigger[t] = max_value
    elif num_triggers > 0:
        a = b = external_trigger_timestamps[0]

        for i in range(num_pulses):
            pt = pulse_timestamps[i]
            if pt > a:
                delay_from_last_trigger[i] = pt - a
                delay_until_next_trigger[i] = max_value
            else:
                delay_from_last_trigger[i] = max_value
                dealay_until_next_trigger = a - pt
    else:
        for i in range(num_pulses):
            delay_from_last_trigger[i] = max_value
            delay_until_next_trigger[i] = max_value

    return np.asarray(delay_from_last_trigger,dtype=np.int64), np.asarray(delay_until_next_trigger, dtype=np.int64)