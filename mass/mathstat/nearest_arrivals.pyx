from numpy.math cimport INFINITY
from libc.stdint cimport uint32_t, int64_t
cimport cython
cimport numpy as cnp
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef nearest_arrivals_float(double[:] pulse_timestamps, double[:] external_trigger_timestamps):
    cdef:
        Py_ssize_t num_pulses, num_triggers
        Py_ssize_t i = 0, j = 0, t
        double[:] delay_from_trigger
        double a = -INFINITY, b, pt

    num_pulses = pulse_timestamps.shape[0]
    num_triggers = external_trigger_timestamps.shape[0]

    if num_pulses < 1:
        return np.array([],dtype=np.float64)

    delay_from_trigger = np.zeros_like(pulse_timestamps, dtype=np.float64)

    if num_triggers > 0:
        b = external_trigger_timestamps[0]
    else:
        b = INFINITY

    while True:
        pt = pulse_timestamps[i]
        if pt < b:
            delay_from_trigger[i] = pulse_timestamps[i] - a
            i += 1
            if i >= num_pulses:
                break
        else:
            j += 1
            if j >= num_triggers:
                for t in range(i, num_pulses):
                    delay_from_trigger[t] = pulse_timestamps[t] - b
                break
            else:
                a, b = b, external_trigger_timestamps[j]

    return np.asarray(delay_from_trigger)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef nearest_arrivals(int64_t[:] pulse_timestamps, int64_t[:] external_trigger_timestamps):
    cdef:
        Py_ssize_t num_pulses, num_triggers
        Py_ssize_t i = 0, j = 0, t
        int64_t[:] delay_from_trigger
        int64_t a = np.iinfo(np.int64).min, b, pt

    num_pulses = pulse_timestamps.shape[0]
    num_triggers = external_trigger_timestamps.shape[0]

    if num_pulses < 1:
        return np.array([],dtype=np.int64)

    delay_from_trigger = np.zeros_like(pulse_timestamps, dtype=np.int64)

    if num_triggers > 0:
        b = external_trigger_timestamps[0]
    else:
        b = np.iinfo(np.int64).max

    while True:
        pt = pulse_timestamps[i]
        if pt < b:
            delay_from_trigger[i] = pulse_timestamps[i] - a
            i += 1
            if i >= num_pulses:
                break
        else:
            j += 1
            if j >= num_triggers:
                for t in range(i, num_pulses):
                    delay_from_trigger[t] = pulse_timestamps[t] - b
                break
            else:
                a, b = b, external_trigger_timestamps[j]

    return np.asarray(delay_from_trigger,dtype=np.int64)