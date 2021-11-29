import numpy as np

""" code for doing 5 lag filtering like used for gamma ray data"""


def calc_5lag_fit_matrix(filter_5lag, basis):
    filter_5lag_in_basis = np.zeros((5, basis.shape[1]))
    for j in range(5):
        if j == 4:
            filter_5lag_in_basis[j, :] = np.matmul(filter_5lag, basis[4:, :])
        else:
            filter_5lag_in_basis[j, :] = np.matmul(filter_5lag, basis[j:j-4, :])
    # These parameters fit a parabola to any 5 evenly-spaced points
    fit_array = np.array((
        (-6, 24, 34, 24, -6),
        (-14, -7, 0, 7, 14),
        (10, -5, -10, -5, 10)), dtype=float)/70.0
    filter_5lag_fit_in_basis = np.matmul(fit_array, filter_5lag_in_basis).T
    return filter_5lag_in_basis, filter_5lag_fit_in_basis


def filtValue5Lag(cba):
    c, b, a = cba.T
    peak_y = c - 0.25 * b**2 / a
    return peak_y


def peakX5Lag(cba):
    c, b, a = cba.T
    peak_x = -0.5 * b / a
    return peak_x
