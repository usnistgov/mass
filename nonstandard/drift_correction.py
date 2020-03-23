# Was previously part of MicrocalDataSet.
# But it was never made to work correctly!

def time_drift_correct_polynomial(self, poly_order=2, attr='p_filt_value_phc', num_lines=None, forceNew=False):
    """Assumes the gain is a polynomial in time.
    Estimates that polynomial by fitting a polynomial to each line in the calibration with
    the same name as the attribute and taking an appropriate average of the polyonomials
    from each line weighted by the counts in each line.
    """
    if not hasattr(self, 'p_filt_value_tdc') or forceNew:
        LOG.info("chan %d doing time_drift_correct_polynomail with order %d" % (self.channum, poly_order))
        cal = self.calibration[attr]
        attr = getattr(self, attr)
        attr_good = attr[self.cuts.good()]

        if num_lines is None:
            num_lines = len(cal.elements)

        t0 = np.median(self.p_timestamp)
        counts = [h[0].sum() for h in cal.histograms]
        pfits = []
        counts = [h[0].sum() for h in cal.histograms]
        for i in np.argsort(counts)[-1:-num_lines-1:-1]:
            line_name = cal.elements[i]
            low, high = cal.histograms[i][1][[0, -1]]
            use = np.logical_and(attr_good > low, attr_good < high)
            use_time = self.p_timestamp[self.cuts.good()][use]-t0
            pfit = np.polyfit(use_time, attr_good[use], poly_order)
            pfits.append(pfit)
        pfits = np.array(pfits)

        pfits_slope = np.average(pfits/np.repeat(np.array(pfits[:, -1], ndmin=2).T,
                                                 pfits.shape[-1], 1),
                                 axis=0, weights=np.array(sorted(counts))[-1:-num_lines-1:-1])

        p_corrector = pfits_slope.copy()
        p_corrector[:-1] *= -1
        corrected = attr*np.polyval(p_corrector, self.p_timestamp-t0)
        self.p_filt_value_tdc[:] = corrected

        new_info = {'poly_gain': p_corrector, 't0': t0, 'type': 'time_gain_polynomial'}
    else:
        LOG.info("chan %d skipping time_drift_correct_polynomial_dataset" % self.channum)
        corrected, new_info = self.p_filt_value_tdc, {}
    return corrected, new_info
