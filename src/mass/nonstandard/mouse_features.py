# Here are 2 functions that used to be methods of TESGroup.
# They let you click on a series of per-channel histograms and return
# the x-value of where you clicked. Clever, but not in common use.

    def find_features_with_mouse(self, channame='p_filt_value', nclicks=1, prange=None, trange=None):
        """
        Plot histograms of each channel's "energy" spectrum, one channel at a time.
        After recording the x-coordinate of <nclicks> mouse clicks per plot, return an
        array of shape (N_channels, N_click) containing the "energy" of each click.

        <channame>  A string to choose the desired energy-like parameter.  Probably you want
                    to start with p_filt_value or p_filt_value_dc and later (once an energy
                    calibration is in place) p_energy.
        <nclicks>   The number of x coordinates to record per detector.  If you want to get
                    for example, a K-alpha and K-beta line in one go, then choose 2.
        <prange>    A 2-element sequence giving the limits to histogram.  If None, then the
                    histogram will show all data.
        <trange>    A 2-element sequence giving the time limits to use (in sec).  If None, then the
                    histogram will show all data.

        Returns:
        A np.ndarray of shape (self.n_channels, nclicks).
        """
        x = []
        for i, ds in enumerate(self.datasets):
            plt.clf()
            g = ds.cuts.good()
            if trange is not None:
                g = np.logical_and(g, ds.p_timestamp > trange[0])
                g = np.logical_and(g, ds.p_timestamp < trange[1])
            plt.hist(ds.__dict__[channame][g], 200, range=prange)
            plt.xlabel(channame)
            plt.title("Detector %d: attribute %s" % (i, channame))
            fig = plt.gcf()
            pf = mass.core.utilities.MouseClickReader(fig)

            for j in range(nclicks):
                while True:
                    plt.waitforbuttonpress()
                    try:
                        pfx = '%g' % pf.x
                    except TypeError:
                        continue
                    print('Click on line #%d at %s' % (j + 1, pfx))
                    x.append(pf.x)
                    break

        xvalues = np.array(x)
        xvalues.shape = (self.n_channels, nclicks)
        return xvalues

    def find_named_features_with_mouse(self, name='Mn Ka1', channame='p_filt_value',
                                       prange=None, trange=None, energy=None):

        if energy is None:
            energy = mass.calibration.energy_calibration.STANDARD_FEATURES[name]

        print("Please click with the mouse on each channel's histogram at the %s line" % name)
        xvalues = self.find_features_with_mouse(channame=channame, nclicks=1,
                                                prange=prange, trange=trange).ravel()
        for ds, xval in zip(self.datasets, xvalues):
            calibration = ds.calibration[channame]
            calibration.add_cal_point(xval, energy, name)
