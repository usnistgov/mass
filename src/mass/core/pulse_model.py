import pylab as plt

class PulseModel():
    version = 1
    def __init__(self, projectors_so_far, basis_so_far, n_basis, pulses_for_svd, v_dv, pretrig_rms_median, pretrig_rms_sigma):
        self.pulses_for_svd = pulses_for_svd
        self.n_basis = n_basis
        if projectors_so_far.shape[0] < n_basis:
            self.projectors, self.basis = self._additional_projectors_tsvd(projectors_so_far, basis_so_far, n_basis, pulses_for_svd)
        elif projectors_so_far.shape[0] == n_basis:
            self.projectors, self.basis = projectors_so_far, basis_so_far
        else:
            raise Exception("n_basis={} < projectors_so_far.shape[0] = {}".format(n_basis, projectors_so_far.shape[0]))
        self.v_dv = v_dv
        self.pretrig_rms_median = pretrig_rms_median
        self.pretrig_rms_sigma = pretrig_rms_sigma

    def toHDF5(self, hdf5_group, save_inverted):
        projectors, basis = self.projectors[()], self.basis[()]
        if save_inverted:
            # flip every component except the mean component if data is being inverted
            basis[:, 1:] *= -1
            projectors[1:, :] *= -1

        # projectors is MxN, where N is samples/record and M the number of basis elements
        # basis is NxM
        hdf5_group["svdbasis/projectors"] = projectors
        hdf5_group["svdbasis/basis"] = basis
        hdf5_group["svdbasis/v_dv"] = self.v_dv
        hdf5_group["svdbasis/training_pulses_for_plots"] = self.pulses_for_svd
        hdf5_group["svdbasis/was_saved_inverted"] = save_inverted
        hdf5_group["svdbasis/pretrig_rms_median"] = self.pretrig_rms_median
        hdf5_group["svdbasis/pretrig_rms_sigma"] = self.pretrig_rms_sigma
        hdf5_group["svdbasis/version"] = self.version

    @classmethod
    def fromHDF5(self, hdf5_group):
        projectors = hdf5_group["svdbasis/projectors"][()]
        n_basis = projectors.shape[0]
        basis = hdf5_group["svdbasis/basis"][()]
        v_dv = hdf5_group["svdbasis/v_dv"][()]
        pulses_for_svd = hdf5_group["svdbasis/training_pulses_for_plots"][()]
        pretrig_rms_median = hdf5_group["svdbasis/pretrig_rms_median"][()]
        pretrig_rms_sigma = hdf5_group["svdbasis/pretrig_rms_sigma"][()]
        version = hdf5_group["svdbasis/version"][()]
        if version != 1:
            raise Exception("loading not implemented for other versions")
        return PulseModel(projectors, basis, n_basis, pulses_for_svd, v_dv, pretrig_rms_median, pretrig_rms_sigma)

    def _additional_projectors_tsvd(self, projectors, basis, n_basis, pulses_for_svd):
        """
        Given an existing basis with projectors, compute a basis with n_basis elements
        by randomized SVD of the residual elements of the training data in pulses_for_svd.
        It should be the case that projectors.dot(basis) is approximately the identity matrix.

        It is assumed that the projectors will have been computed from the basis in some
        noise-optimal way, say, from optimal filtering. However, the additional basis elements
        will be computed from a standard (non-noise-weighted) SVD, and the additional projectors
        will be computed without noise optimization.

        The projectors and basis will be ordered as:
        mean, deriv_ike, pulse_like, any svd components...
        """

        # Check sanity of inputs
        n_samples, n_existing = basis.shape
        assert (n_existing, n_samples) == projectors.shape
        assert n_basis >= n_existing

        if n_basis == n_existing:
            return projectors, basis

        mpc = np.matmul(projectors, pulses_for_svd.T)  # modeled pulse coefs
        mp = np.matmul(basis, mpc)  # modeled pulse
        residuals = pulses_for_svd.T - mp
        Q = mass.mathstat.utilities.find_range_randomly(residuals, n_basis-3)

        projectors2 = np.linalg.pinv(Q)  # = Q.T, perhaps??
        projectors2 -= projectors2.dot(basis).dot(projectors)

        basis = np.hstack([basis, Q])
        projectors = np.vstack([projectors, projectors2])

        return projectors, basis


    def plot(self):
        labels = ["mean", "deriv", "pulse"]
        for i in range(self.n_basis-3):
            labels = ["svd{}".format(i)] + labels
        mpc = self.projectors.data(self.pulses_for_svd)
        mp = pulses_for_svd.dot(self.basis)
        residuals = self.pulses_for_svd-mp

        plt.figure(figsize=(10,12))
        plt.subplot(411)
        plt.plot(self.projectors[::-1,:].T)
        plt.title("projectors")
        plt.legend(labels)
        plt.subplot(412)
        plt.plot(self.basis[:,::-1])
        plt.title("basis")
        plt.legend(labels)
        plt.subplot(413)
        plt.plot(self.pulses_for_svd[:10,:])
        plt.title("from ljh")
        plt.subplot(414)
        plt.plot(residuals[:10,:])
        plt.title("residuals")


