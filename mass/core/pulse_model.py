import pylab as plt
import numpy as np
import mass.mathstat

class PulseModel():
    """Object to hold a "pulse model", meaning a low-dimensional linear basis to express "all" pulses,
    along with a projector such that projector.dot(basis) is the identity matrix.

    Also has the capacity to store to and restore from HDF5, and the ability to compute additional
    basis elements and corresponding projectors with method _additional_projectors_tsvd"""

    version = 1

    def __init__(self, projectors_so_far, basis_so_far, n_basis, pulses_for_svd, v_dv, pretrig_rms_median, pretrig_rms_sigma, file_name):
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
        self.file_name = str(file_name)

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
        hdf5_group["svdbasis/file_name"] = self.file_name

    @classmethod
    def fromHDF5(cls, hdf5_group):
        projectors = hdf5_group["svdbasis/projectors"][()]
        n_basis = projectors.shape[0]
        basis = hdf5_group["svdbasis/basis"][()]
        v_dv = hdf5_group["svdbasis/v_dv"][()]
        pulses_for_svd = hdf5_group["svdbasis/training_pulses_for_plots"][()]
        pretrig_rms_median = hdf5_group["svdbasis/pretrig_rms_median"][()]
        pretrig_rms_sigma = hdf5_group["svdbasis/pretrig_rms_sigma"][()]
        version = hdf5_group["svdbasis/version"][()]
        file_name = hdf5_group["svdbasis/file_name"][()]
        if version != 1:
            raise Exception("loading not implemented for other versions")
        return cls(projectors, basis, n_basis, pulses_for_svd, v_dv, pretrig_rms_median, pretrig_rms_sigma, file_name)

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

        mpc = np.matmul(projectors, pulses_for_svd)  # modeled pulse coefs
        mp = np.matmul(basis, mpc)  # modeled pulse
        residuals = pulses_for_svd - mp
        Q = mass.mathstat.utilities.find_range_randomly(residuals, n_basis-3)

        projectors2 = np.linalg.pinv(Q)  # = Q.T, perhaps??
        projectors2 -= projectors2.dot(basis).dot(projectors)

        basis = np.hstack([basis, Q])
        projectors = np.vstack([projectors, projectors2])

        return projectors, basis


    def plot(self):
        labels = ["pulse", "deriv", "mean"]
        for i in range(self.n_basis-3):
            labels = ["svd{}".format(i)] + labels
        mpc = np.matmul(self.projectors, self.pulses_for_svd)
        mp = np.matmul(self.basis, mpc)
        residuals = self.pulses_for_svd-mp

        fig=plt.figure(figsize=(10,14))
        plt.subplot(511)
        projector_scale = np.amax(np.abs(self.projectors[2,:]))
        plt.plot(self.projectors[::-1,:].T)
        plt.title("projectors")
        plt.ylim(-2*projector_scale,2*projector_scale)
        plt.legend(labels)
        plt.grid(True)
        plt.subplot(512)
        plt.plot(self.basis[:,::-1])
        plt.title("basis")
        plt.legend(labels)
        plt.grid(True)
        plt.subplot(513)
        plt.plot(self.pulses_for_svd[:,:10])
        plt.title("from ljh")
        plt.legend(["{}".format(i) for i in range(10)])
        plt.grid(True)
        plt.subplot(514)
        plt.plot(residuals[:,:10])
        plt.title("residuals")
        plt.legend(["{}".format(i) for i in range(10)])
        plt.grid(True)
        should_be_identity = np.matmul(self.projectors, self.basis)
        identity = np.identity(self.n_basis)
        wrongness = np.abs(should_be_identity-identity)
        plt.subplot(515)
        plt.imshow(np.log10(wrongness))
        plt.title("log10(abs(projectors*basis-identity))")
        plt.colorbar()
        fig.suptitle(self.file_name)

        plt.figure()
        plt.plot(self.pulses_for_svd[:,0], label="from ljh")
        plt.plot(mp[:,0],label="modeled pulse")
        plt.legend()
        plt.title("modeled pulse vs true pulse")
