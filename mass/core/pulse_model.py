import pylab as plt
import numpy as np
import mass.mathstat
from mass.common import tostr


class PulseModel:
    """Object to hold a "pulse model", meaning a low-dimensional linear basis to express "all" pulses,
    along with a projector such that projector.dot(basis) is the identity matrix.

    Also has the capacity to store to and restore from HDF5, and the ability to compute additional
    basis elements and corresponding projectors with method _additional_projectors_tsvd"""

    version = 2

    def __init__(self, projectors_so_far, basis_so_far, n_basis, pulses_for_svd,  # noqa: PLR0917
                 v_dv, pretrig_rms_median, pretrig_rms_sigma, file_name,
                 extra_n_basis_5lag, f_5lag, average_pulse_for_5lag, noise_psd, noise_psd_delta_f,
                 noise_autocorr, _from_hdf5=False):
        self.pulses_for_svd = pulses_for_svd
        self.n_basis = n_basis
        dn = n_basis - extra_n_basis_5lag
        if projectors_so_far.shape[0] < dn:
            self.projectors, self.basis = self._additional_projectors_tsvd(
                projectors_so_far, basis_so_far, dn, pulses_for_svd)
        elif (projectors_so_far.shape[0] == dn) or _from_hdf5:
            self.projectors, self.basis = projectors_so_far, basis_so_far
        else:  # don't throw error on
            s = f"n_basis-extra_n_basis_5lag={dn} < projectors_so_far.shape[0] = {projectors_so_far.shape[0]}"
            s += f", extra_n_basis_5lag={extra_n_basis_5lag}"
            raise Exception(s)
        if (not _from_hdf5) and (extra_n_basis_5lag > 0):
            filters_5lag = np.zeros((len(f_5lag) + 4, 5))
            for i in range(5):
                if i < 4:
                    filters_5lag[i:-4 + i, i] = projectors_so_far[2, 2:-2]
                else:
                    filters_5lag[i:, i] = projectors_so_far[2, 2:-2]
            self.projectors, self.basis = self._additional_projectors_tsvd(
                self.projectors, self.basis, n_basis, filters_5lag)

        self.v_dv = v_dv
        self.pretrig_rms_median = pretrig_rms_median
        self.pretrig_rms_sigma = pretrig_rms_sigma
        self.file_name = str(file_name)
        self.extra_n_basis_5lag = extra_n_basis_5lag
        self.f_5lag = f_5lag
        self.average_pulse_for_5lag = average_pulse_for_5lag
        self.noise_psd = noise_psd
        self.noise_psd_delta_f = noise_psd_delta_f
        self.noise_autocorr = noise_autocorr
        self.was_saved_inverted = None

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
        hdf5_group["svdbasis/extra_n_basis_5lag"] = self.extra_n_basis_5lag
        hdf5_group["svdbasis/5lag_filter"] = self.f_5lag
        hdf5_group["svdbasis/average_pulse_for_5lag"] = self.average_pulse_for_5lag
        hdf5_group["svdbasis/noise_psd"] = self.noise_psd
        hdf5_group["svdbasis/noise_psd_delta_f"] = self.noise_psd_delta_f
        hdf5_group["svdbasis/noise_autocorr"] = self.noise_autocorr

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
        file_name = tostr(hdf5_group["svdbasis/file_name"][()])
        extra_n_basis_5lag = hdf5_group["svdbasis/extra_n_basis_5lag"][()]
        f_5lag = hdf5_group["svdbasis/5lag_filter"][()]
        average_pulse_for_5lag = hdf5_group["svdbasis/average_pulse_for_5lag"][()]
        noise_psd = hdf5_group["svdbasis/noise_psd"][()]
        noise_psd_delta_f = hdf5_group["svdbasis/noise_psd_delta_f"][()]
        noise_autocorr = hdf5_group["svdbasis/noise_autocorr"][()]
        was_saved_inverted = hdf5_group["svdbasis/was_saved_inverted"][()]

        if version != cls.version:
            raise Exception(f"loading not implemented for other versions, version={version}")
        model =  cls(projectors, basis, n_basis, pulses_for_svd, v_dv, pretrig_rms_median,
                   pretrig_rms_sigma, file_name, extra_n_basis_5lag, f_5lag, average_pulse_for_5lag,
                   noise_psd, noise_psd_delta_f, noise_autocorr, _from_hdf5=True)
        model.was_saved_inverted = was_saved_inverted
        return model

    @staticmethod
    def _additional_projectors_tsvd(projectors, basis, n_basis, pulses_for_svd):
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
        Q = mass.mathstat.utilities.find_range_randomly(residuals, n_basis - n_existing)

        projectors2 = np.linalg.pinv(Q)  # = Q.T, perhaps??
        projectors2 -= projectors2.dot(basis).dot(projectors)

        basis = np.hstack([basis, Q])
        projectors = np.vstack([projectors, projectors2])

        return projectors, basis

    def labels(self):
        labels = ["const", "deriv", "pulse"]
        for i in range(self.n_basis - 3):
            if i > self.n_basis - 3 - self.extra_n_basis_5lag:
                labels += [f"5lag{i + 2 - self.extra_n_basis_5lag}"]
            else:
                labels += [f"svd{i}"]
        return labels

    def plot(self, fig1=None, fig2=None):
        # plots information about a pulse model
        # fig1 and fig2 are optional matplotlib.pyplot (plt) figures if you need to embed the plots.
        # you can pass in the reference like fig=plt.figure() call or the figure's number, e.g. fig.number
        #   fig1 has modeled pulse vs true pulse
        #   fig2 has projectors, basis, "from ljh", residuals, and a measure of "wrongness"

        labels = self.labels()
        mpc = np.matmul(self.projectors, self.pulses_for_svd)
        mp = np.matmul(self.basis, mpc)
        residuals = self.pulses_for_svd - mp

        if fig1 is None:
            fig = plt.figure(figsize=(10, 14))
        else:
            fig = plt.figure(fig1)
        plt.subplot(511)
        plt.plot(self.projectors[::-1, :].T)
        plt.title("projectors")
        # projector_scale = np.amax(np.abs(self.projectors[2, :]))
        # plt.ylim(-2*projector_scale, 2*projector_scale)
        plt.legend(labels[::-1])
        plt.grid(True)
        plt.subplot(512)
        plt.plot(self.basis[:, ::-1])
        plt.title("basis")
        plt.legend(labels[::-1])
        plt.grid(True)
        plt.subplot(513)
        plt.plot(self.pulses_for_svd[:, :10])
        plt.title("from ljh")
        plt.legend([f"{i}" for i in range(10)])
        plt.grid(True)
        plt.subplot(514)
        plt.plot(residuals[:, :10])
        plt.title("residuals")
        plt.legend([f"{i}" for i in range(10)])
        plt.grid(True)
        should_be_identity = np.matmul(self.projectors, self.basis)
        identity = np.identity(self.n_basis)
        wrongness = np.abs(should_be_identity - identity)
        wrongness[wrongness < 1e-20] = 1e-20  # avoid warnings
        plt.subplot(515)
        plt.imshow(np.log10(wrongness))
        plt.title("log10(abs(projectors*basis-identity))")
        plt.colorbar()
        fig.suptitle(self.file_name)

        if fig2 is None:
            plt.figure(figsize=(10, 14))
        else:
            plt.figure(fig2)
        plt.plot(self.pulses_for_svd[:, 0], label="from ljh index 0")
        plt.plot(mp[:, 0], label="modeled pulse index 0")
        plt.legend()
        plt.title("modeled pulse vs true pulse")


# how well are the the 5lag filters represented
