r"""
Vectorized parametrization of cross-sections and interaction kernels for particle astrophysics.

.. warning::
    This module is designed as an internal backend engine and is not intended for direct end-user interaction.
    All cross-section models and emission kernels defined here are automatically managed and wrapped by the :py:class:`~saetass.utils.emissions.EmissionCalculator` class.

This module provides high-performance, vectorized implementations of empirical and theoretical cross-sections and emission kernels.
Its architectural design ensures that the SAETASS framework remains extensible and closed to modification when incorporating new interaction physics models.

The module provides concrete implementations for:

Hadronic inelastic collisions (:math:`p + p \to \pi^0 + X \to 2\gamma`, :math:`p + p \to \pi^\pm + X \to \nu`):
    - :py:class:`Kafexhiu2014`: Analytical parametrization across threshold to PeV energies (:cite:p:`Kafexhiu2014`).
    - :py:class:`AAFragPyModel`: LHC-tuned Monte Carlo fragmentation model via ``aafragpy`` (:cite:p:`Kachelriess2019`).

Leptonic radiative kernels (:math:`e^-` interactions):
    - :py:class:`AnalyticalBremsstrahlung`: Relativistic Bethe-Heitler cross-section (:cite:p:`BetheHeitler1934`, :cite:p:`BlumenthalGould1970`).
    - :py:class:`AnalyticalInverseCompton`: Exact Klein-Nishina differential rate kernel on arbitrary photon fields (:cite:p:`BlumenthalGould1970`).
    - :py:class:`AnalyticalSynchrotron`: Synchrotron radiation emission rate in magnetic fields (:cite:p:`BlumenthalGould1970`, :cite:p:`Aharonian2010`).
"""

import abc

import astropy.constants as const
import astropy.units as u
import numpy as np
import scipy.special

from .. import units as su

# -----------------
# BASE CLASSES
# -----------------


class HadronicCrossSectionModel(abc.ABC):
    r"""
    Abstract Base Class for hadronic differential cross-section models.

    Defines the interface for computing 2D differential cross-section matrices :math:`\frac{d\sigma}{dE_{\mathrm{out}}}(E_{\mathrm{out}}, E_{\mathrm{cr}})` for inelastic hadron-hadron collisions.
    """

    @abc.abstractmethod
    def compute_matrix(
        self,
        E_out_grid: np.ndarray,
        E_cr_kin_grid: np.ndarray,
        secondary: str = "gamma",
        **kwargs,
    ) -> np.ndarray:
        r"""
        Compute the 2D differential cross-section matrix.

        Parameters
        ----------
        E_out_grid : numpy.ndarray
            Secondary particle energies with shape ``(N_out,)``, in :math:`\mathrm{GeV}`.
        E_cr_kin_grid : numpy.ndarray
            Primary proton kinetic energies with shape ``(N_cr,)``, in :math:`\mathrm{GeV}`.
        secondary : str, optional
            Secondary particle channel to compute (e.g., ``"gamma"`` or ``"neutrino"``).
            Default is ``"gamma"``.
        **kwargs : dict
            Additional model-specific parameters.

        Returns
        -------
        sigma_matrix : numpy.ndarray
            Differential cross-section matrix with shape ``(N_out, N_cr)``, in :math:`\mathrm{mb\,GeV^{-1}}`.
            Kinematically forbidden transitions are set to zero.
        """
        pass


class LeptonicCrossSectionModel(abc.ABC):
    r"""
    Abstract Base Class for leptonic emission kernels and differential cross-sections.

    Defines the interface for computing 2D differential emission kernels :math:`\frac{d\sigma}{dE_\gamma}(E_\gamma, E_e)` or :math:`\frac{d^2 N}{dt \, dE_\gamma}(E_\gamma, E_e)` for relativistic electron interactions.
    """

    @abc.abstractmethod
    def compute_matrix(
        self,
        E_gamma_grid: np.ndarray,
        E_e_grid: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        r"""
        Compute the 2D differential emission kernel matrix.

        Parameters
        ----------
        E_gamma_grid : numpy.ndarray
            Emitted secondary photon energies with shape ``(N_gamma,)``, in :math:`\mathrm{GeV}`.
        E_e_grid : numpy.ndarray
            Primary electron total energies with shape ``(N_e,)``, in :math:`\mathrm{GeV}`.
        **kwargs : dict
            Additional physical parameters required by the specific leptonic mechanism
            (e.g., ambient photon fields, magnetic field strength, shielding regime).

        Returns
        -------
        kernel_matrix : numpy.ndarray
            Differential interaction matrix with shape ``(N_gamma, N_e)``.
            Units depend on the specific physical process (e.g., :math:`\mathrm{cm^2\,GeV^{-1}}` for Bremsstrahlung or :math:`\mathrm{s^{-1}\,GeV^{-1}}` for Inverse Compton / Synchrotron).
        """
        pass


# -----------------
# HADRONIC MODELS
# -----------------


class Kafexhiu2014(HadronicCrossSectionModel):
    r"""
    Kafexhiu et al. (2014) parametrization for inelastic :math:`p+p` collisions.

    This model provides semi-analytical parametrizations for gamma-ray production from neutral pion decay (:math:`p + p \to \pi^0 + X \to 2\gamma + X`) spanning from the kinematic threshold (:math:`T_{p,\mathrm{th}} \approx 0.2797\text{ GeV}`) up to PeV energies (:cite:p:`Kafexhiu2014`).

    The differential cross-section is formulated as:

    .. math::

        \frac{d\sigma}{dE_\gamma}(T_p, E_\gamma) = A_{\mathrm{max}}(T_p) \, F(T_p, E_\gamma)

    where :math:`A_{\mathrm{max}}(T_p)` is the maximum peak value of the differential cross-section and :math:`F(T_p, E_\gamma)` is the normalized spectral shape function.

    Parameters
    ----------
    he_model : str, optional
        High-energy interaction generator to use for primary proton energies :math:`T_p > 50\text{--}100\text{ GeV}`.
        Supported options are ``'sibyll'``, ``'geant4'``, ``'pythia8'`` and ``'qgsjet'``. Default is ``'sibyll'``.

    Attributes
    ----------
    M_P : float
        Rest mass energy of the proton, :math:`m_p c^2 = 0.938272\,\mathrm{GeV}`, in :math:`\mathrm{GeV}`.
    M_PI : float
        Rest mass energy of the neutral pion, :math:`m_{\pi^0} c^2 = 0.134976\,\mathrm{GeV}`, in :math:`\mathrm{GeV}`.
    TP_TH : float
        Kinematic threshold proton kinetic energy in the laboratory frame, in :math:`\mathrm{GeV}`:

        .. math::

            T_{p,\mathrm{th}} = 2 m_{\pi^0} + \frac{m_{\pi^0}^2}{2 m_p} \approx 0.2797\text{ GeV}
    """

    # Fundamental Constants (PDG)
    M_P = 0.938272  # GeV, proton mass
    M_PI = 0.134976  # GeV, neutral pion mass
    TP_TH = 2.0 * M_PI + (M_PI**2) / (2.0 * M_P)

    def __init__(self, he_model: str = "sibyll"):
        r"""
        Initialize the Kafexhiu (2014) cross-section model.

        Parameters
        ----------
        he_model : str, optional
            High-energy hadronic generator parameterization to apply at :math:`T_p > 50\text{--}100\text{ GeV}`.
            Must be one of ``'geant4'``, ``'pythia8'``, ``'sibyll'``, or ``'qgsjet'``. Default is ``'sibyll'``.

        Raises
        ------
        ValueError
            If an unsupported high-energy model name is provided.
        """
        valid_models = ["geant4", "pythia8", "sibyll", "qgsjet"]
        he_clean = he_model.lower()
        if he_clean not in valid_models:
            raise ValueError(
                f"Unknown high-energy model '{he_model}'. Supported options are: {valid_models}."
            )
        self.he_model = he_clean

    def _epi0_max_lab(self, Tp: np.ndarray) -> np.ndarray:
        r"""
        Compute the maximum neutral pion energy allowed by kinematics in the laboratory frame.

        Parameters
        ----------
        Tp : numpy.ndarray
            Primary proton kinetic energy, in :math:`\mathrm{GeV}`.

        Returns
        -------
        E_pi_max : numpy.ndarray
            Maximum neutral pion energy :math:`E_{\pi^0,\mathrm{max}}^{\mathrm{LAB}}`, in :math:`\mathrm{GeV}`.
        """
        s = 2.0 * self.M_P * (Tp + 2.0 * self.M_P)
        gamma_cm = (Tp + 2.0 * self.M_P) / np.sqrt(s)
        e_pi_cm = (s - 4.0 * self.M_P**2 + self.M_PI**2) / (2.0 * np.sqrt(s))
        p_pi_cm = np.sqrt(np.clip(e_pi_cm**2 - self.M_PI**2, 0.0, None))
        beta_cm = np.sqrt(np.clip(1.0 - gamma_cm ** (-2.0), 0.0, None))
        return gamma_cm * (e_pi_cm + p_pi_cm * beta_cm)

    def _egamma_max(self, Tp: np.ndarray) -> np.ndarray:
        r"""
        Compute the maximum gamma-ray photon energy allowed by kinematics in the laboratory frame.

        Parameters
        ----------
        Tp : numpy.ndarray
            Primary proton kinetic energy, in :math:`\mathrm{GeV}`.

        Returns
        -------
        E_gamma_max : numpy.ndarray
            Maximum gamma-ray energy :math:`E_{\gamma,\mathrm{max}}`, in :math:`\mathrm{GeV}`.
        """
        gamma_pi_lab = self._epi0_max_lab(Tp) / self.M_PI
        beta_pi_lab = np.sqrt(np.clip(1.0 - gamma_pi_lab ** (-2.0), 0.0, None))
        return (self.M_PI / 2.0) * gamma_pi_lab * (1.0 + beta_pi_lab)

    def _sigma_inel(self, Tp: np.ndarray) -> np.ndarray:
        r"""
        Compute the total inelastic proton-proton cross-section :math:`\sigma_{\mathrm{inel}}(T_p)`.

        Parameters
        ----------
        Tp : numpy.ndarray
            Primary proton kinetic energy, in :math:`\mathrm{GeV}`.

        Returns
        -------
        sigma_inel : numpy.ndarray
            Total inelastic cross-section, in :math:`\mathrm{mb}`.
        """
        xs = np.zeros_like(Tp)
        mask = Tp > self.TP_TH
        if not np.any(mask):
            return xs

        tp_safe = Tp[mask]
        lx = np.log(tp_safe / self.TP_TH)
        threshold = np.maximum(0.0, 1.0 - (self.TP_TH / tp_safe) ** 1.9)
        xs[mask] = (30.7 - 0.96 * lx + 0.18 * lx**2) * (threshold**3)
        return xs

    def _sigma_1pi(self, Tp: np.ndarray) -> np.ndarray:
        r"""
        Compute the single neutral pion production cross-section :math:`\sigma_{1\pi^0}(T_p)` (:math:`p + p \to p + p + \pi^0`).

        Valid near the kinematic threshold (:math:`T_{p,\mathrm{th}} < T_p \le 2\text{ GeV}`) using the relativistic Breit-Wigner resonance formulation.

        Parameters
        ----------
        Tp : numpy.ndarray
            Primary proton kinetic energy, in :math:`\mathrm{GeV}`.

        Returns
        -------
        sigma_1pi : numpy.ndarray
            Single-pion production cross-section, in :math:`\mathrm{mb}`.
        """
        m_res = 1.1883
        gamma_res = 0.2264
        sigma_0 = 7.66e-3

        xs = np.zeros_like(Tp)
        mask = (Tp > self.TP_TH) & (Tp <= 2.0)
        if not np.any(mask):
            return xs

        tp_safe = Tp[mask]
        s = 2.0 * self.M_P * (tp_safe + 2.0 * self.M_P)
        x = np.sqrt(s) - self.M_P

        num = np.sqrt(
            np.clip(
                (s - self.M_PI**2 - 4.0 * self.M_P**2) ** 2
                - 16.0 * (self.M_PI**2) * (self.M_P**2),
                0.0,
                None,
            )
        )
        eta = num / (2.0 * self.M_PI * np.sqrt(s))

        g_res = np.sqrt((m_res**2) * (m_res**2 + gamma_res**2))
        k_res = (
            np.sqrt(8.0)
            * m_res
            * gamma_res
            * g_res
            / (np.pi * np.sqrt(m_res**2 + g_res))
        )
        f_bw = self.M_P * k_res / ((x**2 - m_res**2) ** 2 + (m_res * gamma_res) ** 2)

        xs[mask] = sigma_0 * (eta**1.95) * (1.0 + eta + eta**5) * (f_bw**1.86)
        return xs

    def _sigma_2pi(self, Tp: np.ndarray) -> np.ndarray:
        r"""
        Compute the two-pion production cross-section :math:`\sigma_{2\pi}(T_p)`.

        Valid in the low-energy multi-pion regime (:math:`0.56\text{ GeV} \le T_p \le 2.0\text{ GeV}`).

        Parameters
        ----------
        Tp : numpy.ndarray
            Primary proton kinetic energy, in :math:`\mathrm{GeV}`.

        Returns
        -------
        sigma_2pi : numpy.ndarray
            Two-pion production cross-section, in :math:`\mathrm{mb}`.
        """
        xs = np.zeros_like(Tp)
        mask = (Tp >= 0.56) & (Tp <= 2.0)
        xs[mask] = 5.7 / (1.0 + np.exp(-9.3 * (Tp[mask] - 1.4)))
        return xs

    def _multip_pi0(self, Tp: np.ndarray) -> np.ndarray:
        r"""
        Compute the average neutral pion production multiplicity :math:`\langle n_{\pi^0} \rangle(T_p)`.

        Parameters
        ----------
        Tp : numpy.ndarray
            Primary proton kinetic energy, in :math:`\mathrm{GeV}`.

        Returns
        -------
        multiplicity : numpy.ndarray
            Average neutral pion multiplicity.
        """
        multip = np.zeros_like(Tp)

        # Common Geant4 low-energy behavior (Tp <= 2.0 is 0.0, handled by zeros_like)
        mask_low = (Tp > 2.0) & (Tp < 5.0)
        q_p = (Tp[mask_low] - self.TP_TH) / self.M_P
        multip[mask_low] = -6.0e-3 + 0.237 * q_p - 0.023 * (q_p**2)

        # High energy masks depending on model configuration
        mask_g4 = Tp >= 5.0
        if self.he_model == "geant4":
            mask_he = np.zeros_like(Tp, dtype=bool)
        elif self.he_model == "pythia8":
            mask_g4 = (Tp >= 5.0) & (Tp <= 50.0)
            mask_he = Tp > 50.0
            params_he = (0.652, 0.0016, 0.488, 0.1928, 0.483)
        elif self.he_model == "sibyll":
            mask_g4 = (Tp >= 5.0) & (Tp <= 100.0)
            mask_he = Tp > 100.0
            params_he = (5.436, 0.254, 0.072, 0.075, 0.166)
        elif self.he_model == "qgsjet":
            mask_g4 = (Tp >= 5.0) & (Tp <= 100.0)
            mask_he = Tp > 100.0
            params_he = (0.908, 0.0009, 6.089, 0.176, 0.448)

        # Apply Geant4 parameters where applicable
        if np.any(mask_g4):
            xi_p = np.clip((Tp[mask_g4] - 3.0) / self.M_P, 0.0, None)
            a1, a2, a3, a4, a5 = 0.728, 0.596, 0.491, 0.2503, 0.117
            multip[mask_g4] = (
                a1
                * (xi_p**a4)
                * (1.0 + np.exp(-a2 * (xi_p**a5)))
                * (1.0 - np.exp(-a3 * (xi_p**0.25)))
            )

        # Apply high-energy model parameters where applicable
        if np.any(mask_he):
            xi_p = np.clip((Tp[mask_he] - 3.0) / self.M_P, 0.0, None)
            a1, a2, a3, a4, a5 = params_he
            multip[mask_he] = (
                a1
                * (xi_p**a4)
                * (1.0 + np.exp(-a2 * (xi_p**a5)))
                * (1.0 - np.exp(-a3 * (xi_p**0.25)))
            )

        return multip

    def _sigma_pi(self, Tp: np.ndarray) -> np.ndarray:
        r"""
        Compute the total neutral pion production cross-section :math:`\sigma_{\pi^0}(T_p)`.

        Combines resonant channels, two-pion production, and inclusive inelastic multiplicities:

        .. math::

            \sigma_{\pi^0}(T_p) = \sigma_{1\pi^0}(T_p) + \sigma_{2\pi}(T_p) + \sigma_{\mathrm{inel}}(T_p) \, \langle n_{\pi^0} \rangle(T_p)

        Parameters
        ----------
        Tp : numpy.ndarray
            Primary proton kinetic energy, in :math:`\mathrm{GeV}`.

        Returns
        -------
        sigma_pi : numpy.ndarray
            Total neutral pion production cross-section, in :math:`\mathrm{mb}`.
        """
        return (
            self._sigma_1pi(Tp)
            + self._sigma_2pi(Tp)
            + self._sigma_inel(Tp) * self._multip_pi0(Tp)
        )

    def _amax(self, Tp: np.ndarray) -> np.ndarray:
        r"""
        Compute the peak value :math:`A_{\mathrm{max}}(T_p)` of the differential cross-section.

        Parameters
        ----------
        Tp : numpy.ndarray
            Primary proton kinetic energy, in :math:`\mathrm{GeV}`.

        Returns
        -------
        amax : numpy.ndarray
            Peak differential cross-section value, in :math:`\mathrm{mb\,GeV^{-1}}`.
        """
        amax = np.zeros_like(Tp)
        theta_p = np.clip(Tp / self.M_P, 1e-9, None)

        # Prevent log(0) warnings
        safe_theta = np.clip(theta_p, 1e-9, None)
        ltheta_p = np.log(safe_theta)

        # Region 1: Tp_th < Tp < 1.0 GeV
        m_exp = (Tp > self.TP_TH) & (Tp < 1.0)

        # Temporarily use Geant4 normalization for low-energy threshold region
        original_model = self.he_model
        self.he_model = "geant4"
        amax[m_exp] = 5.9 * self._sigma_pi(Tp[m_exp]) / self._epi0_max_lab(Tp[m_exp])
        self.he_model = original_model

        m_he = np.zeros_like(Tp, dtype=bool)
        if self.he_model == "geant4":
            m_g4_low = (Tp >= 1.0) & (Tp < 5.0)
            m_g4_high = Tp >= 5.0
        elif self.he_model == "pythia8":
            m_g4_low = (Tp >= 1.0) & (Tp < 5.0)
            m_g4_high = (Tp >= 5.0) & (Tp <= 50.0)
            m_he = Tp > 50.0
            b_he = (9.06, -0.3795, 0.01105)
        elif self.he_model == "sibyll":
            m_g4_low = (Tp >= 1.0) & (Tp < 5.0)
            m_g4_high = (Tp >= 5.0) & (Tp <= 100.0)
            m_he = Tp > 100.0
            b_he = (10.77, -0.412, 0.01264)
        elif self.he_model == "qgsjet":
            m_g4_low = (Tp >= 1.0) & (Tp < 5.0)
            m_g4_high = (Tp >= 5.0) & (Tp <= 100.0)
            m_he = Tp > 100.0
            b_he = (13.16, -0.4419, 0.01439)

        # Geant4 parameterization for intermediate energy ranges
        self.he_model = "geant4"
        amax[m_g4_low] = (
            9.53
            * (theta_p[m_g4_low] ** -0.52)
            * np.exp(0.054 * ltheta_p[m_g4_low] ** 2)
            * self._sigma_pi(Tp[m_g4_low])
            / self.M_P
        )
        amax[m_g4_high] = (
            9.13
            * (theta_p[m_g4_high] ** -0.35)
            * np.exp(0.0097 * ltheta_p[m_g4_high] ** 2)
            * self._sigma_pi(Tp[m_g4_high])
            / self.M_P
        )
        self.he_model = original_model

        if np.any(m_he):
            b1, b2, b3 = b_he
            amax[m_he] = (
                b1
                * (theta_p[m_he] ** b2)
                * np.exp(b3 * ltheta_p[m_he] ** 2)
                * self._sigma_pi(Tp[m_he])
                / self.M_P
            )

        return amax

    def _f_shape(self, Tp: np.ndarray, Egamma: np.ndarray) -> np.ndarray:
        r"""
        Compute the dimensionless spectral shape function :math:`F(T_p, E_\gamma)`.

        Parameters
        ----------
        Tp : numpy.ndarray
            Primary proton kinetic energies with shape ``(N_cr,)``, in :math:`\mathrm{GeV}`.
        Egamma : numpy.ndarray
            Secondary photon energies with shape ``(N_out,)``, in :math:`\mathrm{GeV}`.

        Returns
        -------
        FF : numpy.ndarray
            2D array of spectral shape values with shape ``(N_out, N_cr)``.
        """
        # Broadcast grids to 2D: Tp across columns (1, N_cr), Egamma across rows (N_out, 1)
        Tp_2d = Tp[np.newaxis, :]
        Eg_2d = Egamma[:, np.newaxis]

        Y = Eg_2d + (self.M_PI**2) / (4.0 * Eg_2d)
        eg_max = self._egamma_max(Tp_2d)
        Y0 = eg_max + (self.M_PI**2) / (4.0 * eg_max)

        # Avoid division by zero in kinematically unphysical regions
        Y0_safe = np.where(Y0 == self.M_PI, self.M_PI + 1e-9, Y0)
        X = (Y - self.M_PI) / (Y0_safe - self.M_PI)
        X_safe = np.clip(X, 0.0, 1.0)

        FF = np.zeros_like(X)
        kin_mask = (X >= 0.0) & (X < 1.0)

        Tp_full = np.broadcast_to(Tp_2d, X.shape)
        Y0_full = np.broadcast_to(Y0, X.shape)

        # Precompute common variables for valid kinematics
        theta = Tp_full / self.M_P
        kappa = 3.29 - 0.2 * (np.clip(theta, 1e-9, None) ** -1.5)
        q = (Tp_full - 1.0) / self.M_P
        q_safe = np.clip(q, 0.0, None)

        if self.he_model == "pythia8":
            c_he = 3.5 * self.M_PI / Y0_full
            m_he = Tp_full > 50.0
            exp_he = 4.0
        elif self.he_model == "sibyll":
            c_he = 3.55 * self.M_PI / Y0_full
            m_he = Tp_full > 100.0
            exp_he = 3.6
        elif self.he_model == "qgsjet":
            c_he = 3.55 * self.M_PI / Y0_full
            m_he = Tp_full > 100.0
            exp_he = 4.5
        else:
            m_he = np.zeros_like(Tp_full, dtype=bool)

        C_g4 = 3.0 * self.M_PI / Y0_full

        # Region 1: Tp_th < Tp < 1.0 GeV
        m1 = kin_mask & (Tp_full > self.TP_TH) & (Tp_full < 1.0)
        FF[m1] = (1.0 - X_safe[m1]) ** kappa[m1]

        # Region 2: 1.0 <= Tp <= 4.0 GeV
        m2 = kin_mask & (Tp_full >= 1.0) & (Tp_full <= 4.0) & ~m_he
        mu2 = 1.25 * (q_safe[m2] ** 1.25) * np.exp(-1.25 * q_safe[m2])
        FF[m2] = ((1.0 - X_safe[m2]) ** (mu2 + 2.45)) / (
            (1.0 + X_safe[m2] / C_g4[m2]) ** (mu2 + 1.45)
        )

        # Region 3: 4.0 < Tp <= 20.0 GeV
        m3 = kin_mask & (Tp_full > 4.0) & (Tp_full <= 20.0) & ~m_he
        mu3 = 1.25 * (q_safe[m3] ** 1.25) * np.exp(-1.25 * q_safe[m3])
        FF[m3] = ((1.0 - X_safe[m3]) ** (1.5 * mu3 + 4.95)) / (
            (1.0 + X_safe[m3] / C_g4[m3]) ** (mu3 + 1.5)
        )

        # Region 4: 20.0 < Tp <= 100.0 GeV (Geant4)
        m4 = kin_mask & (Tp_full > 20.0) & (Tp_full <= 100.0) & ~m_he
        FF[m4] = ((1.0 - np.sqrt(X_safe[m4])) ** 4.2) / (1.0 + X_safe[m4] / C_g4[m4])

        # Region 5: Tp > 100.0 GeV (Geant4)
        m5 = kin_mask & (Tp_full > 100.0) & ~m_he
        FF[m5] = ((1.0 - np.sqrt(X_safe[m5])) ** 4.9) / (1.0 + X_safe[m5] / C_g4[m5])

        # Region HE: High-Energy models (Pythia, Sibyll, QGSJet)
        m_he_kin = kin_mask & m_he
        if np.any(m_he_kin):
            FF[m_he_kin] = ((1.0 - np.sqrt(X_safe[m_he_kin])) ** exp_he) / (
                1.0 + X_safe[m_he_kin] / c_he[m_he_kin]
            )

        return FF

    def compute_matrix(
        self,
        E_out_grid: np.ndarray,
        E_cr_kin_grid: np.ndarray,
        secondary: str = "gamma",
        **kwargs,
    ) -> np.ndarray:
        r"""
        Compute the 2D differential cross-section matrix :math:`\frac{d\sigma}{dE_\gamma}(E_\gamma, T_p)`.

        Parameters
        ----------
        E_out_grid : numpy.ndarray
            Secondary gamma-ray photon energies with shape ``(N_out,)``, in :math:`\mathrm{GeV}`.
        E_cr_kin_grid : numpy.ndarray
            Primary proton **kinetic** energies with shape ``(N_cr,)``, in :math:`\mathrm{GeV}`.
        secondary : str, optional
            Secondary particle type to compute. Must be ``"gamma"`` (or ``"gam"``) for this model.
            Default is ``"gamma"``.
        **kwargs : dict
            Additional arguments (reserved for interface compatibility).

        Returns
        -------
        sigma_matrix : numpy.ndarray
            Differential cross-section matrix with shape ``(N_out, N_cr)``, in :math:`\mathrm{mb\,GeV^{-1}}`.
            Kinematically forbidden transitions are strictly zero-padded.

        Raises
        ------
        ValueError
            If a secondary particle other than gamma-rays is requested.
        """
        secondary_clean = secondary.lower()
        if secondary_clean not in ["gamma", "gam"]:
            raise ValueError(
                f"The Kafexhiu (2014) model only supports gamma-ray production from pi0 decay. Requested secondary: '{secondary}'."
            )

        amax_vals = self._amax(E_cr_kin_grid)
        f_shape = self._f_shape(E_cr_kin_grid, E_out_grid)

        return amax_vals[np.newaxis, :] * f_shape


class AAFragPyModel(HadronicCrossSectionModel):
    r"""
    Wrapper for the external ``aafragpy`` hadronic interaction library.

    Provides access to the LHC-tuned Monte Carlo fragmentation parameterizations for inelastic proton-proton collisions. Computes differential cross-sections for secondary gamma-rays and all-flavor neutrinos.

    Notes
    -----
    ``aafragpy`` takes and tabulates total energies, :math:`E_p = T_p + m_p c^2`, while SAETASS works with kinetic energies: the wrapper converts them. The p-p tables start at :math:`E_p = 4\text{ GeV}`, i.e. :math:`T_p \approx 3.06\text{ GeV}`; below it the cross-sections are set to zero and a user warning is issued.
    """

    #: Rest energy of the proton, in GeV, the value used by ``aafragpy``.
    M_P = 0.938272
    #: Lowest total energy of the primary in the p-p tables of ``aafragpy``, in GeV.
    E_P_MIN = 4.0

    def compute_matrix(
        self,
        E_out_grid: np.ndarray,
        E_cr_kin_grid: np.ndarray,
        secondary: str = "gamma",
        **kwargs,
    ) -> np.ndarray:
        r"""
        Compute the 2D differential cross-section matrix using AAFRAG.

        Parameters
        ----------
        E_out_grid : numpy.ndarray
            Secondary particle energies with shape ``(N_out,)``, in :math:`\mathrm{GeV}`.
        E_cr_kin_grid : numpy.ndarray
            Primary proton kinetic energies with shape ``(N_cr,)``, in :math:`\mathrm{GeV}`.
        secondary : str, optional
            Secondary particle type to compute: ``"gamma"`` (neutral pion decay photons)
            or ``"neutrino"`` (all-flavor neutrinos from charged pion decay chains).
            Default is ``"gamma"``.
        **kwargs : dict
            Additional arguments (reserved for interface compatibility).

        Returns
        -------
        sigma_matrix : numpy.ndarray
            Differential cross-section matrix with shape ``(N_out, N_cr)``, in :math:`\mathrm{mb\,GeV^{-1}}`.

        Raises
        ------
        ValueError
            If an unsupported secondary particle type is requested.
        """
        from aafragpy import get_cross_section

        sec_map = {"gamma": "gam", "neutrino": "nu_all"}
        if secondary not in sec_map:
            raise ValueError(
                f"Unknown secondary '{secondary}' for AAFRAG. Supported options are: {list(sec_map.keys())}."
            )

        # aafragpy expects total energies; for the secondaries (photons and neutrinos)
        # total and kinetic energies coincide
        E_cr_tot_grid = E_cr_kin_grid + self.M_P
        valid_mask = E_cr_tot_grid >= self.E_P_MIN
        if not np.all(valid_mask):
            import warnings

            warnings.warn(
                "AAFRAG parametrization is only valid for primary proton total energies >= 4 GeV "
                "(kinetic energies >= 3.06 GeV). Cross-sections for lower energies have been set to 0.",
                UserWarning,
                stacklevel=2,
            )

        sigma_matrix = np.zeros((len(E_out_grid), len(E_cr_kin_grid)))

        if np.any(valid_mask):
            result = get_cross_section(
                primary_target="p-p",
                secondary=sec_map[secondary],
                E_primaries=E_cr_tot_grid[valid_mask],
                E_secondaries=E_out_grid,
            )
            sigma = result[0] if isinstance(result, tuple) else result
            sigma_matrix[:, valid_mask] = sigma.T

        return sigma_matrix


# -----------------
# LEPTONIC MODELS
# -----------------


class AnalyticalBremsstrahlung(LeptonicCrossSectionModel):
    r"""
    Analytical differential cross-section for relativistic electron Bremsstrahlung.

    Implements the relativistic Bethe-Heitler formulation for electron-ion
    (:math:`e^- + Z \to e^- + Z + \gamma`) and electron-electron
    (:math:`e^- + e^- \to e^- + e^- + \gamma`) Bremsstrahlung (:cite:p:`BetheHeitler1934`, :cite:p:`BlumenthalGould1970`).

    Supports both weak-shielding (fully ionized plasma) and strong-shielding
    (neutral atomic gas) regimes:

    .. math::

        \frac{d\sigma_{\mathrm{ep}}}{dE_\gamma} = 4 \alpha r_0^2 Z^2 \frac{1}{E_\gamma}
        \left[ 1 + (1-y)^2 - \frac{2}{3}(1-y) \right] L_{\mathrm{rad}}

    where :math:`y = E_\gamma / E_e`, :math:`r_0` is the classical electron radius,
    :math:`\alpha` is the fine-structure constant, and :math:`L_{\mathrm{rad}}` is the radiation logarithm.
    """

    def compute_matrix(
        self,
        E_gamma_grid: np.ndarray,
        E_e_grid: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        r"""
        Compute the 2D differential cross-section matrix for electron Bremsstrahlung.

        Parameters
        ----------
        E_gamma_grid : numpy.ndarray
            Emitted secondary photon energies with shape ``(N_gamma,)``, in :math:`\mathrm{GeV}`.
        E_e_grid : numpy.ndarray
            Primary electron total energies with shape ``(N_e,)``, in :math:`\mathrm{GeV}`.
        **kwargs : dict
            Additional physical parameters controlling the interaction regime:

            - ``ionised`` : bool, optional
                If True, uses the weak-shielding formula appropriate for ionized plasma.
                If False, uses the strong-shielding formula for neutral gas. Default is ``True``.
            - ``Z`` : float, optional
                Mean nuclear charge of the ambient medium. Default is ``1.0``.
            - ``include_ee`` : bool, optional
                Whether to include electron-electron Bremsstrahlung contributions. Default is ``True``.
            - ``weight_ep`` : float, optional
                Abundance-weighted multiplicity factor for electron-ion interactions.
                Default is ``1.263``, the value for the standard ISM solar metallicity.
            - ``weight_ee`` : float, optional
                Abundance-weighted multiplicity factor for electron-electron interactions.
                Default is ``1.088``, the value for the standard ISM solar metallicity.

        Returns
        -------
        sigma_matrix : numpy.ndarray
            Differential cross-section matrix with shape ``(N_gamma, N_e)``, in :math:`\mathrm{cm^2\,GeV^{-1}}`.
            Kinematically forbidden transitions (:math:`E_\gamma \ge E_e`) are strictly set to zero.
        """
        ionised = kwargs.get("ionised", True)
        Z = kwargs.get("Z", 1.0)
        include_ee = kwargs.get("include_ee", True)

        # Default astrophysical abundance weights (Standard ISM solar metallicity)
        weight_ep = kwargs.get("weight_ep", 1.263)
        weight_ee = kwargs.get("weight_ee", 1.088)

        Eg = E_gamma_grid[:, np.newaxis]
        Ee = E_e_grid[np.newaxis, :]

        valid = Eg < Ee
        y = np.where(valid, Eg / Ee, 1e-10)

        mec2 = (const.m_e * const.c**2).to_value(u.GeV)
        r_e_cm = const.e.gauss.value**2 / (const.m_e.cgs.value * const.c.cgs.value**2)
        alpha = const.alpha.value

        phi = 1.0 + (1.0 - y) ** 2 - (2.0 / 3.0) * (1.0 - y)

        if ionised:
            # Weak-shielding regime (ionized gas)
            arg = 2.0 * Ee * (Ee - Eg) / (mec2 * Eg)
            arg = np.where(valid & (arg > 1.0), arg, 1.0)
            # The relativistic formula fails as the final electron becomes non-relativistic
            # near the tip of the spectrum, where it would turn negative
            log_term = np.maximum(np.log(arg) - 0.5, 0.0)
        else:
            # Strong-shielding regime (neutral gas with atomic screening)
            log_term = np.log(183.0 / (Z ** (1.0 / 3.0)))

        # Base electron-proton cross section weighted by metallicity
        sigma_ep = 4.0 * alpha * (Z**2) * (r_e_cm**2) / Eg * phi * log_term
        sigma_ep_weighted = sigma_ep * weight_ep

        # Add orbital and free electron-electron contribution (e-e Bremsstrahlung)
        if include_ee:
            sigma_total = sigma_ep_weighted + (sigma_ep * weight_ee)
        else:
            sigma_total = sigma_ep_weighted

        return np.where(valid, sigma_total, 0.0)


class AnalyticalInverseCompton(LeptonicCrossSectionModel):
    r"""
    Analytical differential emission rate kernel for Inverse Compton scattering.

    Implements the exact Klein-Nishina cross-section formulation from :cite:ct:`BlumenthalGould1970`
    integrated over an arbitrary target background photon field :math:`\frac{dn}{d\varepsilon}(\varepsilon)`:

    .. math::

        e^- + \gamma_{\mathrm{target}} \to e^- + \gamma

    The differential photon production rate per electron is computed via:

    .. math::

        \frac{d^2 N}{dt \, dE_\gamma}(E_\gamma, E_e) = \frac{3 \sigma_{\mathrm{T}} c (m_e c^2)^2}{4 E_e^2}
        \int \frac{dn/d\varepsilon}{\varepsilon} \, F(q, \Gamma) \, d\varepsilon

    where :math:`\Gamma = \frac{4 E_e \varepsilon}{(m_e c^2)^2}`, :math:`q = \frac{E_\gamma}{\Gamma (E_e - E_\gamma)}`,
    and :math:`F(q, \Gamma)` is the dimensionless Blumenthal & Gould kernel.
    """

    def compute_matrix(
        self,
        E_gamma_grid: np.ndarray,
        E_e_grid: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        r"""
        Compute the 2D differential emission kernel matrix for Inverse Compton scattering.

        Parameters
        ----------
        E_gamma_grid : numpy.ndarray
            Upscattered secondary photon energies with shape ``(N_gamma,)``, in :math:`\mathrm{GeV}`.
        E_e_grid : numpy.ndarray
            Primary electron total energies with shape ``(N_e,)``, in :math:`\mathrm{GeV}`.
        **kwargs : dict
            Required physical background photon field inputs:

            - ``eps_grid`` : astropy.units.Quantity
                Target seed photon energy grid. Units compatible with :py:data:`~saetass.units.ENERGY`.
            - ``dn_deps`` : astropy.units.Quantity
                1D or 2D target photon differential number density.
                Units compatible with :math:`\mathrm{cm^{-3}\,eV^{-1}}`.

        Returns
        -------
        kernel_matrix : numpy.ndarray
            Differential emission rate kernel with shape ``(N_gamma, N_e)``, in :math:`\mathrm{s^{-1}\,GeV^{-1}}`.
            Kinematically forbidden transitions (:math:`E_\gamma \ge E_e` or :math:`q > 1`) are set to zero.

        Raises
        ------
        ValueError
            If ``eps_grid`` or ``dn_deps`` are not provided in ``kwargs``.
        """
        eps_grid = kwargs.get("eps_grid")
        dn_deps = kwargs.get("dn_deps")

        if eps_grid is None or dn_deps is None:
            raise ValueError(
                "AnalyticalInverseCompton requires 'eps_grid' (u.Quantity) and 'dn_deps' (u.Quantity) in kwargs."
            )

        # Convert physical quantities to consistent GeV-based numerical units
        eps_GeV = su.validate_quantity(
            eps_grid, u.GeV, "AnalyticalInverseCompton argument 'eps_grid'"
        ).value
        dn_deps_GeV = su.validate_quantity(
            dn_deps, u.cm**-3 / u.GeV, "AnalyticalInverseCompton argument 'dn_deps'"
        ).value

        # Broadcast grids to 3D for integration: (N_gamma, N_e, N_eps)
        Eg = E_gamma_grid[:, np.newaxis, np.newaxis]
        Ee = E_e_grid[np.newaxis, :, np.newaxis]
        eps = eps_GeV[np.newaxis, np.newaxis, :]

        mec2 = (const.m_e * const.c**2).to_value(u.GeV)
        Gamma = 4.0 * Ee * eps / (mec2**2)

        # Kinematic parameter q
        diff = Ee - Eg
        diff_safe = np.where(diff > 0.0, diff, 1e-10)
        q = Eg / (Gamma * diff_safe)

        gamma_e = Ee / mec2
        q_min = 1.0 / (4.0 * gamma_e**2)

        # Valid domain: 1/(4 gamma^2) <= q <= 1  AND  E_gamma < E_e
        valid = (q >= q_min) & (q <= 1.0) & (Eg < Ee)
        q_safe = np.where(valid, q, 1.0)

        # Blumenthal & Gould (1970) dimensionless F(q, Gamma) kernel
        f_q = (
            2.0 * q_safe * np.log(q_safe)
            + (1.0 + 2.0 * q_safe) * (1.0 - q_safe)
            + 0.5 * ((Gamma * q_safe) ** 2) / (1.0 + Gamma * q_safe) * (1.0 - q_safe)
        )
        f_q = np.where(valid, f_q, 0.0)

        # Integrand: (dn/deps) / eps * f_q
        integrand = (dn_deps_GeV[np.newaxis, np.newaxis, :] / eps) * f_q

        # Perform numerical integration over seed photon energy eps
        integral = np.trapezoid(integrand, x=eps_GeV, axis=2)

        # Prefactor: 3/4 * sigma_T * c * (m_e c^2)^2 / E_e^2
        sigma_T_cm2 = const.sigma_T.to_value(u.cm**2)
        c_cm_s = const.c.cgs.value
        prefactor = (
            0.75 * sigma_T_cm2 * c_cm_s * (mec2**2) / (E_e_grid[np.newaxis, :] ** 2)
        )

        # Resulting differential emission kernel in s^-1 GeV^-1
        return prefactor * integral


class AnalyticalSynchrotron(LeptonicCrossSectionModel):
    r"""
    Analytical differential emission rate kernel for Synchrotron radiation.

    Computes the photon production rate for relativistic electrons moving in an ambient
    magnetic field (:math:`e^- + B \to e^- + \gamma_{\mathrm{syn}}`) (:cite:p:`BlumenthalGould1970`, :cite:p:`Aharonian2010`).

    Supports two pitch-angle treatments:
    - ``'isotropic'``: Standard pitch-angle averaged approximation (:cite:ct:`Aharonian2010`).
    - ``'perpendicular'``: Exact 90-degree pitch-angle formulation integrated over the modified Bessel function :math:`K_{5/3}`.
    """

    def compute_matrix(
        self,
        E_gamma_grid: np.ndarray,
        E_e_grid: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        r"""
        Compute the 2D differential emission kernel matrix for Synchrotron radiation.

        Parameters
        ----------
        E_gamma_grid : numpy.ndarray
            Emitted synchrotron photon energies with shape ``(N_gamma,)``, in :math:`\mathrm{GeV}`.
        E_e_grid : numpy.ndarray
            Primary electron total energies with shape ``(N_e,)``, in :math:`\mathrm{GeV}`.
        **kwargs : dict
            Required physical magnetic field parameters:

            - ``B_field`` : astropy.units.Quantity
                Ambient magnetic field strength. Units compatible with :py:data:`~saetass.units.MAGNETIC_FIELD`.
            - ``pitch_angle`` : str, optional
                Pitch-angle averaging model, ``'isotropic'`` or ``'perpendicular'``. Default is ``'isotropic'``.

        Returns
        -------
        kernel_matrix : numpy.ndarray
            Differential emission rate kernel with shape ``(N_gamma, N_e)``, in :math:`\mathrm{s^{-1}\,GeV^{-1}}`.

        Raises
        ------
        ValueError
            If ``B_field`` is missing or an invalid ``pitch_angle`` is supplied.
        """
        B_field = kwargs.get("B_field")
        pitch_angle = kwargs.get("pitch_angle", "isotropic").lower()

        if B_field is None:
            raise ValueError("AnalyticalSynchrotron requires 'B_field' in kwargs.")

        B_G = su.validate_quantity(
            B_field, u.G, "AnalyticalSynchrotron argument 'B_field'"
        ).value

        Eg = E_gamma_grid[:, np.newaxis]
        Ee = E_e_grid[np.newaxis, :]

        mec2 = (const.m_e * const.c**2).to_value(u.GeV)
        gamma_e = Ee / mec2

        e_charge = const.e.gauss.value
        m_e_g = const.m_e.cgs.value
        c_g = const.c.cgs.value
        hbar_g = const.hbar.cgs.value

        # Emissivity prefactor (s^-1)
        prefactor_rate = (
            np.sqrt(3.0) * (e_charge**3) * B_G / (const.h.cgs.value * m_e_g * c_g**2)
        )

        if pitch_angle == "perpendicular":
            # Exact formula for 90-degree pitch angle
            E_c_erg = 1.5 * hbar_g * e_charge * B_G / (m_e_g * c_g) * gamma_e**2
            E_c_GeV = (E_c_erg * u.erg).to_value(u.GeV)
            E_c_safe = np.where(E_c_GeV > 0.0, E_c_GeV, 1e-30)
            x = Eg / E_c_safe

            t_shift = np.logspace(-4, 2, 80)
            t = x[..., np.newaxis] + t_shift[np.newaxis, np.newaxis, :]
            integrand = scipy.special.kv(5.0 / 3.0, t)
            F_x = x * np.trapezoid(integrand, x=t, axis=-1)

            Rate_GeV = prefactor_rate * F_x / Eg

        elif pitch_angle == "isotropic":
            # Aharonian et al. (2010) pitch-angle averaged approximation (Astrophysical Standard)
            E_c_erg = 1.5 * hbar_g * e_charge * B_G / (m_e_g * c_g) * gamma_e**2
            E_c_GeV = (E_c_erg * u.erg).to_value(u.GeV)
            E_c_safe = np.where(E_c_GeV > 0.0, E_c_GeV, 1e-30)
            x = Eg / E_c_safe

            num = (
                1.808
                * x ** (1.0 / 3.0)
                * (1.0 + 2.21 * x ** (2.0 / 3.0) + 0.347 * x ** (4.0 / 3.0))
            )
            den = np.sqrt(1.0 + 3.4 * x ** (2.0 / 3.0)) * (
                1.0 + 1.353 * x ** (2.0 / 3.0) + 0.217 * x ** (4.0 / 3.0)
            )
            G_x = (num / den) * np.exp(-x)

            Rate_GeV = prefactor_rate * G_x / Eg

        else:
            raise ValueError(
                f"Unknown pitch_angle: '{pitch_angle}'. Supported options are: 'isotropic' or 'perpendicular'."
            )

        return Rate_GeV


# -----------------
# FACTORY FUNCTIONS
# -----------------


def get_hadronic_cross_section_model(name: str, **kwargs) -> HadronicCrossSectionModel:
    """
    Factory function to instantiate the requested hadronic cross-section model.

    Parameters
    ----------
    name : str
        The identifier of the hadronic model. Supported options are:

        - ``'kafexhiu'`` (or specific sub-models: ``'sibyll'``, ``'pythia8'``, ``'qgsjet'``, ``'geant4'``):
          Parametrizations from :cite:ct:`Kafexhiu2014`.
        - ``'aafragpy'`` (or ``'aafrag'``):
          LHC-tuned fragmentation model from :cite:ct:`Kachelriess2019`.
    **kwargs : dict
        Additional parameters passed to model constructors.

    Returns
    -------
    model : HadronicCrossSectionModel
        An instantiated hadronic cross-section model ready to compute interaction matrices.

    Raises
    ------
    ValueError
        If the requested model name is not recognized.
    """
    name_clean = name.lower()
    kafexhiu_submodels = ["kafexhiu", "sibyll", "pythia8", "qgsjet", "geant4"]

    if name_clean in kafexhiu_submodels:
        he_model = "sibyll" if name_clean == "kafexhiu" else name_clean
        return Kafexhiu2014(he_model=he_model)
    elif name_clean in ["aafragpy", "aafrag"]:
        return AAFragPyModel()
    else:
        raise ValueError(
            f"Cross-section model '{name}' not found. Supported options are: {kafexhiu_submodels + ['aafragpy']}."
        )


def get_leptonic_cross_section_model(name: str, **kwargs) -> LeptonicCrossSectionModel:
    """
    Factory function to instantiate the requested leptonic cross-section or emission kernel model.

    Parameters
    ----------
    name : str
        The identifier of the leptonic model. Supported options are:

        - ``'bremsstrahlung'`` (or ``'brems'``):
          Relativistic electron Bethe-Heitler Bremsstrahlung (:py:class:`AnalyticalBremsstrahlung`).
        - ``'inverse_compton'`` (or ``'ic'``):
          Full Klein-Nishina Inverse Compton scattering kernel (:py:class:`AnalyticalInverseCompton`).
        - ``'synchrotron'`` (or ``'sync'``):
          Synchrotron radiation emission kernel in magnetic fields (:py:class:`AnalyticalSynchrotron`).
    **kwargs : dict
        Additional parameters passed to model constructors.

    Returns
    -------
    model : LeptonicCrossSectionModel
        An instantiated leptonic cross-section model ready to compute interaction matrices.

    Raises
    ------
    ValueError
        If the requested model name is not recognized.
    """
    name_clean = name.lower()
    if name_clean in ["bremsstrahlung", "brems"]:
        return AnalyticalBremsstrahlung()
    elif name_clean in ["inverse_compton", "ic"]:
        return AnalyticalInverseCompton()
    elif name_clean in ["synchrotron", "sync"]:
        return AnalyticalSynchrotron()
    else:
        raise ValueError(
            f"Leptonic emission model '{name}' not found. Supported options are: ['bremsstrahlung', 'inverse_compton', 'synchrotron']."
        )
