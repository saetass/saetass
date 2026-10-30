"""
This module provides the :py:class:`~saetass.utils.emissions.EmissionCalculator` class to compute observable non-thermal multi-messenger emissions (gamma-rays and neutrinos) from cosmic ray distributions.

The module cleanly decouples the static background physical environment and spatial/energy grids from the dynamic particle differential density encoded in :py:class:`~saetass.state.State`.
It performs high-performance vectorized integrations over the particle energy spectra and the spatial volume to compute physical emissivities and observable fluxes.

The structure and API mirror the energy losses module (:py:mod:`~saetass.utils.energy_losses`), supporting the following emission processes:

- Neutral pion decay gamma-rays (:py:meth:`~saetass.utils.emissions.EmissionCalculator.compute_pion_decay_emission`).
- Hadronic secondary neutrino emission (:py:meth:`~saetass.utils.emissions.EmissionCalculator.compute_neutrino_emission`).
- Relativistic electron Bremsstrahlung (:py:meth:`~saetass.utils.emissions.EmissionCalculator.compute_bremsstrahlung_emission`).
- Inverse Compton scattering (:py:meth:`~saetass.utils.emissions.EmissionCalculator.compute_inverse_compton_emission`).
- Non-thermal Synchrotron radiation (:py:meth:`~saetass.utils.emissions.EmissionCalculator.compute_synchrotron_emission`).
- Cumulative multi-component gamma-ray emission (:py:meth:`~saetass.utils.emissions.EmissionCalculator.compute_total_gamma_emission`).

Other key features include:

- Internal caching of individual emissivity and flux components for detailed multi-wavelength analysis.
- Flexible support for both point-source observable fluxes (applying inverse-square geometric dilution) and total volume-integrated photon production rates.
- Automatic integration with internal parametric cross-section models (:py:mod:`~saetass.utils.cross_sections`) and external libraries such as ``aafragpy`` and ``naima``, as well as support for user-supplied custom interaction kernels.
- Strict particle species validation based on :py:class:`~saetass.utils.energy_losses.Particle` to prevent unphysical calculations (e.g., pion decay from electrons).
"""

import logging
from collections.abc import Callable
from typing import Optional

import astropy.constants as const
import astropy.units as u
import numpy as np

from saetass import units as su
from saetass.state import State
from saetass.utils.cross_sections import (
    get_hadronic_cross_section_model,
    get_leptonic_cross_section_model,
)
from saetass.utils.energy_losses import Particle

logger = logging.getLogger(__name__)


class EmissionCalculator:
    r"""
    Main calculator for multi-messenger non-thermal emissions from cosmic ray distributions.

    This class is initialized with the static environment parameters (outgoing photon/neutrino energy grid, primary cosmic ray energy grid, spatial radial grid, and ambient gas density profile) as well as the particle species and optional source distance. It provides vectorized methods to compute differential emissivities and volume-integrated or diluted observable fluxes, automatically interfacing with cross-section models or accepting user-defined interaction matrices.

    Parameters
    ----------
    E_out_grid : astropy.units.Quantity
        Energy grid for the outgoing secondary particles (gamma-ray photons or neutrinos). Must be strictly positive (> 0). Units compatible with :py:data:`~saetass.units.ENERGY`.
    E_cr_grid : astropy.units.Quantity
        Kinetic energy grid of the primary cosmic rays (protons or electrons). Must be strictly positive (> 0). Units compatible with :py:data:`~saetass.units.ENERGY`.
    r_grid : astropy.units.Quantity
        Radial grid for spatial variation. Units compatible with :py:data:`~saetass.units.LENGTH`.
    n_gas : astropy.units.Quantity
        Ambient gas number density profile. Must match the 1D shape of ``r_grid``. Units compatible with :py:data:`~saetass.units.NUMBER_DENSITY`.
    particle : Particle or str
        Particle species identifier, chosen between ``"proton"`` (hadronic) or ``"electron"`` (leptonic). Determines validation for emission mechanisms.
    distance : astropy.units.Quantity, optional
        Distance from the observer to the astrophysical source. If provided, calculation methods return the observable differential flux at Earth, with units compatible with :math:`\mathrm{cm^{-2}\,s^{-1}\,GeV^{-1}}`. If ``None``, they return the total volume-integrated spectral production rate, with units compatible with :math:`\mathrm{s^{-1}\,GeV^{-1}}`. Units compatible with :py:data:`~saetass.units.LENGTH`. Default is ``None``.

    Raises
    ------
    ValueError
        If energy grids or distance contain non-positive values, or if the shape of ``n_gas`` does not match ``r_grid``.
    TypeError
        If an unsupported particle species is specified.
    """

    @u.quantity_input(
        E_out_grid=su.ENERGY,
        E_cr_grid=su.ENERGY,
        r_grid=su.LENGTH,
        n_gas=su.NUMBER_DENSITY,
        distance=su.LENGTH,
    )
    def __init__(
        self,
        E_out_grid: u.Quantity,
        E_cr_grid: u.Quantity,
        r_grid: u.Quantity,
        n_gas: u.Quantity,
        particle: Particle | str,
        distance: u.Quantity | None = None,
    ):
        self._check_parameters(E_out_grid, E_cr_grid, r_grid, n_gas, particle, distance)
        self._compute_kinematics()

        self._emissivity_components = {}
        self._flux_components = {}
        self._total_gamma_flux = None
        self._total_gamma_emissivity = None

    def _compute_kinematics(self) -> None:
        """Precompute differential bin widths, geometric volume elements, and dilution factor."""
        # Energy bin widths for primary cosmic rays
        self.dE_cr = np.gradient(self.E_cr_grid.to_value(u.GeV)) * u.GeV

        # Spatial radial shell thicknesses and differential volume elements (dV = 4 * pi * r^2 * dr)
        self.dr = np.gradient(self.r_grid.to_value(u.cm)) * u.cm
        self.dV = 4 * np.pi * (self.r_grid.to(u.cm) ** 2) * self.dr

        # Geometrical dilution factor: 1 / (4 * pi * d^2)
        if self.distance is not None:
            self.dilution_factor = 1 / (4 * np.pi * self.distance.to(u.cm) ** 2)
        else:
            self.dilution_factor = 1 * u.dimensionless_unscaled

        self._c_cgs = const.c.cgs.value

    def _check_parameters(
        self,
        E_out_grid: u.Quantity,
        E_cr_grid: u.Quantity,
        r_grid: u.Quantity,
        n_gas: u.Quantity,
        particle: Particle | str,
        distance: u.Quantity | None,
    ) -> None:
        """Store the inputs, already validated by ``quantity_input``, in the units used internally and check their values."""
        self.E_out_grid = E_out_grid.to(su.ENERGY)
        self.E_cr_grid = E_cr_grid.to(su.ENERGY)
        self.r_grid = r_grid.to(su.LENGTH)
        self.n_gas = n_gas.to(u.cm**-3)

        if np.any(self.E_out_grid.value <= 0):
            raise ValueError("E_out_grid energies must be strictly positive (> 0).")

        if np.any(self.E_cr_grid.value <= 0):
            raise ValueError("E_cr_grid energies must be strictly positive (> 0).")

        if self.n_gas.shape != self.r_grid.shape:
            raise ValueError(
                f"Shape mismatch: n_gas has shape {self.n_gas.shape}, but r_grid has shape {self.r_grid.shape}. "
                "n_gas must match spatial grid dimensions."
            )

        particle_str = particle.lower() if isinstance(particle, str) else particle
        self.particle = Particle(particle_str)
        self.particle_species = self.particle.species

        if distance is not None:
            self.distance = distance.to(u.kpc)
            if self.distance.value <= 0:
                raise ValueError("distance must be strictly positive (> 0).")
        else:
            self.distance = None

    def _integrate_volume_and_dilute(self, emissivity: u.Quantity) -> u.Quantity:
        r"""
        Integrate differential emissivity over spherical shell volumes and apply geometric dilution.

        Parameters
        ----------
        emissivity : astropy.units.Quantity
            Local differential emissivity with shape ``(len(E_out_grid), len(r_grid))``. Units compatible with :math:`\mathrm{cm^{-3}\,s^{-1}\,GeV^{-1}}`.

        Returns
        -------
        astropy.units.Quantity
            Observable flux at Earth with shape ``(len(E_out_grid),)`` if ``distance`` is set, with units compatible with :math:`\mathrm{cm^{-2}\,s^{-1}\,GeV^{-1}}`. Otherwise, total volume-integrated spectrum with shape ``(len(E_out_grid),)``, with units compatible with :math:`\mathrm{s^{-1}\,GeV^{-1}}`.
        """
        volume_integrated_spectrum = np.dot(
            emissivity.to_value(u.cm**-3 * u.s**-1 * u.GeV**-1),
            self.dV.to_value(u.cm**3),
        ) * (u.s**-1 * u.GeV**-1)

        final_flux = volume_integrated_spectrum * self.dilution_factor

        if self.distance is not None:
            return final_flux.to(u.cm**-2 * u.s**-1 * u.GeV**-1)
        return final_flux

    def _get_hadronic_cross_section(
        self, secondary: str, model_name: str
    ) -> u.Quantity:
        r"""
        Retrieve and evaluate a hadronic differential cross-section matrix using the model factory.

        Parameters
        ----------
        secondary : str
            Secondary particle type to produce, either ``"gamma"`` or ``"neutrino"``.
        model_name : str
            Identifier of the hadronic cross-section model (e.g. ``"aafragpy"``, ``"kafexhiu"``).

        Returns
        -------
        astropy.units.Quantity
            Differential cross-section matrix :math:`\frac{d\sigma}{dE}` with shape ``(len(E_out_grid), len(E_cr_grid))``. Units compatible with :math:`\mathrm{cm^{2}\,GeV^{-1}}`.
        """
        logger.info(
            f"Generating {secondary} hadronic cross-section matrix using {model_name}..."
        )

        # 1. Retrieve the model instance from the factory
        cs_model = get_hadronic_cross_section_model(model_name)

        # 2. Extract numerical grid arrays in GeV and compute cross-section matrix
        E_cr_val = self.E_cr_grid.to_value(u.GeV)
        E_out_val = self.E_out_grid.to_value(u.GeV)
        sigma_matrix = cs_model.compute_matrix(
            E_out_grid=E_out_val, E_cr_kin_grid=E_cr_val, secondary=secondary
        )

        return (sigma_matrix * u.mbarn / u.GeV).to(u.cm**2 / u.GeV)

    def _get_leptonic_cross_section(self, model_name: str, **kwargs) -> u.Quantity:
        r"""
        Retrieve and evaluate a leptonic differential emission kernel or cross-section matrix.

        Parameters
        ----------
        model_name : str
            Identifier of the leptonic interaction model (e.g. ``"bremsstrahlung"``, ``"inverse_compton"``, ``"synchrotron"``).
        **kwargs :
            Additional physical keyword arguments passed to the specific leptonic model (such as magnetic field strength, photon field densities, or target temperatures).

        Returns
        -------
        astropy.units.Quantity
            Differential interaction kernel with shape ``(len(E_out_grid), len(E_cr_grid))``. For Bremsstrahlung, it is a differential cross-section with units compatible with :math:`\mathrm{cm^{2}\,GeV^{-1}}`. For Inverse Compton and Synchrotron, it is an emission rate kernel with units compatible with :math:`\mathrm{s^{-1}\,GeV^{-1}}`.
        """
        logger.info(f"Generating leptonic cross-section matrix using {model_name}...")

        # 1. Retrieve the model instance from the factory
        cs_model = get_leptonic_cross_section_model(model_name)

        # 2. Extract numerical grid arrays in GeV and compute kernel matrix
        E_e_val = self.E_cr_grid.to_value(u.GeV)
        E_gamma_val = self.E_out_grid.to_value(u.GeV)

        sigma_matrix = cs_model.compute_matrix(
            E_gamma_grid=E_gamma_val, E_e_grid=E_e_val, **kwargs
        )

        # IC and Synchrotron kernels return differential rates (s^-1 GeV^-1).
        # Bremsstrahlung returns differential cross sections (cm^2 GeV^-1).
        if model_name.lower() in ["inverse_compton", "ic", "synchrotron", "sync"]:
            return sigma_matrix * u.s**-1 / u.GeV
        else:
            return sigma_matrix * u.cm**2 / u.GeV

    def _convert_psip_to_dndE(self, psi_p: np.ndarray) -> np.ndarray:
        r"""
        Convert SAETASS momentum-space differential density :math:`\psi(p)` to energy-space differential density :math:`\frac{dn}{dE}`.

        The SAETASS state stores the differential density as :math:`\psi(p) = \frac{dn}{dp} = 4\pi p^2 f_\mathrm{ps}(p)`, which is passed here as bare floats in :math:`\mathrm{cm^{-3}\,(GeV/c)^{-1}}`. This method converts it to the differential number density :math:`\frac{dn}{dE} = \psi(p)\,\frac{dp}{dE}` in :math:`\mathrm{cm^{-3}\,GeV^{-1}}` using the relativistic Jacobian:

        .. math::
            \frac{dp}{dE} = \frac{E_{\mathrm{tot}}}{p\,c^2} = \frac{E + m\,c^2}{c\,\sqrt{E^2 + 2\,E\,m\,c^2}}

        Parameters
        ----------
        psi_p : numpy.ndarray
            Cosmic ray differential density with shape ``(len(E_cr_grid), len(r_grid))``, in :math:`\mathrm{cm^{-3}\,(GeV/c)^{-1}}`.

        Returns
        -------
        numpy.ndarray
            Differential number density with shape ``(len(E_cr_grid), len(r_grid))``, in :math:`\mathrm{cm^{-3}\,GeV^{-1}}`.
        """
        # 1. Particle rest mass energy in GeV
        m = (
            (const.m_p * const.c**2).to(u.GeV)
            if self.particle_species == "hadronic"
            else (const.m_e * const.c**2).to(u.GeV)
        )

        # 2. Total relativistic energy in GeV
        E_tot = self.E_cr_grid + m

        # 3. Momentum p in GeV/c
        p = np.sqrt(E_tot**2 - m**2) / const.c

        # 4. Kinematic Jacobian dp/dE = E_tot / (p * c^2)
        dp_dE = E_tot / (p * const.c**2)

        # 5. Multiply psi(p) (shape N_cr, N_r) by the 1D Jacobian (shape N_cr, 1)
        # Note: psi(p) is passed in units of cm^-3 (GeV/c)^-1.
        # dp/dE has units of c^-1, equivalent to (GeV/c) / GeV.
        dp_dE_val = dp_dE.to_value(1 / const.c)
        dn_dE = psi_p * dp_dE_val[:, np.newaxis]

        return dn_dE

    _convert_fp_to_dndE = _convert_psip_to_dndE

    def _compute_gas_density_dependent_emission(
        self,
        state: State,
        diff_cross_section: u.Quantity,
        component_name: str,
        threshold_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> u.Quantity:
        r"""
        Compute emissivity and integrated flux for target gas density-dependent processes.

        This calculation applies to collisional processes such as hadronic pion decay and relativistic electron Bremsstrahlung, evaluating the emissivity:

        .. math::
            j(E_{\mathrm{out}}, r) = c\,n_{\mathrm{gas}}(r)\int \frac{dn}{dE_{\mathrm{cr}}}(E_{\mathrm{cr}}, r)\,\frac{d\sigma}{dE_{\mathrm{out}}}(E_{\mathrm{out}}, E_{\mathrm{cr}})\,dE_{\mathrm{cr}}

        Parameters
        ----------
        state : :py:class:`~saetass.state.State`
            Cosmic ray simulation state containing the differential density :math:`\psi(p)`.
        diff_cross_section : astropy.units.Quantity
            Differential cross-section matrix with shape ``(len(E_out_grid), len(E_cr_grid))``. Units compatible with :math:`\mathrm{cm^2\,GeV^{-1}}`.
        component_name : str
            Identifier used to store and cache the resulting emissivity and flux components.
        threshold_kernel : callable, optional
            Differential cross-section ``kernel(E_out, E_cr)`` of the model, taking bare energies in GeV and returning the matrix with shape ``(len(E_out), len(E_cr))`` in :math:`\mathrm{cm^2\,GeV^{-1}}`, for kernels that vanish below the kinematic threshold :math:`E_{\mathrm{cr}} = E_{\mathrm{out}}` and rise steeply above it, such as Bremsstrahlung. The grid cell that contains the threshold is then integrated from the threshold itself with a Gauss-Legendre quadrature; otherwise the trapezoidal rule misses the part of that cell above the threshold and the result oscillates with the position of :math:`E_{\mathrm{out}}` between the nodes of ``E_cr_grid``. Default is ``None``.

        Returns
        -------
        astropy.units.Quantity
            Volume-integrated spectrum or distance-diluted flux at Earth.

        Raises
        ------
        ValueError
            If the shape of ``state.psi_p`` does not match ``(len(E_cr_grid), len(r_grid))``.
        """
        psi_cr = state.psi_p.to_value(
            u.cm**-3 / (u.GeV / const.c)
        )  # Shape: (N_cr, N_r)
        expected_shape = (len(self.E_cr_grid), len(self.r_grid))
        if psi_cr.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch in cosmic ray State: expected {expected_shape}, got {psi_cr.shape}."
            )

        dn_dE = self._convert_psip_to_dndE(psi_cr)

        # 1. Extract raw numerical values for high-performance vectorized operations
        sigma_val = diff_cross_section.to_value(u.cm**2 / u.GeV)  # Shape: (N_out, N_cr)
        E_cr_val = self.E_cr_grid.to_value(u.GeV)  # Shape: (N_cr,)

        # 2. Broadcast to 3D shape (N_out, N_cr, N_r) to align cross-sections with spatial dn/dE
        integrand = sigma_val[:, :, np.newaxis] * dn_dE[np.newaxis, :, :]

        # 3. Trapezoidal integration along the cosmic ray energy axis (axis=1) -> Shape: (N_out, N_r)
        integral_E = np.trapezoid(integrand, x=E_cr_val, axis=1)

        if threshold_kernel is not None:
            self._integrate_from_threshold(
                integral_E, sigma_val, dn_dE, threshold_kernel
            )

        # 4. Emissivity calculation: j = c * n_gas * integral
        emissivity = (
            self._c_cgs * self.n_gas.to_value(u.cm**-3)[np.newaxis, :] * integral_E
        ) * (u.cm**-3 * u.s**-1 * u.GeV**-1)

        self._emissivity_components[component_name] = emissivity
        flux = self._integrate_volume_and_dilute(emissivity)
        self._flux_components[component_name] = flux

        return flux

    def _integrate_from_threshold(
        self,
        integral_E: np.ndarray,
        sigma_val: np.ndarray,
        dn_dE: np.ndarray,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
        num_points: int = 8,
    ) -> None:
        r"""
        Correct, in place, the trapezoidal integral over the cell that contains the kinematic threshold.

        For each :math:`E_{\mathrm{out}}` inside the cell :math:`[E_k, E_{k+1}]` of ``E_cr_grid``, the trapezoidal rule uses the kernel at :math:`E_k < E_{\mathrm{out}}`, where it vanishes, over the whole cell.
        This replaces that contribution by the integral from the threshold, :math:`[E_{\mathrm{out}}, E_{k+1}]`, computed with a Gauss-Legendre quadrature in :math:`t`, with :math:`E = E_{\mathrm{out}} + (E_{k+1} - E_{\mathrm{out}})\,t^2` to cluster the nodes where the kernel rises, and :math:`dn/dE` interpolated linearly in :math:`\log E`-:math:`\log(dn/dE)`.

        Parameters
        ----------
        integral_E : numpy.ndarray
            Integral over the cosmic ray energies with shape ``(len(E_out_grid), len(r_grid))``, in :math:`\mathrm{cm^{-1}\,GeV^{-1}}`, corrected in place.
        sigma_val : numpy.ndarray
            Differential cross-section matrix with shape ``(len(E_out_grid), len(E_cr_grid))``, in :math:`\mathrm{cm^2\,GeV^{-1}}`.
        dn_dE : numpy.ndarray
            Differential number density with shape ``(len(E_cr_grid), len(r_grid))``, in :math:`\mathrm{cm^{-3}\,GeV^{-1}}`.
        kernel : callable
            Differential cross-section ``kernel(E_out, E_cr)`` in :math:`\mathrm{cm^2\,GeV^{-1}}`, for bare energies in GeV.
        num_points : int, optional
            Number of quadrature nodes in the cell. Default is ``8``.
        """
        E_cr = self.E_cr_grid.to_value(u.GeV)
        E_out = self.E_out_grid.to_value(u.GeV)
        k = np.searchsorted(E_cr, E_out, side="right") - 1
        j = np.flatnonzero((k >= 0) & (k < E_cr.size - 1))
        if j.size == 0:
            return
        k = k[j]
        lower, upper = dn_dE[k], dn_dE[k + 1]  # Shape: (N_j, N_r)

        # Quadrature nodes and weights on [E_out, E_{k+1}]
        t, weights = np.polynomial.legendre.leggauss(num_points)
        t, weights = 0.5 * (t + 1.0), 0.5 * weights
        span = (E_cr[k + 1] - E_out[j])[:, np.newaxis]
        E_nodes = E_out[j, np.newaxis] + span * t**2  # Shape: (N_j, M)
        jacobian = 2.0 * t * span * weights

        # Kernel at the nodes of each E_out: the diagonal blocks of the full matrix
        matrix = kernel(E_out[j], E_nodes.ravel()).reshape(j.size, j.size, num_points)
        sigma_nodes = matrix[np.arange(j.size), np.arange(j.size)]  # Shape: (N_j, M)

        w = (
            np.log(E_nodes / E_cr[k, np.newaxis])
            / np.log(E_cr[k + 1] / E_cr[k])[:, np.newaxis]
        )
        w = w[:, :, np.newaxis]
        lo, up = lower[:, np.newaxis, :], upper[:, np.newaxis, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_log = lo ** (1.0 - w) * up**w
        dn_dE_nodes = np.where((lo > 0.0) & (up > 0.0), log_log, lo + w * (up - lo))
        partial = np.sum(
            (jacobian * sigma_nodes)[:, :, np.newaxis] * dn_dE_nodes, axis=1
        )

        f_lower = sigma_val[j, k, np.newaxis] * lower
        f_upper = sigma_val[j, k + 1, np.newaxis] * upper
        width = (E_cr[k + 1] - E_cr[k])[:, np.newaxis]
        integral_E[j] += partial - 0.5 * (f_lower + f_upper) * width

    def _compute_direct_rate_emission(
        self, state: State, emission_rate_kernel: u.Quantity, component_name: str
    ) -> u.Quantity:
        r"""
        Compute emissivity and integrated flux for field-dependent direct-rate processes.

        This calculation applies to field-interaction processes (such as Inverse Compton scattering on background radiation fields and Synchrotron radiation in magnetic fields) where the interaction rate kernel :math:`\frac{dN}{dt\,dE_{\mathrm{out}}}` already incorporates target field properties:

        .. math::
            j(E_{\mathrm{out}}, r) = \int \frac{dn}{dE_{\mathrm{cr}}}(E_{\mathrm{cr}}, r)\,\frac{dN}{dt\,dE_{\mathrm{out}}}(E_{\mathrm{out}}, E_{\mathrm{cr}})\,dE_{\mathrm{cr}}

        Parameters
        ----------
        state : :py:class:`~saetass.state.State`
            Cosmic ray simulation state containing the differential density :math:`\psi(p)`.
        emission_rate_kernel : astropy.units.Quantity
            Differential emission rate kernel with shape ``(len(E_out_grid), len(E_cr_grid))``. Units compatible with :math:`\mathrm{s^{-1}\,GeV^{-1}}`.
        component_name : str
            Identifier used to store and cache the resulting emissivity and flux components.

        Returns
        -------
        astropy.units.Quantity
            Volume-integrated spectrum or distance-diluted flux at Earth.

        Raises
        ------
        ValueError
            If the shape of ``state.psi_p`` does not match ``(len(E_cr_grid), len(r_grid))``.
        """
        psi_cr = state.psi_p.to_value(
            u.cm**-3 / (u.GeV / const.c)
        )  # Shape: (N_cr, N_r)
        expected_shape = (len(self.E_cr_grid), len(self.r_grid))
        if psi_cr.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch in cosmic ray State: expected {expected_shape}, got {psi_cr.shape}."
            )

        dn_dE = self._convert_psip_to_dndE(psi_cr)

        # 1. Extract raw numerical values for high-performance vectorized operations
        kernel_val = emission_rate_kernel.to_value(
            u.s**-1 * u.GeV**-1
        )  # Shape: (N_out, N_cr)
        E_cr_val = self.E_cr_grid.to_value(u.GeV)  # Shape: (N_cr,)

        # 2. Broadcast to 3D shape (N_out, N_cr, N_r) to align rate kernel with spatial dn/dE
        integrand = kernel_val[:, :, np.newaxis] * dn_dE[np.newaxis, :, :]

        # 3. Trapezoidal integration along the lepton energy axis (axis=1) -> Shape: (N_out, N_r)
        emissivity_val = np.trapezoid(integrand, x=E_cr_val, axis=1)

        # 4. Assign physical emissivity units (cm^-3 * s^-1 * GeV^-1)
        emissivity = emissivity_val * (u.cm**-3 * u.s**-1 * u.GeV**-1)

        self._emissivity_components[component_name] = emissivity
        flux = self._integrate_volume_and_dilute(emissivity)
        self._flux_components[component_name] = flux

        return flux

    @u.quantity_input(custom_matrix=su.DIFFERENTIAL_CROSS_SECTION)
    def compute_pion_decay_emission(
        self,
        state: State,
        model: str = "aafragpy",
        custom_matrix: Optional[u.Quantity] = None,
    ) -> u.Quantity:
        r"""
        Compute hadronic gamma-ray emission from inelastic proton-proton collisions via neutral pion decay.

        Evaluates secondary gamma-ray production from the inclusive hadronic reaction:

        .. math::
            p + p \to \pi^0 + X, \quad \text{with} \quad \pi^0 \to 2\gamma

        where :math:`X` represents all other final-state hadrons. Calculates the local differential emissivity and integrates over the source volume, storing the result in the internal component cache under ``"pion_decay"``.

        Parameters
        ----------
        state : :py:class:`~saetass.state.State`
            Cosmic ray state containing the proton distribution function.
        model : str, optional
            Hadronic cross-section parametrization model:
            - ``"aafragpy"``: LHC-tuned Monte Carlo fragmentation model (:cite:ct:`Koldobskiy2021`, :cite:ct:`Kachelriess2023`, :cite:ct:`Kachelriess2019`).
            - ``"kafexhiu"``: Semi-analytical parameterization from threshold to PeV energies (:cite:ct:`Kafexhiu2014`).
            Default is ``"aafragpy"``.
        custom_matrix : astropy.units.Quantity, optional
            User-provided differential cross-section matrix with shape ``(len(E_out_grid), len(E_cr_grid))``. Units compatible with :math:`\mathrm{cm^2\,GeV^{-1}}`. If supplied, ``model`` is ignored. Default is ``None``.

        Returns
        -------
        flux : astropy.units.Quantity
            Observable differential flux at Earth with shape ``(len(E_out_grid),)`` if ``distance`` is set, with units compatible with :math:`\mathrm{cm^{-2}\,s^{-1}\,GeV^{-1}}`, or volume-integrated production rate otherwise, with units compatible with :math:`\mathrm{s^{-1}\,GeV^{-1}}`.

        Raises
        ------
        TypeError
            If the calculator was initialized for non-hadronic particles (e.g., electrons).
        ValueError
            If ``state`` array dimensions mismatch grid dimensions.
        """
        if self.particle_species != "hadronic":
            raise TypeError(
                "Pion decay emission can only be computed for hadronic particles."
            )

        if custom_matrix is not None:
            sigma = custom_matrix
        else:
            sigma = self._get_hadronic_cross_section("gamma", model)

        flux = self._compute_gas_density_dependent_emission(state, sigma, "pion_decay")
        logger.debug(f"Computed pion decay emission at stage: {state.stage_name}")
        return flux

    @u.quantity_input(custom_matrix=su.DIFFERENTIAL_CROSS_SECTION)
    def compute_neutrino_emission(
        self,
        state: State,
        model: str = "aafragpy",
        custom_matrix: Optional[u.Quantity] = None,
    ) -> u.Quantity:
        r"""
        Compute hadronic all-flavor neutrino emission from inelastic proton-proton collisions via charged pion decay chains.

        Evaluates secondary neutrino production from inclusive charged pion production:

        .. math::
            p + p \to \pi^\pm + X

        followed by the charged pion and muon decay chains:

        .. math::
            \begin{aligned}
            \pi^+ &\to \mu^+ + \nu_\mu \to e^+ + \nu_e + \bar{\nu}_\mu + \nu_\mu \\
            \pi^- &\to \mu^- + \bar{\nu}_\mu \to e^- + \bar{\nu}_e + \nu_\mu + \bar{\nu}_\mu
            \end{aligned}

        Calculates the local differential neutrino emissivity and integrates over the source volume, storing the result in the internal component cache under ``"neutrino"``.

        Parameters
        ----------
        state : :py:class:`~saetass.state.State`
            Cosmic ray state containing the proton distribution function.
        model : str, optional
            Hadronic production model:
            - ``"aafragpy"``: LHC-tuned Monte Carlo fragmentation model (:cite:ct:`Koldobskiy2021`, :cite:ct:`Kachelriess2023`, :cite:ct:`Kachelriess2019`).
            - ``"kafexhiu"``: Semi-analytical parameterization from threshold to PeV energies (:cite:ct:`Kafexhiu2014`).
            Default is ``"aafragpy"``.
        custom_matrix : astropy.units.Quantity, optional
            User-provided differential cross-section matrix with shape ``(len(E_out_grid), len(E_cr_grid))``. Units compatible with :math:`\mathrm{cm^2\,GeV^{-1}}`. If supplied, ``model`` is ignored. Default is ``None``.

        Returns
        -------
        flux : astropy.units.Quantity
            Observable differential flux at Earth with shape ``(len(E_out_grid),)`` if ``distance`` is set, with units compatible with :math:`\mathrm{cm^{-2}\,s^{-1}\,GeV^{-1}}`, or volume-integrated production rate otherwise, with units compatible with :math:`\mathrm{s^{-1}\,GeV^{-1}}`.

        Raises
        ------
        TypeError
            If the calculator was initialized for non-hadronic particles (e.g., electrons).
        ValueError
            If ``state`` array dimensions mismatch grid dimensions.
        """
        if self.particle_species != "hadronic":
            raise TypeError(
                "Neutrino emission can only be computed for hadronic particles."
            )

        if custom_matrix is not None:
            sigma = custom_matrix
        else:
            sigma = self._get_hadronic_cross_section("neutrino", model)

        flux = self._compute_gas_density_dependent_emission(state, sigma, "neutrino")
        logger.debug(f"Computed neutrino emission at stage: {state.stage_name}")
        return flux

    @u.quantity_input(custom_matrix=su.DIFFERENTIAL_CROSS_SECTION)
    def compute_bremsstrahlung_emission(
        self,
        state: State,
        model: str = "bremsstrahlung",
        custom_matrix: Optional[u.Quantity] = None,
        **kwargs,
    ) -> u.Quantity:
        r"""
        Compute leptonic gamma-ray emission from relativistic electron Bremsstrahlung (:cite:ct:`BlumenthalGould1970`).

        Evaluates radiative emission from relativistic electrons scattering in the Coulomb fields of ambient gas ions/nuclei (:math:`Z`) and electrons:

        .. math::
            e^- + Z \to e^- + Z + \gamma \quad \text{and} \quad e^- + e^- \to e^- + e^- + \gamma

        Calculates the differential Bremsstrahlung emissivity on ambient gas and integrates over the source volume, storing the result in the internal component cache under ``"bremsstrahlung"``.

        Parameters
        ----------
        state : :py:class:`~saetass.state.State`
            Cosmic ray state containing the electron distribution function.
        model : str, optional
            Bremsstrahlung cross-section model name. Default is ``"bremsstrahlung"``.
        custom_matrix : astropy.units.Quantity, optional
            User-provided differential cross-section matrix with shape ``(len(E_out_grid), len(E_cr_grid))``. Units compatible with :math:`\mathrm{cm^2\,GeV^{-1}}`. If supplied, ``model`` is ignored. Default is ``None``.
        **kwargs :
            Additional physical keyword arguments passed to the cross-section model.

        Returns
        -------
        flux : astropy.units.Quantity
            Observable differential flux at Earth with shape ``(len(E_out_grid),)`` if ``distance`` is set, with units compatible with :math:`\mathrm{cm^{-2}\,s^{-1}\,GeV^{-1}}`, or volume-integrated production rate otherwise, with units compatible with :math:`\mathrm{s^{-1}\,GeV^{-1}}`.

        Raises
        ------
        TypeError
            If the calculator was initialized for non-leptonic particles (e.g., protons).
        ValueError
            If ``state`` array dimensions mismatch grid dimensions.
        """
        if self.particle_species != "leptonic":
            raise TypeError(
                "Bremsstrahlung emission can only be computed for leptonic particles."
            )

        kernel = None
        if custom_matrix is not None:
            sigma = custom_matrix
        else:
            sigma = self._get_leptonic_cross_section(model, **kwargs)
            cs_model = get_leptonic_cross_section_model(model)

            # The cross-section rises steeply from zero at E_e = E_gamma
            def kernel(E_gamma, E_e):
                return cs_model.compute_matrix(
                    E_gamma_grid=E_gamma, E_e_grid=E_e, **kwargs
                )

        flux = self._compute_gas_density_dependent_emission(
            state, sigma, "bremsstrahlung", threshold_kernel=kernel
        )
        logger.debug(f"Computed Bremsstrahlung emission at stage: {state.stage_name}")
        return flux

    @u.quantity_input(custom_kernel=u.s**-1 / u.GeV)
    def compute_inverse_compton_emission(
        self,
        state: State,
        model: str = "inverse_compton",
        custom_kernel: Optional[u.Quantity] = None,
        **kwargs,
    ) -> u.Quantity:
        r"""
        Compute leptonic gamma-ray emission from Inverse Compton scattering (:cite:ct:`BlumenthalGould1970`).

        Evaluates the upscattering of ambient low-energy target photons (e.g., CMB, infrared, optical) by relativistic electrons to gamma-ray energies:

        .. math::
            e^- + \gamma_{\mathrm{target}} \to e^- + \gamma_{\mathrm{IC}}

        Calculates the differential Inverse Compton emissivity on target photon fields and integrates over the source volume, storing the result in the internal component cache under ``"inverse_compton"``.

        Parameters
        ----------
        state : :py:class:`~saetass.state.State`
            Cosmic ray state containing the electron distribution function.
        model : str, optional
            Inverse Compton kernel model name. Default is ``"inverse_compton"``.
        custom_kernel : astropy.units.Quantity, optional
            User-provided differential emission rate kernel with shape ``(len(E_out_grid), len(E_cr_grid))``. Units compatible with :math:`\mathrm{s^{-1}\,GeV^{-1}}`. If supplied, ``model`` is ignored. Default is ``None``.
        **kwargs :
            Additional photon field parameters passed to the interaction model (such as radiation field temperatures and energy densities).

        Returns
        -------
        flux : astropy.units.Quantity
            Observable differential flux at Earth with shape ``(len(E_out_grid),)`` if ``distance`` is set, with units compatible with :math:`\mathrm{cm^{-2}\,s^{-1}\,GeV^{-1}}`, or volume-integrated production rate otherwise, with units compatible with :math:`\mathrm{s^{-1}\,GeV^{-1}}`.

        Raises
        ------
        TypeError
            If the calculator was initialized for non-leptonic particles (e.g., protons).
        ValueError
            If ``state`` array dimensions mismatch grid dimensions.
        """
        if self.particle_species != "leptonic":
            raise TypeError(
                "Inverse Compton emission can only be computed for leptonic particles."
            )

        if custom_kernel is not None:
            emissivity_kernel = custom_kernel
        else:
            emissivity_kernel = self._get_leptonic_cross_section(model, **kwargs)

        flux = self._compute_direct_rate_emission(
            state, emissivity_kernel, "inverse_compton"
        )
        logger.debug(f"Computed Inverse Compton emission at stage: {state.stage_name}")
        return flux

    @u.quantity_input(custom_kernel=u.s**-1 / u.GeV)
    def compute_synchrotron_emission(
        self,
        state: State,
        model: str = "synchrotron",
        custom_kernel: Optional[u.Quantity] = None,
        **kwargs,
    ) -> u.Quantity:
        r"""
        Compute leptonic non-thermal Synchrotron emission from relativistic electrons (:cite:ct:`Ginzburg1979`, :cite:ct:`BlumenthalGould1970`).

        Evaluates magnetobremsstrahlung radiation emitted by relativistic electrons gyrating in ambient magnetic fields:

        .. math::
            e^- + B \to e^- + \gamma_{\mathrm{syn}}

        Calculates the differential Synchrotron emissivity in ambient magnetic fields and integrates over the source volume, storing the result in the internal component cache under ``"synchrotron"``.

        Parameters
        ----------
        state : :py:class:`~saetass.state.State`
            Cosmic ray state containing the electron distribution function.
        model : str, optional
            Synchrotron emission model name. Default is ``"synchrotron"``.
        custom_kernel : astropy.units.Quantity, optional
            User-provided differential emission rate kernel with shape ``(len(E_out_grid), len(E_cr_grid))``. Units compatible with :math:`\mathrm{s^{-1}\,GeV^{-1}}`. If supplied, ``model`` is ignored. Default is ``None``.
        **kwargs :
            Additional magnetic field parameters passed to the interaction model (e.g., ``B_field``).

        Returns
        -------
        flux : astropy.units.Quantity
            Observable differential flux at Earth with shape ``(len(E_out_grid),)`` if ``distance`` is set, with units compatible with :math:`\mathrm{cm^{-2}\,s^{-1}\,GeV^{-1}}`, or volume-integrated production rate otherwise, with units compatible with :math:`\mathrm{s^{-1}\,GeV^{-1}}`.

        Raises
        ------
        TypeError
            If the calculator was initialized for non-leptonic particles (e.g., protons).
        ValueError
            If ``state`` array dimensions mismatch grid dimensions.
        """
        if self.particle_species != "leptonic":
            raise TypeError(
                "Synchrotron emission can only be computed for leptonic particles."
            )

        if custom_kernel is not None:
            emissivity_kernel = custom_kernel
        else:
            emissivity_kernel = self._get_leptonic_cross_section(model, **kwargs)

        flux = self._compute_direct_rate_emission(
            state, emissivity_kernel, "synchrotron"
        )
        logger.debug(f"Computed Synchrotron emission at stage: {state.stage_name}")
        return flux

    def compute_total_gamma_emission(self) -> u.Quantity:
        r"""
        Compute cumulative total gamma-ray emission by summing all previously computed gamma-ray mechanisms.

        Aggregates computed components from the set ``{"pion_decay", "bremsstrahlung", "inverse_compton", "synchrotron"}``. Stores the resulting total flux and emissivity in :py:attr:`_total_gamma_flux` and :py:attr:`_total_gamma_emissivity`.

        Returns
        -------
        flux_total : astropy.units.Quantity
            Total observable differential gamma-ray flux at Earth with shape ``(len(E_out_grid),)`` if ``distance`` is set, with units compatible with :math:`\mathrm{cm^{-2}\,s^{-1}\,GeV^{-1}}`, or total volume-integrated spectrum otherwise, with units compatible with :math:`\mathrm{s^{-1}\,GeV^{-1}}`.

        Raises
        ------
        RuntimeError
            If no gamma-ray emission mechanisms have been computed prior to calling this method.
        """
        gamma_keys = ["pion_decay", "bremsstrahlung", "inverse_compton", "synchrotron"]
        computed_keys = [k for k in gamma_keys if k in self._flux_components]

        if not computed_keys:
            raise RuntimeError("No gamma-ray emission mechanisms have been computed.")

        flux_total = sum(self._flux_components[k] for k in computed_keys)
        emissivity_total = sum(self._emissivity_components[k] for k in computed_keys)

        self._total_gamma_flux = flux_total
        self._total_gamma_emissivity = emissivity_total

        logger.debug("Total gamma-ray emission computed")
        return flux_total

    def get_emissivity_component(self, component: str) -> u.Quantity:
        r"""
        Retrieve the cached 2D differential emissivity matrix for a specific emission mechanism.

        Parameters
        ----------
        component : str
            Identifier of the component: ``"pion_decay"``, ``"neutrino"``, ``"bremsstrahlung"``, ``"inverse_compton"``, ``"synchrotron"``, or ``"total_gamma"``.

        Returns
        -------
        astropy.units.Quantity
            Differential emissivity matrix with shape ``(len(E_out_grid), len(r_grid))``. Units compatible with :math:`\mathrm{cm^{-3}\,s^{-1}\,GeV^{-1}}`.

        Raises
        ------
        KeyError
            If the requested component has not been computed.
        """
        if component == "total_gamma":
            if self._total_gamma_emissivity is None:
                raise KeyError(
                    "Total gamma emissivity has not been computed yet. Call compute_total_gamma_emission() first."
                )
            return self._total_gamma_emissivity

        if component not in self._emissivity_components:
            raise KeyError(
                f"Component '{component}' has not been computed. "
                f"Available computed components: {list(self._emissivity_components.keys())}"
            )
        return self._emissivity_components[component]

    def get_flux_component(self, component: str) -> u.Quantity:
        r"""
        Retrieve the cached 1D integrated flux or production rate spectrum for a specific emission mechanism.

        Parameters
        ----------
        component : str
            Identifier of the component: ``"pion_decay"``, ``"neutrino"``, ``"bremsstrahlung"``, ``"inverse_compton"``, ``"synchrotron"``, or ``"total_gamma"``.

        Returns
        -------
        astropy.units.Quantity
            Integrated flux with shape ``(len(E_out_grid),)`` if ``distance`` is set, with units compatible with :math:`\mathrm{cm^{-2}\,s^{-1}\,GeV^{-1}}`, or production rate otherwise, with units compatible with :math:`\mathrm{s^{-1}\,GeV^{-1}}`.

        Raises
        ------
        KeyError
            If the requested component has not been computed.
        """
        if component == "total_gamma":
            if self._total_gamma_flux is None:
                raise KeyError(
                    "Total gamma flux has not been computed yet. Call compute_total_gamma_emission() first."
                )
            return self._total_gamma_flux

        if component not in self._flux_components:
            raise KeyError(
                f"Component '{component}' has not been computed. "
                f"Available computed components: {list(self._flux_components.keys())}"
            )
        return self._flux_components[component]

    def get_all_emissivities(self) -> dict[str, u.Quantity]:
        r"""
        Return a copy of all cached differential emissivity components.

        Returns
        -------
        dict of str to astropy.units.Quantity
            Dictionary mapping component names to their 2D emissivity matrices with shape ``(len(E_out_grid), len(r_grid))``. Units compatible with :math:`\mathrm{cm^{-3}\,s^{-1}\,GeV^{-1}}`.
        """
        emissivities = self._emissivity_components.copy()
        if self._total_gamma_emissivity is not None:
            emissivities["total_gamma"] = self._total_gamma_emissivity
        return emissivities

    def get_all_fluxes(self) -> dict[str, u.Quantity]:
        r"""
        Return a copy of all cached 1D integrated flux or production rate components.

        Returns
        -------
        dict of str to astropy.units.Quantity
            Dictionary mapping component names to their 1D flux spectra with shape ``(len(E_out_grid),)``, with units compatible with :math:`\mathrm{cm^{-2}\,s^{-1}\,GeV^{-1}}` if ``distance`` is set or :math:`\mathrm{s^{-1}\,GeV^{-1}}` otherwise.
        """
        fluxes = self._flux_components.copy()
        if self._total_gamma_flux is not None:
            fluxes["total_gamma"] = self._total_gamma_flux
        return fluxes
