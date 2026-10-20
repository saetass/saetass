"""
Cost of the emission calculator, for every emission mechanism.

``test_emission`` computes each emission end to end with the cross-section model of its
mechanism (both hadronic models for pion decay), on a grid of typical size, where most
of the cost is the cross-section matrix. ``test_emission_integration`` isolates the
integration over the distribution on a large grid, with a given cross-section matrix or
emission kernel: the integral weighted by the gas density (pion decay, neutrinos and
bremsstrahlung) and the direct-rate integral (inverse Compton and synchrotron) are the
two paths of every mechanism.
"""

import astropy.units as u
import numpy as np
import problems
import pytest

from saetass import Grid, Particle, State
from saetass import units as su
from saetass.utils.emissions import EmissionCalculator

#: Photon energies of the emissions.
N_OUT = 50


def _calculator(particle: str, n_cr: int, n_r: int):
    """An emission calculator and a power-law distribution of primaries from 10 GeV to 1 PeV."""
    mc2 = problems.MPC2 if particle == "proton" else problems.MEC2
    E_cr = np.logspace(1.0, 6.0, n_cr) * u.GeV
    p = np.sqrt(E_cr.value**2 + 2.0 * E_cr.value * mc2)
    r = np.linspace(0.1, 10.0, n_r) * u.pc
    state = State(
        grid=Grid(r_centers=r, p_centers=p * su.MOMENTUM),
        psi_p=np.outer(p**-2.2, np.ones(n_r)) * su.PSI_P,
        particle=Particle(particle),
    )
    calculator = EmissionCalculator(
        E_out_grid=np.logspace(-1.0, 4.0, N_OUT) * u.GeV,
        E_cr_grid=E_cr,
        r_grid=r,
        n_gas=np.ones(n_r) * u.cm**-3,
        particle=particle,
        distance=1 * u.kpc,
    )
    return calculator, state


def _check(flux):
    assert flux.shape == (N_OUT,)
    assert flux.unit.is_equivalent(u.cm**-2 * u.s**-1 * u.GeV**-1)
    assert np.all(np.isfinite(flux.value)) and np.any(flux.value > 0.0)


EMISSIONS = {
    "pion_decay-kafexhiu": (
        "proton",
        lambda calc, state: calc.compute_pion_decay_emission(state, model="kafexhiu"),
    ),
    "pion_decay-aafragpy": (
        "proton",
        lambda calc, state: calc.compute_pion_decay_emission(state, model="aafragpy"),
    ),
    "neutrino-aafragpy": (
        "proton",
        lambda calc, state: calc.compute_neutrino_emission(state, model="aafragpy"),
    ),
    "bremsstrahlung": (
        "electron",
        lambda calc, state: calc.compute_bremsstrahlung_emission(state),
    ),
    "inverse_compton": (
        "electron",
        lambda calc, state: calc.compute_inverse_compton_emission(
            state,
            eps_grid=problems.EPS_GRID,
            dn_deps=problems.photon_number_density(),
        ),
    ),
    "synchrotron": (
        "electron",
        lambda calc, state: calc.compute_synchrotron_emission(state, B_field=10 * u.uG),
    ),
}


@pytest.mark.parametrize("name", list(EMISSIONS))
def test_emission(bench, name):
    if name.endswith("aafragpy"):
        pytest.importorskip("aafragpy")
    particle, compute = EMISSIONS[name]
    calculator, state = _calculator(particle, n_cr=200, n_r=400)
    bench.group = "emissions"
    flux = bench.pedantic(compute, args=(calculator, state), rounds=5, warmup_rounds=1)
    _check(flux)


INTEGRATIONS = {
    "gas_density": (
        "proton",
        u.cm**2 / u.GeV,
        lambda calc, state, matrix: calc.compute_pion_decay_emission(
            state, custom_matrix=matrix
        ),
    ),
    "direct_rate": (
        "electron",
        1 / (u.s * u.GeV),
        lambda calc, state, kernel: calc.compute_inverse_compton_emission(
            state, custom_kernel=kernel
        ),
    ),
}


@pytest.mark.parametrize("path", list(INTEGRATIONS))
def test_emission_integration(bench, path):
    particle, unit, compute = INTEGRATIONS[path]
    calculator, state = _calculator(particle, n_cr=1000, n_r=1000)
    matrix = np.full((N_OUT, 1000), 1e-27) * unit
    bench.group = "emissions"
    flux = bench(compute, calculator, state, matrix)
    _check(flux)
