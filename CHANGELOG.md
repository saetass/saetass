# Changelog

All notable changes to SAETASS are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and SAETASS follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before version 1.0.0, minor releases may break the public API: such changes are listed under **Changed** or **Removed** and marked **Breaking**.

## [Unreleased]

### Added

- Compiled Numba kernels for the advection, loss and diffusion operators. They run several times faster than the NumPy code of 0.5 on one thread and reproduce its results to rounding error ([#64](https://github.com/saetass/saetass/pull/64)).
- Adaptive threads for the kernels: the number of threads follows the work of each simulation and the idle cores of the machine, and `SAETASS_MAX_THREADS` or `saetass.parallel.set_max_threads` sets a limit ([#64](https://github.com/saetass/saetass/pull/64)).
- `saetass.sweep.run_sweep` runs independent simulations, such as parameter scans, in parallel processes ([#64](https://github.com/saetass/saetass/pull/64)).
- `saetass.Separable` declares time-dependent parameters as a sum of fixed profiles times time factors, which advection and losses advance in one kernel call per step. It can be written as a formula, and `EnergyLossCalculator.get_momentum_loss_rate(time_factors=...)` returns one ([#64](https://github.com/saetass/saetass/pull/64)).
- The log messages of SAETASS go through its Rich console: a summary of each simulation at the `INFO` level, the default, and the setup, CFL sub-cycles, thread teams and kernel compilations at the `DEBUG` level. The level is set with `saetass.cli.set_log_level` or `SAETASS_LOG_LEVEL` ([#68](https://github.com/saetass/saetass/pull/68)).
- `Solver(quiet=True)`, `saetass.cli.set_quiet` and `SAETASS_QUIET` hide the banner and the progress bar, which child processes such as the workers of `run_sweep` hide by default ([#26](https://github.com/saetass/saetass/issues/26), [#43](https://github.com/saetass/saetass/issues/43)).
- The banner fits the width of the console, with a smaller art for narrow ones ([#68](https://github.com/saetass/saetass/pull/68)).
- Golden outputs of every operator and a pytest-benchmark suite in `benchmarks/` ([#64](https://github.com/saetass/saetass/pull/64)).
- This changelog, issue forms and a pull request template ([#66](https://github.com/saetass/saetass/issues/66), [#67](https://github.com/saetass/saetass/issues/67)).
- Links to the GitHub repository and PyPI in the header of the documentation ([#46](https://github.com/saetass/saetass/issues/46)).
- Adiabatic losses: `LossSolver(adiabatic_losses=True)` adds the momentum change `-(p / 3) div v` of the flow, which `Solver` takes from the advection operator, so that advection, diffusion and the adiabatic gains at a shock produce diffusive shock acceleration. `P_dot` is optional when they are on, `v_centers_physical` also accepts callables and `Separable` parameters, and `velocity_divergence` averages the divergence of a flow over the radial cells of a `Grid` ([#29](https://github.com/saetass/saetass/issues/29)).

### Changed

- **Breaking:** `Solver(state, problem_type, ...)` takes the grid from `state.grid` and no longer accepts a `Grid` as first argument ([#11](https://github.com/saetass/saetass/issues/11)).
- **Breaking:** `compute_sychrotron_losses` is renamed `compute_synchrotron_losses`, without an alias ([#61](https://github.com/saetass/saetass/issues/61)).
- **Breaking:** the advection and loss operators default to `order=2`, the `minmod` limiter and `cfl=0.5`, and a limiter given with `order=1` raises a `ValueError` ([#65](https://github.com/saetass/saetass/issues/65)).
- A missing required operator parameter raises a `ValueError` naming the operator and the parameter, instead of a `KeyError` ([#65](https://github.com/saetass/saetass/issues/65)).
- The inflow of the loss operator defaults to zero.
- Advection from `r = 0` needs at least three radial cells, and `DiffusionSolver` rejects unknown boundary conditions and parameters of the wrong shape.
- The documentation shows the version of the installed package instead of a fixed 0.1.0.
- The "Cite this repository" box of GitHub proposes the technical paper, as the README does.

### Removed

- **Breaking:** `DiagnosticsWarning`. The warnings of `solver.diagnostics()` are logged, and remain available in `SimulationDiagnostics.warnings`.
- **Breaking:** `setup_rich_logging`, replaced by `saetass.cli.set_log_level` and `saetass.cli.set_rich_logging`.
- The placeholder version and release date of `CITATION.cff`, which went stale with every release: the citation uses the Zenodo DOI of all versions.

### Fixed

- The CFL step of losses on 2D grids used the last radius at every momentum instead of the last momentum at every radius ([#57](https://github.com/saetass/saetass/issues/57)).
- `solver.diagnostics()` no longer warns about a division by zero where the diffusion coefficient vanishes ([#60](https://github.com/saetass/saetass/issues/60)).
- `run()` after `step()` advances to the final time instead of failing with an `IndexError`.
- A failing step no longer leaves the progress bar running.
- `Grid.volumes` gave the cell centered at the origin twice the volume of the ball inside its outer face, and `Grid.face_areas` an area to its inner face, since the first face of such a grid lies at `-dr / 2`.

## [0.5.1] - 2026-10-14

### Changed

- The repository moved to the [saetass](https://github.com/saetass) GitHub organization. This release only tests the publication from the new repository; the package is identical to 0.5.0.

## [0.5.0] - 2026-10-14

### Added

- `BubbleProfileCalculator.compute_source_term` injects `f ∝ p^-s e^-Γ(p)` from `p_inj` in the cell that contains the termination shock, with the cutoff of Morlino et al. (2021) as fitted by Menchiari et al. (2024), normalized to a fraction `eta_inj` of the wind luminosity ([#55](https://github.com/saetass/saetass/pull/55)).
- The hot bubble holds the mass evaporated from the shell, and the swept-up shell has a given mass `M_shell`. The shell can be uniform or fragmented, with part of its mass penetrating into the bubble.
- `eta_m` accounts for radiative losses in the bubble and termination shock radii.
- `BubbleProfileCalculator` takes a `particle` and accepts momenta or kinetic energies; the diffusion coefficient follows the rigidity.
- Optional dependencies grouped by use case: `tutorials`, `test`, `docs`, `lint` and `dev`.
- Continuous tests of the minimum versions of the dependencies, of the newest versions on every supported Python, on macOS and Windows, and of the built package.

### Changed

- **Breaking:** Python 3.11 or later is required.
- **Breaking:** `E_k` is renamed `p_or_E_k` in `BubbleProfileCalculator`.
- **Breaking:** `compute_source_term` and `get_all_profiles` (Morlino21) require `E_max`, and `compute_source_term` takes `s` and `p_inj` and returns a Quantity in canonical units. A fragmented shell requires `shell_width` and `shell_penetration`.
- The magnetic field is constant inside the cluster core.
- Every minimum version of the dependencies is the oldest one that works.

### Removed

- **Breaking:** the `plotting` extra. Use `tutorials` instead.

### Fixed

- AAFrag cross sections were evaluated at the kinetic energy of the primary instead of its total energy. Gamma-ray and neutrino spectra below a few GeV change.
- Bremsstrahlung emission is integrated from its kinematic threshold. On grids of 10 momenta per decade the spectrum was up to 10-27% too low.
- The wind density had a spurious factor 3.

## [0.4.0] - 2026-10-02

### Added

- `solver.diagnostics()` checks a configured simulation without running it: Courant numbers and CFL sub-cycles of advection and losses, Fourier number of diffusion, variation timescale of time-dependent sources, splitting ratio, and cell Péclet and Damköhler numbers. It warns, with a suggested number of time steps, when the time step is too large ([#53](https://github.com/saetass/saetass/pull/53)).
- Tutorial 07 on simulation diagnostics.
- Tests that the solvers give bitwise-identical results when the problem is rescaled.

### Fixed

- The slope limiters compared signs by multiplying slopes, which overflowed or underflowed when `psi_p` was beyond about 1e±154. Results no longer depend on the units or normalization of the problem.
- The van Leer limiter crashed on 2D grids.
- The end of the CFL sub-cycling and the `r = 0` check of diffusion use relative tolerances.

## [0.3.0] - 2026-10-01

### Added

- The canonical unit system `saetass.units` (pc, Myr, M_sun, GeV, GeV/c) with the validation helpers `validate_quantity` and `require_bare` ([#50](https://github.com/saetass/saetass/pull/50)).
- Tutorial on the unit system.

### Changed

- **Breaking:** every physical input of the public API is an `astropy.units.Quantity`, converted once to canonical units. `Grid`, `State`, the operator parameters and the calculators of `saetass.utils` reject bare numbers, and results are returned as Quantities.
- **Breaking:** `State` stores the momentum differential density `psi_p` instead of `f`, with the views `f_ps` and `psi_E`, and `Particle` is exported from `saetass`.
- Docstrings follow one NumPy style, and the tutorials, examples and validation cases use units.

## [0.2.0] - 2026-08-26

### Added

- Gamma-ray and neutrino emission from hadronic interactions, bremsstrahlung, synchrotron and inverse Compton, in `saetass.utils.emissions` and `saetass.utils.cross_sections`, with a tutorial ([#44](https://github.com/saetass/saetass/pull/44)).
- Tests for the grid, state, bubble profiles, energy losses and the edge cases of advection and losses ([#41](https://github.com/saetass/saetass/pull/41)).
- Validation case for the convergence with the size of the domain ([#38](https://github.com/saetass/saetass/pull/38)).

### Changed

- `EnergyLossCalculator` validates its parameters.

### Fixed

- `Grid` did not raise on non-positive momentum faces.
- `compute_coulomb_losses` ignored a given electron density for leptons, and the bubble profiles called a misspelled method for the Weaver (1977) kinematics.

### Removed

- Unused code of the hyperbolic solver.

## [0.1.1] - 2026-04-09

### Changed

- Author list of `CITATION.cff` and the README.

## [0.1.0] - 2026-04-08

First public, experimental release: transport of cosmic rays in spherical symmetry with advection, diffusion, energy losses and sources combined by operator splitting, bubble profiles of stellar clusters, energy loss rates, tests, tutorials and examples.

[Unreleased]: https://github.com/saetass/saetass/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/saetass/saetass/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/saetass/saetass/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/saetass/saetass/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/saetass/saetass/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/saetass/saetass/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/saetass/saetass/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/saetass/saetass/releases/tag/v0.1.0
