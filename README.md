<div align="center">

<img src="docs/assets/saetass_logo_horizontal.svg" alt="SAETASS Logo" width="500">

<br/>
<br/>

**Solver for Astroparticle Equation of Transport Analysis in Spherical Symmetry**

A Python library for simulating cosmic-ray transport in spherically symmetric astrophysical environments using a finite volume method approach.

<br/>

<!-- ─────────────────────── Badges ─────────────────────── -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19468484.svg)](https://doi.org/10.5281/zenodo.19468484)
[![CI tests](https://img.shields.io/github/actions/workflow/status/saetass/saetass/ci.yml?logo=github&logoColor=white&label=tests)](https://github.com/saetass/saetass/actions)
[![codecov](https://img.shields.io/codecov/c/github/saetass/saetass?logo=codecov&logoColor=white)](https://codecov.io/gh/saetass/saetass)
[![Docs](https://img.shields.io/readthedocs/saetass?logo=readthedocs&logoColor=white)](https://saetass.readthedocs.io/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/saetass?logo=pypi&logoColor=white&color=007EC6)](https://pypi.org/project/saetass/)<!-- [![Python versions](https://img.shields.io/pypi/pyversions/saetass?logo=python&logoColor=white)](https://pypi.org/project/saetass/) -->
[![Project Status: Active](https://img.shields.io/badge/project%20status-active-44CC11.svg?logo=github&logoColor=white)](https://www.repostatus.org/#active)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome!-F3AC4B.svg?logo=github&logoColor=white)](CONTRIBUTING.md)
[![License: BSD-3](https://img.shields.io/badge/license-BSD_3--Clause-F3AC4B.svg?logo=open-source-initiative&logoColor=white)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/saetass/saetass?logo=git&logoColor=white)](https://github.com/saetass/saetass/commits/main)
[![Issues](https://img.shields.io/github/issues/saetass/saetass?logo=github&logoColor=white)](https://github.com/saetass/saetass/issues)
<!-- [![Stars](https://img.shields.io/github/stars/saetass/saetass?logo=github&logoColor=black)](https://github.com/saetass/saetass/stargazers) -->


</div>

## Overview

**SAETASS** numerically solves the **astroparticle transport equation** in spherically symmetric astrophysical environments; this is, the fundamental equation governing the propagation and energy losses of energetic particles within astrophysical environments such as stellar wind bubbles.

The package decomposes the full transport equation into independent physical operators: **diffusion**, **advection**, **energy losses** and **source**. It evolves them via mathematically robust **operator-splitting schemes**. Each operator is implemented as a dedicated finite-volume solver, ensuring modularity, testability and physical transparency.

> **For comprehensive documentation**, visit the [SAETASS Documentation](https://saetass.readthedocs.io/en/latest/).


<div align="center">

<img src="docs/assets/evolution.gif" alt="SAETASS Multi-Energy Diffusion Simulation Evolution" width="900">

</div>


## Key Features

| Feature | Description |
|---------|-------------|
| **Modular solvers** | Independent numerical schemes for advection, diffusion energy losses, and source injection; combined via operator splitting |
| **Flexible grids** | Dedicated grid object with support for diverse spatial and momentum grid configurations |
| **Physical accuracy** | Deep integration with `astropy.units` and `astropy.constants` for strict dimensional analysis |
| **Extensibility** | Plug in custom diffusion coefficients, velocity fields, loss functions and source terms |
| **Fast numerical core** | Compiled Numba kernels, multithreaded on large grids, and parallel parameter sweeps across processes |

## Installation

A quick simple installation to get started using SAETASS can be simply done via `pip`:

```bash
pip install saetass
```

This will install the latest stable version of SAETASS from [PyPI](https://pypi.org/project/saetass/).
The dependencies of other use cases are installed as extras: `tutorials` to run the tutorials and examples, `test`, `docs`, and `dev` for all of them, for instance `pip install "saetass[tutorials]"`.

> See the full [Installation Guide](https://saetass.readthedocs.io/en/latest/installation.html) for additional installation options and troubleshooting.

## Quickstart

For step-by-step guides to get started with SAETASS, check the [Tutorials](https://saetass.readthedocs.io/en/latest/tutorials/index.html) section of the [documentation](https://saetass.readthedocs.io/en/latest/).

For complete, more advanced examples, check the [Examples](https://saetass.readthedocs.io/en/latest/examples/index.html) section of the [documentation](https://saetass.readthedocs.io/en/latest/).


## Mathematical background

SAETASS solves the following **astroparticle transport equation** in spherically symmetric geometry:

$$\frac{\partial \psi}{\partial t}
    + \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2 u_\mathrm{w}\psi\right)
    + \frac{\partial}{\partial p}\left( \dot{p} \psi \right)
    = \frac{1}{r^2} \frac{\partial}{\partial r}
    \left(r^2 D\frac{\partial \psi}{\partial r}\right)
    + Q.$$

Where:
- $\psi(t, r, p)$ is the differential (in momentum) density function ($\psi = 4\pi p^2 f_\mathrm{ps}$),
- $u_\mathrm{w}(t,r,p)$ is the advection (wind) velocity,
- $\dot{p}(t, r, p)$ is the rate of energy loss,
- $D(t, r, p)$ is the spatial diffusion coefficient,
- $Q(t, r, p)$ is the source term.

The equation is discretized using a **finite volume scheme** and solved by an **operator-splitting routine**, which decomposes the full PDE into simpler, independently solvable subproblems.

### References

For a detailed derivation of the numerical schemes, see the [technical paper](https://doi.org/10.1088/1475-7516/2026/09/133).


## Citation

If you use SAETASS in your research, **please cite the technical paper** to acknowledge the development effort:

```bibtex
@article{saetass_technical_paper,
  author  = {García-Morillo, José María and Menchiari, Stefano and López-Coto, Rubén},
  title   = {SAETASS: Solver for Astroparticle Equation of Transport Analysis in Spherical Symmetry},
  journal = {Journal of Cosmology and Astroparticle Physics},
  publisher = {IOP Publishing},
  year    = {2026},
  month   = {sep},
  volume  = {2026},
  number  = {09},
  pages   = {133},
  doi     = {10.1088/1475-7516/2026/09/133},
  url     = {https://doi.org/10.1088/1475-7516/2026/09/133},
}
```

For software-specific citations, you may use the **Zenodo DOI** [10.5281/zenodo.19468484](https://doi.org/10.5281/zenodo.19468484) citation:

```bibtex
@software{saetass_software,
  author       = {García-Morillo, José María and Menchiari, Stefano and López-Coto, Rubén},
  title        = "{SAETASS: Solver for Astroparticle Equation of Transport Analysis in Spherical Symmetry}",
  month        = apr,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19468484},
  url          = {https://doi.org/10.5281/zenodo.19468484}
}
```

## Contributing

Contributions, bug reports and feature suggestions are welcome! Whether you want to fix a typo, improve a solver or add a new physical module: we'd love your help!

Please read the [Contributing Guidelines](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md) before getting started. The [changelog](CHANGELOG.md) lists the changes of every release.

## License

SAETASS is released under the [BSD 3-Clause License](LICENSE).

## Acknowledgements

<div align="center">

SAETASS is developed within the [**VHEGA**](https://vhega.iaa.es) research group at the [**Instituto de Astrofísica de Andalucía (IAA-CSIC)**](https://www.iaa.csic.es), Granada, Spain.

<br/>

[![Author](https://img.shields.io/badge/Author-J.%20M.%20García--Morillo-FFFFFF?logo=github&logoColor=white)](https://github.com/jmgarciamorillo)
[![Email](https://img.shields.io/badge/Email-jmorillo%40iaa.es-007EC6?logo=minutemailer&logoColor=white)](mailto:jmorillo@iaa.es)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--5232--349X-a6ce39?logo=orcid&logoColor=white)](https://orcid.org/0009-0008-5232-349X)

</div>
