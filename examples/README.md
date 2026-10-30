# SAETASS Examples

This directory contains executable example scripts demonstrating standard use cases and workflows with SAETASS. Examples outputs are saved in the `examples/outputs/` directory.

## Available Examples

*   **`01_multi_energy_diffusion.py`**: Simulates cosmic ray transport (advection and diffusion) in a stellar cluster wind bubble, comparing time evolution with the analytical steady-state solutions given by S. Menchiari et al. (2024). The example compares the time evolution of the cosmic ray spatial distribution for different particle energies and diffusion models. Outputs a 3x4 grid comparison figure (`multi_energy_diffusion_comparison.pdf`).

## Running the Examples

You can run these scripts directly from your terminal within the project's root or the `examples/` directory using Python:

```bash
python examples/01_multi_energy_diffusion.py
```
