# fdtd_gradproj

A small, self-contained Python package to (re)create graduate-project style FDTD simulations and figures.

## Features (initial)
- 1D Yee FDTD (TEz) with Mur ABC
- Material regions (epsilon_r, sigma) on grid
- Sources: Gaussian pulse, continuous-wave
- Figure helpers (matplotlib)
- Reproducible scripts in `scripts/`

## Roadmap
- CPML (convolutional PML) boundary
- 2D TEz/TM modes
- Dispersive materials (Drude/Lorentz)
- Benchmarks that mirror figures from the attached paper

## Quickstart
```bash
pip install -r requirements.txt
python -m scripts.fig01_1d_pulse_in_dielectric
```
This produces `fig01_fields.png` and `fig01_snapshots.png` in the `outputs/` directory.
