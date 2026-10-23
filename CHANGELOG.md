# Changelog

## v1.1.0

- Gaussian-process distance reconstruction with Gibbs and Matérn 5/2 kernels,
  tested with the Double Power Law plus Double Peak (DPLDP) mass model.
  Paper citation forthcoming.
- Weakly modelled Dirichlet-process Gaussian-mixture (DPGMM) mass models:
  `DPUC`, `DPUC_TruncM2`, and `DPUC_Primary`.
  See [arXiv:2610.10535](https://arxiv.org/abs/2610.10535).
- JSON settings alongside INI files and command-line overrides.
- Installation and usage guides, with portable examples for published analyses.

## v1.0.0

Parametric population and cosmological inference with GPU-compatible NumPyro
sampling, including DPLDP and DPLDP-z.
See [arXiv:2605.20112](https://arxiv.org/abs/2605.20112).

## v0.1.0

Original PyMC implementation.
See [arXiv:2502.12156](https://arxiv.org/abs/2502.12156).
