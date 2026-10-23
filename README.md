# <img src="assets/pymcpop-gw-logo.svg" width="44" height="44" alt="" valign="middle"> pymcpop-gw

**GPU-accelerated gravitational-wave population inference and cosmology with JAX and NumPyro.**

pymcpop-gw infers the mass distribution, merger-rate evolution, and cosmological
distance relation of a population of compact binaries. It accounts for detection
selection effects and supports both posterior-sample marginalization and
Gaussian-mixture representations of individual-event likelihoods.

[Models](docs/model-support.md) · [Installation](#installation) ·
[Run your analysis](#run-your-analysis) · [Documentation](#documentation) ·
[Papers](#papers-and-citation)

## What's new

- **v1.1.0:** Adds two model families:
  - **Gaussian-process (GP) distance reconstruction**, tested with the Double
    Power Law plus Double Peak (**DPLDP**) mass model. Reference is
    [forthcoming](#papers-and-citation).
  - **Weakly modelled Dirichlet-process Gaussian-mixture (DPGMM) mass
    distributions**, with three mass-coordinate variants. See
    [arXiv:2610.10535](https://arxiv.org/abs/2610.10535).
- **v1.0.0:** Parametric population and cosmological inference, including a Double Power Law plus Double Peak
  (DPLDP) model and its redshift-evolving extension, DPLDP-z, with GPU-compatible NumPyro sampling.
  See
  [arXiv:2605.20112](https://arxiv.org/abs/2605.20112).

- **v0.1.0:** Original pymc implementation. See [arXiv:2502.12156](https://arxiv.org/abs/2502.12156).

See the [changelog](CHANGELOG.md) for supporting interface and reproducibility
updates in v1.1.0.

## Installation

Clone the repository and create a Python 3.12 CPU environment:

```bash
git clone https://github.com/CosmoStatGW/pymcpop-gw.git
cd pymcpop-gw
python3.12 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -r environment/requirements-cpu.txt
python -m pip check
```

The code runs directly from the checkout. The [installation guide](docs/environment.md)
also covers an NVIDIA GPU environment, device checks, cluster setup, and
dependencies for preparing event Gaussian-mixture fits.

## Run your analysis

Prepare event posterior samples or Gaussian-mixture fits and a compatible
selection-injection campaign using the [input guide](docs/input-data.md).
These datasets are obtained separately. To start with a DPLDP analysis using
posterior samples, copy the example settings and priors:

```bash
cp examples/analysis.ini analysis.ini
cp examples/priors/dpldp.json priors.json
```

In `analysis.ini`, replace `<EVENT_POSTERIOR_FILE_OR_PREFIX>` and
`<INJECTION_PREFIX>` with your prepared input paths, and `<OUTPUT_DIRECTORY>`
with a new writable directory. Review `priors.json` and the model and sampling
settings before launching:

```bash
python pymcpop-gw/fit_model_numpyro.py --settings analysis.ini
```

The output directory receives the effective settings, priors, log, posterior
traces, and summary plots. Follow [Run your analysis](docs/usage.md) for the
complete workflow, including cluster execution, opening results with ArviZ,
and assessing sampling diagnostics. `--settings` accepts INI or JSON files;
explicit command-line options override file settings.

## Documentation

- [Installation: CPU and NVIDIA GPU](docs/environment.md)
- [Run your analysis: inputs, configuration, execution, and results](docs/usage.md)
- [Input data and event-mixture fitting](docs/input-data.md)
- [Configuration files and priors](docs/configuration.md)
- [Models and supported workflows](docs/model-support.md)
- [Command-line options by category](docs/cli-options.md)
- [Examples from papers](docs/paper-recipes.md)

## How to modify

Developer instructions for adding new **mass**, **spin**, and **merger-rate**
models will be added here in a future documentation update.

## Papers and citation

Please cite the papers relevant to your analysis and record the software commit,
effective settings, priors, and data provenance:

- Foundational methods: [arXiv:2502.12156](https://arxiv.org/abs/2502.12156).
- Parametric models and NumPyro version: [arXiv:2605.20112](https://arxiv.org/abs/2605.20112).
- Weakly modelled DPGMM populations: [arXiv:2610.10535](https://arxiv.org/abs/2610.10535).
- GP distance reconstruction: **citation forthcoming**.

## Contributing and contact

Maintainers: Michele Mancarella, Alessandro Agapito, Viola De Renzis. Please use the
[issue tracker](https://github.com/CosmoStatGW/pymcpop-gw/issues) for bug reports.
Include the software commit, effective settings, environment, and a minimal
reproducible example.

## License

pymcpop-gw is distributed under the [BSD 3-Clause License](LICENSE).
