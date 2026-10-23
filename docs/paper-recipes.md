# Examples from papers

These examples provide settings, priors, and initialization for the analyses
listed below. For your own analysis, start with the [usage guide](usage.md).
The event and injection datasets must be obtained separately in the
[supported formats](input-data.md).

## Prepare and run an example

For example, copy the DPLDP settings:

```bash
cp examples/papers/dpldp/settings.json paper-settings.json
```

Open `paper-settings.json` and replace **every** angle-bracket placeholder:

- `<EVENT_DATA_GROUP_1>`, etc.: the appropriate event file or filename prefix,
  in the same observing-period order as the analysis. The table lists the
  number of input groups. Retain all groups and their order.
- `<INJECTION_GROUP_1_PREFIX>`, etc.: compatible injection filename prefixes,
  preserving the campaign order and number of groups.
- `<OUTPUT_DIRECTORY>`: a new writable directory for this run.

The example's `fin_priors` and `ivals` already point to its included files.
Run from the repository root so those relative paths resolve correctly.
For mixture inputs, keep each `dLprior` entry matched to its event-data group.
The examples include paper configurations and the listed model variations,
with full numerical and sampling settings; these are not short demonstration
runs.

```bash
python pymcpop-gw/fit_model_numpyro.py --settings paper-settings.json
```

Follow [opening and interpreting results](usage.md#5-open-and-interpret-the-results)
after sampling. Reproducing a posterior requires the paper's actual catalogue,
event selection, injection campaign, distance priors, and analysis choices;
substituting another dataset changes the analysis.

For five-group mixture inputs, use this order:

| Group | Observing period | Distance prior (`dLprior`) |
|---|---|---|
| 1 | O1 and O2 | `dLsq` |
| 2 | O3a | `dLsq` |
| 3 | O3b | `dLsq` |
| 4 | O4a | `UniformSourceFrame-J` |
| 5 | O4b | `UniformSourceFrame-J` |

Posterior-sample examples use one combined file or prefix, with per-sample
prior densities supplied in the data. Use the paper's event selection and
matching selection campaign for either representation.

## Parametric populations

[arXiv:2605.20112](https://arxiv.org/abs/2605.20112).
Model definitions are in [model support](model-support.md).

| Analysis | Event inputs | Input groups |
|---|---|---|
| [DPLDP](../examples/papers/dpldp/settings.json) | Posterior samples | 1 |
| [DPLDP-z](../examples/papers/dpldp-z/settings.json) | Posterior samples | 1 |

## Weakly modelled populations

[arXiv:2610.10535](https://arxiv.org/abs/2610.10535) uses the
`DPUC_TruncM2` conditional secondary-mass mixture. The chirp-mass and
primary-mass examples are alternative model choices.

| Analysis | Event inputs | Input groups |
|---|---|---|
| [Conditional secondary-mass mixture](../examples/papers/dpgmm-truncm2/settings.json) | Bounded event mixtures | 5 |
| [Chirp-mass and mass-ratio mixture](../examples/papers/dpgmm-chirp-mass/settings.json) | Posterior samples | 1 |
| [Primary-mass mixture](../examples/papers/dpgmm-primary/settings.json) | Bounded event mixtures | 5 |
| [Weaker aspect regularization](../examples/papers/dpgmm-weaker-aspect/settings.json) | Bounded event mixtures | 5 |
| [Smaller component-width floors](../examples/papers/dpgmm-smaller-width-floors/settings.json) | Bounded event mixtures | 5 |
| [10 mixture components](../examples/papers/dpgmm-10-components/settings.json) | Bounded event mixtures | 5 |
| [Stronger aspect regularization](../examples/papers/dpgmm-stronger/settings.json) | Bounded event mixtures | 5 |
| [Weaker aspect regularization and smaller width floors](../examples/papers/dpgmm-weaker/settings.json) | Bounded event mixtures | 5 |
| [Marginal posterior-sample likelihood](../examples/papers/dpgmm-marginal/settings.json) | Posterior samples | 1 |
| [Alternative concentration prior](../examples/papers/dpgmm-concentration/settings.json) | Bounded event mixtures | 5 |
| [30 mixture components](../examples/papers/dpgmm-30-components/settings.json) | Bounded event mixtures | 5 |
| [100 mixture components](../examples/papers/dpgmm-100-components/settings.json) | Bounded event mixtures | 5 |
| [Larger component-width floors](../examples/papers/dpgmm-larger-width-floors/settings.json) | Bounded event mixtures | 5 |
| [Lower quantile for component-mean bounds](../examples/papers/dpgmm-lower-mean-quantile/settings.json) | Bounded event mixtures | 5 |

## Gaussian-process distances

GP distance reconstruction with DPLDP; paper citation forthcoming.
Initialization is tied to the selected catalogue and GP grid. Changing the
event set or grid requires corresponding initialization; the launcher builds
its preprocessing arrays from the analysis inputs.

| Analysis | Event inputs | Input groups |
|---|---|---|
| [Gibbs kernel with DPLDP](../examples/papers/gp-dpldp/settings.json) | Bounded event mixtures | 5 |
| [Alternative resolution quantile](../examples/papers/gp-resolution-quantile/settings.json) | Bounded event mixtures | 5 |
| [Shorter length-scale prior](../examples/papers/gp-short-length-scale/settings.json) | Bounded event mixtures | 5 |
| [Amplitude prior with U=12](../examples/papers/gp-amplitude-12/settings.json) | Bounded event mixtures | 5 |
| [Marginal likelihood, variance threshold 10](../examples/papers/gp-marginal/settings.json) | Posterior samples | 1 |
| [Marginal likelihood, variance threshold 1](../examples/papers/gp-marginal-variance-1/settings.json) | Posterior samples | 1 |
| [Wider length-scale prior](../examples/papers/gp-wide-length-scale/settings.json) | Bounded event mixtures | 5 |
| [Smoothed resolution](../examples/papers/gp-smooth-resolution/settings.json) | Bounded event mixtures | 5 |
| [Gibbs kernel, variance threshold 1](../examples/papers/gp-variance-1/settings.json) | Bounded event mixtures | 5 |
| [Matérn 5/2 kernel](../examples/papers/gp-matern/settings.json) | Bounded event mixtures | 5 |
