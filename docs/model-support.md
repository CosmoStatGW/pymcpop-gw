# Models and supported workflows

The population model combines a component-mass distribution, a merger-rate
history, an optional spin distribution, and a cosmological distance relation.
Follow the [usage guide](usage.md) to configure
an analysis; the model name alone does not determine its priors, smoothing,
mixture size, or numerical settings.

## Mass distributions

Here, $m_1$ and $m_2$ are the source-frame component masses, ordered so that
$m_1 > m_2$, and $q=m_2/m_1$ is the mass ratio.

| `mass_model` | Meaning | Distribution |
|---|---|---|
| `DPLDP` | Double Power Law plus Double Peak | A broken power-law primary-mass distribution with two Gaussian peaks, coupled to the secondary-mass distribution and mass-boundary tapers. |
| `DPLDP-z` | Redshift-evolving DPLDP | Extends DPLDP with redshift evolution of the power-law slopes, break, peak locations and widths, and mixture weights. |
| `DPUC` | Two-dimensional mixture in transformed mass coordinates | A Gaussian mixture in log chirp mass and either mass ratio or its logit transform, with configurable correlations and mass envelopes. |
| `DPUC_TruncM2` | Primary-mass and conditional secondary-mass mixtures | A mixture in log primary mass coupled to a conditional secondary-mass mixture, with physical mass ordering and boundary tapers. The reference configurations use log secondary mass. |
| `DPUC_Primary` | Primary-mass mixture with a parametric mass-ratio component | A Gaussian mixture in log primary mass combined with a mass-ratio power law and mass-boundary tapers. |

The three `DPUC` names are configuration identifiers for the weakly modelled
**Dirichlet-process Gaussian-mixture (DPGMM)** class. They distinguish how the
mass distribution is represented; the identifier does not imply that its
components are uncorrelated. These mass distributions are independent of
redshift in the reference configurations.

DPLDP and DPLDP-z are described in the
[parametric-population paper](https://arxiv.org/abs/2605.20112).
The DPGMM class is described in the
[weakly modelled population paper](https://arxiv.org/abs/2610.10535), whose
reference model is `DPUC_TruncM2`.

## Distance relation, merger rate, and spins

| Component | Setting | Meaning |
|---|---|---|
| Parametric distance relation | `is_GP_dL=0` | Use the cosmological parameters and distance prescription selected by the configuration. |
| Gaussian-process distance relation | `is_GP_dL=1` | Reconstruct a monotone gravitational-wave luminosity-distance relation. The tested combination is GP with DPLDP; reference kernels are Gibbs and Matérn 5/2. |
| Merger-rate history | `rate_model=MD` | A Madau–Dickinson-like redshift dependence with low-redshift slope, turnover, and high-redshift decline parameters. |
| No population-spin model | `spin_model=none` | Omit the population-spin density from the analysis. |
| Gaussian spin model | `spin_model=default_gauss` | Spin magnitudes described by truncated Gaussians, with a mixture of aligned and isotropic spin-tilt distributions. |

The GP manuscript citation is forthcoming.

## Event likelihoods and inputs

Two event-likelihood workflows are available:

- **Marginal posterior-sample likelihood:** `pop_only=1`, using event parameter
  estimation (PE) samples and their sampling priors.
- **Bounded Gaussian-mixture likelihood:** `pop_only=0`, `sampling_gw=gauss`,
  and `bounded_gmm_transform=1`, using fitted event mixtures with their
  coordinate names, bounds, and transform metadata.

Population DPGMM components and event Gaussian-mixture fits serve different
purposes: the former describe the astrophysical mass distribution; the latter
approximate individual-event data for the likelihood. An unbounded event fit
cannot be made into a bounded fit merely by changing a runtime flag. See
[input data and PE fitting](input-data.md) for formats and fitting instructions.
Detection-selection calculations also require a compatible injection campaign
and its sampling-density metadata.
