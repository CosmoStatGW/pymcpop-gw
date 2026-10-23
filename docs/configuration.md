# Configure an analysis

An analysis combines data locations, population and cosmology choices, priors,
and numerical settings. Start with the complete
[INI example](../examples/analysis.ini) and follow the [usage guide](usage.md).

## INI syntax

Put settings under a single `[settings]` section. Names are case-sensitive,
without the command-line `--` prefix. Comments start with `#` or `;`.
Use space-separated values for list settings; quote paths containing spaces.

```ini
[settings]
fin_data = "/path with spaces/events.h5"
fin_injections = /path/to/injections_
fin_priors = priors.json
fout = /path/to/new-results
events_use = GW150914_095045 GW170814_103043
pop_only = 1
sampling_gw = samples
```

This is a syntax illustration, not a complete model configuration. Flags such
as `pop_only` use `0` or `1`. Relative paths resolve from the directory where
the launcher is invoked. Replace all angle-bracket placeholders with real
values; settings files do not expand environment variables or `~`.

## Required and analysis-specific choices

| Setting | Purpose |
|---|---|
| `fin_data` | One or more event input files/prefixes |
| `fin_injections` | One or more injection prefixes |
| `fin_priors` | JSON population/cosmology prior file |
| `fout` | Output directory; use a new directory for each analysis |
| `mass_model`, `rate_model`, `spin_model` | Population model components |
| `pop_only`, `sampling_gw`, `bounded_gmm_transform` | Event-likelihood treatment |
| `ntune`, `nsteps`, `nchains` | Warmup steps and retained draws per chain; chain count |
| `ivals` | Optional initialization JSON, consistent with model and selected data |

For marginal posterior-sample likelihoods, use `pop_only=1` and
`sampling_gw=samples`. For bounded event-mixture likelihoods, use
`pop_only=0`, `sampling_gw=gauss`, and `bounded_gmm_transform=1`, with
matching bounded fits. A flag cannot convert an unbounded fit into a bounded fit.

Multiple event inputs retain their order. For mixture analyses, give the matching
`dLprior` for each event input group. Give one `n_inj_use` fraction per injection
input when using that option. Keep data, selection campaign, distance priors,
and observing periods consistent.

## Set priors

Copy a prior file for your model and point `fin_priors` to that copy. The
[example DPLDP prior file](../examples/priors/dpldp.json) is complete; change
its values only after considering the implied scientific assumptions.
For example, its `H0` and `Om` entries set the corresponding parameter ranges.

Prior entries are interpreted by the model: a pair of numbers is not universally
a hard uniform bound. For the DPGMM power-law primary-mass envelope, the
`alpha1` pair specifies the central 95% interval of a Normal prior. Mixture
weights and positive scales have their own parameterizations. A prior JSON
file supplies values; it does not accept arbitrary executable distribution code.

Initialization sets the starting point and does not replace the prior.
Changing GP event selection or grid dimensions requires matching initialization
arrays. Start from a model-compatible example rather than changing only the
`mass_model` name.

## JSON and command-line overrides

`--settings` also accepts a JSON object with typed values: list options use
JSON arrays, numeric options use numbers. Explicit command-line arguments
take precedence over file settings.

```bash
python pymcpop-gw/fit_model_numpyro.py --settings analysis.ini --nsteps 2000
python pymcpop-gw/fit_model_numpyro.py --settings settings.json
```

Unknown options and invalid types produce an error. Defaults and advanced
options are listed by category in the [CLI reference](cli-options.md).
