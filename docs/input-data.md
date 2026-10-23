# Input data and posterior fitting

Event data and selection injections are external to the code repository. Use
the path settings in your [analysis INI file](usage.md#3-write-an-ini-file-and-choose-priors).

## File paths and formats

Except for posterior-sample HDF5 files, data paths are **filename prefixes**.
For example, `/data/O4a/events_` identifies files such as
`/data/O4a/events_allNames.txt` and `/data/O4a/events_gmm_means.npy`.
A directory path alone is insufficient.

| Input | Required contents |
|---|---|
| Event posterior samples | Aligned detector-frame masses, luminosity distance, distance-prior density, event names, and valid samples |
| Event Gaussian-mixture fits | Matching component weights, means, covariance factors, event names, and Gaussian proposal files |
| Bounded Gaussian-mixture fits | The event-mixture files plus coordinate names, bounds, and transformation metadata |
| Selection injections | Aligned masses, luminosity distance, draw density, generated-injection count, observation time, and any spin arrays required by the settings |

### Posterior samples

The marginal likelihood accepts an HDF5 layout with `properties`,
`posteriors`, and `prior` groups, or a NumPy prefix layout with these suffixes:

- `allNames.txt`
- `m1d_samples.npy`, `m2d_samples.npy`, `dL_samples.npy`
- `dL_PE_prior.npy`

The loader first masks invalid or unaligned samples, then applies
`nsamplesmax` to the remaining samples. It does not randomly resample each
event. Spin-dependent analyses also need the corresponding spin samples;
a file without spins is suitable only for analyses that do not use them.

### Event Gaussian mixtures

Event Gaussian-mixture likelihoods require:

- `allNames.txt` and `allNgm.txt`
- `gmm_log_wts.npy`, `gmm_means.npy`, `gmm_icovs.npy`
- `gmm_log_dets.npy` and `gmm_cho_covs.npy`
- a complete proposal pair: `cho-means.npy` with `cho-covs.npy`, or
  `cho_means.npy` with `cho_covs.npy`

With `bounded_gmm_transform=1`, also provide:

- `gmm_fit_coord_bounds.npy`
- `gmm_fit_coord_names.txt`
- `gmm_fit_transform_info.json`

These files describe the coordinates and transformation used to fit the
mixtures. Keep them with the matching mixture arrays. An event-posterior
Gaussian mixture is an input representation; it is distinct from choosing
a DPGMM population mass model.

### Selection injections

Injection prefixes require `m1d.npy`, `m2d.npy`, `dL.npy`, `log_p_draw.npy`,
`Ngen.npy`, and `Tobs.txt`. Preserve any inclusion weights needed by the
selected workflow.

| `spin_inj` setting | Additional array suffixes |
|---|---|
| `default` or `default_gauss` | `chi1.npy`, `chi2.npy`, `ct1.npy`, `ct2.npy` |
| `chi12xyz` | `spin1x.npy`, `spin1y.npy`, `spin1z.npy`, `spin2x.npy`, `spin2y.npy`, `spin2z.npy` |

Cartesian spins are converted to the polar representation used by the
population calculation. Injection thinning keeps weights and spins aligned,
scales the generated count by the same selected fraction, and leaves
observation time unchanged. Reduced injection counts are useful for debugging;
publication runs require their intended selection dataset and sampling level.

## Catalogue and GP alignment

An event whitelist selects rows in source-file order. Returned event names
follow the selected numerical rows, even when the whitelist is ordered
differently or contains one event. Match initialization arrays and
precomputed Gaussian-process (GP) quantities to that resulting order.

GP initialization, event-distance bounds, and resolution arrays depend on
the catalogue and grid. Matching filenames or array lengths alone does not
establish that they describe the same events. Select and align events explicitly, and use initialization consistent with
the chosen catalogue and grid.

Keep input-group order consistent with the distance priors and observing
periods. Verify event/grid alignment when changing the catalogue.

## Fitting event posterior samples

The fitting entry point is `pymcpop-gw/fit_posterior_samples.py`. It supports
bounded-coordinate transformations and produces the mixture
inputs used by the inference launcher. Inspect its arguments in the fitting
environment:

```sh
python pymcpop-gw/fit_posterior_samples.py --help
```

Fitting needs scikit-learn and the external [MGCosmoPop](https://github.com/CosmoStatGW/MGCosmoPop) data readers in addition
to the inference environment. Install the package containing `MGCosmoPop.dataStructures`, or expose
the legacy `dataStructures` package explicitly through `PYTHONPATH`. The
script does not assume an adjacent MGCosmoPop checkout. See
[environment details](environment.md).

Data-coordinate `spins=default` denotes polar spin coordinates. It is separate
from the population spin-model setting used for inference.
