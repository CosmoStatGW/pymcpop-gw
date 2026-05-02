# pymcpop-gw


Python package for running GW population inference with Hamiltonian Monte Carlo in the high-dimensional space of individual events and population parameters or with the customary marginal likelihood approach. 


This version uses ```jax``` and ```numpyro``` for GPU (and not)-accelerated inference .


The main entry point is:

```bash
python fit_model_numpyro.py [options]
```

The same options can also be placed in a simple INI file and passed with:

```bash
python fit_model_numpyro.py --settings settings.ini
```

The INI file must contain a `[settings]` section.  Option names are the command-line names without the leading `--`.  Values for list-like options are written as whitespace-separated values, exactly as they would appear after the command-line option.

Command-line values are still accepted when a settings file is used.  If the same scalar option is present both in the INI file and on the command line, the command-line value is parsed last and takes precedence.

## Contact
This repository is maintained by:

- **Michele Mancarella** — *[add email]* 
- **Alessandro Agapito** — *[add email]* 

For questions, issues, or collaborations, please contact the developers directly.

---


## Index

- [Minimal run](#minimal-run)
- [Multi-chain execution](#multi-chain-execution)
- [Input data layout](#input-data-layout)
  - [Prefix-based file convention](#prefix-convention)
  - [Event data](#event-data)
  - [Posterior samples](#posterior-samples)
  - [Gaussian / GMM surrogate likelihood](#gmm-likelihood)
  - [Injection data](#injection-data)
  - [Injection spin formats](#injection-spin-formats)
  - [Event selection](#event-selection)
  - [JSON configuration inputs](#json-inputs)
- [Output](#output)
- [Main model choices](#main-model-choices)
- [Option reference](#option-reference)
- [Example INI](#example-ini)
- [Citation](#citation)

<a id="minimal-run"></a>
## Minimal run


The output directory `fout` is created automatically by the script if it does not exist.

```bash
mkdir -p results/test_run
python fit_model_numpyro.py \
  --fin_data /path/to/PE_samples_prefix_or_file \
  --fin_injections /path/to/injections_prefix \
  --fin_priors priors_files/priors.json \
  --fout results/test_run \
  --nsteps 100 \
  --ntune 100
```

Equivalent INI run:

```ini
[settings]
fin_data = /path/to/PE_samples_prefix_or_file
fin_injections = /path/to/injections_prefix
fin_priors = priors_files/priors.json
fout = results/test_run
nsteps = 100
ntune = 100
```

```bash
mkdir -p results/test_run
python fit_model_numpyro.py --settings settings.ini
```

<a id="multi-chain-execution"></a>
## Multi-chain execution

By default, multiple chains can be run within a single execution using the `--nchains` argument. The execution strategy is controlled by `--chain_method`, which supports:

- `sequential`  
  Chains are run one after the other in a single process.  
  → Lower memory usage, but slower.

- `vectorized`  
  Chains are run simultaneously using JAX vectorization.  
  → Faster, but more memory intensive.

- `parallel`  
  Chains are distributed across multiple devices/processes.  
  → This mode is generally **not recommended** for NumPyro in this setup.

While using `--nchains > 1` is the standard approach, an alternative is to launch multiple independent runs with `--nchains=1` and different random seeds. This pattern is useful because each run writes its output independently, making it safer in case of interruptions.

In practice, this can be used as a robust alternative to `sequential` multi-chain runs, where all chains are executed within a single process and saved together at the end.

```bash
for SEED in 0 1 2 3; do
     python fit_model_numpyro.py --settings settings.ini --seed "$SEED"
done
```

Each run produces an independent chain (since `--nchains=1`).


<a id="input-data-layout"></a>
## Input data layout

The code expects two separate kinds of input data:

1. **Event data**, passed with `--fin_data`
2. **Injection data**, passed with `--fin_injections`

Both options accept one or more paths. In the INI file, multiple paths are written as a space-separated list:

```ini
fin_data = /path/to/events_chunk_1/ /path/to/events_chunk_2/
fin_injections = /path/to/injections/
```

---

<a id="prefix-convention"></a>
### IMPORTANT: prefix-based file convention

For all non-HDF5 inputs, paths are interpreted as **filename prefixes**, not directories.

For example:

```ini
fin_data = /data/O4a/PE_samples_
```

means the code will look for files such as:

```text
/data/O4a/PE_samples_allNames.txt
/data/O4a/PE_samples_m1d_samples.npy
/data/O4a/PE_samples_m2d_samples.npy
/data/O4a/PE_samples_dL_samples.npy
/data/O4a/PE_samples_dL_PE_prior.npy
```

This convention applies to both event data and injections.

---

<a id="event-data"></a>
## Event data (`fin_data`)

The format of the event data depends on the analysis mode.


<a id="posterior-samples"></a>
### Case 1 — Posterior samples

Activated with:

```ini
pop_only = 1
sampling_gw = samples
```

In this mode, the likelihood is computed directly from posterior samples.

Each `fin_data` entry can be either an HDF5 file or a NumPy prefix.

#### HDF5 input

The file must contain:

```text
properties/gwnames
posteriors/dL
posteriors/m1det
posteriors/m2det
prior/dL
```

Optional spin datasets:

```text
posteriors/chi_1
posteriors/chi_2
posteriors/cos_t_1
posteriors/cos_t_2
```

All posterior arrays must have shape:

```text
(n_events, n_samples)
```

If spin datasets are missing, they are internally set to zero.

#### NumPy prefix input

Required files:

```text
<PREFIX>allNames.txt
<PREFIX>m1d_samples.npy
<PREFIX>m2d_samples.npy
<PREFIX>dL_samples.npy
<PREFIX>dL_PE_prior.npy
```

Optional spin files:

```text
<PREFIX>chi1_samples.npy
<PREFIX>chi2_samples.npy
<PREFIX>cost1_samples.npy
<PREFIX>cost2_samples.npy
```

All arrays must have shape:

```text
(n_events, n_samples)
```

Missing entries can be encoded as `NaN`; these are masked internally.

---

<a id="gmm-likelihood"></a>
### Case 2 — Gaussian / GMM surrogate likelihood

Activated with:

```ini
pop_only = 0
sampling_gw = gauss
```

In this mode, each event likelihood is approximated using a Gaussian mixture model.

Each `fin_data` entry must be a prefix providing:

```text
<PREFIX>allNames.txt
<PREFIX>allNgm.txt
<PREFIX>gmm_log_wts.npy
<PREFIX>gmm_means.npy
<PREFIX>gmm_icovs.npy
<PREFIX>gmm_log_dets.npy
```

Optional files:

```text
<PREFIX>gmm_cho_covs.npy
<PREFIX>gmm_covs.npy
```

These arrays encode the Gaussian mixture representation of each event likelihood.

These GMM fits must be prepared in advance. A dedicated section should describe how to generate them.

---

<a id="injection-data"></a>
## Injection data (`fin_injections`)

Each entry in `fin_injections` is interpreted as a prefix.

Required files:

```text
<PREFIX>dL.npy
<PREFIX>m1d.npy
<PREFIX>m2d.npy
<PREFIX>log_p_draw.npy
<PREFIX>Ngen.npy
<PREFIX>Tobs.txt
```

where:

- `dL.npy`: luminosity distance samples
- `m1d.npy`, `m2d.npy`: detector-frame masses
- `log_p_draw.npy`: log of the injection sampling distribution
- `Ngen.npy`: total number of generated injections
- `Tobs.txt`: observing time

Optional file:

```text
<PREFIX>log_p_incl.npy
```

This is used only if:

```ini
is_compressed_inj = 1
```

Otherwise it is ignored.

---
<a id="injection-spin-formats"></a>
## Injection spin formats

Controlled by:

```ini
spin_inj =
```

Supported options:

### `default` or `default_gauss`

Requires:

```text
<PREFIX>chi1.npy
<PREFIX>chi2.npy
<PREFIX>ct1.npy
<PREFIX>ct2.npy
```

### `chieffchip`

Requires:

```text
<PREFIX>chieff.npy
<PREFIX>chip.npy
```

### `chi12xyz`

Requires:

```text
<PREFIX>spin1x.npy
<PREFIX>spin1y.npy
<PREFIX>spin1z.npy
<PREFIX>spin2x.npy
<PREFIX>spin2y.npy
<PREFIX>spin2z.npy
```

---

<a id="event-selection"></a>
## Event selection

You can restrict the analysis using:

```ini
events_use =
```

Explicit list:

```ini
events_use = GW150914 GW170817
```

or per-input file lists:

```ini
events_use = events_chunk1.txt events_chunk2.txt
```

Each file must contain one event name per line.

Do not combine `events_use` with `nev_min` / `nev_max`; the code does not allow selection by both event names and index ranges.

---
<a id="json-inputs"></a>
## JSON configuration inputs

- `fin_priors` (**required**): JSON file defining the population priors
- `ivals` (optional): JSON file with initial parameter values
- `params_fix` (optional): JSON file specifying fixed parameters
- `priors_for_mmin` (optional): additional prior configuration

All JSON files are read at runtime

---

<a id="output"></a>
## Output
The script writes outputs under `fout`.

| File | Meaning |
| --- | --- |
| `logfile.txt` | Copy of the run output after the logger is installed. |
| `inifile.ini` | Copy of the input file if passed. |
| `input_args.json` | Parsed input options for the run if used. |
| `priors.json` | Copy of the prior JSON used by the run. |
| `priors_for_mmin.json` | Copy of `priors_for_mmin`, only when that option is non-empty. |
| `trace_<i>.nc` | ArviZ NetCDF file for the current run, using the next available index. |
| `trace_<i>.npz` | NumPy `.npz` sample dump for the current run. |
| `trace.nc` | Concatenation of all `trace_<i>.nc` files found in `fout`. |
| `trace.pdf` | ArviZ trace plot, if plotting succeeds. |
| `corner_all.pdf` | Corner plot, if `corner` is installed and plotting succeeds. |

Some modes return early and do not write traces: `debug = 1`, non-empty `check_zres`, or `profile > 0`.

<a id="main-model-choices"></a>
## Main model choices

The population parameter vector is organized into four blocks: **cosmology**, **rate**, **spin**, and **mass**.

---

### Quick reference

| Option | Allowed values |
|------|----------------|
| `rate_model` | `MD`, `PL`, `DPUC`, `DPUC-vol`, `DPUC-vol-MD` |
| `spin_model` | `none`, `default_gauss` |
| `mass_model` | `DPLDP`, `PLDP`, `DPLDP-z` |
| `dLprior` | `none`, `dLsq`, `UniformComovingVolume`, `UniformComovingVolume-J`, `UniformSourceFrame`, `UniformSourceFrame-J`, `UniformSourceFrame-bilby` |
| `integrate_dc` | `gauss_legendre`, `trapz`, `pade`, `quick` |
| `param` | `vanilla`, `polexp` |

---

### Detailed descriptions

#### `rate_model`

- `MD`  :  Madau–Dickinson–like form psi(z) ∝ (1 + z)^γ / [1 + ((1 + z)/(1 + z_p))^(γ + κ)]

  where:
  - γ controls the low-redshift slope  
  - κ controls the high-redshift falloff  
  - z_p sets the peak redshift  

  The full redshift distribution is obtained by multiplying by the comoving volume element p(z) ∝ psi(z) × (dV/dz) /(1+z)


- `PL` : simple power-law psi(z) ∝ (1 + z)^γ

- `DPUC`, `DPUC-vol`, `DPUC-vol-MD`  
  *(Not supported)*

---

#### `spin_model`

- `none`  No spin model. This asumes spin model corresponds to the PE prior

- `default_gauss`  
 The spin distribution assumes:

  - Spin magnitudes χ₁, χ₂ drawn from a truncated Gaussian:
  
    p(χ) ∝ N(χ | μ_χ, σ_χ),   with χ ∈ [0, 1]

  - Spin tilts modeled as a mixture of:
    - an aligned Gaussian component around cosθ = 1  
    - an isotropic component  

  The tilt distribution is:

  p(cosθ) = ζ × N(1 − cosθ | 0, σ_t²) + (1 − ζ) × 1/2

  where:
  - μ_χ, σ_χ control spin magnitudes  
  - σ_t controls alignment width  
  - ζ is the mixing fraction  


---

#### `mass_model`

- `DPLDP`  Broken power-law + 2 peaks model. See the [GWTC-4.0 population paper](https://arxiv.org/abs/2508.18083) for details, and [Agapito et al. 2025](https://arxiv.org/abs/2508.18083) for details about this specific implementation.

- `PLDP`  Single power-law + 2 peaks model. A single shared low-mass end for m1 and m2. Analog to the [GWTC-4.0 cosmology paper](https://arxiv.org/abs/2509.04348) model. See [Agapito et al. 2025](https://arxiv.org/abs/2508.18083) for details about this specific implementation.

- `DPLDP-z` Redshift-evolving Broken power-law + 2 peaks model. The mass distribution is modeled as p(m1 | z, L ) with redshift-evolving hyperparameters L that have a smooth, tanh-like redshift evolution. 
Introduced in [Agapito et al. 2025](https://arxiv.org/abs/2508.18083).

---

#### `dLprior` (only for gmm branch)

This specifies the PE prior on distance to be undone. If sampling_gw = samples, a fle with the original per-sample PE prior is provided, so this option is not used.

- `none`  No PE prior on distance removed

- `dLsq`  PE prior ∝ d_L^2 removed. Used for GWTC events up to O3b.

- `UniformComovingVolume`, `UniformComovingVolume-J`  PE prior ∝ dV / dz . if `-J` is used (correct), a jacobian from distance to redshift is also incuded. This corresponds to the `bilby` prior.
 

- `UniformSourceFrame`, `UniformSourceFrame-J` PE prior ∝ (dV / dz)/(1+z) . if `-J` is used (correct), a jacobian from distance to redshift is also incuded. This corresponds to the `bilby` prior. Used for GWTC events from O4a onward.

- `UniformSourceFrame-bilby`  used pre-defined interpolation of the bilby prior.

---

#### `integrate_dc`

- `gauss_legendre`  : gauss_legendre method
- `trapz`   : standard trapezoidal
- `pade`  : pade approximation to high order (does not support w0 at the moment)
- `quick`  : pade approximation to low order (does not support w0 at the moment)

*(TODO: explain accuracy vs speed trade-offs)*

---

#### `param`
Modified gravity `Xi0-n` parameterization choices

- `vanilla`  : standard
- `polexp`  : polynomial-exponential, used for DHOST theories

*(TODO: better describe GW propagation parameterizations)*

---

### Notes

- These options define the **structure of the hierarchical model**.
- Not all combinations are guaranteed to be valid.
- Some options are only meaningful for specific likelihood branches or data formats.

<a id="option-reference"></a>
## Option reference

Status labels:

- **Active**: the option is used directly in `fit_model_numpyro.py` or passed into active model/likelihood construction.
- **Diagnostic**: the option triggers checks, profiling, debug output, or sampler diagnostics.
- **Currently unused/deprecated**: the option is parsed but no active use is done in the current code path.
- **Limited/conditional**: the option is only meaningful for a specific model branch or data format.

<a id="settings-env"></a>
### Settings and execution environment

| Option | Default | Status | Meaning |
| --- | --- | --- | --- |
| `settings` | `None` | Active | Path to an INI file with a `[settings]` section. Options are named without `--`. |
| `nth` | `None` | Active | Thread count used before JAX imports. It sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `BLIS_NUM_THREADS`, `TF_NUM_INTRAOP_THREADS`, `ACCELERATE_MAX_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `JAX_NUM_THREADS`. If absent, the script uses `1`. |
| `xla_cpu_multi_thread_eigen` | `true` | Active | Used when constructing final `XLA_FLAGS`. If set to `false` with `chain_method` `parallel` or `vectorized`, the script forces it back to `true`. |
| `jax_debug_nans` | `0` | Diagnostic | Toggles `jax.config.update("jax_debug_nans", ...)`. |

<a id="required-options"></a>
### Required file and run-control options

| Option | Default | Status | Meaning |
| --- | --- | --- | --- |
| `fin_data` | required | Active | One or more posterior-data inputs. For `pop_only=1`, these are posterior samples; for `pop_only=0`, these are interpolated/Gaussian-summary inputs. |
| `fin_injections` | required | Active | One or more injection filename prefixes. |
| `fin_priors` | required | Active | JSON prior file. It is loaded and copied to `fout/priors.json`. |
| `fout` | required | Active | Output directory. It must exist before the logger opens `fout/logfile.txt` in the current code. |
| `nsteps` | required | Active | Number of post-warmup MCMC samples passed to NumPyro as `num_samples`. |
| `ntune` | required | Active | Number of warmup/adaptation steps passed to NumPyro as `num_warmup`. |
| `seed` | `0` | Active | Random seed. In multi-run chain loops, override this per chain. |
| `nchains` | `1` | Active | Number of NumPyro chains inside this Python process. |
| `ncores` | `1` | Active | Used to choose exposed JAX host devices for `parallel` mode and for warnings/checks. |
| `chain_method` | `sequential` | Active | NumPyro chain execution mode: `sequential`, `parallel`, or `vectorized`. |

<a id="data-loading"></a>
### Data selection and loading

| Option | Default | Status | Meaning |
| --- | --- | --- | --- |
| `events_use` | empty list | Active | Optional event whitelist. Can be event names, or one `.txt` file per `fin_data` input. Cannot be combined with index selection. |
| `nev_min` | `0` | Active | Start index for event slicing after data loading, when `events_use` is empty. |
| `nev_max` | `-1` | Active | End index for event slicing after data loading. `-1` means no upper cut. |
| `pop_only` | `0` | Active | Selects likelihood/data mode. `1` uses posterior samples directly (standard marginal likelihood); `0` uses the Gaussian/interpolated branch (high-dimensional full posterior). |
| `sampling_gw` | `gauss` | Limited/conditional | For `pop_only=0`, `gauss` is the active supported branch in `fit_model_numpyro.py`; strings containing `gmm` currently raise `NotImplementedError` there. For `pop_only=1`, posterior samples are used directly. |
| `cho_dil` | `1.0` | Active | Multiplicative dilation applied to `samples_cho_covs` in the `pop_only=0` Gaussian branch. |
| `n_inj_use` | `None` | Active | One float per injection input. Values below 1 randomly downsample detected injections and scale `Ngen` accordingly. Values above 1 raise `ValueError`. |
| `nsamplesmax` | `-1` | Active | Maximum number of posterior samples per event for `pop_only=1`. `-1` means no cut. |
| `is_compressed_inj` | `0` | Active | If true, use `log_p_incl` from injection files, which is a compressed form of the injections in mass bins; otherwise use zeros. Not supported. |
| `allTobs` | `None` | Active | Optional list of observing times passed to model construction. Needed if fitting for total rate `R_0` |

<a id="priors-init"></a>
### Prior, initialization, and fixed-parameter files

| Option | Default | Status | Meaning |
| --- | --- | --- | --- |
| `ivals` | empty string | Active | Optional JSON file with initial values. If empty, no explicit initial values are loaded. |
| `params_fix` | empty string | Active | Optional JSON file with values for fixed parameters, used when parameters such as `H0` or `Om` are fixed. |
| `priors_for_mmin` | empty string | Limited/conditional | Optional JSON file; loaded only when non-empty. No further active use is currently made. |
| `eps_init` | `0.01` | Active | Initialization scale used in model construction and for the latent `x` initialization when needed. |
| `reparam_mass` | `1` | Active | Enables special initialization/reparameterized sample-site handling for `DPLDP` and `DPLDP-z`. Recommended with HMC. |
| `reparam_z` | `1` | Currently unused/deprecated | Always used in the current version. |
| `fix_H0` | `1` | Active | If true, use fixed `H0` from `params_fix`; otherwise sample it from priors. |
| `fix_Om` | `1` | Active | If true, use fixed `Om` from `params_fix`; otherwise sample it from priors. |
| `fix_w0` | `1` | Active | If true, fix `w0=-1`; otherwise sample it. Varying `w0` is incompatible with `pade` integration in the visible code. |
| `fix_Xi0n` | `1` | Active | If true, fix modified-propagation parameters `Xi0=1`, `nXi0=0`; otherwise sample them. |
| `remove_spin_prior` | `0` | Active | Passed to model construction; when enabled, the model code adjusts PE prior normalization to remove the PE spin prior. |

<a id="population-model"></a>
### Population model options

| Option | Default | Status | Meaning |
| --- | --- | --- | --- |
| `rate_model` | `MD` | Active | Redshift/rate model. Active options include `MD`, `PL`. |
| `mass_model` | `DPLDP` | Active | Mass-distribution model. Visible options include `DPLDP`, `PLDP`, `DPLDP-z`; support depends on branch. |
| `spin_model` | `none` | Active | Spin population model. The active population path supports `none` and `default_gauss`. |
| `spin_inj` | `none` | Active | Selects which spin columns to use from injection files: `chieffchip`, `chi12xyz`, `default`, or `default_gauss`; `none` uses no injection spins. |
| `use_sel_spin` | `0` | Active | If false, the selection-effect likelihood ignores spin even when the population spin model is enabled. |
| `marginal_R0` | `1` | Active | Passed to likelihood construction; controls whether the overall merger rate normalization is marginalized. |
| `sample_from_pop` | `0` | Limited/conditional | Passed to the non-`pop_only` Gaussian data packing branch. Currently value `1` is not supported. |
| `r` | `0` | Active | Selection regularizer parameter. |
| `vary_mb` | `1` | Active | For `DPLDP-z` reparameterized initialization/modeling, controls whether the mass break parameter `mb` has redshift evolution. |

<a id="mass-model"></a>
### Mass-model and smoothing options

| Option | Default | Status | Meaning |
| --- | --- | --- | --- |
| `smoothing` | `poly` | Active | Low-mass smoothing prescription passed into mass/population likelihoods. Default uses polynomial-style smoothing. |
| `simplex_repair` | `0` | Active | Passed to `DPLDP-z` mass function to repair or handle mixture weights on the simplex. |
| `has_m2_break` | `0` | Active | Enables an additional secondary-mass break in the mass model. |
| `norm_gauss` | `uplow` | Active | Controls Gaussian normalization convention in `DPLDP-z` mass function. |
| `interp_mass` | `0` | Active | If positive, precomputes mass PDFs on grids and evaluates by interpolation; mandatory in `DPLDP-z`. |
| `interp_z` | `0` | Currently unused |  |
| `N_DP_comp_max` | `30` | Currently unused |  |
| `DP_m1_env` | `0` | Currently unused |  |
| `DP_prior` | `SB` | Currently unused/deprecated | |
| `sigma_softmax` | `0.75` | Currently unused/deprecated |  |
| `gamma_DP_params` | `1.0 1.0` | Currently unused/deprecated |  |
| `DP_truncate_up` | `0` | Currently unused/deprecated|
| `DP_truncate_low` | `0` | Currently unused/deprecated |
| `linear_mass` | `0` |Currently unused/deprecated|
| `linear_z` | `0` | Currently unused/deprecated |
| `alpha_tail` | `-1` | Currently unused/deprecated |
| `alpha_small` | `0.01` | Currently unused/deprecated |
| `L_small_1` | `1.0` | Currently unused/deprecated |
| `L_small_2` | `1.0` | Currently unused/deprecated|
| `L_small_3` | `0.5` | Currently unused/deprecated |
| `s_local` | `0.5` | Currently unused/deprecated |
| `find_m_bounds` | `0` | Currently unused/deprecated |
| `q_mbound` | `0.05` | Currently unused/deprecated|
| `alpha_inv_params` | `1.0 1.0` | Currently unused/deprecated |
| `mmin_inj` | `-1` |Currently unused/deprecated |

<a id="cosmology"></a>
### Distance, cosmology, and redshift-grid options

| Option | Default | Status | Meaning |
| --- | --- | --- | --- |
| `dLprior` | `none` | Active | PE luminosity-distance prior correction per data chunk. Accepted strings are listed above. |
| `penorm_lims` | empty list | Active | Optional per-chunk JSON files with event distance limits for PE prior normalization. Use `none` for chunks that do not need normalization. |
| `integrate_dc` | `pade` | Active | Method for comoving-distance computation: options are `gauss_legendre`, `trapz`, `pade`, and `quick`. |
| `param` | `vanilla` | Active | Modified GW-propagation parameterization; options are `vanilla` and `polexp`. |
| `pade` | `0` | Deprecated | |
| `zres` | `1000` | Active | Redshift grid resolution used when building grids. |
| `z_grid_mode` | `man` | Active | Redshift grid construction mode passed to model construction. |
| `zmin_a` | `1e-05` | Not used | Lower redshift-grid parameter passed to model construction. |
| `zmin_b` | `0.001` | Not used  | Low-redshift grid transition parameter passed to model construction. |
| `zmid_b` | `3.0` | Not used  | Mid-redshift grid parameter passed to model construction. |
| `zmax_c` | `10.0` | Not used  | High-redshift grid parameter passed to model construction. |
| `hi_boost` | `0.2` | Not used  | High-redshift grid boost parameter passed to model construction. |
| `find_z_bounds` | `0` | Not used  | Passed to model construction for optional z-bound finding. |
| `z_pivot` | `0` | Currently unused/deprecated | |
| `check_zres` | empty list | Diagnostic | If non-empty, builds models for the listed redshift resolutions, evaluates a fixed test likelihood, prints a summary, and returns without sampling. |

<a id="likelihood"></a>
### Selection, likelihood, and numerical options

| Option | Default | Status | Meaning |
| --- | --- | --- | --- |
| `sel` | `Tobs` | Active | Selection method name passed to model construction as `sel_method`. |
| `sel_smoothing` | `x30` | Active | Option t control the tapering of the likelihood to enforce accuracy requirement on log lik var.   |
| `chunk_inj` | `0` | Active | If positive, enables chunked/streaming selection-effect evaluation over injections. |
| `chunk_reduce` | `0` | Active | For `pop_only=1`, passed as `chunk_pe` to posterior-sample likelihood packing. |
| `use_float32` | `0` | Not used | Allow float32 use in selected internals; not working. |
| `use_float32_bias` | `0` | Not used | Float32 selection-bias internals; not working. |
| `log_lik_var_min` | `1` | Active | Log likelihood variance threshold . If 0, cut on N_efff is used|
| `min_Neff` | `0` | Active | Cut on effective points in selection bias MC integral. In units of Nobs. IF 0, variance cut is used. |
| `Neff_min_lik` | `0` | Limited/conditional |Cut on effective points in per-event MC integral. Only used in pop_only=1 branch. If 0, variance cut is used.  |
| `fix_inj_len` | `0` | Currently unused/deprecated |  |
| `inj_loop` | `scan-GPU` | Currently unused/deprecated |  |
| `interp_inj` | `0` | Currently unused/deprecated | |
| `detach_var` | `0` | Currently unused/deprecated | |
| `Nsamplesuse` | `-1` | Currently unused/deprecated | |
| `sel_uncertainty` | `0` | Currently unused/deprecated | |
| `alpha_beta_prior` | `sigmoid` | Currently unused/deprecated |  |
| `dil_factor` | `1` | Currently unused/deprecated |  |
| `use_log_alpha_beta` | `0` | Currently unused/deprecated | |

<a id="sampler"></a>
### Sampler options

| Option | Default | Status | Meaning |
| --- | --- | --- | --- |
| `target_accept` | `0.9` | Active | NumPyro NUTS target acceptance probability. |
| `max_tree_depth` | `10` | Active | NumPyro NUTS maximum tree depth. |
| `find_heuristic_step_size` | `0` | Active | Passed to NumPyro NUTS as `find_heuristic_step_size`. |
| `regularize_mass_matrix` | `1e-04` | Active | Passed to NumPyro NUTS as `regularize_mass_matrix`. |
| `dense_mass` | `0` | Currently unused/deprecated | The current code always builds dense blocks from sampled non-`x` sites and passes those blocks to NUTS; this flag is not read. |
| `sampler` | `pymc_bar` | Currently unused/deprecated | Parsed/commented legacy option; no active use in this branch. Numpyro is always used |

<a id="diagnostics"></a>
### Diagnostics and developer options

| Option | Default | Status | Meaning |
| --- | --- | --- | --- |
| `debug` | `0` | Diagnostic | If true, runs a fixed-lambda likelihood test and returns without sampling. |
| `check_init` | `0` | Diagnostic | Prints a placeholder message: “Not yet available.” |
| `profile` | `0` | Diagnostic | If positive, runs a selection-gradient block benchmark with at least three repeats and returns without sampling. |
| `debug_sel_batch` | `0` | Currently unused/deprecated | |
| `recompile` | `0` | Currently unused/deprecated |  |
| `save_thetas` | `1` | Currently unused/deprecated | |
| `backend` | `disk` | Currently unused/deprecated |  |
| `draws_per_chunk` | `100` | Currently unused/deprecated | |
| `MAP_init` | `0` | Currently unused/deprecated |  |
| `is_observed` | `0` | Currently unused/deprecated | |

<a id="example-ini"></a>
## Example INI

```ini
[settings]
fin_data = <path_to_samples>
fin_injections = <path_to_injections>
fout = my_ini
fin_priors = priors_files/priors_GWTC4_DPLDP_SS_rpm_high-mmax-100-250.json
ivals = initvals_files/init_GWTC4_lowVar_from_fid.json
rate_model = MD
spin_model = default_gauss
spin_inj = default
use_sel_spin = 1
smoothing = poly
mass_model = DPLDP
sampling_gw = samples
nchains = 1
ncores = 1
chain_method = sequential
ntune = 4
nsteps = 4
target_accept = 0.85
n_inj_use = 1
marginal_R0 = 1
fix_H0 = 0
fix_Om = 0
chunk_inj = 100000
pop_only = 1
reparam_mass = 1
reparam_z = 1
integrate_dc = quick
nsamplesmax = 5000
chunk_reduce = 100000
sel_smoothing = x30
max_tree_depth = 8
nth = 4
```

<a id="citation"></a>
### Citation

If using this code, please cite this repository and the paper [Sampling the full hierarchical population posterior distribution in gravitational-wave astronomy](<https://arxiv.org/abs/2502.12156>). Bibtex:

```
@article{Mancarella:2025uat,
    author = "Mancarella, Michele and Gerosa, Davide",
    title = "{Sampling the full hierarchical population posterior distribution in gravitational-wave astronomy}",
    eprint = "2502.12156",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.111.103012",
    journal = "Phys. Rev. D",
    volume = "111",
    number = "10",
    pages = "103012",
    year = "2025"
}
```