# Command-line options

Use this reference to look up individual options after following
[Run your analysis](usage.md). The same names appear in INI and JSON settings
without the `--` prefix. Explicit command-line values override file values.

`str`, `int`, and `float` denote strings, integers, and real numbers. `+` means
one or more values. `null` means unset, `[]` is an empty list, and integer
switches generally use `0` or `1`. **Required** settings must be supplied even
when the parser defines a default. The tables list parser defaults; complete
examples set the choices needed for each analysis.

Model-specific options apply only to their corresponding model. The parser
also exposes advanced combinations beyond the workflows described in
[model support](model-support.md); accepting an option does not establish that
all combinations are supported.

- [Settings, data, and event selection](#settings-data-and-event-selection)
- [Event likelihood and selection effects](#event-likelihood-and-selection-effects)
- [Population models](#population-models)
- [DPGMM mass distributions](#dpgmm-mass-distributions)
- [Cosmology and redshift grids](#cosmology-and-redshift-grids)
- [Gaussian-process distances](#gaussian-process-distances)
- [Sampler and initialization](#sampler-and-initialization)
- [Runtime, output, and diagnostics](#runtime-output-and-diagnostics)

## Settings, data, and event selection

| Option | Type / shape | Default | Choices / description |
|---|---|---|---|
| `--settings` | str | `null` | Path to an INI or JSON settings file containing command-line options. |
| `--fin_data` | str / + | `null` | **Required.** Event posterior files or event-mixture filename prefixes, in input-group order. |
| `--fin_injections` | str / + | `null` | **Required.** Injection filename prefixes, in campaign order. |
| `--fin_priors` | str | `""` | **Required.** Population/cosmology prior JSON file. |
| `--priors_for_mmin` | str | `""` | — |
| `--events_use` | str / + | `[]` | Event-name whitelist; selected rows retain source-file order. |
| `--events_exclude` | str / + | `[]` | Event names to omit. |
| `--nev_min` | int | `0` | — |
| `--nev_max` | int | `-1` | — |
| `--nsamplesmax` | int | `-1` | Maximum valid posterior samples per event at loading; -1 keeps all. |
| `--Nsamplesuse` | int | `-1` | — |
| `--spin_inj` | str | `"none"` | Choices: default, default_gauss, chi12xyz, none.  |
| `--remove_spin_prior` | int | `0` | — |

## Event likelihood and selection effects

| Option | Type / shape | Default | Choices / description |
|---|---|---|---|
| `--pop_only` | int | `0` | 1: marginalize over event posterior samples. 0: sample event parameters using the selected event-likelihood representation. |
| `--sampling_gw` | str | `"gauss"` | Use samples for the marginal workflow or gauss for event Gaussian-mixture fits. |
| `--bounded_gmm_transform` | int | `0` | 1 uses bounded event-mixture transformations; requires matching fitted metadata. |
| `--dLprior` | str / + | `["dLsq"]` | Event distance-prior prescriptions, one per event-mixture input group. |
| `--penorm_lims` | str / + | `[]` | — |
| `--cho_dil` | float | `1.0` | Scale applied to event Gaussian proposal Cholesky factors. |
| `--marginal_R0` | int | `1` | 1 marginalizes the overall merger-rate normalization. |
| `--sel` | str | `"Tobs"` | — |
| `--skip_sel` | int | `0` | 1 skips the selection contribution; use only when scientifically appropriate. |
| `--use_sel_spin` | int | `0` | — |
| `--n_inj_use` | float / + | `null` | Fractions of injections to use, one value per injection prefix. |
| `--fix_inj_len` | int | `0` | — |
| `--min_Neff` | int | `0` | — |
| `--Neff_min_lik` | int | `0` | — |
| `--log_lik_var_min` | float | `1` | Threshold used by the log-likelihood variance penalty, with sel_smoothing selecting its form. |
| `--chunk_inj` | int | `0` | Injection chunk size for the selection calculation. |
| `--chunk_reduce` | int | `0` | Chunk size for the event posterior-sample reduction. |
| `--use_float32_bias` | int | `0` | Use float32 in the selection calculation. |
| `--inj_loop` | str | `"scan-GPU"` | Implementation of the injection reduction; paper examples use scan-GPU. |
| `--interp_inj` | int | `0` | — |
| `--detach_var` | int | `0` | — |
| `--sel_uncertainty` | int | `0` | — |
| `--sel_smoothing` | str | `"sigmoid"` | Form of the selection/likelihood-variance penalty; examples include x30 and sigmoid. |
| `--dil_factor` | int | `1` | — |
| `--mmin_inj` | float | `-1` | — |
| `--is_compressed_inj` | int | `0` | — |
| `--r` | float | `0` | — |
| `--allTobs` | float / + | `null` | — |

## Population models

| Option | Type / shape | Default | Choices / description |
|---|---|---|---|
| `--mass_model` | str | `"DPLDP"` | Choices: DPLDP, DPLDP-z, DPUC, DPUC_TruncM2, DPUC_Primary, PLPreg. Population mass model; see the model guide for supported scientific workflows. |
| `--rate_model` | str | `"MD"` | Choices: MD. Merger-rate evolution model. |
| `--spin_model` | str | `"none"` | Choices: none, default_gauss. Population spin distribution. |
| `--interp_mass` | int | `0` | — |
| `--interp_z` | int | `0` | — |
| `--linear_mass` | int | `0` | — |
| `--linear_z` | int | `0` | — |
| `--vary_mb` | int | `1` | — |
| `--smoothing` | str | `"poly"` | — |
| `--simplex_repair` | int | `0` | — |
| `--has_m2_break` | int | `0` | — |
| `--norm_gauss` | str | `"uplow"` | — |
| `--taper_kind` | str | `"sigmoid"` | Mass-boundary taper form, such as power or sigmoid. |
| `--taper_p` | float | `15` | Exponent for the power taper. |
| `--reparam_mass` | int | `1` | — |
| `--reparam_z` | int | `1` | — |

## DPGMM mass distributions

| Option | Type / shape | Default | Choices / description |
|---|---|---|---|
| `--N_DP_comp_max` | int | `30` | Maximum number of population mixture components. |
| `--alpha_tail` | float | `-1` | — |
| `--alpha_small` | float | `0.01` | — |
| `--L_small_1` | float | `1.0` | — |
| `--L_small_2` | float | `1.0` | — |
| `--s_local` | float / + | `[0.5, 1.0]` | — |
| `--find_m_bounds` | float | `0` | Infer mixture-coordinate bounds from the event data. |
| `--DPUC_low_end_share` | float | `1` | — |
| `--DPUC_TruncM2_shared_weights` | int | `0` | For DPUC_TruncM2, use one shared stick-breaking weight vector for the conditional 2D components. If 0, use independent w1 and w2 vectors. |
| `--DPUC_TruncM2_mu2_mode` | str | `"free"` | Choices: free, ordered, q_ratio. For DPUC_TruncM2: free samples log-m2 means freely; ordered uses a moving-bound ordered transform; q_ratio samples q_mu in fixed bounds and sets mu2=mu1+log(q_mu). |
| `--DPUC_TruncM2_secondary_coord` | str | `"logm2"` | Choices: logm2, logitq. For DPUC_TruncM2: logm2 uses log secondary mass; logitq uses logit(m2/m1) and requires DPUC_TruncM2_mu2_mode=q_ratio. |
| `--DPUC_TruncM2_area_reg` | int | `0` | For mass_model=DPUC_TruncM2, add a smooth one-sided prior penalty against too-small weighted component determinant area in log(m1),log(m2). |
| `--DPUC_TruncM2_log_area_min` | float | `-4.6` | Soft floor for TruncM2 log component area log(sig1*sig2*sqrt(1-rho^2)); tuned near log(1e-2). |
| `--DPUC_TruncM2_log_area_width` | float | `0.5` | Softplus transition width for TruncM2 area regularizer. |
| `--DPUC_TruncM2_log_area_lambda` | float | `1.0` | Strength of TruncM2 area regularizer. |
| `--DPUC_TruncM2_aspect_reg` | int | `0` | For mass_model=DPUC_TruncM2, add a smooth one-sided prior penalty against large component aspect ratio sig1/sig2. |
| `--DPUC_TruncM2_aspect_max` | float | `1.0` | Soft upper limit for \|log(sig1/sig2)\| in the TruncM2 aspect regularizer. |
| `--DPUC_TruncM2_aspect_width` | float | `0.5` | Softplus transition width for TruncM2 aspect regularizer. |
| `--DPUC_TruncM2_aspect_lambda` | float | `0.5` | Strength of TruncM2 aspect regularizer. |
| `--DPUC_TruncM2_low_center_weight_reg` | int | `0` | For mass_model=DPUC_TruncM2, add a soft one-sided prior penalty against too much total mixture weight in components centered below a mass cut. |
| `--DPUC_TruncM2_weight_cap_reg` | int | `0` | For mass_model=DPUC_TruncM2, enable global max-component-weight cap by shrinking shared weights toward uniform. |
| `--DPUC_TruncM2_weight_cap` | float | `0.85` | Global maximum allowed component weight after shrink-to-uniform transform. Active only if DPUC_TruncM2_weight_cap_reg=1. |
| `--DPUC_TruncM2_low_center_mcut` | float | `5.0` | Mass cut in Msun defining low-centered TruncM2 primary components. |
| `--DPUC_TruncM2_low_center_log_width` | float | `0.15` | Smoothness in log(m1) for the low-center gate around DPUC_TruncM2_low_center_mcut. |
| `--DPUC_TruncM2_low_center_weight_cap` | float | `0.8` | Soft cap for total shared mixture weight assigned to low-centered primary components. |
| `--DPUC_TruncM2_low_center_weight_width` | float | `0.05` | Softplus transition width for the low-center total-weight cap. |
| `--DPUC_TruncM2_low_center_weight_lambda` | float | `1.0` | Strength of the low-center total-weight regularizer. |
| `--DPUC_low_taper_deltas` | float / + | `[2.0, 2.0]` | Low-mass taper widths for DPUC. Use two positive values '[dm1 dm2]' to fix them. Use single negative value '-1' to sample both with bounded log-space priors using prior keys delta_m1 and delta_m2. |
| `--DPUC_low_end` | float / + | `[-1.0]` | Low-mass ends for DPUC. Use two positive values '[m1 m2]' to fix them. Use single negative value '-1' to sample both using prior keys m1_low and m2_low. |
| `--DPUC_high_taper_delta` | float | `10.0` | High-mass taper width for DPUC. If &gt;0, fix deltam_PL_DPUC_high to this value. If &lt;0, sample it. |
| `--q_mbound_mu_low` | float / + | `[0.05]` | Lower quantile for DP/DPUC Gaussian mean bounds. Either scalar or two values in coordinate order [mass coordinate 1, mass coordinate 2]. |
| `--q_mbound_mu_up` | float / + | `[0.05]` | Upper-tail quantile for DP/DPUC Gaussian mean bounds. Upper bound is quantile(1-q_mbound_mu_up). Either scalar or two values in coordinate order [mass coordinate 1, mass coordinate 2]. |
| `--q_mbound_sig_low` | float / + | `[0.05]` | Lower quantile for DP/DPUC Gaussian width-scale bounds. Either scalar or two values in coordinate order [mass coordinate 1, mass coordinate 2]. |
| `--q_mbound_sig_up` | float / + | `[0.05]` | Upper-tail quantile for DP/DPUC Gaussian width-scale bounds. Upper bound is quantile(1-q_mbound_sig_up). Either scalar or two values in coordinate order [mass coordinate 1, mass coordinate 2]. |
| `--DP_mu_bounds_source` | str | `"samples"` | Choices: samples, event_means. How to set DP/DPUC Gaussian mean prior bounds. 'samples' uses posterior-sample quantiles. 'event_means' uses quantiles of event-wise posterior means. |
| `--DP_sig_bounds_source` | str | `"samples"` | Choices: samples, event_means. How to set DP/DPUC Gaussian width-scale bounds U_i. 'samples' uses posterior-sample quantiles. 'event_means' uses quantiles of event-wise posterior means. |
| `--alpha_inv_params` | float / + | `[1.0, 1.0]` | — |
| `--M_active` | int | `5` | — |
| `--tau_prior` | str | `"flat"` | — |
| `--DP_prior` | str | `"SB"` | — |
| `--sigma_softmax` | float | `0.75` | — |
| `--gamma_DP_params` | float / + | `[1.0, 1.0]` | — |
| `--DP_truncate_up` | int | `0` | — |
| `--DPUC_mmax_fixed` | float | `-1.0` | If &gt;0, fix mmax_DP to this value for DPUC when DP_truncate_up=1. If &lt;=0, sample mmax_DP with the usual prior. |
| `--DP_truncate_low` | int | `0` | — |
| `--DP_m1_env` | int | `0` | — |
| `--DP_m1_env_form` | str | `"powerlaw"` | Choices: powerlaw, smooth_broken_pl_2breaks. Envelope for DP_m1_env=1: powerlaw uses m1^(-alpha); smooth_broken_pl_2breaks uses a smooth plateau/PL/PL envelope in log m1 with two sampled breaks. |
| `--DP_m1_env_delta1` | float | `0.2` | Fixed transition width in log(m1) for the first smooth broken-PL envelope break. |
| `--DP_m1_env_delta2` | float | `0.2` | Fixed transition width in log(m1) for the second smooth broken-PL envelope break. |
| `--DP_q_env` | int | `0` | — |
| `--DPUC_Primary_q_norm` | str | `"conditional"` | Choices: conditional, joint. For DPUC_Primary: conditional normalizes the q envelope at each m1; joint treats q^beta as a joint intensity factor. |
| `--DPUC_kernel` | str | `"gaussian"` | Choices: gaussian, local_native_gaussian, local_native_product_gaussian. Kernel family for DPUC: gaussian has infinite support; local_native_gaussian has a smooth radial window; local_native_product_gaussian has product 1D windows. The local kernels require DPUC with rate_model=MD. |
| `--DPUC_kernel_rcut` | float | `4.0` | Local-native DPUC kernel cutoff radius in native Mahalanobis units. |
| `--DPUC_kernel_dr` | float | `0.15` | Smooth transition width for the local-native DPUC kernel cutoff. |
| `--DPUC_kernel_xcut` | float | `3.0` | x=logMc cutoff for DPUC_kernel='local_native_product_gaussian'. |
| `--DPUC_kernel_ycut` | float | `3.0` | y=logit(q) cutoff for DPUC_kernel='local_native_product_gaussian'. |
| `--DPUC_q_coord` | str | `"logitq"` | Choices: logitq, q. DPUC mass-ratio coordinate: logitq uses Gaussian components in logit(q); q uses Gaussian components truncated to 0&lt;=q&lt;=1. |
| `--DPUC_physical_Lsmall_shared_mode` | str | `"max"` | Choices: max, min. Only used when --mass_model DPUC_Primary and --DPUC_Primary. Selects how to combine physical PE-resolution floors from m1 and m2: 'max' is conservative; 'min' is more permissive. |
| `--DPUC_q_mu_bounds_source` | str | `"inferred"` | Choices: inferred, unit. For DPUC_q_coord=q: inferred derives bounds from posterior samples or event means after transformation to q; unit fixes mean support to [0,1]. |
| `--center_eps_widths` | int | `0` | — |
| `--DP_corr_m1m2` | int | `0` | Enable component correlations where supported by the mass model. |
| `--DP_corr_eta_rho` | float | `4.0` | — |
| `--DP_corr_f_minor` | float | `0.5` | — |
| `--PL_DPUC_normalize_gmm` | int | `0` | — |
| `--L_small_source` | str | `"spacing"` | — |
| `--L_small_res_percentile` | float / + | `[25.0]` | PE-resolution percentile for L_small. Either scalar or list in coordinate order [mass coordinate 1, mass coordinate 2]. |
| `--L_small_res_factor` | float / + | `[0.5]` | PE-resolution multiplicative factor for L_small. Either scalar or list in coordinate order [mass coordinate 1, mass coordinate 2]. |
| `--mu_log_alpha_inv` | float | `0.0` | Mean of the Normal prior on log inverse concentration for the corresponding stick-breaking prior. |
| `--sigma_log_alpha_inv` | float | `0.5` | Standard deviation of that log inverse-concentration prior. |

## Cosmology and redshift grids

| Option | Type / shape | Default | Choices / description |
|---|---|---|---|
| `--fix_H0` | int | `1` | 1 holds the Hubble constant fixed; 0 infers it. |
| `--fix_Om` | int | `1` | 1 holds the matter density parameter fixed; 0 infers it. |
| `--fix_w0` | int | `1` | 1 holds the dark-energy equation-of-state parameter fixed; 0 infers it. |
| `--fix_Xi0n` | int | `1` | 1 holds the modified-propagation parameters fixed; 0 infers them. |
| `--z_pivot` | float | `0` | — |
| `--integrate_dc` | str | `"pade"` | Comoving-distance calculation method. |
| `--param` | str | `"vanilla"` | — |
| `--pade` | int | `0` | — |
| `--zres` | int | `1000` | Redshift integration/interpolation grid resolution. |
| `--z_grid_mode` | str | `"man"` | — |
| `--zmin_a` | float | `1e-05` | — |
| `--zmin_b` | float | `0.001` | — |
| `--zmid_b` | float | `3.0` | — |
| `--zmax_c` | float | `10.0` | — |
| `--hi_boost` | float | `0.2` | — |
| `--find_z_bounds` | int | `0` | — |
| `--is_observed` | int | `0` | — |
| `--sample_from_pop` | int | `0` | — |
| `--reparam_cosmo` | int | `1` | — |

## Gaussian-process distances

| Option | Type / shape | Default | Choices / description |
|---|---|---|---|
| `--is_GP_dL` | int | `0` | 1 enables GP reconstruction of the distance relation. |
| `--find_GP_L` | int | `1` | — |
| `--monotonicity` | str | `"mono_reparam"` | — |
| `--GP_kernel` | str | `"stationary"` | GP covariance kernel; the documented examples use gibbs_resolution or matern52. |
| `--U` | float | `0` | Amplitude-prior scale, with P(eta &gt; U) = alpha_eta. |
| `--GP_prior` | str | `"frechet"` | GP hyperprior family; the distance model implements frechet. |
| `--large_ell_penalty` | int | `1` | — |
| `--ell_min` | float | `0.1` | Small-length-scale prior reference for stationary kernels. |
| `--ell_max` | float | `3.0` | Large-length-scale prior reference for stationary kernels. |
| `--ell0_min` | float | `1.0` | Small-length-scale prior reference for the Gibbs kernel. |
| `--ell0_max` | float | `10.0` | Large-length-scale prior reference for the Gibbs kernel. |
| `--nu` | float | `0.5` | — |
| `--lam` | float | `1.0` | — |
| `--res_nbins` | int | `15` | Number of bins used to estimate event resolution. |
| `--res_q` | float | `0.1` | Quantile used in the event-resolution estimate. |
| `--smooth_GP_resolution_loglinear` | int | `0` | 1 smooths the resolution function using log-linear interpolation. |
| `--fine_res` | float | `0.05` | — |
| `--res_highz` | float | `0.1` | — |
| `--res_lowz` | float | `0.05` | — |
| `--eps_DE` | float | `-1.0` | — |
| `--invert_dL_GP` | int | `1` | — |
| `--n_gp_nodes` | int | `150` | Number of GP nodes. |
| `--n_gp_fine` | int | `900` | Number of points in the fine GP grid. |
| `--GP_jitter` | float | `1e-10` | Diagonal covariance jitter used in GP factorization. |
| `--init_GP` | str | `"zeros"` | — |
| `--GP_oob_guard` | int | `1` | If 1, add hard GP dL interpolation out-of-bounds penalty. If 0, only record diagnostics. |
| `--alpha_eta` | float | `0.01` | Upper-tail probability defining the amplitude-prior scale U. |
| `--alpha_ell` | float | `0.01` | Probability controlling the small-length-scale prior. |
| `--alpha_large` | float | `0.05` | Probability controlling the large-length-scale penalty. |
| `--logXi_guard` | int | `0` | — |
| `--logXi_abs_max_allowed` | float | `5.0` | — |
| `--logXi_guard_strength` | float | `1000.0` | — |

## Sampler and initialization

| Option | Type / shape | Default | Choices / description |
|---|---|---|---|
| `--seed` | int | `0` | Random-number seed. |
| `--ivals` | str | `""` | Optional JSON initialization file, consistent with the model and data. |
| `--MAP_init` | int | `0` | — |
| `--eps_init` | float | `0.01` | — |
| `--params_fix` | str | `""` | Optional JSON file of fixed parameter values. |
| `--nsteps` | int | `100` | **Required.** Posterior draws per chain, after warmup. |
| `--ntune` | int | `100` | **Required.** Warmup steps per chain. |
| `--nchains` | int | `1` | Number of chains. |
| `--ncores` | int | `1` | Number of host devices made available for CPU chain parallelism. |
| `--target_accept` | float | `0.9` | Target NUTS acceptance probability. |
| `--chain_method` | str | `"sequential"` | Chain execution mode: sequential, parallel, or vectorized. |
| `--dense_mass` | int | `0` | 1 adapts a dense mass matrix; 0 uses a diagonal matrix. |
| `--max_tree_depth` | int | `10` | Maximum NUTS tree depth. |
| `--find_heuristic_step_size` | int | `0` | — |
| `--regularize_mass_matrix` | float | `0.0001` | — |
| `--init_per_chain_jitter` | float | `0.0` | If &gt;0, build initial unconstrained params by jittering --ivals/init_vals before initialize_model. For nchains&gt;1, one jittered point per chain is built; for nchains=1, the single chain is jittered using the run seed. Use small values such as 0.005--0.02. |

## Runtime, output, and diagnostics

| Option | Type / shape | Default | Choices / description |
|---|---|---|---|
| `--fout` | str | `"results/"` | **Required.** Writable output directory; use a fresh directory for each independent analysis. |
| `--backend` | str | `"disk"` | Trace-saving backend. |
| `--draws_per_chunk` | int | `100` | — |
| `--check_init` | int | `0` | — |
| `--debug` | int | `0` | — |
| `--debug_sel_batch` | int | `0` | — |
| `--profile` | int | `0` | — |
| `--recompile` | int | `0` | — |
| `--save_thetas` | int | `1` | — |
| `--use_float32` | int | `0` | Use float32 instead of the default float64 calculations. |
| `--jax_debug_nans` | int | `0` | Enable JAX NaN debugging. |
| `--check_zres` | int / + | `[]` | — |
| `--xla_cpu_multi_thread_eigen` | str | `"true"` | — |
| `--nth` | int | `null` | Thread-count limit for numerical libraries. |

To display the launcher help:

```bash
python pymcpop-gw/fit_model_numpyro.py --help
```
