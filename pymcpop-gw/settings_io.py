"""Import-light CLI/INI/JSON settings handling for the inference launcher.

JSON preserves typed lists, empty lists, nulls, and strings containing spaces.
Explicit command-line values override settings-file values. This module imports
only the standard library so configuration validation does not initialize JAX.
"""

import argparse
import configparser
import json
from pathlib import Path
import shlex
import sys


_SCALAR_OR_PAIR = {
    "s_local", "q_mbound_mu_low", "q_mbound_mu_up",
    "q_mbound_sig_low", "q_mbound_sig_up",
    "L_small_res_percentile", "L_small_res_factor",
}


class IniSettings(dict):
    """INI strings use shell tokenization, unlike typed JSON values."""


def load_settings_file(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as stream:
            settings = json.load(stream)
        if not isinstance(settings, dict):
            raise ValueError("JSON settings must be an object of option names and values")
        return settings
    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    config.optionxform = str
    with path.open(encoding="utf-8") as stream:
        config.read_file(stream)
    if not config.has_section("settings"):
        raise ValueError(f"Settings file {str(path)!r} must contain a [settings] section")
    return IniSettings(config.items("settings"))


def _convert_value(action, value, *, ini=False):
    if value is None:
        return None
    multiple = action.nargs not in (None, "?")
    if ini:
        values = shlex.split(value)
        if not multiple:
            if len(values) != 1:
                raise ValueError(f"--{action.dest} expects one value")
            value = values[0]
        else:
            value = values
    elif multiple and not isinstance(value, (list, dict)) and action.dest in _SCALAR_OR_PAIR:
        value = [value]
    elif multiple and not isinstance(value, list):
        raise ValueError(f"--{action.dest} expects a JSON array")
    elif not multiple and isinstance(value, (list, dict)):
        raise ValueError(f"--{action.dest} expects a scalar value")

    values = value if multiple else [value]
    if isinstance(action.nargs, int) and len(values) != action.nargs:
        raise ValueError(f"--{action.dest} expects {action.nargs} values")
    converted = []
    for item in values:
        if isinstance(item, (list, dict)) or item is None:
            raise ValueError(f"--{action.dest} contains an invalid value")
        if action.type is int and isinstance(item, float) and not item.is_integer():
            raise ValueError(f"--{action.dest} expects an integer, got {item!r}")
        try:
            item = action.type(item) if action.type is not None else item
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid --{action.dest} value: {item!r}") from exc
        if action.choices is not None and item not in action.choices:
            raise ValueError(f"Invalid --{action.dest} choice: {item!r}")
        converted.append(item)
    return converted if multiple else converted[0]


def settings_to_namespace(settings, parser):
    actions = {a.dest: a for a in parser._actions if a.option_strings}
    unknown = sorted(set(settings) - set(actions))
    if unknown:
        raise ValueError("Unknown option(s) in settings file: " + ", ".join(unknown))
    values = {
        key: _convert_value(actions[key], value, ini=isinstance(settings, IniSettings))
        for key, value in settings.items() if key != "settings"
    }
    return argparse.Namespace(**values)


def settings_to_argv(settings, parser):
    """Render representable settings as tokens; typed parsing preserves null/[] too."""
    namespace = settings_to_namespace(settings, parser)
    actions = {a.dest: a for a in parser._actions}
    tokens = []
    for key, value in vars(namespace).items():
        if value is None or value == []:
            continue
        action = actions[key]
        tokens.append(next(s for s in action.option_strings if s.startswith("--")))
        tokens.extend(str(x) for x in (value if isinstance(value, list) else [value]))
    return tokens


def parse_args_with_optional_settings(parser, argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--settings", default=None)
    settings_args, _ = probe.parse_known_args(argv)
    try:
        namespace = (settings_to_namespace(load_settings_file(settings_args.settings), parser)
                     if settings_args.settings else argparse.Namespace())
    except (OSError, ValueError, configparser.Error) as exc:
        parser.error(str(exc))
    required = [a for a in parser._actions if a.required]
    try:
        for action in required:
            action.required = False
        flags = parser.parse_args(argv, namespace=namespace)
    finally:
        for action in required:
            action.required = True
    missing = [a.dest for a in required
               if getattr(flags, a.dest, None) in (None, "", [])]
    if missing:
        parser.error("the following arguments are required, either on the command line "
                     "or in --settings: " + ", ".join("--" + k for k in missing))
    return flags


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default=None, type=str, required=False,
                        help="Path to an INI or JSON settings file containing command-line options.")
    
    
    parser.add_argument("--fin_data", nargs='+', type=str, required=True)
    parser.add_argument("--fin_injections", nargs='+', type=str, required=True)
    parser.add_argument("--fin_priors", default='', type=str, required=True)
    parser.add_argument("--priors_for_mmin", default='', type=str, required=False)
    parser.add_argument("--events_use", nargs='+', default=[], type=str, required=False)
    parser.add_argument("--events_exclude", nargs='+', default=[], type=str, required=False)
    parser.add_argument("--backend", default='disk', type=str, required=False)
    parser.add_argument("--seed", default=0, type=int, required=False)
    parser.add_argument("--skip_sel", default=0, type=int, required=False)

    
    parser.add_argument("--draws_per_chunk", default=100, type=int, required=False)
    

    parser.add_argument("--nev_min", default=0, type=int, required=False)
    parser.add_argument("--nev_max", default=-1, type=int, required=False)
    
    parser.add_argument("--pop_only", default=0, type=int, required=False)
    
    parser.add_argument("--rate_model", default='MD', type=str, required=False)
    parser.add_argument("--mass_model", default='DPLDP', type=str, required=False)
    parser.add_argument("--spin_model", default='none', type=str, required=False)
    parser.add_argument("--interp_mass", default=0, type=int, required=False)
    parser.add_argument("--interp_z", default=0, type=int, required=False)
    parser.add_argument("--linear_mass", default=0, type=int, required=False)
    parser.add_argument("--linear_z", default=0, type=int, required=False)

    parser.add_argument("--vary_mb", default=1, type=int, required=False)

    
    parser.add_argument("--N_DP_comp_max", default=30, type=int, required=False)
    parser.add_argument("--alpha_tail", default=-1, type=float, required=False)
    parser.add_argument("--alpha_small", default=0.01, type=float, required=False)
    parser.add_argument("--L_small_1", default=1., type=float, required=False)
    parser.add_argument("--L_small_2", default=1., type=float, required=False)
    parser.add_argument("--s_local", nargs='+', default=[0.5, 1.0], type=float, required=False)
    parser.add_argument("--find_m_bounds", default=0, type=float, required=False)
    parser.add_argument("--DPUC_low_end_share", default=1, type=float, required=False)
    parser.add_argument("--DPUC_TruncM2_shared_weights", default=0, type=int, required=False,
                        help="For mass_model=DPUC_TruncM2, use one shared stick-breaking weight vector for the K conditional 2D components. If 0, keep legacy independent w1/w2 behavior.")
    parser.add_argument(
        "--DPUC_TruncM2_mu2_mode",
        default="free",
        choices=("free", "ordered", "q_ratio"),
        type=str,
        required=False,
        help=(
            "Only used when --mass_model DPUC_TruncM2. "
            "'free' keeps the legacy free log-m2 component means. "
            "'ordered' keeps the experimental moving-bound ordered transform. "
            "'q_ratio' samples q_mu in fixed bounds and sets mu2=mu1+log(q_mu)."
        ),
    )
    parser.add_argument(
        "--DPUC_TruncM2_secondary_coord",
        default="logm2",
        choices=("logm2", "logitq"),
        type=str,
        required=False,
        help=(
            "Only used when --mass_model DPUC_TruncM2. "
            "'logm2' keeps the legacy secondary Gaussian coordinate. "
            "'logitq' uses y=logit(m2/m1) as the secondary Gaussian coordinate; "
            "requires --DPUC_TruncM2_mu2_mode q_ratio."
        ),
    )

    parser.add_argument("--DPUC_TruncM2_area_reg", default=0, type=int, required=False,
                        help="For mass_model=DPUC_TruncM2, add a smooth one-sided prior penalty against too-small weighted component determinant area in log(m1),log(m2).")
    parser.add_argument("--DPUC_TruncM2_log_area_min", default=-4.6, type=float, required=False,
                        help="Soft floor for TruncM2 log component area log(sig1*sig2*sqrt(1-rho^2)); tuned near log(1e-2).")
    parser.add_argument("--DPUC_TruncM2_log_area_width", default=0.5, type=float, required=False,
                        help="Softplus transition width for TruncM2 area regularizer.")
    parser.add_argument("--DPUC_TruncM2_log_area_lambda", default=1.0, type=float, required=False,
                        help="Strength of TruncM2 area regularizer.")
    parser.add_argument("--DPUC_TruncM2_aspect_reg", default=0, type=int, required=False,
                        help="For mass_model=DPUC_TruncM2, add a smooth one-sided prior penalty against large component aspect ratio sig1/sig2.")
    parser.add_argument("--DPUC_TruncM2_aspect_max", default=1.0, type=float, required=False,
                        help="Soft upper limit for |log(sig1/sig2)| in the TruncM2 aspect regularizer.")
    parser.add_argument("--DPUC_TruncM2_aspect_width", default=0.5, type=float, required=False,
                        help="Softplus transition width for TruncM2 aspect regularizer.")
    parser.add_argument("--DPUC_TruncM2_aspect_lambda", default=0.5, type=float, required=False,
                        help="Strength of TruncM2 aspect regularizer.")

    parser.add_argument("--DPUC_TruncM2_low_center_weight_reg", default=0, type=int, required=False,
                        help="For mass_model=DPUC_TruncM2, add a soft one-sided prior penalty against too much total mixture weight in components centered below a mass cut.")

    parser.add_argument("--DPUC_TruncM2_weight_cap_reg", default=0, type=int, required=False,
                        help="For mass_model=DPUC_TruncM2, enable global max-component-weight cap by shrinking shared weights toward uniform.")
    parser.add_argument("--DPUC_TruncM2_weight_cap", default=0.85, type=float, required=False,
                        help="Global maximum allowed component weight after shrink-to-uniform transform. Active only if DPUC_TruncM2_weight_cap_reg=1.")
    
    parser.add_argument("--DPUC_TruncM2_low_center_mcut", default=5.0, type=float, required=False,
                        help="Mass cut in Msun defining low-centered TruncM2 primary components.")
    parser.add_argument("--DPUC_TruncM2_low_center_log_width", default=0.15, type=float, required=False,
                        help="Smoothness in log(m1) for the low-center gate around DPUC_TruncM2_low_center_mcut.")
    parser.add_argument("--DPUC_TruncM2_low_center_weight_cap", default=0.8, type=float, required=False,
                        help="Soft cap for total shared mixture weight assigned to low-centered primary components.")
    parser.add_argument("--DPUC_TruncM2_low_center_weight_width", default=0.05, type=float, required=False,
                        help="Softplus transition width for the low-center total-weight cap.")
    parser.add_argument("--DPUC_TruncM2_low_center_weight_lambda", default=1.0, type=float, required=False,
                        help="Strength of the low-center total-weight regularizer.")
    
    
    parser.add_argument(
        "--DPUC_low_taper_deltas",
        nargs="+",
        type=float,
        default=[2.0, 2.0],
        required=False,
        help=(
            "Low-mass taper widths for DPUC. "
            "Use two positive values '[dm1 dm2]' to fix them. "
            "Use single negative value '-1' to sample both with bounded log-space priors "
            "using prior keys delta_m1 and delta_m2."
        ),
    )

    parser.add_argument(
        "--DPUC_low_end",
        nargs="+",
        type=float,
        default=[-1.0],
        required=False,
        help=(
            "Low-mass ends for DPUC. "
            "Use two positive values '[m1 m2]' to fix them. "
            "Use single negative value '-1' to sample both  "
            "using prior keys m1_low and m2_low."
        ),
    )

    parser.add_argument(
        "--DPUC_high_taper_delta",
        default=10.0,
        type=float,
        required=False,
        help=(
            "High-mass taper width for DPUC. "
            "If >0, fix deltam_PL_DPUC_high to this value. "
            "If <0, sample it."
        ),
    )
    
    parser.add_argument(
        "--q_mbound_mu_low",
        nargs="+",
        default=[0.05],
        type=float,
        required=False,
        help=(
            "Lower quantile for DP/DPUC Gaussian mean bounds. "
            "Either scalar or two values in coordinate order [mass coordinate 1, mass coordinate 2]."
        ),
    )

    parser.add_argument(
        "--q_mbound_mu_up",
        nargs="+",
        default=[0.05],
        type=float,
        required=False,
        help=(
            "Upper-tail quantile for DP/DPUC Gaussian mean bounds. "
            "Upper bound is quantile(1-q_mbound_mu_up). "
            "Either scalar or two values in coordinate order [mass coordinate 1, mass coordinate 2]."
        ),
    )

    parser.add_argument(
        "--q_mbound_sig_low",
        nargs="+",
        default=[0.05],
        type=float,
        required=False,
        help=(
            "Lower quantile for DP/DPUC Gaussian width-scale bounds. "
            "Either scalar or two values in coordinate order [mass coordinate 1, mass coordinate 2]."
        ),
    )

    parser.add_argument(
        "--q_mbound_sig_up",
        nargs="+",
        default=[0.05],
        type=float,
        required=False,
        help=(
            "Upper-tail quantile for DP/DPUC Gaussian width-scale bounds. "
            "Upper bound is quantile(1-q_mbound_sig_up). "
            "Either scalar or two values in coordinate order [mass coordinate 1, mass coordinate 2]."
        ),
    )

    parser.add_argument(
        "--DP_mu_bounds_source",
        default="samples",
        type=str,
        choices=["samples", "event_means"],
        required=False,
        help=(
            "How to set DP/DPUC Gaussian mean prior bounds. "
            "'samples' uses posterior-sample quantiles. "
            "'event_means' uses quantiles of event-wise posterior means."
        ),
    )

    parser.add_argument(
        "--DP_sig_bounds_source",
        default="samples",
        type=str,
        choices=["samples", "event_means"],
        required=False,
        help=(
            "How to set DP/DPUC Gaussian width-scale bounds U_i. "
            "'samples' uses posterior-sample quantiles. "
            "'event_means' uses quantiles of event-wise posterior means."
        ),
    )
    parser.add_argument("--alpha_inv_params", nargs='+', type=float, default=[1., 1.], required=False)
    parser.add_argument("--M_active", default=5, type=int, required=False)
    parser.add_argument("--tau_prior", default="flat", type=str, required=False)
    parser.add_argument("--DP_prior", default='SB', type=str, required=False) # SB, dirichelet, softmax
    parser.add_argument("--sigma_softmax", default=0.75, type=float, required=False)
    parser.add_argument("--gamma_DP_params", nargs='+', type=float, default=[1., 1.], required=False)
    parser.add_argument("--DP_truncate_up", default=0, type=int, required=False)
    parser.add_argument(
        "--DPUC_mmax_fixed",
        default=-1.0,
        type=float,
        required=False,
        help=(
            "If >0, fix mmax_DP to this value for DPUC when DP_truncate_up=1. "
            "If <=0, sample mmax_DP with the usual prior."
        ),
    )
    parser.add_argument("--DP_truncate_low", default=0, type=int, required=False)
    parser.add_argument("--DP_m1_env", default=0, type=int, required=False)
    parser.add_argument(
        "--DP_m1_env_form",
        default="powerlaw",
        choices=("powerlaw", "smooth_broken_pl_2breaks"),
        type=str,
        required=False,
        help=(
            "Envelope used when --DP_m1_env 1. 'powerlaw' keeps the old single "
            "m1^{-alpha} envelope. 'smooth_broken_pl_2breaks' uses a smooth "
            "plateau/PL/PL envelope in log m1 with two sampled breaks."
        ),
    )
    parser.add_argument("--DP_m1_env_delta1", default=0.2, type=float, required=False,
                        help="Fixed transition width in log(m1) for the first smooth broken-PL envelope break.")
    parser.add_argument("--DP_m1_env_delta2", default=0.2, type=float, required=False,
                        help="Fixed transition width in log(m1) for the second smooth broken-PL envelope break.")
    parser.add_argument("--DP_q_env", default=0, type=int, required=False)
    parser.add_argument(
        "--DPUC_Primary_q_norm",
        default="conditional",
        choices=("conditional", "joint"),
        type=str,
        required=False,
        help=(
            "Only used for mass_model='DPUC_Primary'. "
            "'conditional' keeps the current per-m1 q-envelope normalization. "
            "'joint' skips the conditional q normalization and treats q^beta as a joint intensity factor."
        ),
    )



    parser.add_argument(
        "--DPUC_kernel",
        default="gaussian",
        choices=("gaussian", "local_native_gaussian", "local_native_product_gaussian"),
        type=str,
        required=False,
        help=(
            "DPUC kernel family. 'gaussian' keeps the current infinite-support "
            "Gaussian basis. 'local_native_gaussian' uses a smooth radial local native-space "
            "kernel; 'local_native_product_gaussian' uses product 1D native windows. Both are only supported for mass_model='DPUC' with rate_model in "
            "('MD',)."
        ),
    )
    parser.add_argument(
        "--DPUC_kernel_rcut",
        default=4.0,
        type=float,
        required=False,
        help="Local-native DPUC kernel cutoff radius in native Mahalanobis units.",
    )
    parser.add_argument(
        "--DPUC_kernel_dr",
        default=0.15,
        type=float,
        required=False,
        help="Smooth transition width for the local-native DPUC kernel cutoff.",
    )
    parser.add_argument(
        "--DPUC_kernel_xcut",
        default=3.0,
        type=float,
        required=False,
        help="x=logMc cutoff for DPUC_kernel='local_native_product_gaussian'.",
    )
    parser.add_argument(
        "--DPUC_kernel_ycut",
        default=3.0,
        type=float,
        required=False,
        help="y=logit(q) cutoff for DPUC_kernel='local_native_product_gaussian'.",
    )

    parser.add_argument(
        "--DPUC_q_coord",
        default="logitq",
        choices=("logitq", "q"),
        type=str,
        required=False,
        help=(
            "DPUC coordinate used for the mass-ratio direction. "
            "'logitq' keeps the current Gaussian in logit(q). "
            "'q' uses q=m2/m1 with Gaussian components truncated to 0<=q<=1."
        ),
    )


    parser.add_argument(
        "--DPUC_physical_Lsmall_shared_mode",
        default="max",
        choices=("max", "min"),
        type=str,
        required=False,
        help=(
            "Only used when --mass_model DPUC_Primary and --DPUC_Primary. "
            "Selects how to combine physical PE-resolution floors from m1 and m2: "
            "'max' is conservative; 'min' is more permissive."
        ),
    )

    parser.add_argument(
        "--DPUC_q_mu_bounds_source",
        default="inferred",
        choices=("inferred", "unit"),
        type=str,
        required=False,
        help=(
            "Only used when --DPUC_q_coord q. "
            "'inferred' uses the existing posterior-sample/event-mean bound machinery "
            "after transforming coordinate 2 to q. "
            "'unit' forces the q-coordinate mean support to [0,1]."
        ),
    )

    parser.add_argument("--center_eps_widths", default=0, type=int, required=False)
    parser.add_argument("--DP_corr_m1m2", default=0, type=int, required=False)
    parser.add_argument("--DP_corr_eta_rho", type=float, default=4.0)
    parser.add_argument("--DP_corr_f_minor", type=float, default=0.5)
    parser.add_argument("--PL_DPUC_normalize_gmm", default=0, type=int, required=False)
    
    parser.add_argument("--L_small_source", default="spacing", type=str, required=False)
    parser.add_argument(
        "--L_small_res_percentile",
        nargs="+",
        type=float,
        default=[25.0],
        required=False,
        help=(
            "PE-resolution percentile for L_small. "
            "Either scalar or list in coordinate order [mass coordinate 1, mass coordinate 2]."
        ),
    )
    
    parser.add_argument(
        "--L_small_res_factor",
        nargs="+",
        type=float,
        default=[0.5],
        required=False,
        help=(
            "PE-resolution multiplicative factor for L_small. "
            "Either scalar or list in coordinate order [mass coordinate 1, mass coordinate 2]."
        ),
    )
    
    parser.add_argument("--mu_log_alpha_inv", default=0., type=float, required=False)
    parser.add_argument("--sigma_log_alpha_inv", default=0.5, type=float, required=False)

    
    parser.add_argument("--marginal_R0", default=1, type=int, required=False)
    parser.add_argument("--smoothing", default='poly', type=str, required=False)
    parser.add_argument("--simplex_repair", default=0, type=int, required=False)

    parser.add_argument("--has_m2_break", default=0, type=int, required=False)
    parser.add_argument("--norm_gauss", default='uplow', type=str, required=False)

    parser.add_argument("--taper_kind", default='sigmoid', type=str, required=False)
    parser.add_argument("--taper_p", default=15, type=float, required=False)
    
    
    
    parser.add_argument("--dLprior", nargs='+', default=['dLsq'], type=str, required=False)
    #parser.add_argument("--normalize_PE_prior",  default=1, type=int, required=False)
    parser.add_argument("--penorm_lims",  nargs='+', default=[], type=str, required=False)
    parser.add_argument("--use_sel_spin", default=0, type=int, required=False)
    
    
    parser.add_argument("--sampling_gw", default='gauss', type=str, required=False)
    parser.add_argument("--bounded_gmm_transform", default=0, type=int, required=False)
    parser.add_argument("--cho_dil", default=1., type=float, required=False)
    parser.add_argument("--sel", default='Tobs', type=str, required=False)
    parser.add_argument("--ivals", default='', type=str, required=False)
    parser.add_argument("--MAP_init", default=0, type=int, required=False)
    parser.add_argument("--eps_init", default=0.01, type=float, required=False)
    parser.add_argument("--params_fix", default='', type=str, required=False)
    parser.add_argument("--check_init", default=0, type=int, required=False)
    parser.add_argument("--debug", default=0, type=int, required=False)
    parser.add_argument("--debug_sel_batch", default=0, type=int, required=False)
    parser.add_argument("--profile", default=0, type=int, required=False)
    parser.add_argument("--recompile", default=0, type=int, required=False)

    parser.add_argument("--save_thetas", default=1, type=int, required=False)
    
    
    
    parser.add_argument("--n_inj_use", nargs='+', type=float, required=False)
    parser.add_argument("--fix_inj_len", default=0, type=int, required=False)
    parser.add_argument("--min_Neff", default=0, type=int, required=False)
    parser.add_argument("--Neff_min_lik", default=0, type=int, required=False)
    parser.add_argument("--log_lik_var_min", default=1, type=float, required=False)
    parser.add_argument("--chunk_inj", default=0, type=int, required=False)
    parser.add_argument("--chunk_reduce", default=0, type=int, required=False)
    parser.add_argument("--use_float32", default=0, type=int, required=False)
    parser.add_argument("--use_float32_bias", default=0, type=int, required=False)
    parser.add_argument("--inj_loop", default='scan-GPU', type=str, required=False)
    parser.add_argument("--interp_inj", default=0, type=int, required=False)
    parser.add_argument("--detach_var", default=0, type=int, required=False)
    
    parser.add_argument("--nsamplesmax", default=-1, type=int, required=False)
    parser.add_argument("--spin_inj", default='none', type=str, required=False)
    parser.add_argument("--Nsamplesuse", default=-1, type=int, required=False)
    parser.add_argument("--sel_uncertainty", default=0, type=int, required=False)
    parser.add_argument("--sel_smoothing", default='sigmoid', type=str, required=False)
    parser.add_argument("--dil_factor", default=1, type=int, required=False)
    
    parser.add_argument("--fout", default='results/', type=str, required=True)
    
    #parser.add_argument("--sampler", default='pymc_bar', type=str, required=False)
    parser.add_argument("--nsteps", default=100, type=int, required=True)
    parser.add_argument("--ntune", default=100, type=int, required=True)
    parser.add_argument("--nchains", default=1, type=int, required=False)
    parser.add_argument("--ncores", default=1, type=int, required=False)
    parser.add_argument("--target_accept", default=0.9, type=float, required=False)
    parser.add_argument("--chain_method", default='sequential', type=str, required=False)
    parser.add_argument("--jax_debug_nans", default=0, type=int, required=False)
    parser.add_argument("--dense_mass", default=0, type=int, required=False)
    parser.add_argument("--max_tree_depth", default=10, type=int, required=False)
    parser.add_argument("--find_heuristic_step_size", default=0, type=int, required=False)
    parser.add_argument("--regularize_mass_matrix", default=1e-04, type=float, required=False)

    parser.add_argument("--check_zres", nargs="+", type=int, default=[], required=False)
    
    
    parser.add_argument("--fix_H0", default=1, type=int, required=False)
    parser.add_argument("--fix_Om", default=1, type=int, required=False)
    parser.add_argument("--fix_w0", default=1, type=int, required=False)
    parser.add_argument("--fix_Xi0n", default=1, type=int, required=False)
    parser.add_argument("--z_pivot", default=0, type=float, required=False)
    parser.add_argument("--integrate_dc", default='pade', type=str, required=False)
    
    
    parser.add_argument("--param", default='vanilla', type=str, required=False)
    parser.add_argument("--pade", default=0, type=int, required=False)
    parser.add_argument("--zres", default=1000, type=int, required=False)
    parser.add_argument("--z_grid_mode", default='man', type=str, required=False)
    
    parser.add_argument("--zmin_a", default=1e-05, type=float, required=False)
    parser.add_argument("--zmin_b", default=1e-03, type=float, required=False)
    parser.add_argument("--zmid_b", default=3., type=float, required=False)
    parser.add_argument("--zmax_c", default=10., type=float, required=False)
    parser.add_argument("--hi_boost", default=.2, type=float, required=False)
    parser.add_argument("--find_z_bounds", default=0, type=int, required=False)
    parser.add_argument("--is_observed", default=0, type=int, required=False)
    parser.add_argument("--sample_from_pop", default=0, type=int, required=False)

    parser.add_argument("--mmin_inj", default=-1, type=float, required=False)
    parser.add_argument("--is_compressed_inj", default=0, type=int, required=False)
    
    parser.add_argument("--r",  default=0, type=float, required=False)
    parser.add_argument("--allTobs", nargs='+', type=float, required=False)

    parser.add_argument("--reparam_mass", default=1, type=int, required=False)
    parser.add_argument("--reparam_z", default=1, type=int, required=False)
    parser.add_argument("--reparam_cosmo", default=1, type=int, required=False)

    parser.add_argument("--remove_spin_prior", default=0, type=int, required=False)

    

    parser.add_argument("--xla_cpu_multi_thread_eigen", default='true', type=str, required=False)

    parser.add_argument("--nth", type=int, default=None)



    parser.add_argument("--init_per_chain_jitter", default=0.0, type=float, required=False,
                        help="If >0, build initial unconstrained params by jittering --ivals/init_vals before initialize_model. For nchains>1, one jittered point per chain is built; for nchains=1, the single chain is jittered using the run seed. Use small values such as 0.005--0.02.")
    parser.add_argument("--is_GP_dL", default=0, type=int, required=False)
    parser.add_argument("--find_GP_L", default=1, type=int, required=False)
    parser.add_argument("--monotonicity", default="mono_reparam", type=str, required=False)
    parser.add_argument("--GP_kernel", default="stationary", type=str, required=False)
    parser.add_argument("--U", default=0, type=float, required=False)
    parser.add_argument("--GP_prior", default="frechet", type=str, required=False)
    parser.add_argument("--large_ell_penalty", default=1, type=int, required=False)
    parser.add_argument("--ell_min", default=0.1, type=float, required=False)
    parser.add_argument("--ell_max", default=3.0, type=float, required=False)
    parser.add_argument("--ell0_min", default=1.0, type=float, required=False)
    parser.add_argument("--ell0_max", default=10.0, type=float, required=False)
    parser.add_argument("--nu", default=0.5, type=float, required=False)
    parser.add_argument("--lam", default=1.0, type=float, required=False)
    parser.add_argument("--res_nbins", default=15, type=int, required=False)
    parser.add_argument("--res_q", default=0.10, type=float, required=False)
    parser.add_argument("--smooth_GP_resolution_loglinear", default=0, type=int, required=False)
    parser.add_argument("--fine_res", default=0.05, type=float, required=False)
    parser.add_argument("--res_highz", default=0.1, type=float, required=False)
    parser.add_argument("--res_lowz", default=0.05, type=float, required=False)
    parser.add_argument("--eps_DE", default=-1.0, type=float, required=False)
    parser.add_argument("--invert_dL_GP", default=1, type=int, required=False)
    parser.add_argument("--n_gp_nodes", default=150, type=int, required=False)
    parser.add_argument("--n_gp_fine", default=900, type=int, required=False)
    parser.add_argument("--GP_jitter", default=1e-10, type=float, required=False)
    parser.add_argument("--init_GP", default="zeros", type=str, required=False)
    parser.add_argument( "--GP_oob_guard", default=1, type=int, required=False, help="If 1, add hard GP dL interpolation out-of-bounds penalty. If 0, only record diagnostics.", )
    parser.add_argument("--alpha_eta", default=0.01, type=float, required=False)
    parser.add_argument("--alpha_ell", default=0.01, type=float, required=False)
    parser.add_argument("--alpha_large", default=0.05, type=float, required=False)
    parser.add_argument("--logXi_guard", default=0, type=int, required=False)
    parser.add_argument("--logXi_abs_max_allowed", default=5.0, type=float, required=False)
    parser.add_argument("--logXi_guard_strength", default=1.0e3, type=float, required=False)
    for action in parser._actions:
        if action.dest == "mass_model":
            action.choices = ("DPLDP", "DPLDP-z", "DPUC", "DPUC_TruncM2", "DPUC_Primary", "PLPreg")
        elif action.dest == "spin_model":
            action.choices = ("none", "default_gauss")
        elif action.dest == "spin_inj":
            action.choices = ("default", "default_gauss", "chi12xyz", "none")
        elif action.dest == "rate_model":
            action.choices = ("MD",)
    return parser
