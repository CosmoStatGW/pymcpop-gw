# jax_models.py

from __future__ import annotations

import json
import os
import logging

import numpy as np
import jax

jax.config.update("jax_debug_nans", True)

import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import transforms


import cosmology as cosmo
import numerical_utils as nutils
from backends import NPBackend, JAXBackend
from likelihood import LikDataGauss, MarginalSampleData, encode_dLprior_list, make_loglik_gauss, make_loglik_marginal_samples, make_loglik_gauss_components, make_loglik_marginal_samples_components
from population import _make_pop_and_sel_core
from constants import PlanckFiducials, PLANCK15_H0, PLANCK15_OM, z_nodes_jax
from numerical_utils import m1m2_from_Mcq, logit, find_mass_redshift_bounds_samples_np, find_mass_redshift_bounds_np, estimate_L_small_from_transformed_event_resolution_np, log_sigmoid
from mass_models import _logit_scalar



def _log_setup(*values, sep=" ", end="\n"):
    logger = logging.getLogger(__name__)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("%s", sep.join(str(value) for value in values))

def make_z_grids_GP_jax(
    zmin=1e-6,
    z0=1e-2,
    zmax=80.0,
    n_nodes=160,
    n_fine=900,
    n_ramp_nodes=12,
    n_ramp_fine=20,
):
    """JAX/NumPy version of the PyMC GP z-node/fine grids."""
    x_min = np.log1p(zmin)
    x_max = np.log1p(zmax)
    z_nodes = np.expm1(np.linspace(x_min, x_max, int(n_nodes)))
    z_fine = np.expm1(np.linspace(x_min, x_max, int(n_fine)))

    def ramp(zmin_, z_target, n):
        if int(n) <= 0 or z_target <= zmin_:
            return np.array([zmin_], dtype=np.float64)
        return np.geomspace(zmin_, z_target, int(n) + 1)[:-1]

    z_nodes = np.concatenate([ramp(zmin, z0, n_ramp_nodes), z_nodes])
    z_fine = np.concatenate([ramp(zmin, z0, n_ramp_fine), z_fine])
    z_nodes = np.unique(np.concatenate([z_nodes, np.array([z0], dtype=np.float64)]))
    z_fine = np.unique(np.concatenate([z_fine, np.array([z0], dtype=np.float64)]))
    return jnp.asarray(z_nodes, dtype=jnp.float64), jnp.asarray(z_fine, dtype=jnp.float64)


def update_gp_redshift_bounds_from_resolution(
    *,
    zmin_a,
    zmin_b,
    zmid_b,
    zmax_c,
    res,
    safety=1.1,
):
    """
    Match PyMC GP redshift-bound update.

    Uses event 68% redshift range and injection redshift range from
    find_event_redshift_resolution / find_event_redshift_resolution_from_dL_samples.
    """
    z_min_data = float(res["z_min_data"])
    z_max_data = float(res["z_max_data"])

    min_z = float(res.get("z_min_inj", z_min_data))
    max_z = float(res.get("z_max_inj", z_max_data))

    zmin_b_new = min(float(zmin_b), min_z, z_min_data)
    zmin_a_new = min(float(zmin_a), min_z, z_min_data)
    zmid_b_new = min(float(zmid_b), z_max_data)
    zmax_c_new = max(float(zmax_c), z_max_data, max_z) * float(safety)

    return zmin_a_new, zmin_b_new, zmid_b_new, zmax_c_new

def smooth_resolution_curve_loglinear_np(x_centers, r_centers):
    """
    Fit log r(x) = a + b x with b > 0.

    Since x = log(1 + z), this is equivalent to

        r(z) = exp(a) * (1 + z)^b.

    The returned x array is unchanged. Only r is replaced by the fitted curve.
    """
    x = np.asarray(x_centers, dtype=np.float64)
    r = np.asarray(r_centers, dtype=np.float64)

    if x.ndim != 1 or r.ndim != 1:
        raise ValueError("x_centers and r_centers must be 1D.")
    if x.shape != r.shape:
        raise ValueError("x_centers and r_centers must have the same shape.")
    if x.size < 2:
        raise ValueError("Need at least two points for log-linear resolution fit.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(r)):
        raise ValueError("Resolution centers contain non-finite values.")
    if np.any(r <= 0.0):
        raise ValueError("Resolution values must be positive for log-linear fit.")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("x_centers must be strictly increasing for log-linear fit.")

    b, a = np.polyfit(x, np.log(r), deg=1)

    if b <= 0.0:
        raise ValueError(
            "Log-linear resolution fit has non-positive slope. "
            f"Got b={b:.6e}, but the requested model assumes increasing resolution."
        )

    r_fit = np.exp(a + b * x)

    return x.copy(), r_fit, float(a), float(b)


def plot_resolution_curve_numpyro_with_loglinear_fit(
    x_res_centers_raw,
    r_res_centers_raw,
    x_res_centers_fit,
    r_res_centers_fit,
    *,
    x_evt,
    w_evt,
    fout,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    if x_evt is not None and w_evt is not None and len(x_evt) > 0:
        ax.scatter(
            np.expm1(x_evt),
            w_evt,
            s=20,
            alpha=0.35,
            label="Event widths",
        )

    ax.plot(
        np.expm1(x_res_centers_raw),
        r_res_centers_raw,
        "o-",
        label="Binned resolution points",
    )

    ax.plot(
        np.expm1(x_res_centers_fit),
        r_res_centers_fit,
        "-",
        label="Log-linear fit used by GP",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"Resolution in $x=\log(1+z)$")
    ax.set_title("Data-driven redshift resolution curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    os.makedirs(fout, exist_ok=True)
    fig.savefig(os.path.join(fout, "gp_resolution_curve.png"), dpi=200)
    plt.close(fig)


def make_z_grid_jax(zres=200):
    nmid = max(2, int(zres / 10))
    ntails = max(2, int(zres / 20))

    z = np.sort(np.unique(np.concatenate([
        np.logspace(-100, -15, ntails),
        np.logspace(-30,  -4, nmid),
        np.logspace( -4,   1, zres),
        np.logspace(  1,   2, nmid),
        np.logspace(  2,   5, ntails),
    ])))

    return jnp.asarray(z, dtype=jnp.float64)
    



def _estimate_L_small_from_physical_m1m2_event_resolution_np(
    logMc_by_event,
    logitq_by_event,
    *,
    percentile=25.0,
    factor=0.5,
    min_samples=10,
    shared_mode="max",
    shared=False,
    primary_only=False,
    secondary_coord="logm2",
):
    """
    PE-resolution floors for DPUC physical-coordinate models in log source masses.

    The DPUC physical-coordinate GMM is defined in x=log(m), not in m.  Therefore the
    width floor returned here is an event-wise PE resolution in log(m1) and
    log(m2).  The physical density later includes the Jacobian 1/m.

    Inputs are the same event-major transformed PE samples already returned by
    the existing bound finder: logMc and logit(q), shape (N_events, N_samples).

    Returns L1, L2, summary in log-mass units.
    """
    logMc_by_event = np.asarray(logMc_by_event, dtype=np.float64)
    logitq_by_event = np.asarray(logitq_by_event, dtype=np.float64)

    if logMc_by_event.shape != logitq_by_event.shape:
        raise ValueError(
            "logMc_by_event and logitq_by_event must have the same shape, "
            f"got {logMc_by_event.shape} and {logitq_by_event.shape}"
        )
    if logMc_by_event.ndim != 2:
        raise ValueError("logMc_by_event/logitq_by_event must be 2D event-major arrays")

    shared_mode = str(shared_mode)
    if shared_mode not in ("max", "min"):
        raise ValueError(f"DPUC_physical_Lsmall_shared_mode must be 'max' or 'min', got {shared_mode!r}")

    def _coord_values(x, name):
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"{name} cannot be empty.")
        if arr.size == 1:
            return float(arr[0]), float(arr[0])
        return float(arr[0]), float(arr[1])

    p1, p2 = _coord_values(percentile, "L_small_res_percentile")
    f1, f2 = _coord_values(factor, "L_small_res_factor")

    secondary_coord = str(secondary_coord)
    if secondary_coord not in ("logm2", "logitq"):
        raise ValueError(f"secondary_coord must be 'logm2' or 'logitq', got {secondary_coord!r}")

    sig_m1 = []
    sig_m2 = []
    sig_logitq = []
    for i in range(logMc_by_event.shape[0]):
        u1 = logMc_by_event[i]
        u2 = logitq_by_event[i]
        ok = np.isfinite(u1) & np.isfinite(u2)
        if np.sum(ok) < min_samples:
            continue
        Mc = np.exp(u1[ok])
        q = 1.0 / (1.0 + np.exp(-u2[ok]))
        m1, m2 = m1m2_from_Mcq(Mc, q)
        ok_m = (m1 > 0.0) & (m2 > 0.0) & np.isfinite(m1) & np.isfinite(m2)
        if np.sum(ok_m) < min_samples:
            continue
        s1 = np.std(np.log(m1[ok_m]))
        s2 = np.std(np.log(m2[ok_m]))
        if np.isfinite(s1) and s1 > 0.0:
            sig_m1.append(float(s1))
        if np.isfinite(s2) and s2 > 0.0:
            sig_m2.append(float(s2))
        s_q = np.std(u2[ok][ok_m])
        if np.isfinite(s_q) and s_q > 0.0:
            sig_logitq.append(float(s_q))

    if len(sig_m1) == 0 or len(sig_m2) == 0 or (secondary_coord == "logitq" and len(sig_logitq) == 0):
        raise ValueError(
            "Could not estimate DPUC physical PE-resolution floors: "
            "no valid event-wise m1/m2/logitq samples."
        )

    sig_m1 = np.asarray(sig_m1, dtype=np.float64)
    sig_m2 = np.asarray(sig_m2, dtype=np.float64)
    sig_logitq = np.asarray(sig_logitq, dtype=np.float64)

    Lm1 = float(f1 * np.percentile(sig_m1, p1))
    Lm2_raw = sig_logitq if secondary_coord == "logitq" else sig_m2
    Lm2 = float(f2 * np.percentile(Lm2_raw, p2))

    if primary_only:
        L1 = L2 = float(Lm1)
    elif shared:
        L = max(Lm1, Lm2) if shared_mode == "max" else min(Lm1, Lm2)
        L1 = L2 = float(L)
    else:
        L1, L2 = Lm1, Lm2

    summary = {
        "DPUC_physical_resolution_coordinate": "log_physical_m1_logitq" if secondary_coord == "logitq" else "log_physical_m1_m2",
        "DPUC_physical_Lsmall_shared": bool(shared),
        "DPUC_physical_Lsmall_primary_only": bool(primary_only),
        "DPUC_physical_Lsmall_shared_mode": shared_mode,
        "resolution_percentile_m1": p1,
        "resolution_percentile_m2": p2,
        "resolution_factor_m1": f1,
        "resolution_factor_m2": f2,
        "n_events_used_m1": int(len(sig_m1)),
        "n_events_used_m2": int(len(sig_m2)),
        "logm1_res_p10": float(np.percentile(sig_m1, 10)),
        "logm1_res_p25": float(np.percentile(sig_m1, 25)),
        "logm1_res_p50": float(np.percentile(sig_m1, 50)),
        "logm1_res_p75": float(np.percentile(sig_m1, 75)),
        "logm2_res_p10": float(np.percentile(sig_m2, 10)),
        "logm2_res_p25": float(np.percentile(sig_m2, 25)),
        "logm2_res_p50": float(np.percentile(sig_m2, 50)),
        "logm2_res_p75": float(np.percentile(sig_m2, 75)),
        "logitq_res_p10": float(np.percentile(sig_logitq, 10)) if sig_logitq.size else None,
        "logitq_res_p25": float(np.percentile(sig_logitq, 25)) if sig_logitq.size else None,
        "logitq_res_p50": float(np.percentile(sig_logitq, 50)) if sig_logitq.size else None,
        "logitq_res_p75": float(np.percentile(sig_logitq, 75)) if sig_logitq.size else None,
        "L_small_logm1_PE": Lm1,
        "L_small_logm2_PE": Lm2,
        "L_small_1_PE_used": L1,
        "L_small_2_PE_used": L2,
    }
    return L1, L2, summary

def _stack_spins_inj(spinsInj, ninj: int, spin_model: str) -> np.ndarray:
    """
    Returns spins_inj as numpy array shape (ninj, nspin) with float64 dtype.
    spinsInj formats allowed:
      - [] or None (when spin_model='none') -> (ninj,0)
      - list/tuple of 2 arrays each (ninj,) -> (ninj,2)
      - list/tuple of 4 arrays each (ninj,) -> (ninj,4)
      - ndarray shape (ninj,2) or (ninj,4) -> passthrough
      - ndarray shape (2,ninj) or (4,ninj) -> transpose to (ninj,2/4)
    """
    if spin_model == "none":
        return np.zeros((ninj, 0), dtype=np.float64)

    if spinsInj is None:
        raise ValueError("spinsInj is None but spin_model != 'none'")

    # list/tuple of components
    if isinstance(spinsInj, (list, tuple)):
        if len(spinsInj) == 0:
            # be strict: empty spins with non-none model is inconsistent
            raise ValueError("spinsInj is empty but spin_model != 'none'")
        comps = [ np.squeeze(np.asarray(s, dtype=np.float64)) for s in spinsInj]
        k = len(comps)
        if k not in (2, 4):
            raise ValueError(f"spinsInj list must have length 2 or 4, got {k}")
        for i, c in enumerate(comps):
            if c.shape != (ninj,):
                raise ValueError(f"spinsInj[{i}] must have shape ({ninj},), got {c.shape}")
        return np.stack(comps, axis=1)  # (ninj,k)

    # ndarray already
    arr = np.asarray(spinsInj, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"spinsInj must be 2D, got shape {arr.shape}")

    # allow (ninj,k)
    if arr.shape[0] == ninj and arr.shape[1] in (0, 2, 4):
        return arr
    # allow (k,ninj)
    if arr.shape[1] == ninj and arr.shape[0] in (0, 2, 4):
        return arr.T

    raise ValueError(f"Unrecognized spinsInj shape {arr.shape} for ninj={ninj}")


def pack_data_gauss_popnot(
    *,
    GWData,
    InjData,
    dLprior,                # list[str], length = nchunks
    Nevs_np,                # (nchunks,) numpy
    all_PE_log_norms,       # (N,) numpy, per-event normalization already computed
    # optional bilby prior grid (both numpy, in Gpc)
    dLgrid_bilby_gpc=None,
    PE_prior_bilby_grid=None,
    # meta/config
    spin_model="none",
    rate_model="MD",
    mass_model="PLP",
    smoothing="LVK",
    simplex_repair=False,
    has_m2_break=False,
    norm_gauss="uplow",
    param="vanilla",
    integrate_dc="trapz",
    subtract_log_p_incl=False,
    sample_from_pop=False,
    marginal_R0=True,
    allTobs = None,
    r = 0,
    taper_kind = "sigmoid", 
     taper_p = 12.0
):
    """
    Build LikDataGauss for pop_only=False, sampling_GW='gauss'.

    Assumptions:
      - ndata_np == 1 (single injection set), consistent with your current port.
      - All distances are in Gpc (events + injections + bilby grids).
    """

    # ---- Unpack GW surrogate data (gauss branch) ----
    # Legacy GWData:
    #   (mus_s, cho_s, log_wts_l, mus_l, icovs_l, log_dets_l, cho_covs_l,
    #    Tobs_np, Nevs, allnames)
    # Bounded-transform GWData:
    #   (mus_s, cho_s, log_wts_l, mus_l, icovs_l, log_dets_l, cho_covs_l,
    #    gmm_fit_coord_bounds, gmm_fit_coord_names, gmm_fit_transform_info,
    #    Tobs_np, Nevs, allnames)
    if len(GWData) == 10:
        (
            mus_s,
            cho_s,
            log_wts_l,
            mus_l,
            icovs_l,
            log_dets_l,
            cho_covs_l,
            _Tobs_np,
            _Nevs,
            _allnames,
        ) = GWData

        gmm_fit_coord_bounds = None
        gmm_fit_coord_names = None
        gmm_fit_transform_mode = "legacy"

    elif len(GWData) == 13:
        (
            mus_s,
            cho_s,
            log_wts_l,
            mus_l,
            icovs_l,
            log_dets_l,
            cho_covs_l,
            gmm_fit_coord_bounds,
            gmm_fit_coord_names,
            gmm_fit_transform_info,
            _Tobs_np,
            _Nevs,
            _allnames,
        ) = GWData

        if not isinstance(gmm_fit_transform_info, dict):
            raise ValueError("gmm_fit_transform_info must be a dict for bounded-transform GWData.")

        gmm_fit_transform_mode = gmm_fit_transform_info.get("transform_mode")
        if gmm_fit_transform_mode != "event_bounded_flogit":
            raise NotImplementedError(
                "Only transform_mode='event_bounded_flogit' is implemented for bounded GMMs. "
                f"Got {gmm_fit_transform_mode!r}."
            )

        gmm_fit_coord_names = tuple(str(x) for x in gmm_fit_coord_names)
        gmm_fit_coord_bounds = np.asarray(gmm_fit_coord_bounds, dtype=np.float64)

    else:
        raise ValueError(
            "GWData must have length 10 for legacy Gaussian surrogates or length 13 "
            f"for event-bounded Gaussian/GMM surrogates. Got length {len(GWData)}."
        )

    mus_s = np.asarray(mus_s, dtype=np.float64)
    cho_s = np.asarray(cho_s, dtype=np.float64)
    mus_l = np.asarray(mus_l, dtype=np.float64)
    icovs_l = np.asarray(icovs_l, dtype=np.float64)
    log_dets_l = np.asarray(log_dets_l, dtype=np.float64)
    log_wts_l = np.asarray(log_wts_l, dtype=np.float64)
    cho_covs_l = np.asarray(cho_covs_l, dtype=np.float64)

    N_evt_full = mus_s.shape[0]
    nd_full = mus_s.shape[1]

    if mus_s.ndim != 2:
        raise ValueError(f"mus_s must have shape (N, nd), got {mus_s.shape}")
    if cho_s.shape != (N_evt_full, nd_full, nd_full):
        raise ValueError(
            f"cho_s must have shape ({N_evt_full}, {nd_full}, {nd_full}), got {cho_s.shape}"
        )
    if mus_l.ndim != 3 or mus_l.shape[0] != N_evt_full or mus_l.shape[2] != nd_full:
        raise ValueError(
            "mus_l must have shape (N, K, nd) matching mus_s; "
            f"got mus_l={mus_l.shape}, mus_s={mus_s.shape}"
        )
    if icovs_l.shape[:3] != mus_l.shape or icovs_l.shape[3] != nd_full:
        raise ValueError(
            "icovs_l must have shape (N, K, nd, nd) matching mus_l; "
            f"got icovs_l={icovs_l.shape}, mus_l={mus_l.shape}"
        )
    if cho_covs_l.shape[:3] != mus_l.shape or cho_covs_l.shape[3] != nd_full:
        raise ValueError(
            "cho_covs_l must have shape (N, K, nd, nd) matching mus_l; "
            f"got cho_covs_l={cho_covs_l.shape}, mus_l={mus_l.shape}"
        )
    if log_dets_l.shape != mus_l.shape[:2]:
        raise ValueError(
            f"log_dets_l must have shape {mus_l.shape[:2]}, got {log_dets_l.shape}"
        )
    if log_wts_l.shape != mus_l.shape[:2]:
        raise ValueError(
            f"log_wts_l must have shape {mus_l.shape[:2]}, got {log_wts_l.shape}"
        )

    if gmm_fit_transform_mode == "event_bounded_flogit":
        if gmm_fit_coord_bounds.shape != (N_evt_full, nd_full, 2):
            raise ValueError(
                "gmm_fit_coord_bounds must have shape (N, nd, 2) matching mus_s; "
                f"got bounds={gmm_fit_coord_bounds.shape}, mus_s={mus_s.shape}"
            )
        if len(gmm_fit_coord_names) != nd_full:
            raise ValueError(
                "gmm_fit_coord_names length must match mus_s coordinate dimension; "
                f"got {len(gmm_fit_coord_names)} and {nd_full}"
            )
        if not np.isfinite(gmm_fit_coord_bounds).all():
            raise ValueError("gmm_fit_coord_bounds contains non-finite values.")
        if not np.all(gmm_fit_coord_bounds[:, :, 1] > gmm_fit_coord_bounds[:, :, 0]):
            raise ValueError("Each gmm_fit_coord_bounds upper bound must be greater than its lower bound.")

    if spin_model == "none":
        d_int = 3
    
        mus_s = mus_s[:, :d_int]
        cho_s = cho_s[:, :d_int, :d_int]
    
        mus_l = mus_l[:, :, :d_int]
    
        # Rebuild marginal 3D GMM precisions and log determinants from
        # the 3D covariance blocks. Do not slice the full 7D precision.
        cho_covs_l_3 = cho_covs_l[:, :, :d_int, :d_int]
        covs_l_3 = np.einsum("...ik,...jk->...ij", cho_covs_l_3, cho_covs_l_3)
    
        # Symmetrize for numerical safety.
        covs_l_3 = 0.5 * (covs_l_3 + np.swapaxes(covs_l_3, -1, -2))
    
        N_evt, K_evt = covs_l_3.shape[:2]
        icovs_l_3 = np.empty_like(covs_l_3)
        log_dets_l_3 = np.empty((N_evt, K_evt), dtype=np.float64)
    
        eye3 = np.eye(d_int, dtype=np.float64)
    
        for i in range(N_evt):
            for k in range(K_evt):
                C = covs_l_3[i, k]
                jitter = 1e-12 * max(np.trace(C) / d_int, 1.0)
    
                # Robustify if needed.
                for ntry in range(8):
                    try:
                        Cj = C + jitter * (10.0 ** ntry) * eye3
                        L = np.linalg.cholesky(Cj)
                        break
                    except np.linalg.LinAlgError:
                        if ntry == 7:
                            evals, evecs = np.linalg.eigh(C)
                            evals = np.clip(evals, jitter, None)
                            Cj = (evecs * evals) @ evecs.T
                            L = np.linalg.cholesky(Cj + jitter * eye3)
    
                icovs_l_3[i, k] = np.linalg.inv(Cj)
                log_dets_l_3[i, k] = 2.0 * np.sum(np.log(np.diag(L)))
    
        icovs_l = icovs_l_3
        log_dets_l = log_dets_l_3
        cho_covs_l = cho_covs_l_3

        if gmm_fit_transform_mode == "event_bounded_flogit":
            gmm_fit_coord_bounds = gmm_fit_coord_bounds[:, :d_int, :]
            gmm_fit_coord_names = gmm_fit_coord_names[:d_int]
    
    else:
        # Spinful branch uses all coordinates as before.
        pass

    # ---- Unpack injections (ndata_np==1) ----
    # InjData = (dLinj, m1inj, m2inj, spinsInj, lpdinj, Ndraw, Ndet_np, lp_incl_inj)
    dLinj, m1inj, m2inj, spinsInj, lpdinj, Ndraw, _Ndet_np, lp_incl_inj = InjData

    dLinj0 = np.asarray(dLinj[0], dtype=np.float64)
    m1inj0 = np.asarray(m1inj[0], dtype=np.float64)
    m2inj0 = np.asarray(m2inj[0], dtype=np.float64)
    lpdinj0 = np.asarray(lpdinj[0], dtype=np.float64)
    lp_incl0 = np.asarray(lp_incl_inj[0], dtype=np.float64)

    ninj = int(m1inj0.shape[0])
    if dLinj0.shape != (ninj,) or m2inj0.shape != (ninj,) or lpdinj0.shape != (ninj,) or lp_incl0.shape != (ninj,):
        _log_setup("\nInjections shape: ninj, dLinj0.shape, m2inj0.shape, lpdinj0.shape, lp_incl0.shape")
        _log_setup(ninj, dLinj0.shape, m2inj0.shape, lpdinj0.shape, lp_incl0.shape )
        raise ValueError("Injection arrays must all be shape (ninj,) for ndata_np==1.")

    spins_inj = _stack_spins_inj(spinsInj, ninj=ninj, spin_model=spin_model)  # (ninj,nspin)

    Ndraw = float(np.asarray(Ndraw).reshape(()))

    # ---- Event/chunk bookkeeping ----
    Nevs_np = np.asarray(Nevs_np, dtype=np.int32)
    nchunks = int(Nevs_np.shape[0])
    N = int(np.sum(Nevs_np))

    labels_evt = np.repeat(np.arange(nchunks, dtype=np.int32), Nevs_np)  # (N,)
    prior_code_per_chunk = encode_dLprior_list(dLprior)                  # (nchunks,)

    all_PE_log_norms = np.asarray(all_PE_log_norms, dtype=np.float64)
    if all_PE_log_norms.shape != (N,):
        raise ValueError(f"all_PE_log_norms must have shape ({N},), got {all_PE_log_norms.shape}")

    if prior_code_per_chunk.shape != (nchunks,):
        raise ValueError(f"dLprior length mismatch: expected {nchunks}, got {prior_code_per_chunk.shape[0]}")

    # ---- Planck15 grids (Gpc) ----
    bk = NPBackend()
    zgrid_dLp = np.logspace(np.log10(1e-5), np.log10(100.0), 1200).astype(np.float64)
    dc_grid = cosmo.dcfun_quad(bk, zgrid_dLp, PLANCK15_H0, PLANCK15_OM, -1.0)
    dL_grid = cosmo.dLfun(
        bk,
        zgrid_dLp,
        PLANCK15_H0,
        PLANCK15_OM,
        -1.0,
        1.0,
        0.0,
        dc=dc_grid,
        Xi=None,
        param="vanilla",
    )

    allTobs=None if allTobs is None else jnp.asarray(allTobs, dtype=jnp.float64)


    # ---- Build LikDataGauss (device arrays) ----

    if jnp.log(r)==-1:
        raise ValueError()
    
    logr = jnp.log(r) if r!=0 else -1
    
    data = LikDataGauss(
        # GW surrogate
        mus_s=jnp.asarray(mus_s, dtype=jnp.float64),
        cho_s=jnp.asarray(cho_s, dtype=jnp.float64),
        mus_l=jnp.asarray(mus_l, dtype=jnp.float64),
        icovs_l=jnp.asarray(icovs_l, dtype=jnp.float64),
        log_dets_l=jnp.asarray(log_dets_l, dtype=jnp.float64),
        log_wts_l=jnp.asarray(log_wts_l, dtype=jnp.float64),

        gmm_fit_coord_bounds=(
            None
            if gmm_fit_coord_bounds is None
            else jnp.asarray(gmm_fit_coord_bounds, dtype=jnp.float64)
        ),
        gmm_fit_transform_mode=str(gmm_fit_transform_mode),
        gmm_fit_coord_names=gmm_fit_coord_names,

        # injections / selection
        m1inj=jnp.asarray(m1inj0, dtype=jnp.float64),
        m2inj=jnp.asarray(m2inj0, dtype=jnp.float64),
        dLinj=jnp.asarray(dLinj0, dtype=jnp.float64),
        spins_inj=jnp.asarray(spins_inj, dtype=jnp.float64),
        log_p_draw=jnp.asarray(lpdinj0, dtype=jnp.float64),
        log_p_incl=jnp.asarray(lp_incl0, dtype=jnp.float64),
        Ndraw=jnp.asarray(Ndraw, dtype=jnp.float64),

        # PE-prior bookkeeping
        labels_evt=jnp.asarray(labels_evt, dtype=jnp.int32),
        prior_code_per_chunk=jnp.asarray(prior_code_per_chunk, dtype=jnp.int32),
        all_PE_log_norms=jnp.asarray(all_PE_log_norms, dtype=jnp.float64),

        # Planck15 grids (Gpc)
        zgrid_dLp=jnp.asarray(zgrid_dLp, dtype=jnp.float64),
        dL_grid_Planck15=jnp.asarray(dL_grid, dtype=jnp.float64),
        dc_grid_Planck15=jnp.asarray(dc_grid, dtype=jnp.float64),

        # bilby grids (optional)
        dLgrid_bilby_gpc=None if dLgrid_bilby_gpc is None else jnp.asarray(dLgrid_bilby_gpc, dtype=jnp.float64),
        PE_prior_bilby_grid=None if PE_prior_bilby_grid is None else jnp.asarray(PE_prior_bilby_grid, dtype=jnp.float64),

        # optional poisson term inputs (wired later)
        Nevs_per_chunk=jnp.asarray(Nevs_np, dtype=jnp.int32),
        allTobs= None if allTobs is None else jnp.asarray(allTobs, dtype=jnp.float64),
        
        Nobs = jnp.asarray(N, dtype=jnp.float64),
        logNobs = jnp.log(N),

        # meta
        spin_model=str(spin_model),
        rate_model=str(rate_model),
        mass_model=str(mass_model),
        smoothing=str(smoothing),
        simplex_repair=bool(simplex_repair),
        has_m2_break=bool(has_m2_break),
        norm_gauss=str(norm_gauss),
        param=str(param),
        integrate_dc=str(integrate_dc),
        subtract_log_p_incl=bool(subtract_log_p_incl),
        sample_from_pop=bool(sample_from_pop),
        marginal_R0=bool(marginal_R0),
        logr = logr,

        taper_kind = taper_kind,
        taper_p = taper_p
    )
    return data

    

def _stack_spins_pe_flat(spins, mask_flat, spin_model: str, n_tot: int) -> np.ndarray:
    if spin_model == "none":
        return np.zeros((int(mask_flat.sum()), 0), dtype=np.float64)

    arr = np.asarray(spins, dtype=np.float64)

    # Accept:
    #   (4,N,S)
    #   (N,4,S)
    #   (N,S,4)
    if arr.ndim == 3 and arr.shape[0] in (2, 4):
        arr = np.moveaxis(arr, 0, -1)   # -> (N,S,nspin)
    elif arr.ndim == 3 and arr.shape[1] in (2, 4):
        arr = np.moveaxis(arr, 1, -1)   # -> (N,S,nspin)
    elif arr.ndim == 3 and arr.shape[-1] in (2, 4):
        pass
    else:
        raise ValueError(f"Unexpected posterior spin sample shape: {arr.shape}")

    arr_flat = arr.reshape((n_tot, arr.shape[-1]))
    return arr_flat[mask_flat]



def pack_data_samples_marginal(
    *,
    GWData,
    InjData,
    spin_model="none",
    rate_model="MD",
    mass_model="DPLDP",
    smoothing="LVK",
    simplex_repair=False,
    has_m2_break=False,
    norm_gauss="uplow",
    param="vanilla",
    integrate_dc="trapz",
    subtract_log_p_incl=False,
    marginal_R0=True,
    allTobs=None,
    chunk_pe=0,
    r=0,
    taper_kind = "sigmoid", 
    taper_p = 12.
):
    
    """Pack flattened posterior samples for the marginalized NumPyro likelihood."""
    # GWData = [m1d_samples, m2d_samples, dL_samples, spin_samples,
    #           dL_prior, Tobs, allNsamples, where_compute, Nevents, allnames]
    m1det, m2det, dL, spins, dL_prior, Tobs_np, allNsamples, where_compute, Nevs, allnames = GWData

    m1det = np.asarray(m1det, dtype=np.float64)
    m2det = np.asarray(m2det, dtype=np.float64)
    dL = np.asarray(dL, dtype=np.float64)
    dL_prior = np.asarray(dL_prior, dtype=np.float64)
    where_compute = np.asarray(where_compute, dtype=bool)

    if m1det.shape != m2det.shape or m1det.shape != dL.shape or m1det.shape != dL_prior.shape:
        raise ValueError("posterior sample arrays m1det/m2det/dL/dL_prior must have identical shape")
    if where_compute.shape != m1det.shape:
        raise ValueError("where_compute must have same shape as posterior sample arrays")

    N, S = m1det.shape
    n_tot = N * S

    valid = (
        where_compute
        & np.isfinite(m1det)
        & np.isfinite(m2det)
        & np.isfinite(dL)
        & np.isfinite(dL_prior)
        & (m1det > 0.0)
        & (m2det > 0.0)
        & (dL > 0.0)
    )

    if spin_model != "none":
        spins_arr = np.asarray(spins, dtype=np.float64)

        if spins_arr.ndim == 3 and spins_arr.shape[0] in (2, 4):
            spins_arr_evt = np.moveaxis(spins_arr, 0, -1)
        elif spins_arr.ndim == 3 and spins_arr.shape[1] in (2, 4):
            spins_arr_evt = np.moveaxis(spins_arr, 1, -1)
        elif spins_arr.ndim == 3 and spins_arr.shape[-1] in (2, 4):
            spins_arr_evt = spins_arr
        else:
            raise ValueError(f"Unexpected posterior spin sample shape: {spins_arr.shape}")

        valid &= np.all(np.isfinite(spins_arr_evt), axis=-1)

    mask_flat = valid.reshape(n_tot)

    m1_flat = m1det.reshape(n_tot)[mask_flat]
    m2_flat = m2det.reshape(n_tot)[mask_flat]
    dL_flat = dL.reshape(n_tot)[mask_flat]
    prior_flat_raw = dL_prior.reshape(n_tot)[mask_flat]
    if np.all(prior_flat_raw > 0.0):
        log_pe_flat = np.log(np.clip(prior_flat_raw, 1e-300, np.inf))
    else:
        log_pe_flat = prior_flat_raw

    event_id_flat = np.repeat(np.arange(N, dtype=np.int32), S)[mask_flat]

    Nsamples_evt = np.bincount(event_id_flat, minlength=N).astype(np.float64)
    if Nsamples_evt.shape != (N,):
        raise ValueError(f"Nsamples_evt must have shape ({N},), got {Nsamples_evt.shape}")
    is_single_PE = bool(np.all(Nsamples_evt == 1))
    if not is_single_PE and not np.all(Nsamples_evt > 1):
        raise ValueError("Need at least two finite posterior samples per event, or exactly one for every event")

    spins_flat = _stack_spins_pe_flat(spins, mask_flat, spin_model, n_tot)

    # Injections: same convention as pack_data_gauss_popnot
    dLinj, m1inj, m2inj, spinsInj, lpdinj, Ndraw, _Ndet_np, lp_incl_inj = InjData
    dLinj0 = np.asarray(dLinj[0], dtype=np.float64)
    m1inj0 = np.asarray(m1inj[0], dtype=np.float64)
    m2inj0 = np.asarray(m2inj[0], dtype=np.float64)
    lpdinj0 = np.asarray(lpdinj[0], dtype=np.float64)
    lp_incl0 = np.asarray(lp_incl_inj[0], dtype=np.float64)
    ninj = int(m1inj0.shape[0])
    spins_inj = _stack_spins_inj(spinsInj, ninj=ninj, spin_model=spin_model)
    Ndraw = float(np.asarray(Ndraw).reshape(()))

    Nevs_np = np.atleast_1d(Nevs).astype(np.int32)

    if jnp.log(r) == -1:
        raise ValueError()
    logr = jnp.log(r) if r != 0 else -1

    _log_setup(f"[PE marginal] M={m1_flat.shape[0]}, chunk_pe={int(chunk_pe)}")

    return MarginalSampleData(
        m1det_pe=jnp.asarray(m1_flat, dtype=jnp.float64),
        m2det_pe=jnp.asarray(m2_flat, dtype=jnp.float64),
        dL_pe=jnp.asarray(dL_flat, dtype=jnp.float64),
        spins_pe=jnp.asarray(spins_flat, dtype=jnp.float64),
        log_PE_prior_pe=jnp.asarray(log_pe_flat, dtype=jnp.float64),
        event_id_pe=jnp.asarray(event_id_flat, dtype=jnp.int32),
        Nsamples_evt=jnp.asarray(Nsamples_evt, dtype=jnp.float64),
        is_single_PE=is_single_PE,

        m1inj=jnp.asarray(m1inj0, dtype=jnp.float64),
        m2inj=jnp.asarray(m2inj0, dtype=jnp.float64),
        dLinj=jnp.asarray(dLinj0, dtype=jnp.float64),
        spins_inj=jnp.asarray(spins_inj, dtype=jnp.float64),
        log_p_draw=jnp.asarray(lpdinj0, dtype=jnp.float64),
        log_p_incl=jnp.asarray(lp_incl0, dtype=jnp.float64),
        Ndraw=jnp.asarray(Ndraw, dtype=jnp.float64),

        Nevs_per_chunk=jnp.asarray(Nevs_np, dtype=jnp.int32),
        allTobs=None if allTobs is None else jnp.asarray(allTobs, dtype=jnp.float64),
        Nobs=int(N),
        logNobs=jnp.log(jnp.asarray(N, dtype=jnp.float64)),
        logr=logr,

        spin_model=str(spin_model),
        rate_model=str(rate_model),
        mass_model=str(mass_model),
        smoothing=str(smoothing),
        simplex_repair=bool(simplex_repair),
        has_m2_break=bool(has_m2_break),
        norm_gauss=str(norm_gauss),
        param=str(param),
        integrate_dc=str(integrate_dc),
        subtract_log_p_incl=bool(subtract_log_p_incl),
        marginal_R0=bool(marginal_R0),
        chunk_pe=int(chunk_pe),
    
        taper_kind = taper_kind,
        taper_p = taper_p
    )




def build_core_and_loglik_gauss_popnot(
    data,
    *,
    chunk_inj=0,
    K_dp=30,
    DP_truncate=False,
    DP_m1_env=False,
    DP_m1_env_form="powerlaw",
    DP_m1_env_delta1=0.2,
    DP_m1_env_delta2=0.2,
    DP_q_env=False,
    DPUC_Primary_q_norm="conditional",
    PL_DPUC_normalize_gmm = False,
    DPUC_kernel="gaussian",
    DPUC_kernel_rcut=4.0,
    DPUC_kernel_dr=0.15,
    DPUC_kernel_xcut=3.0,
    DPUC_kernel_ycut=3.0,
    DPUC_q_coord="logitq",
    DPUC_TruncM2_shared_weights=False,
    DPUC_TruncM2_secondary_coord="logm2",
    DP_corr_m1m2 = False,
    DP_corr_eta_rho = 2.0,
    DP_corr_f_minor = 0.25,
    interp_mass=0,
    stop_grad_var_u=True,
    # selection handling
    skip_sel=False,              # keep False for your current pop_only=False workflow
    verbose=False,
    z_nodes = None,
):
    """
    Build the population/selection core and the final log-likelihood callable:
        core_fn(m1det,m2det,dLdet,spins_evt, m1inj,m2inj,dLinj,spins_inj, log_p_draw,log_p_incl, Lambda, Ndraw)
          -> (logp_pop_evt (N,), log_mu (scalar), aux)
        loglik(Lambda, x) -> scalar

    Assumes `data` is LikDataGauss returned by pack_data_gauss_popnot().
    """

    bk = JAXBackend()

    core = _make_pop_and_sel_core(
        bk=bk,
        rate_model=data.rate_model,
        mass_model=data.mass_model,
        spin_model=data.spin_model,
        smoothing=data.smoothing,
        simplex_repair=data.simplex_repair,
        has_m2_break=data.has_m2_break,
        norm_gauss=data.norm_gauss,
        taper_kind = data.taper_kind,   
        taper_p = data.taper_p,           
        
        param=data.param,
        verbose=bool(verbose),
        z_nodes = z_nodes,

        # matches your likelihood core signature expectations
        subtract_log_p_incl=bool(data.subtract_log_p_incl),
        skip_sel=bool(skip_sel),

        # selection / injections controls
        chunk_inj=int(chunk_inj),
        K_dp=int(K_dp),
        
        DP_truncate=bool(DP_truncate),
        DP_m1_env=bool(DP_m1_env),
        DP_m1_env_form=DP_m1_env_form,
        DP_m1_env_delta1=DP_m1_env_delta1,
        DP_m1_env_delta2=DP_m1_env_delta2,
        DP_q_env=bool(DP_q_env),
        DPUC_Primary_q_norm=DPUC_Primary_q_norm,
        PL_DPUC_normalize_gmm = bool(PL_DPUC_normalize_gmm),
        DPUC_kernel = DPUC_kernel,
        DPUC_kernel_rcut = DPUC_kernel_rcut,
        DPUC_kernel_dr = DPUC_kernel_dr,
        DPUC_kernel_xcut = DPUC_kernel_xcut,
        DPUC_kernel_ycut = DPUC_kernel_ycut,
        DPUC_q_coord = DPUC_q_coord,
        DPUC_TruncM2_shared_weights=bool(DPUC_TruncM2_shared_weights),
            DPUC_TruncM2_secondary_coord=DPUC_TruncM2_secondary_coord,
        DP_corr_m1m2 = bool(DP_corr_m1m2),
        DP_corr_eta_rho = DP_corr_eta_rho,
        DP_corr_f_minor = DP_corr_f_minor,
        interp_mass=int(interp_mass),
        integrate_dc=data.integrate_dc,

        # we are in pop_only=False branch
        pop_only=False,

        # you are not using var_u in the gauss likelihood path right now
        stop_grad_var_u=bool(stop_grad_var_u),
        return_var=True,
    )

    loglik = make_loglik_gauss(core, data)
    loglik_components = make_loglik_gauss_components(core, data, include_eventwise=True)
    # optional: force compilation once (useful for debugging shapes)
    # _ = loglik(jnp.zeros((1,)), jnp.zeros((data.Nobs, data.mus_s.shape[1])))

    return core, loglik, loglik_components



def build_core_and_loglik_samples_marginal(
    data,
    *,
    chunk_inj=0,
    K_dp=30,
    DP_truncate=False,
    DP_m1_env=False,
    DP_m1_env_form="powerlaw",
    DP_m1_env_delta1=0.2,
    DP_m1_env_delta2=0.2,
    DP_q_env=False,
    DPUC_Primary_q_norm="conditional",
    DP_corr_m1m2  = False,
    DP_corr_eta_rho = 3.0,
    DP_corr_f_minor = 0.35,
    PL_DPUC_normalize_gmm=False,
    DPUC_kernel="gaussian",
    DPUC_kernel_rcut=4.0,
    DPUC_kernel_dr=0.15,
    DPUC_kernel_xcut=3.0,
    DPUC_kernel_ycut=3.0,
    DPUC_q_coord="logitq",
    DPUC_TruncM2_shared_weights=False,
    DPUC_TruncM2_secondary_coord="logm2",
    interp_mass=0,
    stop_grad_var_u=True,
    verbose=False,
    z_nodes=None,
    skip_sel = False
):
    bk = JAXBackend()

    core = _make_pop_and_sel_core(
        bk=bk,
        rate_model=data.rate_model,
        mass_model=data.mass_model,
        spin_model=data.spin_model,
        smoothing=data.smoothing,
        simplex_repair=data.simplex_repair,
        has_m2_break=data.has_m2_break,
        norm_gauss=data.norm_gauss,
        taper_kind = data.taper_kind,
        taper_p = data.taper_p,
        param=data.param,
        verbose=bool(verbose),
        z_nodes=z_nodes,
        subtract_log_p_incl=bool(data.subtract_log_p_incl),
        skip_sel=skip_sel,
        chunk_inj=int(chunk_inj),
        chunk_evt=int(data.chunk_pe),
        K_dp=int(K_dp),
        DP_truncate=bool(DP_truncate),
        DP_m1_env=bool(DP_m1_env),
        DP_m1_env_form=DP_m1_env_form,
        DP_m1_env_delta1=DP_m1_env_delta1,
        DP_m1_env_delta2=DP_m1_env_delta2,
        DP_q_env=bool(DP_q_env),
        DPUC_Primary_q_norm=DPUC_Primary_q_norm,
        PL_DPUC_normalize_gmm = bool(PL_DPUC_normalize_gmm),
        DPUC_kernel = DPUC_kernel,
        DPUC_kernel_rcut = DPUC_kernel_rcut,
        DPUC_kernel_dr = DPUC_kernel_dr,
        DPUC_kernel_xcut = DPUC_kernel_xcut,
        DPUC_kernel_ycut = DPUC_kernel_ycut,
        DPUC_q_coord = DPUC_q_coord,
        DPUC_TruncM2_shared_weights=bool(DPUC_TruncM2_shared_weights),
            DPUC_TruncM2_secondary_coord=DPUC_TruncM2_secondary_coord,
        DP_corr_m1m2 = bool(DP_corr_m1m2),
            DP_corr_eta_rho = DP_corr_eta_rho,
        DP_corr_f_minor = DP_corr_f_minor,
        interp_mass=int(interp_mass),
        integrate_dc=data.integrate_dc,
        pop_only=True,
        stop_grad_var_u=bool(stop_grad_var_u),
        return_var=True,
        is_single_PE=data.is_single_PE,
    )

    _log_setup("[PE marginal] using event streaming:", int(data.chunk_pe) > 0)
    _log_setup("[PE marginal] chunk_pe:", int(data.chunk_pe))

    loglik = make_loglik_marginal_samples(core, data)
    loglik_components = make_loglik_marginal_samples_components(
        core,
        data,
        include_eventwise=True,
    )
    
    return core, loglik, loglik_components



##########################################################################

NORM_Q95 = 1.959963984540054
NORM_Q99 = 2.5758293035489004
RAW_SD_95 = 1.502  # for sigmoid-reparam params; keep as-is if you use it elsewhere


def normal_from_bounds_95(name: str, low: float, high: float):
    """
    Interpret [low, high] as the central 95% interval of a Normal.
    Returns the sampled scalar (jnp.ndarray scalar).
    Init values are handled outside via numpyro.infer.init_to_value.
    """
    mu = 0.5 * (low + high)
    sigma = (high - low) / (2.0 * NORM_Q95)
    return numpyro.sample(name, dist.Normal(mu, sigma))

    

def floored_lognormal_q95(
    name: str,
    floor: float,
    typical_max_total: float,
    median_frac: float = 0.2,
):
    """
    sigma = floor + x, with x ~ LogNormal(mu, sigma_ln)

    We set:
      Q95(x)     = raw_typ = typical_max_total - floor
      median(x)  = median_frac * raw_typ   (default 0.2)

    Returns the shifted parameter (Deterministic) as a scalar.
    Also samples an internal variable f"{name}_raw".
    """
    raw_typ = max(1e-12, typical_max_total - floor)
    med = max(1e-12, median_frac * raw_typ)

    # LogNormal: median = exp(mu), Q95 = exp(mu + z95*sigma_ln)
    mu = np.log(med)
    sigma_ln = (np.log(raw_typ) - mu) / NORM_Q95

    x = numpyro.sample(f"{name}_raw", dist.LogNormal(loc=mu, scale=sigma_ln))
    out = floor + x
    numpyro.deterministic(name, out)
    return out


def alpha_bar_init_from_alpha1(ivals: dict, *, fallback_mid: float) -> float:
    """
    If ivals contains alpha_bar, keep it.
    Else if ivals contains alpha1 (and optionally alpha2), convert to alpha_bar.
    Else fallback to the prior midpoint.
    """
    if ivals is None:
        return float(fallback_mid)

    if "alpha_bar" in ivals and ivals["alpha_bar"] is not None:
        return float(ivals["alpha_bar"])

    a1 = ivals.get("alpha1", None)
    a2 = ivals.get("alpha2", None)

    if a1 is not None and a2 is not None:
        return 0.5 * (float(a1) + float(a2))
    if a1 is not None:
        return float(a1)

    return float(fallback_mid)


def alpha_diff_init(ivals: dict, *, default: float = 0.0) -> float:
    """
    If ivals contains alpha_diff, keep it. Otherwise default (0).
    If ivals contains alpha1 and alpha2, use alpha2-alpha1 as a good init.
    """
    if ivals is None:
        return float(default)

    if "alpha_diff" in ivals and ivals["alpha_diff"] is not None:
        return float(ivals["alpha_diff"])

    a1 = ivals.get("alpha1", None)
    a2 = ivals.get("alpha2", None)
    if a1 is not None and a2 is not None:
        return float(a2) - float(a1)

    return float(default)



def unit_interval_sigmoid(name: str, raw_sigma: float = 1.0):
    """
    Unconstrained raw ~ Normal(0, raw_sigma) mapped to (0,1) via sigmoid.

    NumPyro note: init values are handled via init_to_value on f"{name}_raw".
    """
    raw = numpyro.sample(f"{name}_raw", dist.Normal(0.0, raw_sigma))
    out = jax.nn.sigmoid(raw)
    numpyro.deterministic(name, out)
    return out


def unit_interval_sigmoid_raw_init(initval):
    """
    Helper to build init_to_value dict entry for {name}_raw from initval in (0,1).
    """
    if initval is None:
        return None
    x = float(np.clip(initval, 1e-6, 1.0 - 1e-6))
    return np.log(x / (1.0 - x))


def bounded_sigmoid(name: str, low: float, high: float, raw_sigma: float = RAW_SD_95):
    """
    low + (high-low)*sigmoid(raw), with raw ~ Normal(0, raw_sigma)

    NumPyro: provide init via init_to_value on f"{name}_raw" if desired.
    """
    raw = numpyro.sample(f"{name}_raw", dist.Normal(0.0, raw_sigma))
    val = low + (high - low) * jax.nn.sigmoid(raw)
    numpyro.deterministic(name, val)
    return val


def bounded_sigmoid_vec(name: str, low: float, high: float, size: int, raw_sigma: float = 1.0):
    """
    Vector version of bounded_sigmoid.

    Samples:
        {name}_raw : shape (size,)

    Returns deterministic:
        {name} : shape (size,)

    This replaces a vector Uniform(low, high) prior by a logistic-normal
    prior on the same interval, suppressing edge pile-up in the unconstrained
    geometry.
    """
    raw = numpyro.sample(
        f"{name}_raw",
        dist.Normal(0.0, raw_sigma).expand((int(size),)).to_event(1),
    )

    val = (
        jnp.asarray(low, dtype=jnp.float64)
        + (jnp.asarray(high, dtype=jnp.float64) - jnp.asarray(low, dtype=jnp.float64))
        * jax.nn.sigmoid(raw)
    )

    numpyro.deterministic(name, val)
    return val
    
def floored_lognormal_raw_init(ivals, name: str, priors):
    if ivals.get(name) is None:
        return None
    floor = float(priors[name][0])
    return max(1e-12, float(ivals[name]) - floor)


def bounded_sigmoid_raw_init(initval, low: float, high: float):
    """
    Helper to compute raw init (logit) corresponding to initval in [low, high].
    Put result in init_to_value dict under f"{name}_raw".
    """
    if initval is None:
        return None
    t = float((initval - low) / (high - low))
    t = np.clip(t, 1e-6, 1.0 - 1e-6)
    return np.log(t / (1.0 - t))


def _dp_m1_env_extra_count(DP_m1_env_form):
    return 4 if str(DP_m1_env_form) == "smooth_broken_pl_2breaks" else 0

def bounded_sigmoid_logpositive(name: str, low: float, high: float, raw_sigma: float = 1.5):
    raw = numpyro.sample(f"{name}_raw", dist.Normal(0.0, raw_sigma))
    lowlog, highlog = np.log(low), np.log(high)
    logx = lowlog + (highlog - lowlog) * jax.nn.sigmoid(raw)
    numpyro.deterministic(f"{name}_log", logx)
    val = jnp.exp(logx)
    numpyro.deterministic(name, val)
    return val

def bounded_sigmoid_logpositive_median(
    name: str,
    low: float,
    high: float,
    median: float,
    raw_sigma: float = 1.0,
):
    """
    Bounded positive logistic-normal in log-space.

    Support:
        val in (low, high)

    Median is approximately `median`, provided low < median < high.
    Sample site:
        {name}_raw
    Deterministic:
        {name}
    """
    low = float(low)
    high = float(high)
    median = float(median)

    if not (0.0 < low < high):
        raise ValueError(f"{name}: need 0 < low < high, got low={low}, high={high}")
    if not (low < median < high):
        raise ValueError(
            f"{name}: need low < median < high, got low={low}, median={median}, high={high}"
        )

    lowlog = np.log(low)
    highlog = np.log(high)
    medlog = np.log(median)

    t_med = (medlog - lowlog) / (highlog - lowlog)
    t_med = np.clip(t_med, 1.0e-6, 1.0 - 1.0e-6)
    raw_loc = np.log(t_med / (1.0 - t_med))

    raw = numpyro.sample(f"{name}_raw", dist.Normal(raw_loc, raw_sigma))

    logx = lowlog + (highlog - lowlog) * jax.nn.sigmoid(raw)
    numpyro.deterministic(f"{name}_log", logx)

    val = jnp.exp(logx)
    numpyro.deterministic(name, val)
    return val

def shifted_lognormal_q95(
    name: str,
    floor: float,
    q95: float,
    *,
    median=None,
    eps: float = 1e-6,
):
    """
    Positive shifted lognormal prior:

        x = floor + delta
        delta ~ LogNormal(mu_delta, sigma_delta)

    The input q95 is interpreted as the 95% quantile of x, not as a
    hard upper bound.

    If median is None, use the midpoint between floor and q95 as the
    median of x.

    This is useful for mass cutoffs where a hard bounded-sigmoid prior
    can put awkward geometry near the edges.
    """
    floor = float(floor)
    q95 = float(q95)

    if q95 <= floor:
        raise ValueError(
            f"Invalid shifted_lognormal_q95 bounds for {name}: "
            f"floor={floor}, q95={q95}"
        )

    if median is None:
        median = 0.5 * (floor + q95)

    median = float(median)

    if not (floor < median < q95):
        raise ValueError(
            f"Invalid shifted_lognormal_q95 median for {name}: "
            f"need floor < median < q95, got floor={floor}, "
            f"median={median}, q95={q95}"
        )

    delta_med = jnp.asarray(max(median - floor, eps), dtype=jnp.float64)
    delta_q95 = jnp.asarray(max(q95 - floor, eps), dtype=jnp.float64)

    mu_delta = jnp.log(delta_med)
    sigma_delta = (jnp.log(delta_q95) - mu_delta) / NORM_Q95

    delta = numpyro.sample(
        f"delta_{name}",
        dist.LogNormal(loc=mu_delta, scale=sigma_delta),
    )

    val = floor + delta
    numpyro.deterministic(name, val)
    return val

    

def stick_breaking_jax(beta):
    beta = jnp.asarray(beta, dtype=jnp.float64)
    one_minus = 1.0 - beta
    if beta.shape[0] == 1:
        return jnp.ones((1,), dtype=jnp.float64)
    remaining = jnp.concatenate([
        jnp.ones((1,), dtype=jnp.float64),
        jnp.cumprod(one_minus[:-1]),
    ])
    return beta * remaining



def _z_from_dL_numpy(dL, H0, Om, w0, Xi0, nXi0, *, param, integrate_dc):
    bk = NPBackend()
    return np.asarray(
        cosmo.z_from_dL(
            bk,
            np.asarray(dL, dtype=np.float64),
            H0=H0,
            Om=Om,
            w0=w0,
            Xi0=Xi0,
            nXi0=nXi0,
            z_nodes=None,
            d_nodes=None,
            param=param,
            integrate_dc=integrate_dc,
        ),
        dtype=np.float64,
    )




def evo_triplet_numpyro(
    name,
    theta0,
    ivals,
    priors,
    positive=False,
    eps_pos=1e-6,
):
    dtype = jnp.float64

    sigma_key = f"delta_{name}_sigma"
    delta_sigma = jnp.asarray(priors.get(sigma_key, 1.0), dtype=dtype)

    theta0 = jnp.asarray(theta0, dtype=dtype)

    # ------------------------------------------------------------
    # delta evolution
    # ------------------------------------------------------------
    if positive:
        lower = -theta0 + jnp.asarray(eps_pos, dtype=dtype)

        delta_theta = numpyro.sample(
            f"delta_{name}",
            dist.TruncatedNormal(
                loc=jnp.asarray(0.0, dtype=dtype),
                scale=delta_sigma,
                low=lower,
            ),
        )

        # This should be strictly positive by construction.
        theta_inf = numpyro.deterministic(
            f"{name}_inf",
            theta0 + delta_theta,
        )

    else:
        delta_theta = numpyro.sample(
            f"delta_{name}",
            dist.Normal(
                loc=jnp.asarray(0.0, dtype=dtype),
                scale=delta_sigma,
            ),
        )

        theta_inf = numpyro.deterministic(
            f"{name}_inf",
            theta0 + delta_theta,
        )

    # ------------------------------------------------------------
    # transition redshift
    # ------------------------------------------------------------
    z_low, z_high = priors.get("z_t", (0.05, 2.5))

    z_t = numpyro.sample(
        f"z_{name}",
        dist.Uniform(
            jnp.asarray(z_low, dtype=dtype),
            jnp.asarray(z_high, dtype=dtype),
        ),
    )

    # ------------------------------------------------------------
    # transition width
    #
    # This is
    # log_dz ~ Normal(log(dz_mid), 0.5)
    # dz = clip(exp(log_dz), dz_low, dz_high)
    # ------------------------------------------------------------
    dz_low, dz_high = priors.get("dz", (0.05, 3.0))
    dz_low = jnp.asarray(dz_low, dtype=dtype)
    dz_high = jnp.asarray(dz_high, dtype=dtype)

    dz_mid = 0.5 * (dz_low + dz_high)

    log_dz = numpyro.sample(
        f"log_dz_{name}",
        dist.Normal(
            loc=jnp.log(dz_mid),
            scale=jnp.asarray(0.5, dtype=dtype),
        ),
    )

    dz_raw = jnp.exp(log_dz)

    dz = numpyro.deterministic(
        f"dz_{name}",
        jnp.clip(dz_raw, dz_low, dz_high),
    )

    return theta_inf, z_t, dz




def _parse_q_mbounds_local(q_mbound_low, q_mbound_up):
    return nutils._parse_q_mbounds_np(q_mbound_low, q_mbound_up)


def _bounds_from_values_local(vals, qlow, qup):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("Cannot compute bounds from empty/non-finite values.")
    return (
        float(np.quantile(vals, qlow)),
        float(np.quantile(vals, 1.0 - qup)),
    )



def _twosector_sigmoid_np_local(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


def _twosector_m1m2_from_logMc_logitq_np(logMc_by_event, logitq_by_event):
    logMc_by_event = np.asarray(logMc_by_event, dtype=np.float64)
    logitq_by_event = np.asarray(logitq_by_event, dtype=np.float64)
    q = _twosector_sigmoid_np_local(logitq_by_event)
    q = np.clip(q, 1.0e-12, 1.0)
    Mc = np.exp(logMc_by_event)
    m1 = Mc * (1.0 + q)**0.2 / q**0.6
    m2 = q * m1
    return m1, m2, q


def _twosector_flat_event_names_local(allnames, N):
    names = []
    try:
        for chunk_names in allnames:
            if isinstance(chunk_names, (list, tuple, np.ndarray)):
                for nm in chunk_names:
                    names.append(str(nm))
            else:
                names.append(str(chunk_names))
    except Exception:
        names = []
    if len(names) != int(N):
        names = [f"event_{i}" for i in range(int(N))]
    return names





def _convert_scales_coord2_logitq_to_q(
    scales,
    *,
    q_mbound_low,
    q_mbound_up,
    DP_bounds_source,
    DPUC_q_mu_bounds_source="inferred",
):
    """
    Convert the coordinate-2 entries of a find_mass_redshift_bounds* result
    from logit(q) to q.

    The existing bound-finding functions return coordinate-2 samples as
    res_logitq_by_event.  For DPUC_q_coord='q', we reinterpret coordinate 2
    as q and recompute:

      lq_min_data / lq_max_data
      lq_diff
      lq_res_p25_fac05
      res_logitq_by_event  -> q samples, under the same key for downstream code

    The key names are preserved intentionally to minimize downstream changes.
    """
    if DP_bounds_source not in ("samples", "event_means"):
        raise ValueError(
            f"DP_bounds_source must be 'samples' or 'event_means', got {DP_bounds_source!r}."
        )

    if DPUC_q_mu_bounds_source not in ("inferred", "unit"):
        raise ValueError(
            "DPUC_q_mu_bounds_source must be 'inferred' or 'unit'. "
            f"Got {DPUC_q_mu_bounds_source!r}."
        )


    out = dict(scales)

    _, qlow_coord2, _, qup_coord2 = _parse_q_mbounds_local(
        q_mbound_low,
        q_mbound_up,
    )

    y_by_event = np.asarray(scales["res_logitq_by_event"], dtype=np.float64)
    q_by_event = 1.0 / (1.0 + np.exp(-y_by_event))
    q_by_event = np.clip(q_by_event, 1.0e-12, 1.0 - 1.0e-12)

    if DPUC_q_mu_bounds_source == "unit":
        q_min_data = 0.0
        q_max_data = 1.0
    elif DP_bounds_source == "event_means":
        event_q_mean = np.nanmean(q_by_event, axis=1)
        q_min_data, q_max_data = _bounds_from_values_local(
            event_q_mean,
            qlow_coord2,
            qup_coord2,
        )
    else:
        q_mins = []
        q_maxs = []
        for it in range(q_by_event.shape[1]):
            qlo, qhi = _bounds_from_values_local(
                q_by_event[:, it],
                qlow_coord2,
                qup_coord2,
            )
            q_mins.append(qlo)
            q_maxs.append(qhi)
        q_min_data = float(np.min(q_mins))
        q_max_data = float(np.max(q_maxs))

    q_diffs = []
    tol = 1.0e-12
    for it in range(q_by_event.shape[1]):
        qq = np.sort(q_by_event[:, it].copy())
        qq = qq[np.insert(np.diff(qq) > tol, 0, True)]
        dq = np.diff(qq)
        dq = dq[dq > tol]
        if dq.size > 0:
            q_diffs.append(float(np.min(dq)))

    if not q_diffs:
        raise ValueError("No positive q spacings found while estimating DPUC q-coordinate bounds.")

    q_diff = float(np.mean(q_diffs))

    L1_res, L2_res, Lres_summary = estimate_L_small_from_transformed_event_resolution_np(
        np.asarray(scales["res_logMc_by_event"], dtype=np.float64),
        q_by_event,
        percentile=25.0,
        factor=0.5,
        min_samples=max(10, int(0.1 * q_by_event.shape[1])),
    )

    out["lq_min_data"] = q_min_data
    out["lq_max_data"] = q_max_data
    out["lq_min_inj"] = 0.0
    out["lq_max_inj"] = 1.0
    out["lq_diff"] = q_diff
    out["lq_res_p25_fac05"] = L2_res
    out["Lres_summary_default"] = Lres_summary
    out["res_logitq_by_event"] = q_by_event

    _log_setup("DPUC_q_coord='q': converted coordinate-2 bounds from logit(q) to q.")
    _log_setup("DPUC_q_mu_bounds_source=%s" % DPUC_q_mu_bounds_source)
    _log_setup("coord2=q bounds: %s, %s" % (q_min_data, q_max_data))
    _log_setup("coord2=q spacing floor: %s" % q_diff)

    return out

def _restrict_logitq_mu_bounds_to_event_intersection(
    *,
    lowmu2,
    upmu2,
    gmm_fit_transform_mode,
    gmm_fit_coord_bounds,
    gmm_fit_coord_names,
):
    """
    For bounded_gmm_transform=1, restrict the global DP/DPUC logit(q)
    component-mean range to the common physical q-support of all events.

    Event bounded GMM stores q bounds in physical q, not logit(q).
    We convert q_min/q_max -> logit(q_min)/logit(q_max).

    Final rule:
        lowmu2 = max(proposed lowmu2, max_i logit(q_min_i))
        upmu2  = min(proposed upmu2,  min_i logit(q_max_i))

    No fallback. If the intersection is empty, raise.
    """

    if gmm_fit_transform_mode != "event_bounded_flogit":
        return lowmu2, upmu2

    if gmm_fit_coord_bounds is None:
        raise ValueError(
            "event_bounded_flogit is active but gmm_fit_coord_bounds is None."
        )

    if gmm_fit_coord_names is None:
        raise ValueError(
            "event_bounded_flogit is active but gmm_fit_coord_names is None."
        )

    coord_names = tuple(str(x) for x in gmm_fit_coord_names)

    if "q" not in coord_names:
        raise ValueError(
            "event_bounded_flogit q-intersection requires coordinate name 'q'. "
            "Got gmm_fit_coord_names=%s" % (coord_names,)
        )

    iq = coord_names.index("q")

    bounds = np.asarray(gmm_fit_coord_bounds, dtype=np.float64)

    if bounds.ndim != 3 or bounds.shape[2] != 2:
        raise ValueError(
            "gmm_fit_coord_bounds must have shape (Nevents, ndim, 2). "
            "Got %s." % (bounds.shape,)
        )

    qlo_evt = bounds[:, iq, 0]
    qhi_evt = bounds[:, iq, 1]

    if not np.all(np.isfinite(qlo_evt)) or not np.all(np.isfinite(qhi_evt)):
        raise ValueError("Non-finite event q bounds found.")

    if not np.all(qhi_evt > qlo_evt):
        raise ValueError("Some event q bounds have upper <= lower.")

    if np.any(qlo_evt <= 0.0) or np.any(qhi_evt > 1.0):
        raise ValueError(
            "Event q bounds must satisfy 0 < q_min and q_max <= 1 before logit. "
            "Found qlo min=%s, qhi max=%s."
            % (np.min(qlo_evt), np.max(qhi_evt))
        )
    
    q_eps = 1.0e-12
    qlo_evt_safe = np.clip(qlo_evt, q_eps, 1.0 - q_eps)
    qhi_evt_safe = np.clip(qhi_evt, q_eps, 1.0 - q_eps)
    
    ylo_evt = np.log(qlo_evt_safe / (1.0 - qlo_evt_safe))
    yhi_evt = np.log(qhi_evt_safe / (1.0 - qhi_evt_safe))

    common_low = float(np.max(ylo_evt))
    common_high = float(np.min(yhi_evt))

    if not common_high > common_low:
        raise ValueError(
            "Empty common event logit(q) support for bounded GMMs: "
            "max_i logit(q_min_i)=%s, min_i logit(q_max_i)=%s."
            % (common_low, common_high)
        )

    old_low = float(lowmu2)
    old_high = float(upmu2)

    new_low = max(old_low, common_low)
    new_high = min(old_high, common_high)

    if not new_high > new_low:
        raise ValueError(
            "DP/DPUC logit(q) mean range became empty after intersecting "
            "with event bounded-GMM supports. "
            "proposed=[%s,%s], common_event=[%s,%s], final=[%s,%s]."
            % (old_low, old_high, common_low, common_high, new_low, new_high)
        )

    _log_setup("event_bounded_flogit active: restricting DP/DPUC logit(q) mean range.")
    _log_setup("event q bounds converted to logit(q).")
    _log_setup("proposed lowmu2/upmu2: %s, %s" % (old_low, old_high))
    _log_setup("common event logit(q) support: %s, %s" % (common_low, common_high))
    _log_setup("final lowmu2/upmu2: %s, %s" % (new_low, new_high))
    _log_setup("event logit(q) lower bounds 5/50/95:", np.quantile(ylo_evt, [0.05, 0.50, 0.95]))
    _log_setup("event logit(q) upper bounds 5/50/95:", np.quantile(yhi_evt, [0.05, 0.50, 0.95]))

    return new_low, new_high    




def log_lik_var_high_sigmoid_params(
    bk,
    log_lik_var_min,
    width,
    *,
    p_at_min=0.99,
    eps_high=1.0e-6,
    eps=1.0e-12,
):
    """
    High-edge sigmoid gate for log_lik_var.

    Enforces approximately

        gate(log_lik_var_min) = p_at_min
        gate(log_lik_var_min + width) = eps_high

    so the likelihood is killed when log_lik_var is above threshold.
    """

    width = bk.maximum(width, eps)

    a = _logit_scalar(bk, p_at_min)
    b = _logit_scalar(bk, eps_high)

    s = width / (a - b)
    center = log_lik_var_min + s * a

    return center, s


def log_lik_var_high_gate(
    bk,
    log_lik_var_sg,
    log_lik_var_min,
    *,
    kind="sigmoid",
    width=0.05,
    p_at_min=0.99,
    eps_high=1.0e-6,
):
    if kind == "sigmoid":
        center, s = log_lik_var_high_sigmoid_params(
            bk,
            log_lik_var_min,
            width,
            p_at_min=p_at_min,
            eps_high=eps_high,
        )
        return log_sigmoid(bk, center, log_lik_var_sg, s)

    elif kind == "none":
        return bk.where(log_lik_var_sg > log_lik_var_min, -1e30, 0.0)

    else:
        raise ValueError(f"Unknown likelihood-variance gate kind: {kind}")




        

    

def make_model_jax(  priors,
                 GWData,
                 InjData,
                 ivals=None,
                 eps_init = 0.01,
                 sampling_GW = 'gmm',
                 rate_model = 'MD',
                 mass_model = 'PLP',
                 smoothing='LVK',
                 simplex_repair=False,
                 interp_mass = 0,
                 interp_z = 0,
                 has_m2_break = False,
                 norm_gauss = 'uplow',
                 spin_model = 'none',
                 spin_inj = 'none',
                 marginal_R0 = True,
                 dLprior = ['none'],
                 fix_inj_len = False,
                 chunk_inj = -1,
                 chunk_reduce = 0,
                 use_float32 = False,
                 use_float32_bias=False,
                 sel_method='Tobs',
                 N_DP_comp_max = 100,
                 alpha_tail = 0.2,
                 alpha_small = 0.01,
                 L_small_1 = 0.5,
                 L_small_2 = 0.5,
                 s_local = [0.5, 1],
                    L_small_source="spacing",
                L_small_res_percentile=25.0,
                L_small_res_factor=0.5,
                     mu_log_alpha_inv = 0.,
                     sigma_log_alpha_inv =1.,
                 find_m_bounds = False,
                 q_mbound_mu_low = 0.05,
                 q_mbound_mu_up = 0.05,
                 q_mbound_sig_low = 0.05,
                 q_mbound_sig_up = 0.05,
                 DP_mu_bounds_source = "samples",
                 DP_sig_bounds_source = "samples",
                 alpha_inv_params = (1, 1),
                 M_active = 5,
                     center_eps_widths=False,
                 tau_prior = 'flat',
                 fix_H0 = True,
                fix_Om = True,
               fix_w0 = True,
                 fix_Xi0n = True,
                 integrate_dc = 'trapz',
                 z_pivot=0.5,
               pade=False,
               zres=200,
                z_grid_mode='cst',
                zmin_a=1e-05, zmin_b=1e-03, zmid_b=3.0, zmax_c=10.0, hi_boost=0.20,
                 find_z_bounds = False,
               params_fix=None,
                 Neff_min=4,
                Neff_min_lik=1,
               log_lik_var_min=1,
                 use_sel_spin=True,
                 pop_only = False,
               N_successes_l=None,
               Nsamplesuse = -1,
               include_sel_uncertainty=False,
               sel_smoothing='sigmoid',
               dil_factor=1,
               allTobs=None,
                 use_updates=True,
                 inj_loop='vec',
                 save_thetas=False,
                 interp_inj=True,
                 param='vanilla',
                 DP_prior='SB',
                 sigma_softmax=0.75,
                 gamma_DP_params = (4, 0.8),
                 is_observed = False,
                 sample_from_pop = False,
                 mmin_inj=-1,
                 is_compressed_inj=False,
                 debug_sel_batch=False,
                 reparam_z = True,
                 reparam_mass = True,
                reparam_cosmo = True,
                 priors_for_mmin='',
                 penorm_lims=[],
                 linear_mass=False,
                 linear_z=False,
                 DP_truncate_up=False,
                     DPUC_mmax_fixed=-1.0,
                 DP_truncate_low=False,
                 DP_m1_env = False,
                 DP_m1_env_form = "powerlaw",
                 DP_m1_env_delta1 = 0.2,
                 DP_m1_env_delta2 = 0.2,
                 DP_q_env = False,
                 DPUC_Primary_q_norm = "conditional",
                DP_corr_m1m2 = False,
                DP_corr_eta_rho = 3.0,
                DP_corr_f_minor = 0.35,
                     DPUC_low_taper_deltas = (2.0, 2.0),
                     DPUC_low_end = (-1.0,), 
                     DPUC_low_end_share = True,
                     DPUC_high_taper_delta=10.0,
                    PL_DPUC_normalize_gmm = False,
                    DPUC_kernel = "gaussian",
                    DPUC_kernel_rcut = 4.0,
                    DPUC_kernel_dr = 0.15,
                    DPUC_kernel_xcut = 3.0,
                    DPUC_kernel_ycut = 3.0,
                    DPUC_q_coord = "logitq",
                    DPUC_TruncM2_shared_weights = False,
                    DPUC_TruncM2_mu2_mode = "free",
                    DPUC_TruncM2_secondary_coord = "logm2",
                    DPUC_TruncM2_area_reg = False,
                    DPUC_TruncM2_log_area_min = -4.6,
                    DPUC_TruncM2_log_area_width = 0.5,
                    DPUC_TruncM2_log_area_lambda = 1.0,
                    DPUC_TruncM2_aspect_reg = False,
                    DPUC_TruncM2_aspect_max = 1.0,
                    DPUC_TruncM2_aspect_width = 0.5,
                    DPUC_TruncM2_aspect_lambda = 0.5,
                    DPUC_TruncM2_low_center_weight_reg = False,
                    DPUC_TruncM2_low_center_mcut = 5.0,
                    DPUC_TruncM2_low_center_log_width = 0.15,
                    DPUC_TruncM2_low_center_weight_cap = 0.8,
                    DPUC_TruncM2_low_center_weight_width = 0.05,
                    DPUC_TruncM2_low_center_weight_lambda = 1.0,
                    DPUC_TruncM2_weight_cap_reg = False,
                    DPUC_TruncM2_weight_cap = 0.85,
                    DPUC_physical_Lsmall_shared_mode = "max",
                    DPUC_q_mu_bounds_source = "inferred",
                     
                 detach_var = False,
                remove_spin_prior = False,
                r = 0,
                     vary_mb = True,
                     taper_kind = "sigmoid",
                taper_p = 12.0,
                     skip_sel=False,
                    is_GP_dL=False,
                    find_GP_L=True,
                    monotonicity="mono_reparam",
                    GP_kernel="stationary",
                    U=None,
                    GP_prior="frechet",
                     GP_jitter = 1e-10,
                    large_ell_penalty=True,
                    ell_min=0.1,
                    ell_max=3.0,
                    ell0_min=1.0,
                    ell0_max=10.0,
                    nu=0.5,
                    lam=1.0,
                    res_nbins=15,
                    res_q=0.10,
                     smooth_GP_resolution_loglinear=False,
                    fine_res=0.05,
                    res_highz=0.1,
                    res_lowz=0.05,
                    eps_DE=-1.0,
                    invert_dL_GP=True,
                     n_gp_nodes = 150,
                     n_gp_fine = 900,
                    fout=None,
                     GP_oob_guard=True,
                     alpha_eta=0.01,
                    alpha_ell=0.01,
                    alpha_large=0.05,
                    logXi_guard=False,
                    logXi_abs_max_allowed=5.0,
                    logXi_guard_strength=1.0e3,
                ):




    if rate_model != "MD":
        raise ValueError("The publication release supports rate_model='MD'.")
    if spin_model not in ("none", "default_gauss"):
        raise ValueError("spin_model must be 'none' or 'default_gauss'.")
    if mass_model not in ("DPLDP", "DPLDP-z", "DPUC", "DPUC_Primary", "DPUC_TruncM2"):
        raise ValueError(
            "NumPyro inference supports DPLDP, DPLDP-z, DPUC, DPUC_Primary, and DPUC_TruncM2. "
            "PLPreg retains density/core support only."
        )

    s_local = nutils._coordinate_pair(s_local, "s_local")
    L_small_res_percentile = nutils._coordinate_pair(L_small_res_percentile, "L_small_res_percentile")
    L_small_res_factor = nutils._coordinate_pair(L_small_res_factor, "L_small_res_factor")

    if DPUC_kernel not in ("gaussian", "local_native_gaussian", "local_native_product_gaussian"):
        raise ValueError(
            "DPUC_kernel must be 'gaussian', 'local_native_gaussian', or "
            "'local_native_product_gaussian'. "
            f"Got {DPUC_kernel!r}."
        )

    if DPUC_q_coord not in ("logitq", "q"):
        raise ValueError(
            "DPUC_q_coord must be 'logitq' or 'q'. "
            f"Got {DPUC_q_coord!r}."
        )

    if DPUC_q_mu_bounds_source not in ("inferred", "unit"):
        raise ValueError(
            "DPUC_q_mu_bounds_source must be 'inferred' or 'unit'. "
            f"Got {DPUC_q_mu_bounds_source!r}."
        )

    DP_m1_env_form = str(DP_m1_env_form)
    if DP_m1_env_form not in ("powerlaw", "smooth_broken_pl_2breaks"):
        raise ValueError(
            "DP_m1_env_form must be 'powerlaw' or 'smooth_broken_pl_2breaks'. "
            f"Got {DP_m1_env_form!r}."
        )
    DP_m1_env_delta1 = float(DP_m1_env_delta1)
    DP_m1_env_delta2 = float(DP_m1_env_delta2)
    if DP_m1_env_delta1 <= 0.0 or DP_m1_env_delta2 <= 0.0:
        raise ValueError("DP_m1_env_delta1 and DP_m1_env_delta2 must be > 0.")

    if str(mass_model) == "DPUC_Primary":
        mass_model = "DPUC_Primary"
        DP_q_env = True

    DPUC_Primary_q_norm = str(DPUC_Primary_q_norm)
    if DPUC_Primary_q_norm not in ("conditional", "joint"):
        raise ValueError(
            "DPUC_Primary_q_norm must be 'conditional' or 'joint', "
            f"got {DPUC_Primary_q_norm!r}."
        )
    if str(mass_model) != "DPUC_Primary" and DPUC_Primary_q_norm != "conditional":
        raise ValueError(
            "DPUC_Primary_q_norm='joint' is only supported for mass_model='DPUC_Primary'."
        )

    DPUC_TruncM2_shared_weights = bool(DPUC_TruncM2_shared_weights)
    DPUC_TruncM2_mu2_mode = str(DPUC_TruncM2_mu2_mode)
    if DPUC_TruncM2_mu2_mode not in ("free", "ordered", "q_ratio"):
        raise ValueError(
            "DPUC_TruncM2_mu2_mode must be 'free', 'ordered', or 'q_ratio'. "
            f"Got {DPUC_TruncM2_mu2_mode!r}."
        )


    DPUC_TruncM2_secondary_coord = str(DPUC_TruncM2_secondary_coord)
    if DPUC_TruncM2_secondary_coord not in ("logm2", "logitq"):
        raise ValueError(
            "DPUC_TruncM2_secondary_coord must be 'logm2' or 'logitq'. "
            f"Got {DPUC_TruncM2_secondary_coord!r}."
        )
    if DPUC_TruncM2_secondary_coord == "logitq" and DPUC_TruncM2_mu2_mode != "q_ratio":
        raise ValueError(
            "DPUC_TruncM2_secondary_coord='logitq' currently requires "
            "DPUC_TruncM2_mu2_mode='q_ratio', so the secondary means are sampled as q_mu."
        )

    DPUC_TruncM2_area_reg = bool(DPUC_TruncM2_area_reg)
    DPUC_TruncM2_aspect_reg = bool(DPUC_TruncM2_aspect_reg)
    DPUC_TruncM2_log_area_min = float(DPUC_TruncM2_log_area_min)
    DPUC_TruncM2_log_area_width = float(DPUC_TruncM2_log_area_width)
    DPUC_TruncM2_log_area_lambda = float(DPUC_TruncM2_log_area_lambda)
    DPUC_TruncM2_aspect_max = float(DPUC_TruncM2_aspect_max)
    DPUC_TruncM2_aspect_width = float(DPUC_TruncM2_aspect_width)
    DPUC_TruncM2_aspect_lambda = float(DPUC_TruncM2_aspect_lambda)
    if DPUC_TruncM2_log_area_width <= 0.0:
        raise ValueError("DPUC_TruncM2_log_area_width must be > 0.")
    if DPUC_TruncM2_log_area_lambda < 0.0:
        raise ValueError("DPUC_TruncM2_log_area_lambda must be >= 0.")
    if DPUC_TruncM2_aspect_width <= 0.0:
        raise ValueError("DPUC_TruncM2_aspect_width must be > 0.")
    if DPUC_TruncM2_aspect_lambda < 0.0:
        raise ValueError("DPUC_TruncM2_aspect_lambda must be >= 0.")
    if DPUC_TruncM2_aspect_max < 0.0:
        raise ValueError("DPUC_TruncM2_aspect_max must be >= 0.")

    DPUC_physical_Lsmall_shared_mode = str(DPUC_physical_Lsmall_shared_mode)
    if DPUC_physical_Lsmall_shared_mode not in ("max", "min"):
        raise ValueError(
            "DPUC_physical_Lsmall_shared_mode must be 'max' or 'min'. "
            f"Got {DPUC_physical_Lsmall_shared_mode!r}."
        )

    q_mu_low = None
    q_mu_up = None



    if DPUC_kernel in ("local_native_gaussian", "local_native_product_gaussian"):
        if mass_model != "DPUC":
            raise ValueError(
                "DPUC_kernel local-native branches are only supported for "
                f"mass_model='DPUC'. Got mass_model={mass_model!r}."
            )

        DPUC_low_taper_deltas = tuple(float(x) for x in DPUC_low_taper_deltas)


    if len(DPUC_low_end) == 1 and DPUC_low_end[0] < 0.0:
        DPUC_low_end_sample = True
        DPUC_low_end_m1_fixed = None
        DPUC_low_end_m2_fixed = None

    elif len(DPUC_low_end) == 2:
        DPUC_low_end_sample = False
        DPUC_low_end_m1_fixed = float(DPUC_low_end[0])
        DPUC_low_end_m2_fixed = float(DPUC_low_end[1])

        if DPUC_low_end_m1_fixed <= 0.0 or DPUC_low_end_m2_fixed <= 0.0:
            raise ValueError(
                "Fixed DPUC_low_end must be positive. "
                f"Got {DPUC_low_end}."
            )

    else:
        raise ValueError(
            "DPUC_low_end must be either two positive values "
            "[m1_low m2_low], or a single negative value [-1] to sample them. "
            f"Got {DPUC_low_end}."
        )
    
    if len(DPUC_low_taper_deltas) == 1 and DPUC_low_taper_deltas[0] < 0.0:
        DPUC_low_taper_sample = True
        DPUC_low_taper_delta_m1_fixed = None
        DPUC_low_taper_delta_m2_fixed = None

    elif len(DPUC_low_taper_deltas) == 2:
        DPUC_low_taper_sample = False
        DPUC_low_taper_delta_m1_fixed = float(DPUC_low_taper_deltas[0])
        DPUC_low_taper_delta_m2_fixed = float(DPUC_low_taper_deltas[1])

        if DPUC_low_taper_delta_m1_fixed <= 0.0 or DPUC_low_taper_delta_m2_fixed <= 0.0:
            raise ValueError(
                "Fixed DPUC_low_taper_deltas must be positive. "
                f"Got {DPUC_low_taper_deltas}."
            )

    else:
        raise ValueError(
            "DPUC_low_taper_deltas must be either two positive values "
            "[delta_m1 delta_m2], or a single negative value [-1] to sample them. "
            f"Got {DPUC_low_taper_deltas}."
        )

    DPUC_mmax_fixed = float(DPUC_mmax_fixed)

    if DPUC_mmax_fixed > 0.0 and not DP_truncate_up:
        raise ValueError(
            "DPUC_mmax_fixed > 0 requires DP_truncate_up=True. "
            "Otherwise the high-mass truncation is not active."
        )

    DPUC_high_taper_delta = float(DPUC_high_taper_delta)

    if DPUC_high_taper_delta == 0.0:
        raise ValueError(
            "DPUC_high_taper_delta must be >0 for fixed mode or <0 for sampled mode."
        )
    
    DPUC_high_taper_sample = DPUC_high_taper_delta < 0.0

    ################################################
    # Read in data and set dimensions
    ################################################

    
    ## GW data
    if not pop_only:
        # gw data are interpolants of single-event posteriors
        if sampling_GW=='gauss':
            # we sample single-event parameters from broad gaussian approximations of the posteriors
            if len(GWData) == 10:
                (
                    mus_s,
                    cho_s,
                    log_wts_l,
                    mus_l,
                    icovs_l,
                    log_dets_l,
                    cho_covs_l,
                    Tobs_np,
                    Nevs,
                    allnames,
                ) = GWData
                gmm_fit_coord_bounds = None
                gmm_fit_coord_names = None
                gmm_fit_transform_info = None
                gmm_fit_transform_mode = "legacy"
            elif len(GWData) == 13:
                (
                    mus_s,
                    cho_s,
                    log_wts_l,
                    mus_l,
                    icovs_l,
                    log_dets_l,
                    cho_covs_l,
                    gmm_fit_coord_bounds,
                    gmm_fit_coord_names,
                    gmm_fit_transform_info,
                    Tobs_np,
                    Nevs,
                    allnames,
                ) = GWData
                if not isinstance(gmm_fit_transform_info, dict):
                    raise ValueError("gmm_fit_transform_info must be a dict for bounded-transform GWData.")

                gmm_fit_transform_mode = gmm_fit_transform_info.get("transform_mode", "legacy")
                gmm_fit_coord_names = tuple(str(x) for x in gmm_fit_coord_names)
                gmm_fit_coord_bounds = np.asarray(gmm_fit_coord_bounds, dtype=np.float64)
            else:
                raise ValueError(
                    "GWData for sampling_GW='gauss' must have length 10 for legacy mode "
                    "or length 13 for event-bounded mode. Got length %s." % len(GWData)
                )
            wts_l = np.exp(log_wts_l)
            
        elif 'gmm' in sampling_GW or sampling_GW=='gumbel':
            # we sample single-event parameters from the actual single-event posteriors
            wts_l, mus_l, cho_covs_l, Tobs_np, Nevs, allnames = GWData
        else:
            raise ValueError('sampling_GW can be gmm, gmm_cat, gumbel,  gauss ')
            
        
    else:
        # gw data are single-event posterior samples
        # shape of each has to be n_events, n_samples
        m1det, m2det, d, spins, dL_prior, Tobs_np, allNsamples, where_compute, Nevs, allnames = GWData
        d_matrix_for_resolution = np.asarray(d, dtype=np.float64)
            
        if spin_model == 'default_gauss':
           chi1, chi2, cost1, cost2 = spins
        elif spin_model=='none':
            pass
        else:
            raise NotImplementedError()

    ## Injections data

    dLinj, m1inj, m2inj, spinsInj, lpdinj, Ndraw, Ndet_np, lp_incl_inj = InjData
    
    ndata_np = m1inj.shape[0]
    ninj_np  = m1inj.shape[1]
    Ndraw = float(np.asarray(Ndraw).reshape(()))
    
    if ndata_np != 1:
        raise NotImplementedError("Current JAX port assumes a single injection set (ndata_np==1).")
            
        
    # If you don't want spins in the selection effect, just tell the likelihood/core it's "none".
    spin_model_sel = spin_model if use_sel_spin else "none"
    
    # Optional: normalize spinsInj to [] when not used (helps avoid accidental misuse upstream)
    if not use_sel_spin:
        spinsInj = []
    

    
    N_DP_comp_max_np = int(N_DP_comp_max)
    N_DP_comp_global_np = int(N_DP_comp_max_np)
    #Nevs_np = np.atleast_1d(Nevs)
    Nevs_np = np.atleast_1d(Nevs).astype(np.int32)
    N_total = int(np.sum(Nevs_np))

    # Optional global anti-collapse transform for DPUC_TruncM2 shared weights.
    # If enabled, the final weights are shrunk toward uniform so that
    # max_k w_k <= DPUC_TruncM2_low_center_weight_cap by construction.
    DPUC_TruncM2_low_center_weight_reg = bool(DPUC_TruncM2_low_center_weight_reg)
    DPUC_TruncM2_low_center_weight_cap = float(DPUC_TruncM2_low_center_weight_cap)

    DPUC_TruncM2_weight_cap_reg = bool(DPUC_TruncM2_weight_cap_reg)
    DPUC_TruncM2_weight_cap = float(DPUC_TruncM2_weight_cap)

    if DPUC_TruncM2_weight_cap_reg:
        min_allowed_cap = 1.0 / float(N_DP_comp_max_np)
        if not (min_allowed_cap < DPUC_TruncM2_weight_cap <= 1.0):
            raise ValueError(
                "DPUC_TruncM2_weight_cap must satisfy 1/K < cap <= 1 "
                f"when DPUC_TruncM2_weight_cap_reg=1. "
                f"Got cap={DPUC_TruncM2_weight_cap}, K={N_DP_comp_max_np}."
            )

    if DPUC_TruncM2_low_center_weight_reg:
        min_allowed_cap = 1.0 / float(N_DP_comp_max_np)
        if not (min_allowed_cap < DPUC_TruncM2_low_center_weight_cap <= 1.0):
            raise ValueError(
                "DPUC_TruncM2_low_center_weight_cap is now used as a global "
                "max-component-weight cap. It must satisfy "
                f"1/K < cap <= 1, with K={N_DP_comp_max_np}. "
                f"Got {DPUC_TruncM2_low_center_weight_cap}."
            )


        
    if not pop_only:
        N = mus_l.shape[0]
        N_np = N 
        ngmm = mus_l.shape[1]
        ngmm_np = ngmm
        nd = mus_l.shape[2]
        nd_np = nd
        _log_setup('N:%s, max ngmm: %s, nd: %s '%(N_np, ngmm_np, nd_np))
        _log_setup('N evs is %s'%Nevs_np)
        _log_setup('Tobs is %s'%Tobs_np)
    else:
        N = m1det.shape[0] 
        N_np = N 
        Nsamples = m1det.shape[1]
        Nsamples_np = Nsamples 

        allNsamples_np = np.asarray(allNsamples, dtype=int)
        if allNsamples_np.ndim == 0:
            allNsamples_np = np.full(N, int(allNsamples_np), dtype=int)
        if allNsamples_np.shape != (N,):
            raise ValueError(f"allNsamples must have shape ({N},), got {allNsamples_np.shape}")

        if Nsamplesuse !=-1 :
            if Nsamplesuse>Nsamples_np:
                raise ValueError("Must use less samples than those available.")
            _log_setup("Nsamples_np available is %s, but %s will be used"%(Nsamples_np, Nsamplesuse))
            
            m1det, m2det, d = m1det[:, :Nsamplesuse], m2det[:, :Nsamplesuse], d[:, :Nsamplesuse]
            dL_prior = dL_prior[:, :Nsamplesuse]
            spins = np.asarray([s[:, :Nsamplesuse] for s in spins ])
            if spin_model == 'default_gauss':
               chi1, chi2, cost1, cost2 = chi1[:, :Nsamplesuse], chi2[:, :Nsamplesuse], cost1[:, :Nsamplesuse], cost2[:, :Nsamplesuse]

            allNsamples_np = np.minimum(allNsamples_np, Nsamplesuse)
            allNsamples = allNsamples_np

            Nsamples = m1det.shape[1]
            Nsamples_np = Nsamples 

        if np.any(allNsamples_np > Nsamples_np):
            raise ValueError("Some allNsamples entries exceed the posterior array width.")

        _log_setup("Posterior array width is ")
        _log_setup(Nsamples_np)
        _log_setup("Per-event finite sample counts are ")
        _log_setup(allNsamples_np)
        _log_setup('N:%s, padded n samples: %s '%(N_np, Nsamples_np))


        ### reshape

        if spin_model in ('default_gauss',):
            spins = np.stack([chi1, chi2, cost1, cost2], axis=1)  # (N,4)

             
        logd = np.log(d)
        
        NsamplesTot = N*Nsamples

        _log_setup("Reshaping samples to %s"%NsamplesTot)

        m1det_matrix = np.array(m1det, copy=True)
        m2det_matrix = np.array(m2det, copy=True)
        d_matrix = np.array(d, copy=True)
        
        m1det = m1det.reshape(NsamplesTot)
        m2det = m2det.reshape(NsamplesTot)
        d = d.reshape(NsamplesTot)
        logd = logd.reshape(NsamplesTot)
        dL_prior = dL_prior.reshape(NsamplesTot)
        
        # spins: if you store (Ne, S, nspin) -> flatten first two axes
        #spins = spins.reshape((NsamplesTot, spins.shape[-1]))
        spins = np.asarray(spins, dtype=np.float64)

        
        if spin_model == "none":
            spins = np.zeros((NsamplesTot, 0), dtype=np.float64)
        else:
            if spins.ndim == 3 and spins.shape[0] in (2, 4):
                # (nspin, N, S) -> (N, S, nspin)
                spins = np.moveaxis(spins, 0, -1)
        
            elif spins.ndim == 3 and spins.shape[1] in (2, 4):
                # (N, nspin, S) -> (N, S, nspin)
                spins = np.moveaxis(spins, 1, -1)
        
            elif spins.ndim == 3 and spins.shape[-1] in (2, 4):
                # already (N, S, nspin)
                pass
        
            else:
                raise ValueError(f"Unexpected posterior spin sample shape: {spins.shape}")
        
            spins = spins.reshape((NsamplesTot, spins.shape[-1]))

       


    
    logN = np.log(N)


    
    event_index = np.arange(N_np, dtype=int)
    Ttot = np.sum(Tobs_np)

    
    _log_setup('Injections: :%s, '%(ninj_np))

    _log_setup('ninj: :%s, %s datasets,'%(Ndet_np, ndata_np))

    coords = {'event_index': event_index}

    

    if mass_model in ('DPUC', 'DPUC_Primary', 'DPUC_TruncM2'):
        coords['component'] = np.arange(N_DP_comp_max_np, dtype=int)
        
        if mass_model == "DPUC_Primary":
            ndim_GMM = 1
        else:
            ndim_GMM = 2

        _log_setup('GMM dimension is %s'%ndim_GMM)
        
        coords['GMMdimension'] = np.arange(ndim_GMM, dtype=int)
        coords['GMMdimension_1'] = np.arange(ndim_GMM, dtype=int)
        coords['GMMdimension_2'] = np.arange(ndim_GMM, dtype=int)
        p = ndim_GMM*(ndim_GMM+1)//2  # packed length = 3 for n=2
        
        coords["packed_cholesky"] = np.arange(p)

    if pop_only:
        coords['nsamples'] = np.arange( Nsamples_np, dtype=int )
    else:
         coords['GWdimension'] = np.arange(nd_np, dtype=int)


    if params_fix is None:
        _log_setup('No values for parameters to fix passed. Default values will be used. If fixing parameters, check that the values are consistent. Values of fixed parameters:')
        _log_setup(PlanckFiducials)
        params_fix=PlanckFiducials


 
    ################################################
    # Build grids and other data-dependend quantities
    ################################################
    

    if find_z_bounds and not is_GP_dL:
        raise NotImplementedError("find_z_bounds is not ported in this NumPyro branch.")

    if mmin_inj != -1:
        raise NotImplementedError("mmin_inj prefiltering is not ported in this NumPyro branch.")

    is_dpuc_like_mass_model = mass_model in ('DPUC', 'DPUC_Primary', 'DPUC_TruncM2')
    
    if is_dpuc_like_mass_model and (not find_m_bounds):

        def _prior_scalar(name):
            if name not in priors:
                raise KeyError(
                    f"find_m_bounds=False for {mass_model} requires {name!r} "
                    "in the prior file."
                )

            val = priors[name]

            # Accept either:
            #   "DP_lowmu1": 1.5
            # or:
            #   "DP_lowmu1": [1.5]
            if isinstance(val, (list, tuple, np.ndarray)):
                if len(val) != 1:
                    raise ValueError(
                        f"{name!r} must be a scalar or a one-element list. "
                        f"Got {val!r}."
                    )
                val = val[0]

            return float(val)

        # --------------------------
        # Bounds: prior file only
        # --------------------------
        if mass_model in ("DPUC_Primary", "DPUC_TruncM2", ):
            if "DP_lowmu1" in priors and "DP_upmu1" in priors:
                lowmu1 = _prior_scalar("DP_lowmu1")
                upmu1 = _prior_scalar("DP_upmu1")
            else:
                lowmu1 = float(priors.get("m1_low", priors.get("mmin_DP"))[0])
                upmu1 = float(priors.get("mmax_DP", priors.get("m_high"))[1])

            if mass_model == "DPUC_Primary":
                lowmu2 = lowmu1
                upmu2 = upmu1
            elif "DP_lowmu2" in priors and "DP_upmu2" in priors:
                lowmu2 = _prior_scalar("DP_lowmu2")
                upmu2 = _prior_scalar("DP_upmu2")
            else:
                lowmu2 = float(priors.get("m2_low", priors.get("mmin_DP"))[0])
                upmu2 = float(priors.get("mmax_DP", priors.get("m_high"))[1])

            # DPUC_Primary/DPUC_TruncM2 GMM coordinates are x=log(m), while gates remain in physical m.
            if not (lowmu1 > 0.0 and upmu1 > lowmu1 and lowmu2 > 0.0 and upmu2 > lowmu2):
                raise ValueError(
                    f"Invalid positive physical mass bounds for DPUC_Primary/DPUC_TruncM2 before log: "
                    f"m1=({lowmu1},{upmu1}), m2=({lowmu2},{upmu2})"
                )
            lowmu1 = float(np.log(lowmu1)); upmu1 = float(np.log(upmu1))
            lowmu2 = float(np.log(lowmu2)); upmu2 = float(np.log(upmu2))
        else:
            lowmu1 = _prior_scalar("DP_lowmu1")
            upmu1  = _prior_scalar("DP_upmu1")
            lowmu2 = _prior_scalar("DP_lowmu2")
            upmu2  = _prior_scalar("DP_upmu2")

        if mass_model in ("DPUC_TruncM2") and DPUC_TruncM2_mu2_mode == "q_ratio":
            if DPUC_q_mu_bounds_source == "unit":
                q_mu_low = 1.0e-6
                q_mu_up = 1.0 - 1.0e-6
            else:
                raise ValueError(
                    "DPUC_TruncM2_mu2_mode='q_ratio' with find_m_bounds=False "
                    "requires DPUC_q_mu_bounds_source='unit'. No new prior-file key is used."
                )
            if DPUC_TruncM2_secondary_coord == "logitq":
                lowmu2 = float(logit(JAXBackend(), q_mu_low))
                upmu2 = float(logit(JAXBackend(), q_mu_up))

        if mass_model in ('DPUC',) and DPUC_q_coord == 'q' and (DPUC_q_mu_bounds_source == 'unit'):
            lowmu2 = 0.0
            upmu2 = 1.0

        if not np.isfinite(lowmu1) or not np.isfinite(upmu1) or upmu1 <= lowmu1:
            raise ValueError(
                f"Invalid prior-file coordinate-1 bounds: "
                f"DP_lowmu1={lowmu1}, DP_upmu1={upmu1}"
            )

        if not np.isfinite(lowmu2) or not np.isfinite(upmu2) or upmu2 <= lowmu2:
            raise ValueError(
                f"Invalid prior-file coordinate-2 bounds: "
                f"DP_lowmu2={lowmu2}, DP_upmu2={upmu2}"
            )


        # Metadata names expected later.
        lowmu1_inj = lowmu1
        upmu1_inj = upmu1
        lowmu2_inj = lowmu2
        upmu2_inj = upmu2



        # find_m_bounds=False: mean bounds come from explicit prior-file coordinate bounds.
        # For DPUC_Primary/DPUC_TruncM2 the sigma support is a log-mass width.
        # Keep lowsig=0 so the existing U=(upsig-lowsig) machinery uses that width.
        if mass_model in ("DPUC_Primary", "DPUC_TruncM2"):
            lowsig1 = 0.0
            upsig1 = upmu1 - lowmu1
            lowsig2 = 0.0
            upsig2 = upmu2 - lowmu2
        else:
            lowsig1 = lowmu1
            upsig1 = upmu1
            lowsig2 = lowmu2
            upsig2 = upmu2

        # Preserve user-requested floors before any overwrite.
        L_small_1_user = float(L_small_1)
        L_small_2_user = float(L_small_2)

        Lsmall_res_summary = None
        L_small_1_spacing = None
        L_small_2_spacing = None
        L_small_1_res = None
        L_small_2_res = None
        L_small_m1 = None
        L_small_m2 = None

        # --------------------------
        # L_small floors
        # --------------------------
        if L_small_source == "user":

            if L_small_1_user <= 0.0 or L_small_2_user <= 0.0:
                raise ValueError(
                    "L_small_source='user' requires positive "
                    "L_small_1 and L_small_2 in transformed coordinates."
                )


            L_small_1_data = L_small_1_user
            L_small_2_data = L_small_2_user

        else:

            if L_small_source not in ("spacing", "pe_res", "max"):
                raise ValueError(
                    f"Unknown L_small_source={L_small_source!r}. "
                    "Use 'spacing', 'pe_res', 'max', or 'user'."
                )

            # Compute the same scales as the automatic branch, but use ONLY
            # the L_small information. The automatic lowmu/upmu are discarded.
            if not pop_only:
                scales = find_mass_redshift_bounds_np(
                    _z_from_dL_numpy,
                    wts_l,
                    mus_l,
                    cho_covs_l,
                    priors["H0"],
                    priors["Om"],
                    priors["w0"],
                    priors["Xi0"],
                    priors["nXi0"],
                    int(N),
                    int(nd),
                    dLinj[0],
                    m1inj[0],
                    m2inj[0],
                    param=param,
                    integrate_dc=integrate_dc,
                    sampling_GW=sampling_GW,
                    trials=1000,
                    is_observed=False,
                    rng=np.random.default_rng(),
                    q_mbound_low=q_mbound_mu_low,
                    q_mbound_up=q_mbound_mu_up,
                    DP_bounds_source=DP_mu_bounds_source,
                    gmm_fit_transform_mode=gmm_fit_transform_mode,
                    gmm_fit_coord_bounds=(
                        None
                        if gmm_fit_coord_bounds is None
                        else np.asarray(gmm_fit_coord_bounds, dtype=np.float64)[:, :int(nd), :]
                    ),
                    gmm_fit_coord_names=(
                        None
                        if gmm_fit_coord_names is None
                        else tuple(gmm_fit_coord_names[:int(nd)])
                    ),
                )
            else:
                scales = find_mass_redshift_bounds_samples_np(
                    _z_from_dL_numpy,
                    m1det_matrix,
                    m2det_matrix,
                    d_matrix,
                    priors['H0'],
                    priors['Om'],
                    priors['w0'],
                    priors['Xi0'],
                    priors['nXi0'],
                    int(N),
                    dLinj[0],
                    m1inj[0],
                    m2inj[0],
                    param=param,
                    integrate_dc=integrate_dc,
                    trials=1000,
                    is_observed=False,
                    rng=np.random.default_rng(),
                    q_mbound_low=q_mbound_mu_low,
                    q_mbound_up=q_mbound_mu_up,
                    DP_bounds_source=DP_mu_bounds_source,
                )

            # PE samples are always in logit(q).  Only the DPUC
            # kernels may request population coordinate-2 in q.  Do not convert
            # these scale arrays for DPUC_Primary/DPUC_TruncM2,
            # because their physical-mass and sector-bound code expects raw logit(q).
            if mass_model in ('DPUC',) and DPUC_q_coord == 'q':
                scales = _convert_scales_coord2_logitq_to_q(
                    scales,
                    q_mbound_low=q_mbound_mu_low,
                    q_mbound_up=q_mbound_mu_up,
                    DP_bounds_source=DP_mu_bounds_source,
                    DPUC_q_mu_bounds_source=DPUC_q_mu_bounds_source,
                )

            L_small_1_spacing = scales["lMc_diff"]
            L_small_2_spacing = scales["lq_diff"]

            L_small_m1 = scales["m1_diff"]
            L_small_m2 = scales["m2_diff"]

            if mass_model in ("DPUC_Primary", "DPUC_TruncM2", ):
                L_small_1_spacing = scales["m1_diff"] / max(np.exp(lowmu1), 1.0e-300)
                L_small_2_spacing = L_small_1_spacing if (mass_model == "DPUC_Primary") else scales["m2_diff"] / max(np.exp(lowmu2), 1.0e-300)

            if L_small_source in ("pe_res", "max"):
                if mass_model in ("DPUC_Primary", "DPUC_TruncM2", ):
                    L_small_1_res, L_small_2_res, Lsmall_res_summary = (
                        _estimate_L_small_from_physical_m1m2_event_resolution_np(
                            scales["res_logMc_by_event"],
                            scales["res_logitq_by_event"],
                            percentile=L_small_res_percentile,
                            factor=L_small_res_factor,
                            min_samples=10,
                            shared_mode=DPUC_physical_Lsmall_shared_mode,
                            shared=(mass_model == "DPUC_Primary" and False),
                            primary_only=(mass_model == "DPUC_Primary" and True),
                        )
                    )
                else:
                    L_small_1_res, L_small_2_res, Lsmall_res_summary = (
                        estimate_L_small_from_transformed_event_resolution_np(
                            scales["res_logMc_by_event"],
                            scales["res_logitq_by_event"],
                            percentile=L_small_res_percentile,
                            factor=L_small_res_factor,
                            min_samples=10,
                        )
                    )
            else:
                L_small_1_res = None
                L_small_2_res = None
                Lsmall_res_summary = None

            if L_small_source == "spacing":
                L_small_1_data = L_small_1_spacing
                L_small_2_data = L_small_2_spacing

            elif L_small_source == "pe_res":
                L_small_1_data = L_small_1_res
                L_small_2_data = L_small_2_res

            elif L_small_source == "max":
                L_small_1_data = max(L_small_1_spacing, L_small_1_res)
                L_small_2_data = max(L_small_2_spacing, L_small_2_res)


            if L_small_1_data is None or L_small_2_data is None:
                raise ValueError(
                    f"Automatic L_small failed with L_small_source={L_small_source!r}: "
                    f"L_small_1_data={L_small_1_data}, L_small_2_data={L_small_2_data}"
                )

            if L_small_1_data <= 0.0 or L_small_2_data <= 0.0:
                raise ValueError(
                    f"Automatic L_small failed with L_small_source={L_small_source!r}: "
                    f"L_small_1={L_small_1_data}, L_small_2={L_small_2_data}"
                )


        # User hand-floor logic, same convention as the automatic branch:
        # positive input L_small_* overrides the automatic data-driven floor.
        if L_small_1_user > 0.0:
            L_small_1_final = L_small_1_user
        else:
            L_small_1_final = L_small_1_data

        if L_small_2_user > 0.0:
            L_small_2_final = L_small_2_user
        else:
            L_small_2_final = L_small_2_data


        L_small_1 = float(L_small_1_final)
        L_small_2 = float(L_small_2_final)

        _log_setup("\nUsing prior-file DP-GMM coordinate bounds.")
        _log_setup("find_m_bounds=False")
        _log_setup("lowmu1=%s, upmu1=%s, lowmu2=%s, upmu2=%s" % (lowmu1, upmu1, lowmu2, upmu2))
        _log_setup("lowsig1=%s, upsig1=%s, lowsig2=%s, upsig2=%s" % (lowsig1, upsig1, lowsig2, upsig2))
        _log_setup("L_small source: %s" % L_small_source)
        _log_setup("DP_mu_bounds_source: %s" % DP_mu_bounds_source)
        _log_setup("DP_sig_bounds_source: %s" % DP_sig_bounds_source)
        _log_setup("L_small_res_percentile=%s, L_small_res_factor=%s" % (
            L_small_res_percentile,
            L_small_res_factor,
        ))

        if L_small_source in ("spacing", "pe_res", "max"):
            _log_setup("Automatic lowmu/upmu from scales were computed but discarded.")



            if L_small_m1 is not None and L_small_m2 is not None:
                _log_setup("L_small_m1=%s, L_small_m2=%s" % (L_small_m1, L_small_m2))

            if Lsmall_res_summary is not None:
                _log_setup("PE-resolution summary:")
                for k, v in Lsmall_res_summary.items():
                    _log_setup("  %s: %s" % (k, v))

        _log_setup("Transformed-coordinate user floor:")
        _log_setup("  input L_small_1 in logMc    : %s" % L_small_1_user)
        _log_setup("  input L_small_2 in %s : %s" % ("q" if DPUC_q_coord == "q" else "logit(q)", L_small_2_user))



        
    if (is_dpuc_like_mass_model and find_m_bounds):

        _log_setup("\nFinding prior range for DP-GMM.")

        if not pop_only:
            scales_mu = find_mass_redshift_bounds_np(
                _z_from_dL_numpy,
                wts_l,
                mus_l,
                cho_covs_l,
                priors["H0"],
                priors["Om"],
                priors["w0"],
                priors["Xi0"],
                priors["nXi0"],
                int(N),
                int(nd),
                dLinj[0],
                m1inj[0],
                m2inj[0],
                param=param,
                integrate_dc=integrate_dc,
                sampling_GW=sampling_GW,
                trials=1000,
                is_observed=False,
                rng=np.random.default_rng(),
                q_mbound_low=q_mbound_mu_low,
                q_mbound_up=q_mbound_mu_up,
                DP_bounds_source=DP_mu_bounds_source,
                gmm_fit_transform_mode=gmm_fit_transform_mode,
                gmm_fit_coord_bounds=(
                    None
                    if gmm_fit_coord_bounds is None
                    else np.asarray(gmm_fit_coord_bounds, dtype=np.float64)[:, :int(nd), :]
                ),
                gmm_fit_coord_names=(
                    None
                    if gmm_fit_coord_names is None
                    else tuple(gmm_fit_coord_names[:int(nd)])
                ),
            )
            
        else:


            scales_mu = find_mass_redshift_bounds_samples_np(
                _z_from_dL_numpy,
                m1det_matrix,
                m2det_matrix,
                d_matrix,
                priors['H0'],
                priors['Om'],
                priors['w0'],
                priors['Xi0'],
                priors['nXi0'],
                int(N),
                dLinj[0],
                m1inj[0],
                m2inj[0],
                param=param,
                integrate_dc=integrate_dc,
                trials=1000,
                is_observed=False,
                rng=np.random.default_rng(),
                q_mbound_low=q_mbound_mu_low,
                q_mbound_up=q_mbound_mu_up,
                DP_bounds_source=DP_mu_bounds_source,
            )

        if not pop_only:
            scales_sig = find_mass_redshift_bounds_np(
                _z_from_dL_numpy,
                wts_l,
                mus_l,
                cho_covs_l,
                priors["H0"],
                priors["Om"],
                priors["w0"],
                priors["Xi0"],
                priors["nXi0"],
                int(N),
                int(nd),
                dLinj[0],
                m1inj[0],
                m2inj[0],
                param=param,
                integrate_dc=integrate_dc,
                sampling_GW=sampling_GW,
                trials=1000,
                is_observed=False,
                rng=np.random.default_rng(),
                q_mbound_low=q_mbound_sig_low,
                q_mbound_up=q_mbound_sig_up,
                DP_bounds_source=DP_sig_bounds_source,
                gmm_fit_transform_mode=gmm_fit_transform_mode,
                gmm_fit_coord_bounds=(
                    None
                    if gmm_fit_coord_bounds is None
                    else np.asarray(gmm_fit_coord_bounds, dtype=np.float64)[:, :int(nd), :]
                ),
                gmm_fit_coord_names=(
                    None
                    if gmm_fit_coord_names is None
                    else tuple(gmm_fit_coord_names[:int(nd)])
                ),
            )
        else:
            scales_sig = find_mass_redshift_bounds_samples_np(
                _z_from_dL_numpy,
                m1det_matrix,
                m2det_matrix,
                d_matrix,
                priors['H0'],
                priors['Om'],
                priors['w0'],
                priors['Xi0'],
                priors['nXi0'],
                int(N),
                dLinj[0],
                m1inj[0],
                m2inj[0],
                param=param,
                integrate_dc=integrate_dc,
                trials=1000,
                is_observed=False,
                rng=np.random.default_rng(),
                q_mbound_low=q_mbound_sig_low,
                q_mbound_up=q_mbound_sig_up,
                DP_bounds_source=DP_sig_bounds_source,
            )

        if mass_model in ("DPUC_TruncM2") and DPUC_TruncM2_mu2_mode == "q_ratio":
            qratio_scales_mu = _convert_scales_coord2_logitq_to_q(
                scales_mu,
                q_mbound_low=q_mbound_mu_low,
                q_mbound_up=q_mbound_mu_up,
                DP_bounds_source=DP_mu_bounds_source,
                DPUC_q_mu_bounds_source=DPUC_q_mu_bounds_source,
            )
            q_mu_low = float(np.clip(qratio_scales_mu["lq_min_data"], 1.0e-6, 1.0 - 1.0e-6))
            q_mu_up = float(np.clip(qratio_scales_mu["lq_max_data"], 1.0e-6, 1.0 - 1.0e-6))
            if not (np.isfinite(q_mu_low) and np.isfinite(q_mu_up) and 0.0 < q_mu_low < q_mu_up < 1.0):
                raise ValueError(
                    "Invalid q_mu bounds for DPUC_TruncM2_mu2_mode='q_ratio': "
                    f"q_mu_low={q_mu_low}, q_mu_up={q_mu_up}"
                )

        # PE samples are always stored as logit(q).  Convert to q only for the
        # DPUC kernels whose population coordinate is q.
        # DPUC_Primary/DPUC_TruncM2/ must keep raw logit(q) here:
        # their physical-mass reconstruction, PE-resolution floors, q_ratio bounds,
        # and TwoSector event split all assume logit(q) input.
        if mass_model in ('DPUC',) and DPUC_q_coord == 'q':
            scales_mu = _convert_scales_coord2_logitq_to_q(
                scales_mu,
                q_mbound_low=q_mbound_mu_low,
                q_mbound_up=q_mbound_mu_up,
                DP_bounds_source=DP_mu_bounds_source,
                DPUC_q_mu_bounds_source=DPUC_q_mu_bounds_source,
            )
            scales_sig = _convert_scales_coord2_logitq_to_q(
                scales_sig,
                q_mbound_low=q_mbound_sig_low,
                q_mbound_up=q_mbound_sig_up,
                DP_bounds_source=DP_sig_bounds_source,
                DPUC_q_mu_bounds_source="inferred",
            )

        if mass_model in ("DPUC_Primary", "DPUC_TruncM2", ):
            # DPUC_Primary/DPUC_TruncM2 one-body GMM coordinates are x=log(m).
            # Bound the physical mass range first, then transform to log mass.
            prior_m1_low = float(priors.get("m1_low", priors.get("mmin_DP"))[0])
            prior_m2_low = float(priors.get("m2_low", priors.get("mmin_DP"))[0])
            prior_m_low_shared = float(priors.get("mmin_DP", [min(prior_m1_low, prior_m2_low)])[0])
            prior_m_high = float(priors.get("mmax_DP", priors.get("m_high"))[1])

            raw_lowmu1_m = float(scales_mu['m1_min_data'])
            raw_upmu1_m = float(scales_mu['m1_max_data'])
            raw_lowmu2_m = float(scales_mu['m2_min_data'])
            raw_upmu2_m = float(scales_mu['m2_max_data'])

            if mass_model == "DPUC_Primary":
                lowmu1_m = max(min(raw_lowmu1_m, raw_lowmu2_m), prior_m_low_shared)
                upmu1_m = min(max(raw_upmu1_m, raw_upmu2_m), prior_m_high)
                lowmu2_m = lowmu1_m
                upmu2_m = upmu1_m
            else:
                lowmu1_m = max(raw_lowmu1_m, prior_m1_low)
                upmu1_m = min(raw_upmu1_m, prior_m_high)
                lowmu2_m = max(raw_lowmu2_m, prior_m2_low)
                upmu2_m = min(raw_upmu2_m, prior_m_high)

            if not (np.isfinite(lowmu1_m) and np.isfinite(upmu1_m) and lowmu1_m > 0.0 and upmu1_m > lowmu1_m):
                raise ValueError(f"Invalid DPUC_Primary/DPUC_TruncM2 m1 physical mu bounds after clipping: low={lowmu1_m}, up={upmu1_m}")
            if not (np.isfinite(lowmu2_m) and np.isfinite(upmu2_m) and lowmu2_m > 0.0 and upmu2_m > lowmu2_m):
                raise ValueError(f"Invalid DPUC_Primary/DPUC_TruncM2 m2 physical mu bounds after clipping: low={lowmu2_m}, up={upmu2_m}")

            lowmu1 = float(np.log(lowmu1_m)); upmu1 = float(np.log(upmu1_m))
            if mass_model in ("DPUC_TruncM2") and DPUC_TruncM2_secondary_coord == "logitq":
                if q_mu_low is None or q_mu_up is None:
                    raise ValueError("DPUC_TruncM2_secondary_coord='logitq' requires q_mu bounds.")
                lowmu2 = float(logit(JAXBackend(), q_mu_low)); upmu2 = float(logit(JAXBackend(), q_mu_up))
            else:
                lowmu2 = float(np.log(lowmu2_m)); upmu2 = float(np.log(upmu2_m))

            # Sigma support is a width in log-mass, not a physical-mass endpoint.
            raw_lowsig1_m = float(scales_sig['m1_min_data'])
            raw_upsig1_m = float(scales_sig['m1_max_data'])
            raw_lowsig2_m = float(scales_sig['m2_min_data'])
            raw_upsig2_m = float(scales_sig['m2_max_data'])

            if mass_model == "DPUC_Primary":
                sig_low_m = max(min(raw_lowsig1_m, raw_lowsig2_m), prior_m_low_shared)
                sig_up_m = min(max(raw_upsig1_m, raw_upsig2_m), prior_m_high)
                if not (sig_low_m > 0.0 and sig_up_m > sig_low_m):
                    sig_low_m, sig_up_m = lowmu1_m, upmu1_m
                lowsig1 = 0.0
                upsig1 = float(np.log(sig_up_m) - np.log(sig_low_m))
                lowsig2 = 0.0
                upsig2 = upsig1
            else:
                sig1_low_m = max(raw_lowsig1_m, prior_m1_low)
                sig1_up_m = min(raw_upsig1_m, prior_m_high)
                sig2_low_m = max(raw_lowsig2_m, prior_m2_low)
                sig2_up_m = min(raw_upsig2_m, prior_m_high)
                if not (sig1_low_m > 0.0 and sig1_up_m > sig1_low_m):
                    sig1_low_m, sig1_up_m = lowmu1_m, upmu1_m
                if not (sig2_low_m > 0.0 and sig2_up_m > sig2_low_m):
                    sig2_low_m, sig2_up_m = lowmu2_m, upmu2_m
                lowsig1 = 0.0
                upsig1 = float(np.log(sig1_up_m) - np.log(sig1_low_m))
                lowsig2 = 0.0
                upsig2 = float(np.log(sig2_up_m) - np.log(sig2_low_m))

            if mass_model in ("DPUC_TruncM2") and DPUC_TruncM2_secondary_coord == "logitq":
                lowsig2 = 0.0
                upsig2 = float(upmu2 - lowmu2)



            if not (np.isfinite(upsig1) and upsig1 > 0.0):
                raise ValueError(f"Invalid DPUC_Primary/DPUC_TruncM2 m1 log-sigma width support: upsig1={upsig1}")
            if not (np.isfinite(upsig2) and upsig2 > 0.0):
                raise ValueError(f"Invalid DPUC_Primary/DPUC_TruncM2 m2 log-sigma width support: upsig2={upsig2}")
        else:
            lowmu1 = scales_mu['lMc_min_data']
            upmu1 = scales_mu['lMc_max_data']

            lowmu2 = scales_mu['lq_min_data']
            upmu2 = scales_mu['lq_max_data']

            lowsig1 = scales_sig['lMc_min_data']
            upsig1 = scales_sig['lMc_max_data']

            lowsig2 = scales_sig['lq_min_data']
            upsig2 = scales_sig['lq_max_data']

        # lowmu2, upmu2 = _restrict_logitq_mu_bounds_to_event_intersection(
        #     lowmu2=lowmu2,
        #     upmu2=upmu2,
        #     gmm_fit_transform_mode=gmm_fit_transform_mode,
        #     gmm_fit_coord_bounds=gmm_fit_coord_bounds,
        #     gmm_fit_coord_names=gmm_fit_coord_names,
        # )



        lowmu1_inj = scales_mu['lMc_min_inj']
        upmu1_inj = scales_mu['lMc_max_inj']

        lowmu2_inj = scales_mu['lq_min_inj']
        upmu2_inj = scales_mu['lq_max_inj']


        # ------------------------------------------------------------------
        # L_small floors in transformed DPUC coordinates
        # ------------------------------------------------------------------
        # Existing spacing-based estimates
        L_small_1_spacing = scales_mu["lMc_diff"]
        L_small_2_spacing = scales_mu["lq_diff"]

        L_small_m1 = scales_mu["m1_diff"]
        L_small_m2 = scales_mu["m2_diff"]

        if mass_model in ("DPUC_Primary", "DPUC_TruncM2", ):
            # Convert physical-mass spacing floors to log-mass floors.  These are
            # usually negligible compared to PE-resolution floors.  For the new
            # DPUC_TruncM2 logitq secondary coordinate, keep the native logit(q) spacing.
            L_small_1_spacing = scales_mu["m1_diff"] / max(np.exp(lowmu1), 1.0e-300)
            if mass_model in ("DPUC_TruncM2") and DPUC_TruncM2_secondary_coord == "logitq":
                L_small_2_spacing = scales_mu["lq_diff"]
            else:
                L_small_2_spacing = L_small_1_spacing if (mass_model == "DPUC_Primary") else scales_mu["m2_diff"] / max(np.exp(lowmu2), 1.0e-300)

        # Preserve user-requested physical floors before overwriting L_small_*
        L_small_1_user = float(L_small_1)
        L_small_2_user = float(L_small_2)

        # Optional PE-resolution estimates
        if L_small_source in ("pe_res", "max"):
            if mass_model in ("DPUC_Primary", "DPUC_TruncM2", ):
                L_small_1_res, L_small_2_res, Lsmall_res_summary = (
                    _estimate_L_small_from_physical_m1m2_event_resolution_np(
                        scales_mu["res_logMc_by_event"],
                        scales_mu["res_logitq_by_event"],
                        percentile=L_small_res_percentile,
                        factor=L_small_res_factor,
                        min_samples=10,
                        shared_mode=DPUC_physical_Lsmall_shared_mode,
                        shared=(mass_model == "DPUC_Primary" and False),
                        primary_only=(mass_model == "DPUC_Primary" and True),
                        secondary_coord=("logitq" if (mass_model in ("DPUC_TruncM2") and DPUC_TruncM2_secondary_coord == "logitq") else "logm2"),
                    )
                )
            else:
                L_small_1_res, L_small_2_res, Lsmall_res_summary = (
                    estimate_L_small_from_transformed_event_resolution_np(
                        scales_mu["res_logMc_by_event"],
                        scales_mu["res_logitq_by_event"],
                        percentile=L_small_res_percentile,
                        factor=L_small_res_factor,
                        min_samples=10,
                    )
                )
        else:
            L_small_1_res = None
            L_small_2_res = None
            Lsmall_res_summary = None

        if L_small_source == "spacing":
            L_small_1_data = L_small_1_spacing
            L_small_2_data = L_small_2_spacing

        elif L_small_source == "pe_res":
            L_small_1_data = L_small_1_res
            L_small_2_data = L_small_2_res

        elif L_small_source == "max":
            L_small_1_data = max(L_small_1_spacing, L_small_1_res)
            L_small_2_data = max(L_small_2_spacing, L_small_2_res)

        elif L_small_source == "user":
            if L_small_1_user <= 0.0 or L_small_2_user <= 0.0:
                raise ValueError(
                    "L_small_source='user' requires positive "
                    "L_small_1 and L_small_2 in transformed coordinates."
                )

            L_small_1_data = L_small_1_user
            L_small_2_data = L_small_2_user

        else:
            raise ValueError(
                f"Unknown L_small_source={L_small_source!r}. "
                "Use 'spacing', 'pe_res', 'max', or 'user'."
            )


        _log_setup("Mass/redshift DP-GMM prior values found, overwriting default:")
        _log_setup("lowmu1=%s, upmu1=%s, lowmu2=%s, upmu2=%s" % (lowmu1, upmu1, lowmu2, upmu2))
        _log_setup("lowsig1=%s, upsig1=%s, lowsig2=%s, upsig2=%s" % (lowsig1, upsig1, lowsig2, upsig2))
        _log_setup("L_small source: %s" % L_small_source)
        _log_setup("DP_mu_bounds_source: %s" % DP_mu_bounds_source)
        _log_setup("DP_sig_bounds_source: %s" % DP_sig_bounds_source)
        _log_setup("L_small_res_percentile=%s, L_small_res_factor=%s" % (
            L_small_res_percentile,
            L_small_res_factor,
        ))


        _log_setup("L_small_m1=%s, L_small_m2=%s" % (L_small_m1, L_small_m2))

        if Lsmall_res_summary is not None:
            _log_setup("PE-resolution summary:")
            for k, v in Lsmall_res_summary.items():
                _log_setup("  %s: %s" % (k, v))

        # ------------------------------------------------------------------
        # Optional user hand-floor directly in transformed DPUC coordinates.
        #
        # New interpretation:
        #   L_small_1 input = direct floor in log(Mc)
        #   L_small_2 input = direct floor in logit(q)
        #
        # If user provides a positive value, use it exactly.
        # Otherwise use the automatic data-driven floor.
        # ------------------------------------------------------------------
        if L_small_1_user > 0:
            L_small_1_final = L_small_1_user
        else:
            L_small_1_final = L_small_1_data
        
        if L_small_2_user > 0:
            L_small_2_final = L_small_2_user
        else:
            L_small_2_final = L_small_2_data
        
        if mass_model in ("DPUC_Primary", "DPUC_TruncM2",):
            _log_setup("Physical-mass user floor:")
            _log_setup("  input L_small_1 in m1       : %s" % L_small_1_user)
            _log_setup("  input L_small_2 in m2       : %s" % L_small_2_user)
            _log_setup("  final physical floors       : L1=%s, L2=%s" % (
                L_small_1_final, L_small_2_final
            ))
        else:
            _log_setup("Transformed-coordinate user floor:")
            _log_setup("  input L_small_1 in logMc    : %s" % L_small_1_user)
            _log_setup("  input L_small_2 in %s : %s" % ("q" if DPUC_q_coord == "q" else "logit(q)", L_small_2_user))
            _log_setup("  final transformed floors    : L1=%s, L2=%s" % (
                L_small_1_final, L_small_2_final
            ))

        
        _log_setup("Transformed-coordinate user floor:")

        L_small_1 = L_small_1_final
        L_small_2 = L_small_2_final




    
        
    ################################################
    # Redshift grid
        
    if z_grid_mode=='cst':

        _log_setup("Using pre-defined z grid with 1200 points")
        z_nodes = z_nodes_jax

    elif z_grid_mode=='man':

        _log_setup("Using custom z grid with %s points"%zres)
        z_nodes = make_z_grid_jax( zres )

    else:
        raise ValueError()

    _log_setup("Min, max redshift grid: %s, %s"%(z_nodes.min(), z_nodes.max()))


    
    # ------------------------------------------------
    # GP dL: checks, redshift bounds, and optional Gibbs-resolution curve
    # ------------------------------------------------
    x_res_centers = None
    r_res_centers = None
    z_gp_nodes = None
    z_gp_fine = None

    if is_GP_dL:
        if not invert_dL_GP:
            raise ValueError("GP dL branch currently requires invert_dL_GP=True")

        if monotonicity != "mono_reparam":
            raise NotImplementedError(
                "NumPyro GP dL port currently supports monotonicity='mono_reparam' only"
            )

        if GP_kernel not in ("gibbs_resolution", "stationary", "matern52", "matern"):
            raise ValueError(f"Unknown GP_kernel={GP_kernel!r}")

        bkNP = NPBackend()

        def _z_from_dL_np(dL_, H0_, Om_, w0_, Xi0_, nXi0_):
            return np.asarray(
                cosmo.z_from_dL(
                    bkNP,
                    np.asarray(dL_, dtype=np.float64),
                    H0=float(H0_),
                    Om=float(Om_),
                    w0=float(w0_),
                    Xi0=float(Xi0_),
                    nXi0=float(nXi0_),
                    param=param,
                    integrate_dc=integrate_dc,
                ),
                dtype=np.float64,
            )

        # ------------------------------------------------
        # PyMC-equivalent redshift-bound finding.
        # This is independent of GP kernel.
        # ------------------------------------------------
        if find_z_bounds:
            _log_setup("Finding GP redshift bounds from reference prior / PE distances...")

            H0_range = priors["H0"]
            Om_range = priors["Om"]
            w0_range = priors["w0"]
            Xi0_range = priors["Xi0"]
            nXi0_range = priors["nXi0"]

            # Injection range: PyMC checks corners of cosmology/MG prior box.
            z_min_inj = 1.0e100
            z_max_inj = 0.0

            for H0_ in (H0_range[0], H0_range[1]):
                for Om_ in (Om_range[0], Om_range[1]):
                    for w0_ in (w0_range[0], w0_range[1]):
                        for Xi0_ in (Xi0_range[0], Xi0_range[1]):
                            for nXi0_ in (nXi0_range[0], nXi0_range[1]):
                                z_inj_ = _z_from_dL_np(
                                    np.squeeze(dLinj),
                                    H0_,
                                    Om_,
                                    w0_,
                                    Xi0_,
                                    nXi0_,
                                )
                                z_min_inj = min(z_min_inj, float(np.min(z_inj_)))
                                z_max_inj = max(z_max_inj, float(np.max(z_inj_)))

            _log_setup("min, max injection redshift:", z_min_inj, z_max_inj)

            # Data range: random one draw per event, random cosmology in prior box.
            rng_bounds = np.random.default_rng(123)
            z_maxs = []
            z_mins = []
            z_diffs = []
            z_spans = []

            trials_bounds = 1000

            for _ in range(trials_bounds):
                if not pop_only:
                    Xwhite = rng_bounds.standard_normal((int(N), int(nd)))
                    samples = nutils.sample_from_per_event_gmm(
                        wts_l,
                        mus_l,
                        cho_covs_l,
                        Xwhite,
                        rng=rng_bounds,
                    )
                    d_nodes_trial = np.exp(samples[:, 2])
                else:
                    d_samples = np.asarray(d_matrix_for_resolution, dtype=np.float64)
                    n_events, n_samples = d_samples.shape
                    idx = rng_bounds.integers(0, n_samples, size=n_events)
                    d_nodes_trial = d_samples[np.arange(n_events), idx]

                H0_ = rng_bounds.uniform(*H0_range) if H0_range[0] != H0_range[1] else H0_range[0]
                Om_ = rng_bounds.uniform(*Om_range) if Om_range[0] != Om_range[1] else Om_range[0]
                w0_ = rng_bounds.uniform(*w0_range) if w0_range[0] != w0_range[1] else w0_range[0]
                Xi0_ = rng_bounds.uniform(*Xi0_range) if Xi0_range[0] != Xi0_range[1] else Xi0_range[0]
                nXi0_ = (
                    rng_bounds.uniform(*nXi0_range)
                    if nXi0_range[0] != nXi0_range[1]
                    else nXi0_range[0]
                )

                z_data = _z_from_dL_np(d_nodes_trial, H0_, Om_, w0_, Xi0_, nXi0_)
                z_data = np.asarray(z_data, dtype=np.float64)

                z_data_max = float(np.quantile(z_data, 0.99))
                z_data_min = float(np.quantile(z_data, 0.01))

                z_maxs.append(z_data_max)
                z_mins.append(z_data_min)

                z_span_trial = float(np.quantile(z_data, 0.95) - np.quantile(z_data, 0.05))
                z_spans.append(z_span_trial)

                z_sorted = np.sort(z_data)
                tol = 1e-12
                z_sorted = z_sorted[np.insert(np.diff(z_sorted) > tol, 0, True)]
                dz_pos = np.diff(z_sorted)
                dz_pos = dz_pos[dz_pos > tol]
                if dz_pos.size > 0:
                    z_diffs.append(float(np.mean(dz_pos)))

            z_max_data = max(z_maxs)
            z_min_data = min(z_mins)

            z_diff = 2.0 * float(np.quantile(z_diffs, 0.05)) if len(z_diffs) > 0 else None
            z_span = float(np.quantile(z_spans, 0.95)) if len(z_spans) > 0 else None

            _log_setup("min, max data redshift:", z_min_data, z_max_data)
            _log_setup("data min redshift separation:", z_diff)
            _log_setup("data max redshift separation:", z_span)

            old_bounds = (zmin_a, zmin_b, zmid_b, zmax_c)

            min_z = z_min_inj
            max_z = z_max_inj

            zmin_b = min(min_z, z_min_data)
            zmin_a = min(zmin_a, min(min_z, z_min_data))
            zmid_b = min(zmid_b, z_max_data)
            zmax_c = max(zmax_c, max(z_max_data, max_z)) * 1.1

            _log_setup("Redshift values found, overwriting default:")
            _log_setup("  old zmin_a, zmin_b, zmid_b, zmax_c =", old_bounds)
            _log_setup("  new zmin_a, zmin_b, zmid_b, zmax_c =", (zmin_a, zmin_b, zmid_b, zmax_c))

            z_max_mono = max(z_max_data, max_z)

            if z_diff is not None:
                if ell_min > 0:
                    _log_setup("z_diff found to be %s" % z_diff)
                    _log_setup("Min length scale passed by hand = %s. Using max(ell_min, z_diff)" % ell_min)
                    ell_min = max(z_diff, ell_min)
                else:
                    ell_min = z_diff

            if z_span is not None:
                if ell_max > 0:
                    _log_setup("z_span found to be %s" % z_span)
                    _log_setup("Max length scale passed by hand = %s. Using max(ell_max, z_span)" % ell_max)
                    ell_max = max(z_span, ell_max)
                else:
                    _log_setup("z_span found to be %s" % z_span)
                    _log_setup("Using this z_span as ell_max.")
                    ell_max = z_span

            _log_setup(f"ell_min:     {ell_min:.6g}")
            _log_setup(f"ell_max:     {ell_max:.6g}")
            _log_setup(f"z_max_mono:  {z_max_mono:.6g}")

        # ------------------------------------------------
        # Gibbs-resolution curve only for the Gibbs kernel.
        # This is separate from redshift-bound finding.
        # ------------------------------------------------
        if GP_kernel == "gibbs_resolution":
            _log_setup("Building Gibbs-resolution curve for GP dL")

            if not pop_only:
                res = nutils.find_event_redshift_resolution(
                    wts_l,
                    mus_l,
                    cho_covs_l,
                    priors["H0"],
                    priors["Om"],
                    priors["w0"],
                    priors["Xi0"],
                    priors["nXi0"],
                    int(N),
                    int(nd),
                    dLinj,
                    _z_from_dL_np,
                    sampling_GW,
                    trials=1000,
                    rng=np.random.default_rng(123),
                )
            else:
                res = nutils.find_event_redshift_resolution_from_dL_samples(
                    priors["H0"],
                    priors["Om"],
                    priors["w0"],
                    priors["Xi0"],
                    priors["nXi0"],
                    dLinj,
                    d_matrix_for_resolution,
                    _z_from_dL_np,
                    trials=1000,
                    rng=np.random.default_rng(123),
                )

            x_evt = np.asarray(res["x_median"], dtype=np.float64)
            w_evt = np.asarray(res["x_w68"], dtype=np.float64)

            mask_valid = np.isfinite(x_evt) & np.isfinite(w_evt) & (w_evt > 0.0)
            x_evt = x_evt[mask_valid]
            w_evt = w_evt[mask_valid]

            if len(x_evt) == 0 or len(w_evt) == 0 or np.all(w_evt <= 0):
                _log_setup("WARNING: no usable eventwise resolution widths; using debug reference curve.")
                x_res_centers, r_res_centers = nutils.reference_resolution_curve_debug()
            else:
                x_res_centers, r_res_centers = nutils.estimate_resolution_curve_binned(
                    x_evt,
                    w_evt,
                    nbins=res_nbins,
                    q=res_q,
                )

            if len(x_res_centers) < 2:
                raise ValueError(
                    "Need at least two resolution bins for right-tail linear extrapolation. "
                    "Increase event count, reduce res_nbins, or lower min_per_bin."
                )

            x_res_centers_raw = np.asarray(x_res_centers, dtype=np.float64).copy()
            r_res_centers_raw = np.asarray(r_res_centers, dtype=np.float64).copy()

            if smooth_GP_resolution_loglinear:
                (
                    x_res_centers,
                    r_res_centers,
                    res_loglinear_a,
                    res_loglinear_b,
                ) = smooth_resolution_curve_loglinear_np(
                    x_res_centers_raw,
                    r_res_centers_raw,
                )

                _log_setup("Using log-linear smooth Gibbs-resolution curve:")
                _log_setup(
                    f"  log r(x) = a + b x with "
                    f"a={res_loglinear_a:.6g}, b={res_loglinear_b:.6g}"
                )
                _log_setup("  equivalently r(z) = exp(a) * (1 + z)^b")

            if fout is not None:
                os.makedirs(fout, exist_ok=True)

                np.save(os.path.join(fout, "x_resolution_centers.npy"), x_res_centers)
                np.save(os.path.join(fout, "r_resolution_centers.npy"), r_res_centers)
                np.save(os.path.join(fout, "x_resolution_events.npy"), x_evt)
                np.save(os.path.join(fout, "w_resolution_events.npy"), w_evt)

                if smooth_GP_resolution_loglinear:
                    np.save(os.path.join(fout, "x_resolution_centers_raw.npy"), x_res_centers_raw)
                    np.save(os.path.join(fout, "r_resolution_centers_raw.npy"), r_res_centers_raw)

            _log_setup("Resolution curve for Gibbs GP:")
            _log_setup(f"  Number of events used: {len(x_evt)}")
            _log_setup(f"  Number of bins kept:   {len(x_res_centers)}")
            _log_setup(f"  res_nbins requested:   {res_nbins}")
            _log_setup(f"  res_q used:            {res_q}")
            if len(x_evt) > 0:
                _log_setup(f"  x range events:        [{x_evt.min():.6g}, {x_evt.max():.6g}]")
                _log_setup(f"  z range events:        [{np.expm1(x_evt.min()):.6g}, {np.expm1(x_evt.max()):.6g}]")
                _log_setup(f"  r range bins:          [{r_res_centers.min():.6g}, {r_res_centers.max():.6g}]")
            else:
                _log_setup("  x range events:        EMPTY")

            _log_setup(f"  x centers:             {x_res_centers}")
            _log_setup(f"  r centers:             {r_res_centers}")
            if smooth_GP_resolution_loglinear:
                _log_setup(f"  raw x centers:         {x_res_centers_raw}")
                _log_setup(f"  raw r centers:         {r_res_centers_raw}")

            if fout is not None:
                if smooth_GP_resolution_loglinear:
                    plot_resolution_curve_numpyro_with_loglinear_fit(
                        x_res_centers_raw,
                        r_res_centers_raw,
                        x_res_centers,
                        r_res_centers,
                        x_evt=x_evt,
                        w_evt=w_evt,
                        fout=fout,
                    )
                else:
                    nutils.plot_resolution_curve_numpyro(
                        x_res_centers,
                        r_res_centers,
                        x_evt=x_evt,
                        w_evt=w_evt,
                        fout=fout,
                    )

        # ------------------------------------------------
        # GP dL: dedicated nodes and fine grid, matching PyMC GP branch.
        # IMPORTANT: build this after redshift-bound update.
        # ------------------------------------------------
        z_gp_nodes, z_gp_fine = make_z_grids_GP_jax(
            zmin=zmin_a,
            zmax=zmax_c,
            n_nodes=n_gp_nodes,
            n_fine=n_gp_fine,
            n_ramp_nodes=12,
            n_ramp_fine=20,
        )

        _log_setup("GP z grid for interpolation built.")
        _log_setup("  nodes len:", z_gp_nodes.shape[0])
        _log_setup("  fine len:", z_gp_fine.shape[0])
        _log_setup("  z fine min/max:", z_gp_fine.min(), z_gp_fine.max())


    ################################################
    # PE prior on distance, if needed
    ################################################

    if not pop_only:
        

        vol_in_prior = any(
            (("UniformSourceFrame" in s or "UniformComovingVolume" in s) and ("bilby" not in s))
            for s in dLprior
        )
        vol_in_prior_from_bilby = any(s == "UniformSourceFrame-bilby" for s in dLprior)

    
        edges = [0]
        for n in Nevs_np:
            edges.append(edges[-1] + int(n))
    
    
        if vol_in_prior_from_bilby:
            
            _log_setup("Loading bilby pre-computed PE prior from distance for later interpolation")
            dat = np.load("dLgrid_gpc_bilby_prior_grid_O4a.npz")
            dLgrid_bilby_gpc = dat["dLgrid_gpc"]
            PE_prior_bilby_grid =  dat["prior_grid"]

        else:
            dLgrid_bilby_gpc = None
            PE_prior_bilby_grid = None
            

        if ( ( vol_in_prior or vol_in_prior_from_bilby) and (penorm_lims != []) ):
    
            _log_setup("Normalization of PE volume prior on distance required.")
                
      
            bkNP = NPBackend()
            
            Nchunks = len(Nevs_np)
            assert len(allnames) == Nchunks
            j = 0
            all_PE_log_norms = np.zeros(N, dtype=np.float64)
            for i in range(Nchunks):
                
                if  penorm_lims[i]=='none':
                    _log_setup("No normalization of PE prior on distance included for chunk %s"%i)
                    for key in allnames[i]:
                        all_PE_log_norms[j] = 0.
                        j+=1
                else:
                    with open( penorm_lims[i] , 'r') as fp:
                        plims_ = json.load(fp)
                    
                    _log_setup("Normalization of PE prior on distance for chunk %s loaded"%i)
                    
                    for key in allnames[i]:
                        try:
                            lims_ = plims_[key]
                        except:
                            raise ValueError("limits for %s not present"%key)    
                         
                        log_norm_PE_prior_, za, zb = cosmo.compute_log_norm_UniformSourceFrame(bkNP, lims_[0]/1000, lims_[1]/1000, 67.9, 0.3065, -1)
                        #_log_setup("event, dmin [Gpc], dmax [Gpc], zmin, zmax, log_norm")
                        #
                        
                        
                        _log_setup(key, lims_[0]/1000, lims_[1]/1000, za, zb, log_norm_PE_prior_)
                
                        all_PE_log_norms[j] = log_norm_PE_prior_
                        j+=1
                
                _log_setup("at the end of chunk %s, index j is %s"%(i,j))
    
            all_PE_log_norms = np.asarray(all_PE_log_norms)
        else:
            _log_setup("No normalization of PE volume prior on distance required.")
            all_PE_log_norms = np.zeros(Nevs_np.sum(), dtype=np.float64)

    
        if remove_spin_prior:
            _log_setup("Removing PE spin prior")
            amax = 0.99
            spinp= (1./amax)*(1./amax)*0.5*0.5  

            # all_PE_log_norms is later subtraceted to log_PE_prior, so we subtract log(spinp) here
            # this will result in adding log(spinp) to the log_PE_prior which is subtracted itself
            # to the likelihood. 
            all_PE_log_norms -= np.log(spinp)

             
        _log_setup("All PE log norms is ")
        _log_setup("Shape: %s"%all_PE_log_norms.shape)

 
        
    
    ################################################
    # Build model
    ################################################



    if not pop_only:
        if int(mus_l.shape[0]) != int(np.sum(Nevs_np)):
            raise ValueError("Sum(Nevs_np) != mus_l.shape[0] (event count mismatch).")

        data = pack_data_gauss_popnot(
            GWData=GWData,
            InjData=InjData,
            dLprior=dLprior,
            Nevs_np=Nevs_np,
            all_PE_log_norms=all_PE_log_norms,
            dLgrid_bilby_gpc=(dLgrid_bilby_gpc if vol_in_prior_from_bilby else None),
            PE_prior_bilby_grid=(PE_prior_bilby_grid if vol_in_prior_from_bilby else None),
            spin_model=spin_model_sel,
            rate_model=rate_model,
            mass_model=mass_model,
            smoothing=smoothing,
            simplex_repair=simplex_repair,
            has_m2_break=has_m2_break,
            norm_gauss=norm_gauss,
            param=param,
            integrate_dc=integrate_dc,
            subtract_log_p_incl=False,
            sample_from_pop=sample_from_pop,
            marginal_R0=marginal_R0,
            allTobs=allTobs,
            r=r,
            taper_kind = taper_kind, 
            taper_p = taper_p
        )
        _log_setup("pack_data_gauss_popnot ok")
        _log_setup()
        if spin_model_sel == "none":
            assert int(data.mus_s.shape[1]) == 3
            assert int(data.mus_l.shape[2]) == 3
            assert int(data.icovs_l.shape[2]) == 3
            _log_setup("[check] spin_model none: using 3D PE surrogate for latent x")
        else:
            _log_setup("[check] spin_model with spins: using PE surrogate dimension", int(data.mus_s.shape[1]))
        
        core, loglik, loglik_components = build_core_and_loglik_gauss_popnot(
            data,
            chunk_inj=(0 if chunk_inj in (-1, None) else chunk_inj),
            K_dp=N_DP_comp_max_np,
            DP_truncate=bool(DP_truncate_up or DP_truncate_low),
            PL_DPUC_normalize_gmm = bool(PL_DPUC_normalize_gmm),
            DPUC_kernel = DPUC_kernel,
            DPUC_kernel_rcut = DPUC_kernel_rcut,
            DPUC_kernel_dr = DPUC_kernel_dr,
        DPUC_kernel_xcut = DPUC_kernel_xcut,
        DPUC_kernel_ycut = DPUC_kernel_ycut,
        DPUC_q_coord = DPUC_q_coord,
            DPUC_TruncM2_shared_weights=DPUC_TruncM2_shared_weights,
            DPUC_TruncM2_secondary_coord=DPUC_TruncM2_secondary_coord,
            DP_m1_env=DP_m1_env,
            DPUC_Primary_q_norm=DPUC_Primary_q_norm,
            DP_corr_m1m2 = DP_corr_m1m2,
            DP_corr_eta_rho = DP_corr_eta_rho,
        DP_corr_f_minor = DP_corr_f_minor,
            interp_mass=interp_mass,
            stop_grad_var_u=True,
            skip_sel=skip_sel,
            verbose=False,
            z_nodes=z_nodes,
        )
    else:
        data = pack_data_samples_marginal(
            GWData=GWData,
            InjData=InjData,
            spin_model=spin_model_sel,
            rate_model=rate_model,
            mass_model=mass_model,
            smoothing=smoothing,
            simplex_repair=simplex_repair,
            has_m2_break=has_m2_break,
            norm_gauss=norm_gauss,
            param=param,
            integrate_dc=integrate_dc,
            subtract_log_p_incl=False,
            marginal_R0=marginal_R0,
            allTobs=allTobs,
            chunk_pe=(0 if chunk_reduce in (False, 0, None) else int(chunk_reduce)),
            r=r,
            taper_kind = taper_kind, 
            taper_p = taper_p
        )

        core, loglik, loglik_components = build_core_and_loglik_samples_marginal(
            data,
            chunk_inj=(0 if chunk_inj in (-1, None) else chunk_inj),
            K_dp=N_DP_comp_max_np,
            DP_truncate=bool(DP_truncate_up or DP_truncate_low),
            DP_m1_env=DP_m1_env,
            DPUC_Primary_q_norm=DPUC_Primary_q_norm,
            PL_DPUC_normalize_gmm = bool(PL_DPUC_normalize_gmm),
            DPUC_kernel = DPUC_kernel,
            DPUC_kernel_rcut = DPUC_kernel_rcut,
            DPUC_kernel_dr = DPUC_kernel_dr,
        DPUC_kernel_xcut = DPUC_kernel_xcut,
        DPUC_kernel_ycut = DPUC_kernel_ycut,
        DPUC_q_coord = DPUC_q_coord,
            DPUC_TruncM2_shared_weights=DPUC_TruncM2_shared_weights,
            DPUC_TruncM2_secondary_coord=DPUC_TruncM2_secondary_coord,
            DP_corr_m1m2=DP_corr_m1m2,
            DP_corr_eta_rho = DP_corr_eta_rho,
        DP_corr_f_minor = DP_corr_f_minor,
            interp_mass=interp_mass,
            stop_grad_var_u=True,
            verbose=False,
            z_nodes=z_nodes,
            skip_sel=skip_sel
        )

    # `loglik` is now a jitted callable:
    #   loglik(Lambda: (npar,), x: (N, nd)) -> scalar



    def model_numpyro():
        # -------------------------
        # Cosmology (matches PyMC)
        # -------------------------
        # NOTE: priors is your dict of [low, high] etc; params_fix is dict for fixed values
        # ivals optional: use for init_strategy outside (numpyro.infer.init_to_value) if you want
    
        if fix_Om:
            Om_ = jnp.asarray(params_fix["Om"], dtype=jnp.float64)
        else:
            if not reparam_cosmo:
                Om_ = numpyro.sample("Om", dist.Uniform(priors["Om"][0], priors["Om"][1]))
            else:
                Om_ = bounded_sigmoid("Om", priors["Om"][0], priors["Om"][1], raw_sigma=1)
    
        if fix_w0:
            w0_ = jnp.asarray(-1.0, dtype=jnp.float64)
        else:
            if pade or integrate_dc == "pade":
                raise NotImplementedError("Pade with varying w0 not implemented yet.")
            if not reparam_cosmo:
                w0_ = numpyro.sample("w0", dist.Uniform(priors["w0"][0], priors["w0"][1]))
            else:
                w0_ = bounded_sigmoid("w0", priors["w0"][0], priors["w0"][1], raw_sigma=1.0)
    
        if fix_H0:
            H0_ = jnp.asarray(params_fix["H0"], dtype=jnp.float64)
        else:
            if not reparam_cosmo:
                H0_ = numpyro.sample("H0", dist.Uniform(priors["H0"][0], priors["H0"][1]))
            else:
                H0_ = bounded_sigmoid("H0", priors["H0"][0], priors["H0"][1], raw_sigma=1.5)
    
        if fix_Xi0n:
            Xi0_  = jnp.asarray(1.0, dtype=jnp.float64)
            nXi0_ = jnp.asarray(0.0, dtype=jnp.float64)
        else:
            if reparam_cosmo:
                raise NotImplementedError()
            nXi0_ = bounded_sigmoid("nXi0", priors["nXi0"][0], priors["nXi0"][1], raw_sigma=1.0)
            Xi0_  = bounded_sigmoid_logpositive("Xi0", priors["Xi0"][0], priors["Xi0"][1], raw_sigma=1.5)
    
        Lambda_list = [H0_, Om_, w0_, Xi0_, nXi0_]
    
        # -------------------------
        # Rate model (MD)
        # -------------------------
        if not reparam_z:
            gamma_ = numpyro.sample("gamma", dist.Uniform(priors["gamma"][0], priors["gamma"][1]))
            kappa_ = numpyro.sample("kappa", dist.Uniform(priors["kappa"][0], priors["kappa"][1]))
            zp_    = numpyro.sample("zp",    dist.Uniform(priors["zp"][0],    priors["zp"][1]))

        else:
            # Uniform[a,b]  ->  raw ~ Normal, then affine(sigmoid(raw))
            gamma_a, gamma_b = priors["gamma"]
            kappa_a, kappa_b = priors["kappa"]
            zp_a, zp_b       = priors["zp"]
                
            gamma_ = bounded_sigmoid("gamma", gamma_a, gamma_b, raw_sigma = 1.5 )
            kappa_ = bounded_sigmoid("kappa", kappa_a, kappa_b, raw_sigma=1 )
            zp_    = bounded_sigmoid("zp",    zp_a,    zp_b,    raw_sigma=1 )


        Lambda_list += [gamma_, kappa_, zp_]
    
        # -------------------------
        # Spin (default_gauss)
        # -------------------------
        if spin_model == "default_gauss":
            # assumes you have these helpers with same signatures as in PyMC world,
            # but implemented for numpyro (or made backend-agnostic).
            # If not, replace these blocks with direct numpyro.sample definitions.
    
            # muChi in [a,b] via sigmoid reparam
            muChi_a, muChi_b = priors["muChi"]
            muChi_ = bounded_sigmoid("muChi", muChi_a, muChi_b, raw_sigma = 1.5 )
            

            sigmaChi_a, sigmaChi_b = priors["sigmaChi"]
            sigmaChi_ = bounded_sigmoid("sigmaChi", sigmaChi_a, sigmaChi_b, raw_sigma=1.5)
            
            # zeta in [a,b] via sigmoid reparam
            zeta_a, zeta_b = priors["zeta"]
            zeta_ = bounded_sigmoid("zeta", zeta_a, zeta_b, raw_sigma = 1.5)
            
            # sigmat = floor + HalfNormal(raw), with typmax interpreted as ~95% point
            HN_Q95_TO_SIGMA = 1.959963984540054
            sigmat_floor, sigmat_typmax = priors["sigmat"]
            raw_typ = max(1e-12, sigmat_typmax - sigmat_floor)
            sigmat_sigma = raw_typ / HN_Q95_TO_SIGMA
            
            sigmat_raw = numpyro.sample("sigmat_raw", dist.HalfNormal(sigmat_sigma))
            sigmat_ = sigmat_floor + sigmat_raw
            numpyro.deterministic("sigmat", sigmat_)
    


            
    
            Lambda_list += [muChi_, sigmaChi_, zeta_, sigmat_]
    
        # else: spin_model == "none" -> nothing added
    
        # -------------------------
        # Mass model (DPLDP)
        # -------------------------
        if mass_model == 'DPLDP':
            # epsilon fixed 
            epsilon_ = jnp.asarray(0.1, dtype=jnp.float64)
            numpyro.deterministic("epsilon", epsilon_)


            if reparam_mass:


                if priors["alpha1"] != priors["alpha2"]:
                        raise ValueError(f"alpha1/alpha2 priors differ: {priors['alpha1']} vs {priors['alpha2']}")
            
                # bounds -> mid and sigma
                a_low, a_high = priors["alpha1"][0], priors["alpha1"][1]
                a_mid = 0.5 * (a_low + a_high)
                a_sig = (a_high - a_low) / (2.0 * NORM_Q95)
                
                a_bar  = numpyro.sample("alpha_bar",  dist.Normal(a_mid, a_sig), )

                               
                u = unit_interval_sigmoid("u", raw_sigma=1 )
                m1_low_ = 3.0 + (10.0 - 3.0) * u**1.5 #jnp.sqrt(u)
                numpyro.deterministic("m1_low", m1_low_)
                delta_m1_ = floored_lognormal_q95("delta_m1", priors["delta_m1"][0], priors["delta_m1"][1], median_frac=0.3 )
 

                a_diff = numpyro.sample("alpha_diff", dist.Normal(0.0, jnp.sqrt(2.0) * a_sig), )
                
                # deterministics
                alpha1_ = numpyro.deterministic("alpha1", a_bar - 0.5 * a_diff)
                alpha2_ = numpyro.deterministic("alpha2", a_bar + 0.5 * a_diff)

                mb_a, mb_b = priors["mb"][0], priors["mb"][1]
                mb_ = bounded_sigmoid("mb", mb_a, mb_b, raw_sigma=1 )

                v = unit_interval_sigmoid("v",raw_sigma=1)
                m2_low_ = 3.0 + v * (m1_low_ - 3.0)
                numpyro.deterministic("m2_low", m2_low_)
        
                delta_m2_ = floored_lognormal_q95("delta_m2", priors["delta_m2"][0], priors["delta_m2"][1], median_frac=0.3 )
                    
                
                beta_ = normal_from_bounds_95("beta", priors["beta"][0], priors["beta"][1] )
        
                 
                
                sigma1_          = floored_lognormal_q95("sigma1", priors["sigma1"][0], priors["sigma1"][1], median_frac=0.2)
                sigma2_          = floored_lognormal_q95("sigma2", priors["sigma2"][0], priors["sigma2"][1], median_frac=0.3 )
    
   
                mu1_ = bounded_sigmoid("mu1", priors["mu1"][0], priors["mu1"][1], raw_sigma=1.25 )
                mu2_ = bounded_sigmoid("mu2", priors["mu2"][0], priors["mu2"][1], raw_sigma=1.25 )

                
        
                mhigh_floor = priors["m_high"][0]   # e.g. 80
                mmax_median = 0.5 * (priors["m_high"][0] + priors["m_high"][1])
                mmax_q95    = priors["m_high"][1]
                
                delta_med = jnp.maximum(mmax_median - mhigh_floor, 1e-6)
                delta_q95 = jnp.maximum(mmax_q95    - mhigh_floor, 1e-6)
                
                mu_delta = jnp.log(delta_med)
                sigma_delta = (jnp.log(delta_q95) - mu_delta) / NORM_Q95
                
                delta_mhigh = numpyro.sample("delta_mhigh", dist.LogNormal(loc=mu_delta, scale=sigma_delta))
                m_high_ = mhigh_floor + delta_mhigh
                numpyro.deterministic("m_high", m_high_)
                

            else:
                # Uniform priors

                alpha1_ = numpyro.sample("alpha1", dist.Uniform(priors["alpha1"][0], priors["alpha1"][1]))

                delta_m1_ = numpyro.sample("delta_m1", dist.Uniform(priors["delta_m1"][0], priors["delta_m1"][1]))

                m_high_ = numpyro.sample("m_high", dist.Uniform(priors["m_high"][0], priors["m_high"][1]))

                alpha2_ = numpyro.sample("alpha2", dist.Uniform(priors["alpha2"][0], priors["alpha2"][1]))
                mb_ = numpyro.sample("mb", dist.Uniform(priors["mb"][0], priors["mb"][1]))

                u = numpyro.sample("u", dist.Uniform(0, 1))
                m1_low_ = 3.0 + (10.0 - 3.0) * jnp.sqrt(u)
                numpyro.deterministic("m1_low", m1_low_)
            
                v = numpyro.sample("v", dist.Uniform(0, 1))
                m2_low_ = 3.0 + v * (m1_low_ - 3.0)
                numpyro.deterministic("m2_low", m2_low_)

                delta_m2_ = numpyro.sample("delta_m2", dist.Uniform(priors["delta_m2"][0], priors["delta_m2"][1]))
                    

                beta_ = numpyro.sample("beta", dist.Uniform(priors["beta"][0], priors["beta"][1]))

                
                sigma1_ = numpyro.sample("sigma1", dist.Uniform(priors["sigma1"][0], priors["sigma1"][1]))
                sigma2_ = numpyro.sample("sigma2", dist.Uniform(priors["sigma2"][0], priors["sigma2"][1]))
                mu1_ = numpyro.sample("mu1", dist.Uniform(priors["mu1"][0], priors["mu1"][1]))
                mu2_ = numpyro.sample("mu2", dist.Uniform(priors["mu2"][0], priors["mu2"][1]))
                
  

                
            # Dirichlet for lambda weights
            lambda_vec = numpyro.sample("lambda", dist.Dirichlet(jnp.asarray([1.0, 1.0, 1.0])))
    
            lambda0_, lambda1_, lambda2_ = lambda_vec[0], lambda_vec[1], lambda_vec[2]
            numpyro.deterministic("lambda0", lambda0_)
            numpyro.deterministic("lambda1", lambda1_)
            numpyro.deterministic("lambda2", lambda2_)
    
            if has_m2_break:
                m_g_ = numpyro.sample("m_g", dist.Uniform(priors["m_g"][0], priors["m_g"][1]))
                w_g_ = numpyro.sample("w_g", dist.Uniform(priors["w_g"][0], priors["w_g"][1]))
                sig_g_l_ = jnp.asarray(1e-2, dtype=jnp.float64)
                sig_g_h_ = jnp.asarray(1e-2, dtype=jnp.float64)
            else:
                m_g_ = jnp.asarray(45.0, dtype=jnp.float64)
                w_g_ = jnp.asarray(70.0, dtype=jnp.float64)
                sig_g_l_ = jnp.asarray(1e-2, dtype=jnp.float64)
                sig_g_h_ = jnp.asarray(1e-2, dtype=jnp.float64)
    
            # add to Lambda in *the exact order your core expects*
            Lambda_list += [
                alpha1_, alpha2_, mb_, mu1_, sigma1_, mu2_, sigma2_,
                m1_low_, m_high_, delta_m1_,
                lambda0_, lambda1_, lambda2_,
                beta_, m2_low_, delta_m2_,
                epsilon_, m_g_, w_g_, sig_g_l_, sig_g_h_
            ]


        elif mass_model in ('DPUC', 'DPUC_Primary', 'DPUC_TruncM2'):

            #_log_setup("Modeling mass distribution as Dirichelet Process. Max number of components: %s"%N_DP_comp_max)

            if 'SB' in DP_prior:

                #_log_setup("Prior for the process is stick-breaking")

                if DP_prior=='SB_gamma_alpha_inv':
                    alpha_inv = numpyro.sample("alpha_inv", dist.Gamma(alpha_inv_params[0], alpha_inv_params[1]))
                    #_log_setup("alpha_inv prior is Gamma with parameters %s"%str(alpha_inv_params))
                    alpha = 1/alpha_inv

                elif DP_prior=='SB_gamma_alpha':
                    alpha_SB = numpyro.sample("alpha_SB", dist.Gamma(alpha_inv_params[0], alpha_inv_params[1]))
                    #_log_setup("alpha prior is Gamma with parameters %s"%str(alpha_inv_params))
                    alpha_inv = numpyro.deterministic("alpha_inv", 1 / alpha_SB)
                    alpha = 1/alpha_inv

                elif DP_prior=='SB_ln':
                    #mu_log_alpha_inv = np.log(1.0)
                    #sigma_softmax = 0.5
                    log_alpha_inv = numpyro.sample(
                        "log_alpha_inv",
                        dist.Normal(mu_log_alpha_inv, sigma_log_alpha_inv),
                    )
                    alpha_inv = numpyro.deterministic("alpha_inv", jnp.exp(log_alpha_inv))
                    alpha = jnp.exp(-log_alpha_inv)
                    #_log_setup(f"log(alpha_inv) prior: Normal(mu={mu_log_alpha_inv}, sigma={sigma_log_alpha_inv})")
                    #_log_setup("alpha_inv init =", np.exp(np.log(1.0)))
                    #_log_setup("alpha init =", np.exp(-np.log(1.0)))

                else:
                    raise ValueError()

                M_active_np = min(int(M_active), N_DP_comp_max_np)

                if M_active_np==0:
                    beta_init = np.full(N_DP_comp_max_np, 0.02)
                    beta_init[:5] = [0.80, 0.60, 0.35, 0.20, 0.10]
                    #_log_setup("Initializing beta with manual values")

                else:
                    #_log_setup("Initializing beta with %s active components"%M_active_np)
                    tail_mass_frac = 0.02
                    active_mass = 1.0 - tail_mass_frac
                    tail_mass = tail_mass_frac
                    n_tail = N_DP_comp_max_np - M_active_np
                    active_profile = np.linspace(1.0, 0.9, M_active_np)
                    active_profile = active_profile / active_profile.sum()
                    w_active = active_mass * active_profile
                    if n_tail > 0:
                        w_tail = np.full(n_tail, tail_mass / n_tail)
                        w_init = np.concatenate([w_active, w_tail])
                    else:
                        w_init = w_active.copy()
                    beta_init = np.zeros(N_DP_comp_max_np)
                    remaining = 1.0
                    eps_beta = 1e-6
                    for k in range(N_DP_comp_max_np):
                        if k == N_DP_comp_max_np - 1:
                            beta_init[k] = 0.5
                        else:
                            frac = w_init[k] / max(remaining, 1e-12)
                            beta_init[k] = np.clip(frac, eps_beta, 1.0 - eps_beta)
                            remaining -= w_init[k]

                beta = numpyro.sample(
                    "beta",
                    dist.Beta(
                        jnp.ones((N_DP_comp_max_np,), dtype=jnp.float64),
                        alpha * jnp.ones((N_DP_comp_max_np,), dtype=jnp.float64),
                    ).to_event(1),
                )

                #_log_setup("beta init top10:", np.sort(beta_init)[::-1][:10])
                #_log_setup("beta init first10:", beta_init[:10])

                w_raw = stick_breaking_jax(beta)
                w_raw = w_raw / jnp.sum(w_raw)
                numpyro.deterministic("w_stick_raw", w_raw)

                if (
                    mass_model == "DPUC_TruncM2"
                    and bool(DPUC_TruncM2_weight_cap_reg)
                ):
                    K_w = w_raw.shape[0]
                    K_w_f = jnp.asarray(K_w, dtype=jnp.float64)
                    cap_w = jnp.asarray(DPUC_TruncM2_weight_cap, dtype=jnp.float64)

                    # eps chosen so that even if max(w_raw)=1,
                    # max((1-eps)*w_raw + eps/K) = cap_w.
                    eps_cap = (1.0 - cap_w) / (1.0 - 1.0 / K_w_f)

                    w = (1.0 - eps_cap) * w_raw + eps_cap / K_w_f
                    w = w / jnp.sum(w)

                    numpyro.deterministic("weight_cap_used", cap_w)
                    numpyro.deterministic("weight_cap_epsilon", eps_cap)
                    numpyro.deterministic("weight_floor_used", eps_cap / K_w_f)
                else:
                    w = w_raw
                    numpyro.deterministic("weight_cap_used", jnp.asarray(1.0, dtype=jnp.float64))
                    numpyro.deterministic("weight_cap_epsilon", jnp.asarray(0.0, dtype=jnp.float64))
                    numpyro.deterministic("weight_floor_used", jnp.asarray(0.0, dtype=jnp.float64))

                w = numpyro.deterministic("w", w)

                w0 = np.asarray(beta_init, dtype=np.float64)
                w0 = np.asarray(w0 * np.concatenate([[1.0], np.cumprod((1.0 - w0)[:-1])]), dtype=np.float64)
                w0 = w0 / np.sum(w0)

                numpyro.deterministic("max_w", jnp.max(w))
                numpyro.deterministic("n_w_gt_001", jnp.sum(w > 0.01))
                numpyro.deterministic(
                    "weight_entropy",
                    -jnp.sum(w * jnp.log(jnp.clip(w, 1e-300, 1.0)))
                )

            elif DP_prior=='dirichelet':
                #_log_setup("Prior for the process is dirichelet")
                #_log_setup("alpha_total prior is Gamma with parameters %s"%str(gamma_DP_params))
                alpha_total = numpyro.sample("alpha_total", dist.Gamma(gamma_DP_params[0], gamma_DP_params[1]))
                a = alpha_total / N_DP_comp_max
                w = numpyro.sample(
                    "w",
                    dist.Dirichlet(jnp.ones((N_DP_comp_max_np,), dtype=jnp.float64) * a),
                )

            elif DP_prior=='softmax':
                #_log_setup("Prior for the process is softmax")
                #_log_setup("sigma_w sampled from halfnormal with std=%s"%sigma_softmax)
                sigma_w = numpyro.sample("sigma_w", dist.HalfNormal(sigma_softmax))
                raw_w = numpyro.sample(
                    "raw_w",
                    dist.Normal(0, sigma_w).expand((N_DP_comp_max_np,)).to_event(1),
                )
                w = numpyro.deterministic("w", jax.nn.softmax(raw_w))

            else:
                raise ValueError()

            logw = numpyro.deterministic("logw", jnp.log(w))



            if mass_model in ("DPUC_TruncM2") and DPUC_TruncM2_mu2_mode == "q_ratio":
                if q_mu_low is None or q_mu_up is None:
                    raise ValueError(
                        "DPUC_TruncM2_mu2_mode='q_ratio' requires q_mu bounds. "
                        "Use find_m_bounds=1 or DPUC_q_mu_bounds_source='unit' with find_m_bounds=0."
                    )

            U1 = (upsig1 - lowsig1)

            mu1_center = (lowmu1 + upmu1) / 2.0
            mu1_init = np.full(N_DP_comp_max_np, mu1_center)

            M_active_np = min(int(M_active), N_DP_comp_max_np)
            if M_active_np > 0:
                mu1_init[:M_active_np] = np.linspace(
                    lowmu1 + 0.1 * (upmu1 - lowmu1),
                    upmu1 - 0.1 * (upmu1 - lowmu1),
                    M_active_np,
                )

            mu1 = bounded_sigmoid_vec(
                "mulMc",
                lowmu1,
                upmu1,
                N_DP_comp_max_np,
                raw_sigma=1.0,
            )

            if mass_model == 'DPUC_Primary':
                mus = jnp.stack([mu1], axis=0)
            else:
                U2 = (upsig2 - lowsig2)

                mu2_center = (lowmu2 + upmu2) / 2.0
                mu2_init = np.full(N_DP_comp_max_np, mu2_center)

                if M_active_np > 0:
                    mu2_init[:M_active_np] = np.linspace(
                        lowmu2 + 0.1 * (upmu2 - lowmu2),
                        upmu2 - 0.1 * (upmu2 - lowmu2),
                        M_active_np,
                    )

                if mass_model == "DPUC_TruncM2" and DPUC_TruncM2_mu2_mode == "ordered":
                    # Legacy experimental option: component-wise moving upper bound.
                    # Kept for backwards compatibility.
                    if not (float(lowmu1) > float(lowmu2)):
                        raise ValueError(
                            "DPUC_TruncM2_mu2_mode='ordered' requires lowmu1 > lowmu2 "
                            "so every sampled mu1 can define a valid [lowmu2, mu1] interval. "
                            f"Got lowmu1={lowmu1}, lowmu2={lowmu2}."
                        )
                    mu2 = bounded_sigmoid_vec(
                        "mulq",
                        lowmu2,
                        mu1,
                        N_DP_comp_max_np,
                        raw_sigma=1.0,
                    )
                elif mass_model in ("DPUC_TruncM2") and DPUC_TruncM2_mu2_mode == "q_ratio":
                    # Stable ordered parameterization.  In the legacy logm2 branch,
                    # mu2_k = mu1_k + log(q_mu_k).  In the logitq-secondary branch,
                    # the second GMM coordinate is y=logit(q) directly.
                    q_mu = bounded_sigmoid_vec(
                        "q_mu",
                        q_mu_low,
                        q_mu_up,
                        N_DP_comp_max_np,
                        raw_sigma=1.0,
                    )
                    if DPUC_TruncM2_secondary_coord == "logitq":
                        mu2 = numpyro.deterministic("mulq", logit(JAXBackend(), q_mu))
                    else:
                        mu2 = numpyro.deterministic("mulq", mu1 + jnp.log(q_mu))
                else:
                    mu2 = bounded_sigmoid_vec(
                        "mulq",
                        lowmu2,
                        upmu2,
                        N_DP_comp_max_np,
                        raw_sigma=1.0,
                    )

                mus = jnp.stack([mu1, mu2], axis=0)

            mu = numpyro.deterministic("mu", mus)

            if mass_model == "DPUC_TruncM2":
                log_mcut_low_center = jnp.log(
                    jnp.asarray(DPUC_TruncM2_low_center_mcut, dtype=jnp.float64)
                )
                low_center_log_width = jnp.asarray(
                    DPUC_TruncM2_low_center_log_width, dtype=jnp.float64
                )

                # Smooth indicator for primary component centers below mcut.
                # low_center_gate ~ 1 for exp(mu1) < mcut, ~0 above mcut.
                low_center_gate = jax.nn.sigmoid(
                    (log_mcut_low_center - mu1) / low_center_log_width
                )

                W_low_center = numpyro.deterministic(
                    "TruncM2_low_center_weight",
                    jnp.sum(w * low_center_gate),
                )

                if (
                    bool(DPUC_TruncM2_low_center_weight_reg)
                    and DPUC_TruncM2_low_center_weight_lambda > 0.0
                ):
                    low_center_cap = jnp.asarray(
                        DPUC_TruncM2_low_center_weight_cap, dtype=jnp.float64
                    )
                    low_center_weight_width = jnp.asarray(
                        DPUC_TruncM2_low_center_weight_width, dtype=jnp.float64
                    )
                    low_center_lambda = jnp.asarray(
                        DPUC_TruncM2_low_center_weight_lambda, dtype=jnp.float64
                    )

                    hW = low_center_weight_width * jax.nn.softplus(
                        (W_low_center - low_center_cap) / low_center_weight_width
                    )

                    low_center_penalty = -low_center_lambda * hW**2

                    numpyro.deterministic(
                        "TruncM2_low_center_weight_reg_penalty",
                        low_center_penalty,
                    )
                    numpyro.factor(
                        "TruncM2_low_center_weight_reg",
                        low_center_penalty,
                    )


            tau_floor_eps = 1e-6

            U1_excess = float(U1 - L_small_1)
            if U1_excess <= tau_floor_eps:
                raise ValueError(
                    f"Invalid width bounds for dim 1: U1={U1}, L_small_1={L_small_1}, "
                    f"so U1-L_small_1={U1_excess} <= {tau_floor_eps}"
                )

            if tau_prior == 'flat':
                tau1 = numpyro.sample(
                    "tau1",
                    dist.Uniform(
                        jnp.asarray(tau_floor_eps, dtype=jnp.float64),
                        jnp.asarray(U1_excess, dtype=jnp.float64),
                    ),
                )
            elif tau_prior == 'bounded_sigmoid':
                tau1 = bounded_sigmoid(
                    "tau1",
                    tau_floor_eps,
                    U1_excess,
                    raw_sigma=1.5,
                )
            else:
                raise ValueError("unknown tau prior: %s" % tau_prior)

            numpyro.deterministic("tau1_excess", tau1)
            numpyro.deterministic("U1_excess", jnp.asarray(U1_excess, dtype=jnp.float64))
            numpyro.deterministic("U1_width_support", jnp.asarray(U1, dtype=jnp.float64))

            if center_eps_widths:
                eps1_raw = numpyro.sample(
                    "eps1_raw",
                    dist.Normal(0.0, s_local[0]).expand((N_DP_comp_max_np,)).to_event(1),
                )
                eps1 = numpyro.deterministic("eps1", eps1_raw - jnp.mean(eps1_raw))
            else:
                eps1 = numpyro.sample(
                    "eps1",
                    dist.Normal(0.0, s_local[0]).expand((N_DP_comp_max_np,)).to_event(1),
                )

            sig1_excess = tau1 * jnp.exp(eps1)
            sig1 = numpyro.deterministic("sig1", L_small_1 + sig1_excess)

            numpyro.deterministic("min_sig1", jnp.min(sig1))
            numpyro.deterministic("max_sig1", jnp.max(sig1))

            if mass_model == 'DPUC_Primary':
                sigs = jnp.stack([sig1], axis=0)
            else:
                U2_excess = float(U2 - L_small_2)
                if U2_excess <= tau_floor_eps:
                    raise ValueError(
                        f"Invalid width bounds for dim 2: U2={U2}, L_small_2={L_small_2}, "
                        f"so U2-L_small_2={U2_excess} <= {tau_floor_eps}"
                    )

                if tau_prior == 'flat':
                    tau2 = numpyro.sample(
                        "tau2",
                        dist.Uniform(
                            jnp.asarray(tau_floor_eps, dtype=jnp.float64),
                            jnp.asarray(U2_excess, dtype=jnp.float64),
                        ),
                    )
                elif tau_prior == 'bounded_sigmoid':
                    tau2 = bounded_sigmoid(
                        "tau2",
                        tau_floor_eps,
                        U2_excess,
                        raw_sigma=1.5,
                    )
                else:
                    raise ValueError("unknown tau prior: %s" % tau_prior)

                numpyro.deterministic("tau2_excess", tau2)
                numpyro.deterministic("U2_excess", jnp.asarray(U2_excess, dtype=jnp.float64))
                numpyro.deterministic("U2_width_support", jnp.asarray(U2, dtype=jnp.float64))

                if center_eps_widths:
                    eps2_raw = numpyro.sample(
                        "eps2_raw",
                        dist.Normal(0.0, s_local[1]).expand((N_DP_comp_max_np,)).to_event(1),
                    )
                    eps2 = numpyro.deterministic("eps2", eps2_raw - jnp.mean(eps2_raw))
                else:
                    eps2 = numpyro.sample(
                        "eps2",
                        dist.Normal(0.0, s_local[1]).expand((N_DP_comp_max_np,)).to_event(1),
                    )

                sig2_excess = tau2 * jnp.exp(eps2)
                sig2 = numpyro.deterministic("sig2", L_small_2 + sig2_excess)

                numpyro.deterministic("min_sig2", jnp.min(sig2))
                numpyro.deterministic("max_sig2", jnp.max(sig2))

                sigs = jnp.stack([sig1, sig2], axis=0)

            numpyro.deterministic("L_small_1_used", jnp.asarray(L_small_1, dtype=jnp.float64))
            if not mass_model == 'DPUC_Primary':
                numpyro.deterministic("L_small_2_used", jnp.asarray(L_small_2, dtype=jnp.float64))

            if alpha_tail != -1:
                lambda_large_1 = -np.log(alpha_tail) / U1
                numpyro.factor("pc_large_ell_1", -lambda_large_1 * tau1)

                if not mass_model == 'DPUC_Primary':
                    lambda_large_2 = -np.log(alpha_tail) / U2
                    numpyro.factor("pc_large_ell_2", -lambda_large_2 * tau2)
            
            if DP_truncate_low:

                if DPUC_low_end_sample:
        
            
                    if "m1_low" not in priors:
                        raise KeyError(f"{mass_model} with DP_truncate_low requires priors['m1_low'].")
    
                    u = unit_interval_sigmoid("u", raw_sigma=1.0)
                    m1_low_ = priors["m1_low"][0] + (priors["m1_low"][1] - priors["m1_low"][0]) * u**1.5
                    numpyro.deterministic("m1_low", m1_low_)
    
                    if not DPUC_low_end_share:
                        if "m2_low" not in priors:
                            raise KeyError(f"{mass_model} with DP_truncate_low requires priors['m2_low'].")
                    
        
                    
                        v = unit_interval_sigmoid("v", raw_sigma=1.0)
                        m2_low_ = priors["m2_low"][0] + v * (m1_low_ - priors["m2_low"][0])
                            
                    else:
                        m2_low_ = m1_low_
    
    
                    numpyro.deterministic("m2_low", m2_low_)
                    # legacy alias only
                    mmin_ = m1_low_

                else:
                    # fixed low-end 
                    m1_low_ = DPUC_low_end_m1_fixed
                    m2_low_ = DPUC_low_end_m2_fixed

                    numpyro.deterministic("m1_low", m1_low_)
                    numpyro.deterministic("m2_low", m2_low_)
                        
            else:
                mmin_ = jnp.asarray(0.0, dtype=jnp.float64)
                m1_low_ = mmin_
                m2_low_ = mmin_
            
            
            if DP_truncate_up:
            
                if DPUC_mmax_fixed > 0.0:
                    mmax_ = numpyro.deterministic(
                        "mmax_DP",
                        jnp.asarray(DPUC_mmax_fixed, dtype=jnp.float64),
                    )
            
                else:
                    # Interpret priors["mmax_DP"] = [floor, q95], not [low, high].
                    # This mirrors the m_high prior used in the parametric mass model.
                    mmax_DP_floor = float(priors["mmax_DP"][0])
                    mmax_DP_q95 = float(priors["mmax_DP"][1])
            
                    mmax_ = bounded_sigmoid(
                        "mmax_DP",
                        mmax_DP_floor,
                        mmax_DP_q95,
                        raw_sigma=1.0,
                    )
            
            else:
                mmax_ = jnp.asarray(10000.0, dtype=jnp.float64)

            if DP_m1_env:
                if DP_m1_env_form == "smooth_broken_pl_2breaks":
                    for key in (
                        "DP_m1_env_alpha0",
                        "DP_m1_env_alpha1",
                        "DP_m1_env_alpha2",
                        "DP_m1_env_mb1",
                        "DP_m1_env_mb2",
                    ):
                        if key not in priors:
                            raise KeyError(
                                "DP_m1_env_form='smooth_broken_pl_2breaks' requires "
                                f"priors[{key!r}]."
                            )

                    alpha_ = bounded_sigmoid(
                        "DP_m1_env_alpha0",
                        priors["DP_m1_env_alpha0"][0], priors["DP_m1_env_alpha0"][1],
                        raw_sigma=1.0,
                    )
                    alpha1_env_ = bounded_sigmoid(
                        "DP_m1_env_alpha1",
                        priors["DP_m1_env_alpha1"][0], priors["DP_m1_env_alpha1"][1],
                        raw_sigma=1.0,
                    )
                    alpha2_env_ = bounded_sigmoid(
                        "DP_m1_env_alpha2",
                        priors["DP_m1_env_alpha2"][0], priors["DP_m1_env_alpha2"][1],
                        raw_sigma=1.0,
                    )
                    mb1_env_ = bounded_sigmoid(
                        "DP_m1_env_mb1",
                        priors["DP_m1_env_mb1"][0], priors["DP_m1_env_mb1"][1],
                        raw_sigma=1.0,
                    )
                    mb2_env_ = bounded_sigmoid(
                        "DP_m1_env_mb2",
                        priors["DP_m1_env_mb2"][0], priors["DP_m1_env_mb2"][1],
                        raw_sigma=1.0,
                    )
                    if priors["DP_m1_env_mb2"][0] <= priors["DP_m1_env_mb1"][1]:
                        numpyro.factor(
                            "DP_m1_env_order_mb",
                            jnp.where(mb2_env_ > mb1_env_, 0.0, -1.0e100),
                        )
                else:
                    # Historical Normal prior for every single power-law envelope.
                    # alpha1 specifies its central 95% interval, not hard bounds.
                    alpha_ = normal_from_bounds_95(
                        "alpha",
                        priors["alpha1"][0], priors["alpha1"][1],
                    )
                    alpha1_env_ = alpha_
                    alpha2_env_ = alpha_
                    mb1_env_ = jnp.asarray(1.0, dtype=jnp.float64)
                    mb2_env_ = jnp.asarray(1.0, dtype=jnp.float64)
            else:
                alpha_ = jnp.asarray(0.0, dtype=jnp.float64)
                alpha1_env_ = alpha_
                alpha2_env_ = alpha_
                mb1_env_ = jnp.asarray(1.0, dtype=jnp.float64)
                mb2_env_ = jnp.asarray(1.0, dtype=jnp.float64)

            if DP_q_env:
                #beta_ = normal_from_bounds_95("beta_q", priors["beta"][0], priors["beta"][1])

                beta_ = bounded_sigmoid(
                        "beta_q",
                        priors["beta"][0], priors["beta"][1],
                        raw_sigma=1.0,
                    )

                
            else:
                beta_ = jnp.asarray(0.0, dtype=jnp.float64)

            sd = numpyro.deterministic("sig", sigs)

            if DP_corr_m1m2 and mass_model == "DPUC_Primary":
                raise ValueError("DPUC_Primary/DPUC_TruncM2 does not use DP_corr_m1m2.")

            if DP_corr_m1m2:
                #eta_rho = 4.0
                #f_minor = 0.5
                eta_rho = jnp.asarray(DP_corr_eta_rho, dtype=jnp.float64)
                f_minor = jnp.asarray(DP_corr_f_minor, dtype=jnp.float64)
                rho_max = jnp.sqrt(jnp.maximum(0.0, 1.0 - f_minor**2))

                rho_u = numpyro.sample(
                    "rho_u",
                    dist.Beta(
                        eta_rho * jnp.ones((N_DP_comp_max_np,), dtype=jnp.float64),
                        eta_rho * jnp.ones((N_DP_comp_max_np,), dtype=jnp.float64),
                    ).to_event(1),
                )

                rho = numpyro.deterministic(
                    "rho",
                    jnp.asarray(rho_max, dtype=jnp.float64) * (2.0 * rho_u - 1.0),
                )
            else:
                rho = None


            if mass_model == "DPUC_TruncM2":
                rho_reg = rho if rho is not None else jnp.zeros((N_DP_comp_max_np,), dtype=jnp.float64)
                rho_reg = jnp.clip(rho_reg, -1.0 + 1e-12, 1.0 - 1e-12)
                log_sig1_reg = jnp.log(jnp.clip(sd[0], 1e-300, jnp.inf))
                log_sig2_reg = jnp.log(jnp.clip(sd[1], 1e-300, jnp.inf))
                log_area_reg = log_sig1_reg + log_sig2_reg + 0.5 * jnp.log(jnp.clip(1.0 - rho_reg**2, 1e-300, jnp.inf))
                aspect_raw_reg = log_sig1_reg - log_sig2_reg
                aspect_reg_smooth = jnp.sqrt(aspect_raw_reg**2 + 1e-12)

                numpyro.deterministic("TruncM2_log_area_components", log_area_reg)
                numpyro.deterministic("TruncM2_aspect_components", aspect_reg_smooth)

                if DPUC_TruncM2_area_reg and DPUC_TruncM2_log_area_lambda > 0.0:
                    sA = jnp.asarray(DPUC_TruncM2_log_area_width, dtype=jnp.float64)
                    hA = sA * jax.nn.softplus((jnp.asarray(DPUC_TruncM2_log_area_min, dtype=jnp.float64) - log_area_reg) / sA)
                    area_penalty = -jnp.asarray(DPUC_TruncM2_log_area_lambda, dtype=jnp.float64) * jnp.sum(w * hA**2)
                    numpyro.deterministic("TruncM2_area_reg_penalty", area_penalty)
                    numpyro.factor("TruncM2_area_reg", area_penalty)

                if DPUC_TruncM2_aspect_reg and DPUC_TruncM2_aspect_lambda > 0.0:
                    sR = jnp.asarray(DPUC_TruncM2_aspect_width, dtype=jnp.float64)
                    hR = sR * jax.nn.softplus((aspect_reg_smooth - jnp.asarray(DPUC_TruncM2_aspect_max, dtype=jnp.float64)) / sR)
                    aspect_penalty = -jnp.asarray(DPUC_TruncM2_aspect_lambda, dtype=jnp.float64) * jnp.sum(w * hR**2)
                    numpyro.deterministic("TruncM2_aspect_reg_penalty", aspect_penalty)
                    numpyro.factor("TruncM2_aspect_reg", aspect_penalty)


            # Shared DPUC / PL_DPUC physical gate widths.
            #
            # Low-side widths can be either fixed:
            #     DPUC_low_taper_deltas = [dm1, dm2]
            # or sampled:
            #     DPUC_low_taper_deltas = [-1]
            #
            # Sampled mode uses bounded log-space logistic-normal priors
            # with hard support from priors["delta_m1"], priors["delta_m2"].
            if DPUC_low_taper_sample:
                # deltam_PL_DPUC_m1 = bounded_sigmoid_logpositive_median(
                #     "deltam_PL_DPUC_m1",
                #     priors["delta_m1"][0],
                #     priors["delta_m1"][1],
                #     median=2.0,
                #     raw_sigma=1.0,
                # )

                deltam_PL_DPUC_m1 = bounded_sigmoid_logpositive("deltam_PL_DPUC_m1", priors["delta_m1"][0], priors["delta_m1"][1], raw_sigma=1.0 )

                if not DPUC_low_end_share:
                    # deltam_PL_DPUC_m2 = bounded_sigmoid_logpositive_median(
                    #     "deltam_PL_DPUC_m2",
                    #     priors["delta_m2"][0],
                    #     priors["delta_m2"][1],
                    #     median=2.0,
                    #     raw_sigma=1.0,
                    # )
                    deltam_PL_DPUC_m2 = bounded_sigmoid_logpositive("deltam_PL_DPUC_m2", priors["delta_m2"][0], priors["delta_m2"][1], raw_sigma=1.0 )
                else:
                    deltam_PL_DPUC_m2 = numpyro.deterministic( "deltam_PL_DPUC_m2", deltam_PL_DPUC_m1)

            else:
                deltam_PL_DPUC_m1 = numpyro.deterministic(
                    "deltam_PL_DPUC_m1",
                    jnp.asarray(DPUC_low_taper_delta_m1_fixed, dtype=jnp.float64),
                )
                if not DPUC_low_end_share:
                    deltam_PL_DPUC_m2 = numpyro.deterministic(
                        "deltam_PL_DPUC_m2",
                        jnp.asarray(DPUC_low_taper_delta_m2_fixed, dtype=jnp.float64),
                    )
                else:
                    deltam_PL_DPUC_m2 = numpyro.deterministic(
                        "deltam_PL_DPUC_m2",
                        deltam_PL_DPUC_m1,
                    )

            if DPUC_high_taper_sample:
                # Positive, bounded high-end smoothing width.
                # Uses priors["delta_m2"] support by default unless you add a dedicated prior key.
                delta_high_low = float(priors.get("delta_mhigh", [5, 50.0] )[0])
                delta_high_up = float(priors.get("delta_mhigh", [5, 50.0] )[1])

                deltam_PL_DPUC_high = bounded_sigmoid(
                    "deltam_PL_DPUC_high",
                    delta_high_low,
                    delta_high_up,
                    raw_sigma=1.0,
                )
            else:
                deltam_PL_DPUC_high = numpyro.deterministic(
                    "deltam_PL_DPUC_high",
                    jnp.asarray(DPUC_high_taper_delta, dtype=jnp.float64),
                )
            
            numpyro.deterministic(
                "PL_DPUC_m1_taper_end",
                m1_low_ + deltam_PL_DPUC_m1,
            )

            numpyro.deterministic(
                "PL_DPUC_m2_taper_end",
                m2_low_ + deltam_PL_DPUC_m2,
            )

            numpyro.deterministic(
                "PL_DPUC_high_taper_end",
                mmax_ + deltam_PL_DPUC_high,
            )

            if mass_model in ("DPUC_Primary", "DPUC_TruncM2", ):
                if DP_corr_m1m2 and mass_model == "DPUC_Primary":
                    raise ValueError("DPUC_Primary/DPUC_TruncM2 does not support DP_corr_m1m2.")

                needs_logw2 = (
                    mass_model == 'DPUC_TruncM2' and (not DPUC_TruncM2_shared_weights)
                )

                if mass_model == "DPUC_TruncM2" and DPUC_TruncM2_shared_weights:
                    logw2 = logw
                elif needs_logw2:
                    if "SB" in DP_prior:
                        beta_w2 = numpyro.sample(
                            "beta_w2",
                            dist.Beta(
                                jnp.ones((N_DP_comp_max_np,), dtype=jnp.float64),
                                alpha * jnp.ones((N_DP_comp_max_np,), dtype=jnp.float64),
                            ).to_event(1),
                        )
                        w2 = beta_w2 * jnp.cumprod(
                            jnp.concatenate([
                                jnp.ones((1,), dtype=jnp.float64),
                                1.0 - beta_w2[:-1],
                            ])
                        )
                        w2 = w2 / jnp.sum(w2)
                        logw2 = numpyro.deterministic("logw2", jnp.log(w2))
                    elif DP_prior == "dirichelet":
                        w2 = numpyro.sample(
                            "w2",
                            dist.Dirichlet(jnp.ones((N_DP_comp_max_np,), dtype=jnp.float64)),
                        )
                        logw2 = numpyro.deterministic("logw2", jnp.log(w2))
                    elif DP_prior == "softmax":
                        raw_w2 = numpyro.sample(
                            "raw_w2",
                            dist.Normal(0.0, sigma_softmax).expand((N_DP_comp_max_np,)).to_event(1),
                        )
                        logw2 = numpyro.deterministic("logw2", jax.nn.log_softmax(raw_w2))
                    else:
                        raise ValueError(f"Unknown DP_prior={DP_prior!r}")



                if mass_model == "DPUC_TruncM2":
                    rho_for_lambda = rho if rho is not None else jnp.zeros((N_DP_comp_max_np,), dtype=jnp.float64)
                    if DPUC_TruncM2_shared_weights:
                        Lambda_list += [
                            mu,
                            sd,
                            logw,
                            rho_for_lambda,
                            m1_low_,
                            m2_low_,
                            mmax_,
                            alpha_,
                            beta_,
                            *([alpha1_env_, alpha2_env_, mb1_env_, mb2_env_] if DP_m1_env_form == "smooth_broken_pl_2breaks" else []),
                            deltam_PL_DPUC_m1,
                            deltam_PL_DPUC_m2,
                            deltam_PL_DPUC_high,
                        ]
                    else:
                        Lambda_list += [
                            mu,
                            sd,
                            logw,
                            logw2,
                            rho_for_lambda,
                            m1_low_,
                            m2_low_,
                            mmax_,
                            alpha_,
                            beta_,
                            *([alpha1_env_, alpha2_env_, mb1_env_, mb2_env_] if DP_m1_env_form == "smooth_broken_pl_2breaks" else []),
                            deltam_PL_DPUC_m1,
                            deltam_PL_DPUC_m2,
                            deltam_PL_DPUC_high,
                        ]
                else:
                    Lambda_list += [
                        mu,
                        sd,
                        logw,
                        m1_low_,
                        m2_low_,
                        mmax_,
                        alpha_,
                        beta_,
                        *([alpha1_env_, alpha2_env_, mb1_env_, mb2_env_] if DP_m1_env_form == "smooth_broken_pl_2breaks" else []),
                        deltam_PL_DPUC_m1,
                        deltam_PL_DPUC_m2,
                        deltam_PL_DPUC_high,
                    ]

            else:
                if DP_corr_m1m2:
                    Lambda_list += [
                        mu,
                        sd,
                        logw,
                        m1_low_,
                        m2_low_,
                        mmax_,
                        alpha_,
                        beta_,
                        *([alpha1_env_, alpha2_env_, mb1_env_, mb2_env_] if DP_m1_env_form == "smooth_broken_pl_2breaks" else []),
                        rho,
                        deltam_PL_DPUC_m1,
                        deltam_PL_DPUC_m2,
                        deltam_PL_DPUC_high,
                    ]
                else:
                    Lambda_list += [
                        mu,
                        sd,
                        logw,
                        m1_low_,
                        m2_low_,
                        mmax_,
                        alpha_,
                        beta_,
                        *([alpha1_env_, alpha2_env_, mb1_env_, mb2_env_] if DP_m1_env_form == "smooth_broken_pl_2breaks" else []),
                        deltam_PL_DPUC_m1,
                        deltam_PL_DPUC_m2,
                        deltam_PL_DPUC_high,
                    ]



        elif mass_model == "DPLDP-z":

            if priors["alpha1_0"] != priors["alpha2_0"]:
                raise ValueError(
                    f"alpha1/alpha2 priors differ: {priors['alpha1_0']} vs {priors['alpha2_0']}"
                )
        
            # -------------------------
            # Low-z alpha reparam
            # -------------------------
            a_low, a_high = priors["alpha1_0"][0], priors["alpha1_0"][1]
            a_mid = 0.5 * (a_low + a_high)
            a_sig = (a_high - a_low) / (2.0 * NORM_Q95)
        
            a_bar = numpyro.sample(
                "alpha_bar",
                dist.Normal(a_mid, a_sig),
            )
            a_diff = numpyro.sample(
                "alpha_diff",
                dist.Normal(0.0, jnp.sqrt(2.0) * a_sig),
            )
        
            alpha1_0 = numpyro.deterministic("alpha1_0", a_bar - 0.5 * a_diff)
            alpha2_0 = numpyro.deterministic("alpha2_0", a_bar + 0.5 * a_diff)
        
            beta_ = normal_from_bounds_95(
                "beta",
                priors["beta"][0],
                priors["beta"][1],
            )
        
            mb_0 = bounded_sigmoid(
                "mb_0",
                priors["mb_0"][0],
                priors["mb_0"][1],
                raw_sigma=1.0,
            )
        
            mu1_0 = bounded_sigmoid(
                "mu1_0",
                priors["mu1_0"][0],
                priors["mu1_0"][1],
                raw_sigma=1.25,
            )
            mu2_0 = bounded_sigmoid(
                "mu2_0",
                priors["mu2_0"][0],
                priors["mu2_0"][1],
                raw_sigma=1.25,
            )
        
            sigma1_0 = floored_lognormal_q95(
                "sigma1_0",
                priors["sigma1_0"][0],
                priors["sigma1_0"][1],
                median_frac=0.2,
            )
            sigma2_0 = floored_lognormal_q95(
                "sigma2_0",
                priors["sigma2_0"][0],
                priors["sigma2_0"][1],
                median_frac=0.3,
            )
        
            # -------------------------
            # m1_low, m2_low triangle
            # -------------------------
            u = unit_interval_sigmoid("u", raw_sigma=1.0)
            m1_low_ = 3.0 + (10.0 - 3.0) * u**1.5
            numpyro.deterministic("m1_low", m1_low_)
        
            v = unit_interval_sigmoid("v", raw_sigma=1.0)
            m2_low_ = 3.0 + v * (m1_low_ - 3.0)
            numpyro.deterministic("m2_low", m2_low_)
        
            # -------------------------
            # m_high: same as PyMC, force_uniform_mhigh=False
            # -------------------------
            mhigh_floor = priors["m_high"][0]
            mmax_median = 0.5 * (priors["m_high"][0] + priors["m_high"][1])
            mmax_q95 = priors["m_high"][1]
        
            delta_med = jnp.maximum(mmax_median - mhigh_floor, 1e-6)
            delta_q95 = jnp.maximum(mmax_q95 - mhigh_floor, 1e-6)
        
            mu_delta = jnp.log(delta_med)
            sigma_delta = (jnp.log(delta_q95) - mu_delta) / NORM_Q95
        
            delta_mhigh = numpyro.sample(
                "delta_mhigh",
                dist.LogNormal(loc=mu_delta, scale=sigma_delta),
            )
            m_high_ = mhigh_floor + delta_mhigh
            numpyro.deterministic("m_high", m_high_)
        
            # -------------------------
            # tapers
            # -------------------------
            delta_m1_ = floored_lognormal_q95(
                "delta_m1",
                priors["delta_m1"][0],
                priors["delta_m1"][1],
                median_frac=0.3,
            )
            numpyro.deterministic("m1_taper_end", m1_low_ + delta_m1_)
        
            delta_m2_ = floored_lognormal_q95(
                "delta_m2",
                priors["delta_m2"][0],
                priors["delta_m2"][1],
                median_frac=0.3,
            )
            numpyro.deterministic("m2_taper_end", m2_low_ + delta_m2_)
        
            epsilon_ = jnp.asarray(0.1, dtype=jnp.float64)
            numpyro.deterministic("epsilon", epsilon_)
        
            # -------------------------
            # secondary-mass gap
            # -------------------------
            if has_m2_break:
                m_g_ = numpyro.sample(
                    "m_g",
                    dist.Uniform(priors["m_g"][0], priors["m_g"][1]),
                )
                w_g_ = numpyro.sample(
                    "w_g",
                    dist.Uniform(priors["w_g"][0], priors["w_g"][1]),
                )
                sig_g_l_ = jnp.asarray(1e-2, dtype=jnp.float64)
                sig_g_h_ = jnp.asarray(1e-2, dtype=jnp.float64)
            else:
                m_g_ = jnp.asarray(45.0, dtype=jnp.float64)
                w_g_ = jnp.asarray(70.0, dtype=jnp.float64)
                sig_g_l_ = jnp.asarray(1e-2, dtype=jnp.float64)
                sig_g_h_ = jnp.asarray(1e-2, dtype=jnp.float64)
        
            # -------------------------
            # low-z mixture weights
            # -------------------------
            lam0_prior = priors.get("lambda0_vec_0", "Dirichlet(1,1,1)")
        
            if isinstance(lam0_prior, str) and lam0_prior.startswith("Dirichlet"):
                inside = lam0_prior[len("Dirichlet("):-1]
                alphas = [float(x.strip()) for x in inside.split(",")]
            else:
                alphas = lam0_prior
        
            lambda_vec0 = numpyro.sample(
                "lambda0_vec",
                dist.Dirichlet(jnp.asarray(alphas, dtype=jnp.float64)),
            )
        
            lambda0_0 = numpyro.deterministic("lambda0_0", lambda_vec0[0])
            lambda1_0 = numpyro.deterministic("lambda1_0", lambda_vec0[1])
            lambda2_0 = numpyro.deterministic("lambda2_0", lambda_vec0[2])
        
            # -------------------------
            # evolving hyperparameters
            # -------------------------
            alpha1_inf_, z_alpha1_, dz_alpha1_ = evo_triplet_numpyro(
                "alpha1",
                theta0=alpha1_0,
                ivals=ivals,
                priors=priors,
            )
        
            alpha2_inf_, z_alpha2_, dz_alpha2_ = evo_triplet_numpyro(
                "alpha2",
                theta0=alpha2_0,
                ivals=ivals,
                priors=priors,
            )
        
            if not vary_mb:
                mb_inf_ = numpyro.deterministic("mb_inf", mb_0)
                z_mb_ = numpyro.deterministic(
                    "z_mb",
                    jnp.asarray(0.0, dtype=jnp.float64),
                )
                dz_mb_ = numpyro.deterministic(
                    "dz_mb",
                    jnp.asarray(1.0, dtype=jnp.float64),
                )
            else:
                mb_inf_, z_mb_, dz_mb_ = evo_triplet_numpyro(
                    "mb",
                    theta0=mb_0,
                    ivals=ivals,
                    priors=priors,
                    positive=True,
                    eps_pos=5.0,
                )


        
            mu1_inf_, z_mu1_, dz_mu1_ = evo_triplet_numpyro(
                "mu1",
                theta0=mu1_0,
                ivals=ivals,
                priors=priors,
                positive=True,
                eps_pos=5.0,
            )
        
            sigma1_inf_, z_sigma1_, dz_sigma1_ = evo_triplet_numpyro(
                "sigma1",
                theta0=sigma1_0,
                ivals=ivals,
                priors=priors,
                positive=True,
                eps_pos=0.1,
            )
        
            mu2_inf_, z_mu2_, dz_mu2_ = evo_triplet_numpyro(
                "mu2",
                theta0=mu2_0,
                ivals=ivals,
                priors=priors,
                positive=True,
                eps_pos=5.0,
            )
        
            sigma2_inf_, z_sigma2_, dz_sigma2_ = evo_triplet_numpyro(
                "sigma2",
                theta0=sigma2_0,
                ivals=ivals,
                priors=priors,
                positive=True,
                eps_pos=0.1,
            )
        
            # -------------------------
            # high-z mixture weights + shared transition
            # -------------------------
            lambda_vec_inf = numpyro.sample(
                "lambda_inf_vec",
                dist.Dirichlet(jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float64)),
            )
        
            lambda0_inf_ = numpyro.deterministic("lambda0_inf", lambda_vec_inf[0])
            lambda1_inf_ = numpyro.deterministic("lambda1_inf", lambda_vec_inf[1])
            lambda2_inf_ = numpyro.deterministic("lambda2_inf", lambda_vec_inf[2])
        
            z_t_prior = priors.get("z_t", (0.05, 1.5))
            dz_prior = priors.get("dz", (0.05, 2.0))
        
            z_lambda_ = numpyro.sample(
                "z_lambda",
                dist.Uniform(z_t_prior[0], z_t_prior[1]),
            )
        
            log_dz_lambda_ = numpyro.sample(
                "log_dz_lambda",
                dist.Uniform(jnp.log(dz_prior[0]), jnp.log(dz_prior[1])),
            )
        
            dz_lambda_ = numpyro.deterministic("dz_lambda", jnp.exp(log_dz_lambda_))
        
            if simplex_repair:
                raise ValueError("simplex_repair for DPLDP-z is not implemented in NumPyro branch.")
        
            # -------------------------
            # Pack in exact core order: 21 low-z + 26 evo = 47
            # -------------------------
            lambdaBBHmass_lowz_ = [
                alpha1_0, alpha2_0, mb_0,
                mu1_0, sigma1_0, mu2_0, sigma2_0,
                m1_low_, m_high_, delta_m1_,
                lambda0_0, lambda1_0, lambda2_0,
                beta_, m2_low_, delta_m2_,
                epsilon_, m_g_, w_g_, sig_g_l_, sig_g_h_,
            ]
        
            evo_params_ = [
                alpha1_inf_, z_alpha1_, dz_alpha1_,
                alpha2_inf_, z_alpha2_, dz_alpha2_,
                mb_inf_, z_mb_, dz_mb_,
                mu1_inf_, z_mu1_, dz_mu1_,
                sigma1_inf_, z_sigma1_, dz_sigma1_,
                mu2_inf_, z_mu2_, dz_mu2_,
                sigma2_inf_, z_sigma2_, dz_sigma2_,
                lambda0_inf_, lambda1_inf_, lambda2_inf_,
                z_lambda_, dz_lambda_,
            ]
        
            Lambda_list += [*lambdaBBHmass_lowz_, *evo_params_]

        if not marginal_R0:
            R0 = numpyro.sample("R0", dist.Uniform(priors["R0"][0], priors["R0"][1]))
        else:
            R0 = jnp.asarray(1.0, dtype=jnp.float64)
        
        lR0 = jnp.log(R0)
        numpyro.deterministic("lR0", lR0)


        # -------------------------
        # Pack Lambda -> jnp array
        # -------------------------
        Lambda = jnp.concatenate([
            jnp.ravel(jnp.asarray(z, dtype=jnp.float64))
            for z in Lambda_list
        ], axis=0)

        numpyro.deterministic("Lambda", Lambda)

    


        # -------------------------
        # GP dL branch (alternative to parametric Xi0/nXi0; Lambda layout unchanged)
        # -------------------------
        gp_state = None
        if is_GP_dL:
            if U is None:
                raise ValueError("U must be provided for GP dL amplitude prior")

            alpha_eta_ = jnp.asarray(alpha_eta, dtype=jnp.float64)
            alpha_ell_ = jnp.asarray(alpha_ell, dtype=jnp.float64)
            alpha_large_ = jnp.asarray(alpha_large, dtype=jnp.float64)

            lambda_eta = jnp.log(1.0 / alpha_eta_) / jnp.asarray(U, dtype=jnp.float64)

            d_GP = 1.0

            if GP_kernel == "gibbs_resolution":
                lambda_ell = -jnp.log(alpha_ell_) * jnp.asarray(ell0_min, dtype=jnp.float64) ** (d_GP / 2.0)
                lambda_large = -jnp.log(alpha_large_) / jnp.asarray(ell0_max, dtype=jnp.float64)
            else:
                lambda_ell = -jnp.log(alpha_ell_) * jnp.asarray(ell_min, dtype=jnp.float64) ** (d_GP / 2.0)
                lambda_large = -jnp.log(alpha_large_) / jnp.asarray(ell_max, dtype=jnp.float64)

            if GP_prior != "frechet":
                raise NotImplementedError("NumPyro GP dL port currently implements GP_prior='frechet' only")

            ell = numpyro.sample(
                "ell",
                dist.TransformedDistribution(
                    dist.Exponential(lambda_ell),
                    transforms.PowerTransform(-2.0),
                ),
            )
            if large_ell_penalty:
                numpyro.factor("pc_large_ell", -lambda_large * ell)

            eta = numpyro.sample("eta", dist.Exponential(lambda_eta))

            bk = JAXBackend()

            x_gp_nodes = jnp.log1p(z_gp_nodes)
            if GP_kernel == "gibbs_resolution":
                K = nutils.gibbs_kernel_matrix(
                    bk,
                    x_gp_nodes,
                    ell,
                    eta,
                    jnp.asarray(x_res_centers, dtype=jnp.float64),
                    jnp.asarray(r_res_centers, dtype=jnp.float64),
                    jitter=GP_jitter,
                )
            else:
                K = nutils.matern52_kernel_matrix(bk, x_gp_nodes, ell, eta, jitter=GP_jitter)

            L_gp = jnp.linalg.cholesky(K)
            f_rotated = numpyro.sample(
                "f_rotated",
                dist.Normal(0.0, 1.0).expand((z_gp_nodes.shape[0],)).to_event(1),
            )
            f_nodes = L_gp @ f_rotated
            numpyro.deterministic("f_nodes", f_nodes)

            eps_q = 1e-12 if eps_DE < 0 else eps_DE
            gp_grid = cosmo.z_from_dL_at_monotone(
                JAXBackend(),
                jnp.asarray([1.0], dtype=jnp.float64),
                z_nodes=z_gp_nodes,
                f_nodes=f_nodes,
                H0=H0_,
                Om=Om_,
                w0=w0_,
                z_fine=z_gp_fine,
                eps_q=eps_q,
                integrate_dc=integrate_dc,
            )



            gp_state = {
                "z_fine": gp_grid["z_fine"],
                "dL_gw_fine": gp_grid["dL_gw_fine"],
                "log_ddL_dz_fine": gp_grid["log_ddL_dz_fine"],
            }

            ###########
            # stability diagnostics only
            logXi_fine = gp_grid["logXi_fine"]
            dL_gw_fine = gp_grid["dL_gw_fine"]
            q_fine = gp_grid["q_fine"]

            numpyro.deterministic("logXi_fine", gp_grid["logXi_fine"])
            numpyro.deterministic("g_fine", gp_grid["g_fine"])
            numpyro.deterministic("q_fine", gp_grid["q_fine"])
            numpyro.deterministic("b_fine", gp_grid["b_fine"])
            numpyro.deterministic("dL_gw_fine", gp_grid["dL_gw_fine"])
            numpyro.deterministic("log_ddL_dz_fine", gp_grid["log_ddL_dz_fine"])

            min_dL_step = jnp.min(jnp.diff(dL_gw_fine))


            numpyro.deterministic("min_dL_step", jnp.min(jnp.diff(gp_grid["dL_gw_fine"])))
            numpyro.deterministic("min_q_fine", jnp.min(gp_grid["q_fine"]))

            logXi_abs_max = numpyro.deterministic("logXi_abs_max", jnp.max(jnp.abs(gp_grid["logXi_fine"])))
            numpyro.deterministic("logXi_range", jnp.max(gp_grid["logXi_fine"]) - jnp.min(gp_grid["logXi_fine"]))

            dL_steps = jnp.diff(gp_grid["dL_gw_fine"])
            numpyro.deterministic("dL_step_min", jnp.min(dL_steps))
            numpyro.deterministic("dL_step_max", jnp.max(dL_steps))
            numpyro.deterministic("dL_step_ratio", jnp.max(dL_steps) / jnp.maximum(jnp.min(dL_steps), 1e-300))


            if logXi_guard:
                excess = jnp.maximum(
                    logXi_abs_max - jnp.asarray(logXi_abs_max_allowed, dtype=jnp.float64),
                    0.0,
                )
                numpyro.factor(
                    "logXi_amp_guard",
                    -jnp.asarray(logXi_guard_strength, dtype=jnp.float64) * excess**2,
                )


            def add_gp_support_guard(x_evt=None):
                if gp_grid is None:
                    return

                dL_min = gp_grid["dL_gw_fine"][0]
                dL_max = gp_grid["dL_gw_fine"][-1]

                if x_evt is not None:
                    # Historical GP guard uses latent coordinate 2 here. Its physical
                    # decoder discrepancy is recorded for a separate scientific decision.
                    dL_evt = jnp.exp(x_evt[:, 2])
                else:
                    # PE marginal sample path
                    dL_evt = data.dL_pe

                below = dL_evt < dL_min
                above = dL_evt > dL_max
                oob = jnp.any(below | above)

                numpyro.deterministic("gp_dL_min", dL_min)
                numpyro.deterministic("gp_dL_max", dL_max)
                numpyro.deterministic("evt_dL_min", jnp.min(dL_evt))
                numpyro.deterministic("evt_dL_max", jnp.max(dL_evt))

                # Extra diagnostics for checking whether the guard is causing bad geometry.
                numpyro.deterministic("gp_margin_low", jnp.min(dL_evt - dL_min))
                numpyro.deterministic("gp_margin_high", jnp.min(dL_max - dL_evt))
                numpyro.deterministic("gp_oob_count", jnp.sum(below | above))
                numpyro.deterministic("gp_oob_below_count", jnp.sum(below))
                numpyro.deterministic("gp_oob_above_count", jnp.sum(above))

                if GP_oob_guard:
                    numpyro.factor(
                        "interp_oob_penalty",
                        jnp.where(oob, -1.0e6, 0.0),
                    )





        # -------------------------
        # Likelihood
        # -------------------------
        
        if not pop_only:
            N = int(data.Nobs)
            nd = int(data.mus_s.shape[1])
            x = numpyro.sample(
                "x",
                dist.Normal(0.0, 1.0).expand((N, nd)).to_event(2),
            )
            if is_GP_dL:
                add_gp_support_guard(x_evt=x)
            ll, log_lik_var_sg = loglik(Lambda, x, lR0=lR0, gp_state=gp_state)
            log_px_total = (
                -0.5 * jnp.sum(x * x)
                -0.5 * x.size * jnp.log(2.0 * jnp.pi)
            )
        
            numpyro.deterministic("loglik_raw_importance", ll)
            numpyro.deterministic("log_px_total", log_px_total)
            numpyro.deterministic("loglik_corrected", ll + log_px_total)
        
        else:
            if is_GP_dL:
                add_gp_support_guard(x_evt=None)
            ll, log_lik_var_sg = loglik(Lambda, lR0=lR0, gp_state=gp_state)
            numpyro.deterministic("loglik_corrected", ll)



        numpyro.factor("likelihood", ll)


        numpyro.deterministic("log_lik_var_min_used", jnp.asarray(log_lik_var_min, dtype=jnp.float64))
        numpyro.deterministic("log_lik_var_sg_used", jnp.asarray(log_lik_var_sg, dtype=jnp.float64))
        numpyro.deterministic(
            "passes_log_lik_var_gate_used",
            jnp.asarray(log_lik_var_sg <= log_lik_var_min)
        )

        ## likelihood variance bound:
        if sel_smoothing=='sigmoid':
            #gate_llv = jax.nn.log_sigmoid( (log_lik_var_min - log_lik_var_sg) / 0.01 )
            gate_llv = log_lik_var_high_gate(
                JAXBackend(),
                log_lik_var_sg,
                log_lik_var_min,
                kind="sigmoid",
                # Preserve the historical GP and DPGMM gate widths.
                width=0.01 if is_GP_dL else 0.05,
                p_at_min=0.99,
                eps_high=1.0e-7,
            )
        elif sel_smoothing=='none':
            ## hard version 
            gate_llv = jnp.where(log_lik_var_sg <= log_lik_var_min, 0.0, -1e30)
        elif sel_smoothing=='PL':
            # Heinzel et al 2025
            v = jnp.maximum(log_lik_var_sg - log_lik_var_min, 0.0)
            gate_llv = -100.0 * v**2
        elif sel_smoothing=='x30':
            # Callister Farr 2023
            x = jnp.maximum(log_lik_var_sg / log_lik_var_min, 0.0)
            log_x = jnp.log(jnp.maximum(x, 1e-300))
            gate_llv = -jnp.logaddexp(0.0, 30.0 * log_x)
        else:
            raise ValueError("sel_smoothing takes values sigmoid, PL, none, x30, got %s"%sel_smoothing)

        
        numpyro.factor("bound_log_lik_var", gate_llv)


        # optional for debugging/outputs
        numpyro.deterministic("loglik", ll)
        numpyro.deterministic("log_lik_var", log_lik_var_sg)
        
        

    model_init_info = {
        "is_GP_dL": bool(is_GP_dL),
        "GP_kernel": GP_kernel,
        "GP_jitter": float(GP_jitter),
        "eps_DE": float(eps_DE),
        "z_gp_nodes": None if z_gp_nodes is None else np.asarray(z_gp_nodes).tolist(),
        "z_gp_fine": None if z_gp_fine is None else np.asarray(z_gp_fine).tolist(),
        "x_res_centers": None if x_res_centers is None else np.asarray(x_res_centers).tolist(),
        "r_res_centers": None if r_res_centers is None else np.asarray(r_res_centers).tolist(),
        "mass_model": mass_model,
        "rate_model": rate_model,
        "DPUC_Primary_q_norm": DPUC_Primary_q_norm,
        "N_DP_comp_max": int(N_DP_comp_max_np),
        "M_active": int(M_active),
        "DPUC_TruncM2_shared_weights": bool(DPUC_TruncM2_shared_weights),
        "DPUC_TruncM2_mu2_mode": DPUC_TruncM2_mu2_mode,
            "DPUC_TruncM2_secondary_coord": DPUC_TruncM2_secondary_coord,
        "DPUC_TruncM2_area_reg": bool(DPUC_TruncM2_area_reg),
        "DPUC_TruncM2_log_area_min": float(DPUC_TruncM2_log_area_min),
        "DPUC_TruncM2_log_area_width": float(DPUC_TruncM2_log_area_width),
        "DPUC_TruncM2_log_area_lambda": float(DPUC_TruncM2_log_area_lambda),
        "DPUC_TruncM2_aspect_reg": bool(DPUC_TruncM2_aspect_reg),
        "DPUC_TruncM2_aspect_max": float(DPUC_TruncM2_aspect_max),
        "DPUC_TruncM2_aspect_width": float(DPUC_TruncM2_aspect_width),
        "DPUC_TruncM2_aspect_lambda": float(DPUC_TruncM2_aspect_lambda),
    }

    if mass_model in ('DPUC', 'DPUC_Primary', 'DPUC_TruncM2'):
        model_init_info.update({
            "ndim_GMM": int(ndim_GMM),
            "lowmu1": float(lowmu1),
            "upmu1": float(upmu1),
            "lowmu2": float(lowmu2),
            "upmu2": float(upmu2),
            "lowsig1": float(lowsig1),
            "upsig1": float(upsig1),
            "lowsig2": float(lowsig2),
            "upsig2": float(upsig2),
            "U1_width_support": float(upsig1 - lowsig1),
            "U2_width_support": float(upsig2 - lowsig2),
            "DP_mu_bounds_source": DP_mu_bounds_source,
            "DP_sig_bounds_source": DP_sig_bounds_source,
            "q_mbound_mu_low": q_mbound_mu_low,
            "q_mbound_mu_up": q_mbound_mu_up,
            "q_mbound_sig_low": q_mbound_sig_low,
            "q_mbound_sig_up": q_mbound_sig_up,
            "L_small_1": float(L_small_1),
            "L_small_2": float(L_small_2),
            "center_eps_widths" : bool(center_eps_widths),
        "DP_corr_m1m2":bool(DP_corr_m1m2),
        "DP_corr_eta_rho":bool(DP_corr_eta_rho),
        "DP_corr_f_minor":bool(DP_corr_f_minor),
            "PL_DPUC_normalize_gmm": bool(PL_DPUC_normalize_gmm),
            "DPUC_kernel": DPUC_kernel,
            "DPUC_kernel_rcut": float(DPUC_kernel_rcut),
            "DPUC_kernel_dr": float(DPUC_kernel_dr),
            "DPUC_kernel_xcut": float(DPUC_kernel_xcut),
            "DPUC_kernel_ycut": float(DPUC_kernel_ycut),
            "DPUC_q_coord": DPUC_q_coord,
            "DPUC_TruncM2_mu2_mode": DPUC_TruncM2_mu2_mode,
            "DPUC_TruncM2_secondary_coord": DPUC_TruncM2_secondary_coord,
        "DPUC_TruncM2_area_reg": bool(DPUC_TruncM2_area_reg),
        "DPUC_TruncM2_log_area_min": float(DPUC_TruncM2_log_area_min),
        "DPUC_TruncM2_log_area_width": float(DPUC_TruncM2_log_area_width),
        "DPUC_TruncM2_log_area_lambda": float(DPUC_TruncM2_log_area_lambda),
        "DPUC_TruncM2_aspect_reg": bool(DPUC_TruncM2_aspect_reg),
        "DPUC_TruncM2_aspect_max": float(DPUC_TruncM2_aspect_max),
        "DPUC_TruncM2_aspect_width": float(DPUC_TruncM2_aspect_width),
        "DPUC_TruncM2_aspect_lambda": float(DPUC_TruncM2_aspect_lambda),
        "DPUC_TruncM2_weight_cap_reg": bool(DPUC_TruncM2_low_center_weight_reg),
        "DPUC_TruncM2_weight_cap": float(DPUC_TruncM2_low_center_weight_cap),
            "q_mu_low": None if q_mu_low is None else float(q_mu_low),
            "q_mu_up": None if q_mu_up is None else float(q_mu_up),
            "DPUC_q_mu_bounds_source": DPUC_q_mu_bounds_source,
            "DPUC_low_taper_deltas": list(DPUC_low_taper_deltas),
            "DPUC_low_taper_sample": bool(DPUC_low_taper_sample),
            "DPUC_mmax_fixed": float(DPUC_mmax_fixed),
        "DPUC_high_taper_delta": float(DPUC_high_taper_delta),
        "DPUC_high_taper_sample": bool(DPUC_high_taper_sample),
        })



    return model_numpyro, data, core, loglik, loglik_components, model_init_info
