from __future__ import annotations

from typing import Tuple
import numpy as np
import logging
import jax.numpy as jnp
from constants import _PI
import jax


# ---------------------------------------------------------------------
# GW stuff
# 
# ---------------------------------------------------------------------



def _log_setup(*values, sep=" ", end="\n"):
    logger = logging.getLogger(__name__)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("%s", sep.join(str(value) for value in values))

def Mcq_from_m1m2(m1, m2):
   
    Mc  = ((m1*m2)**(3./5.))/((m1+m2)**(1./5.))
    q = m2/m1
    
    return Mc, q



# ---------------------------------------------------------------------
# PE-resolution floors for DPUC / DP-GMM
# ---------------------------------------------------------------------



def estimate_L_small_from_transformed_event_resolution_np(
    logMc_by_event,
    logitq_by_event,
    percentile=25.0,
    factor=0.5,
    min_samples=10,
):
    """Estimate lower width floors for log(Mc) and coordinate 2 from PE resolution."""
    logMc_by_event = np.asarray(logMc_by_event, dtype=np.float64)
    logitq_by_event = np.asarray(logitq_by_event, dtype=np.float64)

    if logMc_by_event.shape != logitq_by_event.shape:
        raise ValueError(
            "logMc_by_event and logitq_by_event must have the same shape, "
            f"got {logMc_by_event.shape} and {logitq_by_event.shape}"
        )

    if logMc_by_event.ndim != 2:
        raise ValueError(
            "logMc_by_event and logitq_by_event must be 2D arrays "
            "(N_events, N_samples_like)"
        )

    sig_logMc = []
    sig_logitq = []

    n_events = logMc_by_event.shape[0]

    for i in range(n_events):
        u1 = logMc_by_event[i]
        u2 = logitq_by_event[i]

        ok12 = np.isfinite(u1) & np.isfinite(u2)

        if np.sum(ok12) >= min_samples:
            s1 = np.std(u1[ok12])
            s2 = np.std(u2[ok12])

            if np.isfinite(s1) and s1 > 0.0:
                sig_logMc.append(float(s1))
            if np.isfinite(s2) and s2 > 0.0:
                sig_logitq.append(float(s2))

    if len(sig_logMc) == 0 or len(sig_logitq) == 0:
        raise ValueError(
            "Could not estimate L_small_1/L_small_2 from PE resolution: "
            "no valid event-wise transformed samples."
        )

    sig_logMc = np.asarray(sig_logMc, dtype=np.float64)
    sig_logitq = np.asarray(sig_logitq, dtype=np.float64)

    p1, p2 = _coordinate_pair(percentile, "percentile")
    f1, f2 = _coordinate_pair(factor, "factor")

    L1 = float(f1 * np.percentile(sig_logMc, p1))
    L2 = float(f2 * np.percentile(sig_logitq, p2))

    summary = {
        "resolution_percentile": percentile,
        "resolution_factor": factor,

        "resolution_percentile_logMc": p1,
        "resolution_percentile_logitq": p2,
        "resolution_factor_logMc": f1,
        "resolution_factor_logitq": f2,

        "n_events_used_logMc": int(len(sig_logMc)),
        "n_events_used_logitq": int(len(sig_logitq)),

        "logMc_res_p10": float(np.percentile(sig_logMc, 10)),
        "logMc_res_p25": float(np.percentile(sig_logMc, 25)),
        "logMc_res_p50": float(np.percentile(sig_logMc, 50)),
        "logMc_res_p75": float(np.percentile(sig_logMc, 75)),

        "logitq_res_p10": float(np.percentile(sig_logitq, 10)),
        "logitq_res_p25": float(np.percentile(sig_logitq, 25)),
        "logitq_res_p50": float(np.percentile(sig_logitq, 50)),
        "logitq_res_p75": float(np.percentile(sig_logitq, 75)),

        "L_small_1_PE": L1,
        "L_small_2_PE": L2,
    }

    return L1, L2, summary


def sample_from_per_event_gmm_np(wts_l, mus_l, cho_covs_l, x, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    wts_l = np.asarray(wts_l, dtype=np.float64)
    mus_l = np.asarray(mus_l, dtype=np.float64)
    cho_covs_l = np.asarray(cho_covs_l, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    N, K = wts_l.shape
    u = rng.random(N)
    cdf = np.cumsum(wts_l, axis=1)
    ig = (u[:, None] <= cdf).argmax(axis=1).astype(int)
    idx = np.arange(N)
    mu_sel = mus_l[idx, ig, :]
    L_sel = cho_covs_l[idx, ig, :, :]
    Lx = np.einsum("nij,nj->ni", L_sel, x)
    return mu_sel + Lx


def min_sep_logMc_logitq_np(m1_min, m1_max, m2_min, m2_max, dm1, dm2):
    m1 = np.arange(m1_min, m1_max + dm1 * 0.5, dm1)
    m2 = np.arange(m2_min, m2_max + dm2 * 0.5, dm2)
    M1, M2 = np.meshgrid(m1, m2, indexing="ij")
    mask = M1 >= M2
    Mc = (M1 * M2) ** (3.0 / 5.0) / (M1 + M2) ** (1.0 / 5.0)
    q = M2 / M1
    Mc[~mask] = np.nan
    q[~mask] = np.nan
    logMc = np.log(Mc)
    logit_q = np.log(q / (1.0 - q))
    dlogMc_m1 = np.diff(logMc, axis=0)
    dlogMc_m2 = np.diff(logMc, axis=1)
    dlogit_m1 = np.diff(logit_q, axis=0)
    dlogit_m2 = np.diff(logit_q, axis=1)
    min_logMc = min(np.nanmin(np.abs(dlogMc_m1)), np.nanmin(np.abs(dlogMc_m2)))
    min_logitq = min(np.nanmin(np.abs(dlogit_m1)), np.nanmin(np.abs(dlogit_m2)))
    return min_logMc, min_logitq

def _coordinate_pair(value, name):
    """Resolve a scalar or the two retained mass-coordinate values."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    if arr.size == 2:
        return float(arr[0]), float(arr[1])
    raise ValueError(f"{name} must be scalar or length 2 [logMc, coord2]; got length {arr.size}.")


def _parse_q_mbounds_np(q_mbound_low, q_mbound_up):
    """Parse lower/upper quantiles for the two retained mass coordinates."""
    lo = _coordinate_pair(q_mbound_low, "q_mbound_low")
    up = _coordinate_pair(q_mbound_up, "q_mbound_up")
    for name, values in (("q_mbound_low", lo), ("q_mbound_up", up)):
        for coord, value in zip(("logMc", "coord2"), values):
            if not np.isfinite(value) or value < 0.0 or value >= 0.5:
                raise ValueError(f"{name} for {coord} must be finite and satisfy 0 <= q < 0.5; got {value}.")
    return lo[0], lo[1], up[0], up[1]

def _bounds_from_values_np(values, q_low, q_up):
    """
    Finite-value asymmetric quantile bounds.

    lower = quantile(q_low)
    upper = quantile(1 - q_up)

    If q_low=q_up=0, use min/max.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("Cannot compute bounds: no finite values.")

    if q_low == 0.0:
        lo = float(np.nanmin(values))
    else:
        lo = float(np.nanquantile(values, q_low))

    if q_up == 0.0:
        hi = float(np.nanmax(values))
    else:
        hi = float(np.nanquantile(values, 1.0 - q_up))

    if not hi > lo:
        raise ValueError(
            "Asymmetric quantile bounds are empty: "
            "q_low=%s, q_up=%s, low=%s, high=%s."
            % (q_low, q_up, lo, hi)
        )

    return lo, hi


def find_mass_redshift_bounds_samples_np(z_from_dL_fn,
    m1det_mat, m2det_mat, dLdet_mat,
    H0_range, Om_range, w0_range, Xi0_range, nXi0_range,
    N,
    dLinj, m1inj, m2inj,
    *,
    param,
    integrate_dc,
    trials=1000,
    is_observed=False,
    rng=None,
    q_mbound_low=0.05,
    q_mbound_up=0.05,
    DP_bounds_source="samples",
):
    if DP_bounds_source not in ("samples", "event_means"):
        raise ValueError(
            f"DP_bounds_source must be 'samples' or 'event_means', got {DP_bounds_source!r}."
        )

    if rng is None:
        rng = np.random.default_rng()

    qlow_logMc, qlow_logitq, qup_logMc, qup_logitq = _parse_q_mbounds_np(q_mbound_low, q_mbound_up)



    _log_setup("DP-GMM bounds source: %s" % DP_bounds_source)

    H0_max, H0_min = H0_range[1], H0_range[0]
    Om_max, Om_min = Om_range[1], Om_range[0]
    w0_max, w0_min = w0_range[1], w0_range[0]
    Xi0_max, Xi0_min = Xi0_range[1], Xi0_range[0]
    nXi0_max, nXi0_min = nXi0_range[1], nXi0_range[0]

    max_dL = np.max(dLinj)
    min_dL = np.min(dLinj)

    Mc_det_inj, q_inj = Mcq_from_m1m2(m1inj, m2inj)
    Mc_det_inj = np.asarray(Mc_det_inj, dtype=np.float64)
    q_inj = np.asarray(q_inj, dtype=np.float64)
    q_inj = np.clip(q_inj, 1e-6, 1.0 - 1e-6)
    logit_q_inj = np.log(q_inj / (1.0 - q_inj))
    lq_min_inj = np.min(logit_q_inj)
    lq_max_inj = np.max(logit_q_inj)

    lMc_max_inj = 0.0
    lMc_min_inj = 1e10
    m1_max_inj = 0.0
    m1_min_inj = 1e10
    m2_max_inj = 0.0
    m2_min_inj = 1e10

    for H0_ in (H0_max, H0_min):
        for Om_ in (Om_min, Om_max):
            for w0_ in (w0_min, w0_max):
                for Xi0_ in (Xi0_min, Xi0_max):
                    for nXi0_ in (nXi0_min, nXi0_max):
                        if not is_observed:
                            zinj = z_from_dL_fn(np.squeeze(dLinj), float(H0_), float(Om_), float(w0_), float(Xi0_), float(nXi0_), param=param, integrate_dc=integrate_dc)
                        else:
                            zinj = np.squeeze(dLinj)

                        if not is_observed:
                            log_Mc_src = np.log(Mc_det_inj) - np.log1p(zinj)
                            m1inj_src = m1inj / (1.0 + zinj)
                            m2inj_src = m2inj / (1.0 + zinj)
                        else:
                            log_Mc_src = np.log(Mc_det_inj)
                            m1inj_src = m1inj
                            m2inj_src = m2inj

                        log_Mc_src_min_ = np.min(log_Mc_src)
                        log_Mc_src_max_ = np.max(log_Mc_src)
                        if log_Mc_src_max_ > lMc_max_inj:
                            lMc_max_inj = log_Mc_src_max_
                        if log_Mc_src_min_ < lMc_min_inj:
                            lMc_min_inj = log_Mc_src_min_

                        m1_src_min_ = np.min(m1inj_src)
                        m1_src_max_ = np.max(m1inj_src)
                        m2_src_min_ = np.min(m2inj_src)
                        m2_src_max_ = np.max(m2inj_src)
                        if m1_src_max_ > m1_max_inj:
                            m1_max_inj = m1_src_max_
                        if m1_src_min_ < m1_min_inj:
                            m1_min_inj = m1_src_min_
                        if m2_src_max_ > m2_max_inj:
                            m2_max_inj = m2_src_max_
                        if m2_src_min_ < m2_min_inj:
                            m2_min_inj = m2_src_min_

    _log_setup("min, max injection distance: %s, %s Gpc" % (min_dL, max_dL))
    if not is_observed:
        _log_setup("min, max injection log(Mc_src): %s, %s " % (lMc_min_inj, lMc_max_inj))
        _log_setup("min, max injection m1: %s, %s " % (m1_min_inj, m1_max_inj))
        _log_setup("min, max injection m2: %s, %s " % (m2_min_inj, m2_max_inj))
    else:
        _log_setup("min, max injection log(Mc_det): %s, %s " % (lMc_min_inj, lMc_max_inj))
    _log_setup("min, max injection logit(q): %s, %s " % (lq_min_inj, lq_max_inj))
    _log_setup("Finding data redshift and mass range...")

    lMc_maxs, lMc_mins = [], []
    lqs_maxs, lqs_mins = [], []
    m1_maxs, m1_mins = [], []
    m2_maxs, m2_mins = [], []
    lqs_diffs, lMc_diffs = ([], [])
    m1_diffs, m2_diffs = [], []

    # For PE-resolution-based L_small floors.
    # Each entry is one trial containing one draw per event.
    res_logMc_trials = []
    res_logitq_trials = []

    _log_setup("H0_range used in prior search will  be %s" % str(H0_range))
    _log_setup("Om_range used in prior search will  be %s" % str(Om_range))

    m1det_mat = np.asarray(m1det_mat)
    m2det_mat = np.asarray(m2det_mat)
    dLdet_mat = np.asarray(dLdet_mat)
    if m1det_mat.ndim != 2 or m2det_mat.ndim != 2 or dLdet_mat.ndim != 2:
        raise ValueError("m1det_mat, m2det_mat, dLdet_mat must be 2D arrays (N_events, N_samples)")
    if m1det_mat.shape != m2det_mat.shape or m1det_mat.shape != dLdet_mat.shape:
        raise ValueError("m1det_mat, m2det_mat, dLdet_mat must have the same shape (N_events, N_samples)")
    N_events, N_samp = m1det_mat.shape
    if int(N) != int(N_events):
        raise ValueError(f"N={N} but m1det_mat has {N_events} events")

    valid_mat = (
        np.isfinite(m1det_mat)
        & np.isfinite(m2det_mat)
        & np.isfinite(dLdet_mat)
        & (m1det_mat > 0.0)
        & (m2det_mat > 0.0)
        & (dLdet_mat > 0.0)
    )

    valid_indices = [np.where(valid_mat[i])[0] for i in range(N_events)]
    bad_events = [i for i, idxs in enumerate(valid_indices) if len(idxs) == 0]
    if len(bad_events) > 0:
        raise ValueError(
            "Some events have no finite posterior samples for bounds: %s"
            % bad_events
        )


    
    _log_setup("\nDEBUG posterior-sample bounds input")
    _log_setup("N_events, N_samp:", N_events, N_samp)
    _log_setup("dL finite:", np.all(np.isfinite(dLdet_mat)))
    _log_setup("dL min/max:", np.nanmin(dLdet_mat), np.nanmax(dLdet_mat))
    _log_setup("dL per-event median range:",
          np.nanmin(np.nanmedian(dLdet_mat, axis=1)),
          np.nanmax(np.nanmedian(dLdet_mat, axis=1)))
    _log_setup("number of unique event median dL:",
          np.unique(np.round(np.nanmedian(dLdet_mat, axis=1), 12)).size)

    
    for _ in range(trials):
        jj = np.array(
            [idxs[rng.integers(low=0, high=len(idxs))] for idxs in valid_indices],
            dtype=int,
        )
        idx = np.arange(N_events)
        m1det = m1det_mat[idx, jj]
        m2det = m2det_mat[idx, jj]
        d_nodes = dLdet_mat[idx, jj]

        H0 = rng.uniform(*H0_range) if H0_range[0] != H0_range[1] else H0_range[0]
        Om = rng.uniform(*Om_range) if Om_range[0] != Om_range[1] else Om_range[0]
        w0 = rng.uniform(*w0_range) if w0_range[0] != w0_range[1] else w0_range[0]
        Xi0 = rng.uniform(*Xi0_range) if Xi0_range[0] != Xi0_range[1] else Xi0_range[0]
        nXi0 = rng.uniform(*nXi0_range) if nXi0_range[0] != nXi0_range[1] else nXi0_range[0]

        if not is_observed:
            z_nodes = z_from_dL_fn(d_nodes, float(H0), float(Om), float(w0), float(Xi0), float(nXi0), param=param, integrate_dc=integrate_dc)
        else:
            z_nodes = d_nodes
        z_data = np.asarray(z_nodes, dtype=np.float64)
        logz_data = np.log1p(z_data)
        Mc_det, q = Mcq_from_m1m2(m1det, m2det)
        Mc_det = np.asarray(Mc_det, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        log_Mc_det = np.log(Mc_det)

        log_Mc_src = log_Mc_det - logz_data if not is_observed else log_Mc_det
        q = np.clip(q, 1e-6, 1.0 - 1e-6)
        logit_q = np.log(q / (1.0 - q))

        res_logMc_trials.append(np.asarray(log_Mc_src, dtype=np.float64).copy())
        res_logitq_trials.append(np.asarray(logit_q, dtype=np.float64).copy())

        if not is_observed:
            log_Mc_src = log_Mc_det - logz_data
            m1_src = m1det / (1.0 + z_data)
            m2_src = m2det / (1.0 + z_data)
        else:
            log_Mc_src = log_Mc_det
            m1_src = m1det
            m2_src = m2det
        logit_q = np.log(q / (1.0 - q))

        # Bounds for this trial.
        #
        # For DP_bounds_source == "samples", these trial-level sample bounds are
        # the intended quantity. For DP_bounds_source == "event_means", we still
        # collect them here so the code path is well-defined, and later overwrite
        # the transformed-coordinate bounds with event-mean bounds after the
        # trial-major arrays have been assembled.
        lMc_min_data, lMc_max_data = _bounds_from_values_np(
            log_Mc_src,
            qlow_logMc,
            qup_logMc,
        )

        lq_min_data, lq_max_data = _bounds_from_values_np(
            logit_q,
            qlow_logitq,
            qup_logitq,
        )


        # Physical mass bounds are kept as true min/max of the trial.
        m1_min_data, m1_max_data = _bounds_from_values_np(m1_src, 0.0, 0.0)
        m2_min_data, m2_max_data = _bounds_from_values_np(m2_src, 0.0, 0.0)

        lMc_maxs.append(lMc_max_data); lMc_mins.append(lMc_min_data)
        lqs_maxs.append(lq_max_data); lqs_mins.append(lq_min_data)
        m1_mins.append(m1_min_data); m1_maxs.append(m1_max_data)
        m2_mins.append(m2_min_data); m2_maxs.append(m2_max_data)

        tol = 1e-12
        
        logMc_sorted = np.sort(np.asarray(log_Mc_src, dtype=np.float64).copy())
        logitq_sorted = np.sort(np.asarray(logit_q, dtype=np.float64).copy())
        m1_sorted = np.sort(np.asarray(m1_src, dtype=np.float64).copy())
        m2_sorted = np.sort(np.asarray(m2_src, dtype=np.float64).copy())
        
        logMc_sorted = logMc_sorted[np.insert(np.diff(logMc_sorted) > tol, 0, True)]
        logitq_sorted = logitq_sorted[np.insert(np.diff(logitq_sorted) > tol, 0, True)]
        m1_sorted = m1_sorted[np.insert(np.diff(m1_sorted) > tol, 0, True)]
        m2_sorted = m2_sorted[np.insert(np.diff(m2_sorted) > tol, 0, True)]
        
        
        dMc_pos = np.diff(logMc_sorted)
        dMc_pos = dMc_pos[dMc_pos > tol]
        
        dq_pos = np.diff(logitq_sorted)
        dq_pos = dq_pos[dq_pos > tol]
        
        dm1_pos = np.diff(m1_sorted)
        dm1_pos = dm1_pos[dm1_pos > tol]
        
        dm2_pos = np.diff(m2_sorted)
        dm2_pos = dm2_pos[dm2_pos > tol]


        if dMc_pos.size > 0:
            lMc_diffs.append(float(np.min(dMc_pos)))

        if dq_pos.size > 0:
            lqs_diffs.append(float(np.min(dq_pos)))

        if dm1_pos.size > 0:
            m1_diffs.append(float(np.min(dm1_pos)))

        if dm2_pos.size > 0:
            m2_diffs.append(float(np.min(dm2_pos)))
    
    lMc_max_data = max(lMc_maxs); lMc_min_data = min(lMc_mins)
    lq_max_data = max(lqs_maxs); lq_min_data = min(lqs_mins)
    m1_max_data = max(m1_maxs); m1_min_data = min(m1_mins)
    m2_max_data = max(m2_maxs); m2_min_data = min(m2_mins)

    if len(lMc_diffs) == 0:
        raise ValueError("No positive logMc spacings found while estimating bounds.")
    if len(lqs_diffs) == 0:
        raise ValueError("No positive logit(q) spacings found while estimating bounds.")
    if len(m1_diffs) == 0:
        raise ValueError("No positive m1 spacings found while estimating bounds.")
    if len(m2_diffs) == 0:
        raise ValueError("No positive m2 spacings found while estimating bounds.")
        
    lMc_diff = np.mean(lMc_diffs)
    lq_diff = np.mean(lqs_diffs)
    m1_diff = np.mean(m1_diffs)
    m2_diff = np.mean(m2_diffs)

    # Convert trial-major arrays (trials, N_events) -> event-major (N_events, trials)
    res_logMc_by_event = np.asarray(res_logMc_trials, dtype=np.float64).T
    res_logitq_by_event = np.asarray(res_logitq_trials, dtype=np.float64).T

    if DP_bounds_source == "event_means":
        event_logMc_mean = np.nanmean(res_logMc_by_event, axis=1)
        event_logitq_mean = np.nanmean(res_logitq_by_event, axis=1)
    
        lMc_min_data, lMc_max_data = _bounds_from_values_np(
            event_logMc_mean,
            qlow_logMc,
            qup_logMc,
        )
        
        lq_min_data, lq_max_data = _bounds_from_values_np(
            event_logitq_mean,
            qlow_logitq,
            qup_logitq,
        )
        
    
        _log_setup("DP-GMM bounds computed from event posterior means.")
        _log_setup("event-mean logMc bounds: %s, %s" % (lMc_min_data, lMc_max_data))
        _log_setup("event-mean logit(q) bounds: %s, %s" % (lq_min_data, lq_max_data))

    # ------------------------------------------------------------------
    # Diagnostic: PE resolution in q / logit(q)
    # ------------------------------------------------------------------
    q_by_event = 1.0 / (1.0 + np.exp(-res_logitq_by_event))
    
    std_logMc = np.std(res_logMc_by_event, axis=1)
    std_logitq = np.std(res_logitq_by_event, axis=1)
    std_q = np.std(q_by_event, axis=1)
    
    q_med = np.median(q_by_event, axis=1)
    q_p05 = np.percentile(q_by_event, 5, axis=1)
    q_p95 = np.percentile(q_by_event, 95, axis=1)
    
    _log_setup("\nPE-resolution diagnostic for q")
    _log_setup("-" * 80)
    _log_setup("std(logMc) percentiles  :", np.percentile(std_logMc, [0, 5, 10, 25, 50, 75, 90, 95, 100]))
    _log_setup("std(logitq) percentiles :", np.percentile(std_logitq, [0, 5, 10, 25, 50, 75, 90, 95, 100]))
    _log_setup("std(q) percentiles      :", np.percentile(std_q, [0, 5, 10, 25, 50, 75, 90, 95, 100]))
    _log_setup("median q percentiles    :", np.percentile(q_med, [0, 5, 10, 25, 50, 75, 90, 95, 100]))
    _log_setup("q 90%-width percentiles :", np.percentile(q_p95 - q_p05, [0, 5, 10, 25, 50, 75, 90, 95, 100]))
    
    idx = np.argsort(std_logitq)[:10]
    _log_setup("\n10 narrowest logit(q) events:")
    for ii in idx:
        _log_setup(
            "event %4d | std_logitq=% .6g | std_q=% .6g | "
            "q_med=% .6g | q05=% .6g | q95=% .6g | std_logMc=% .6g"
            % (
                ii,
                std_logitq[ii],
                std_q[ii],
                q_med[ii],
                q_p05[ii],
                q_p95[ii],
                std_logMc[ii],
            )
        )
    _log_setup("-" * 80)
    
    L1_res_p25_fac05, L2_res_p25_fac05, Lres_summary_default = (
        estimate_L_small_from_transformed_event_resolution_np(
            res_logMc_by_event,
            res_logitq_by_event,
            percentile=25.0,
            factor=0.5,
            min_samples=max(10, int(0.1 * trials)),
        )
    )

    if not is_observed:
        _log_setup("min, max data log(Mc)_src: %s, %s " % (lMc_min_data, lMc_max_data))
        _log_setup("min, max data logit(q): %s, %s " % (lq_min_data, lq_max_data))
        _log_setup("min, max data m1: %s, %s " % (m1_min_data, m1_max_data))
        _log_setup("min, max data m2: %s, %s " % (m2_min_data, m2_max_data))
        _log_setup("min log(Mc)_src scale: %s " % (lMc_diff))
        _log_setup("min logit(q) scale: %s " % (lq_diff))
        _log_setup("min m1 src scale: %s " % (m1_diff))
        _log_setup("min m2 src scale: %s " % (m2_diff))
    else:
        _log_setup("min, max data log(Mc_D): %s, %s " % (lMc_min_data, lMc_max_data))
        _log_setup("min, max data logit(q): %s, %s " % (lq_min_data, lq_max_data))
        _log_setup("min, max data m1_D: %s, %s " % (m1_min_data, m1_max_data))
        _log_setup("min, max data m2_D: %s, %s " % (m2_min_data, m2_max_data))
        _log_setup("min log(Mc_D) scale: %s " % (lMc_diff))
        _log_setup("min logit(q) scale: %s " % (lq_diff))
        _log_setup("min m1_D scale: %s " % (m1_diff))
        _log_setup("min m2_D scale: %s " % (m2_diff))

    return dict(lMc_min_data=lMc_min_data, lMc_max_data=lMc_max_data, lq_min_data=lq_min_data, lq_max_data=lq_max_data, lMc_diff=lMc_diff, lq_diff=lq_diff, m1_diff=m1_diff, m2_diff=m2_diff, lMc_min_inj=lMc_min_inj, lMc_max_inj=lMc_max_inj, lq_max_inj=lq_max_inj, lq_min_inj=lq_min_inj, m1_min_inj=m1_min_inj, m2_min_inj=m2_min_inj, m1_max_inj=m1_max_inj, m2_max_inj=m2_max_inj, m1_min_data=m1_min_data, m1_max_data=m1_max_data, m2_min_data=m2_min_data, m2_max_data=m2_max_data, lMc_res_p25_fac05=L1_res_p25_fac05, lq_res_p25_fac05=L2_res_p25_fac05, Lres_summary_default=Lres_summary_default, res_logMc_by_event=res_logMc_by_event, res_logitq_by_event=res_logitq_by_event)


def _sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def inv_flogit_bounds_np(z, xmin, xmax):
    """
    Inverse of bounded flogit coordinate used by event_bounded_flogit GMMs.

    z is the Gaussian surrogate coordinate.
    output is in the original PE coordinate bounded by [xmin, xmax].
    """
    z = np.asarray(z, dtype=np.float64)
    xmin = np.asarray(xmin, dtype=np.float64)
    xmax = np.asarray(xmax, dtype=np.float64)
    return xmin + (xmax - xmin) * _sigmoid_np(z)


def _decode_gauss_bound_finder_samples_np(
    samples,
    *,
    gmm_fit_transform_mode="legacy",
    gmm_fit_coord_bounds=None,
    gmm_fit_coord_names=None,
):
    """
    Convert samples drawn from the PE Gaussian/GMM surrogate into the
    physical coordinates needed by the DPUC bound finder.

    Returns:
        log_Mc_det, q_det, logd
    """
    samples = np.asarray(samples, dtype=np.float64)

    if gmm_fit_transform_mode in (None, "legacy"):
        log_Mc_det = samples[:, 0]
        logit_q = samples[:, 1]
        q_det = 1.0 / (1.0 + np.exp(-logit_q))
        logd = samples[:, 2]
        return log_Mc_det, q_det, logd

    if gmm_fit_transform_mode != "event_bounded_flogit":
        raise NotImplementedError(
            f"Unsupported gmm_fit_transform_mode={gmm_fit_transform_mode!r}"
        )

    if gmm_fit_coord_bounds is None:
        raise ValueError("gmm_fit_coord_bounds is required for event_bounded_flogit.")
    if gmm_fit_coord_names is None:
        raise ValueError("gmm_fit_coord_names is required for event_bounded_flogit.")

    gmm_fit_coord_bounds = np.asarray(gmm_fit_coord_bounds, dtype=np.float64)
    gmm_fit_coord_names = tuple(str(x) for x in gmm_fit_coord_names)

    expected = ("logMc", "q", "logdL")
    if tuple(gmm_fit_coord_names[:3]) != expected:
        raise NotImplementedError(
            "Unsupported bounded GMM coordinate names for bound finder. "
            f"Expected {expected}, got {tuple(gmm_fit_coord_names[:3])}."
        )

    if gmm_fit_coord_bounds.shape[0] != samples.shape[0]:
        raise ValueError(
            "gmm_fit_coord_bounds first dimension must match number of events/samples. "
            f"Got bounds={gmm_fit_coord_bounds.shape}, samples={samples.shape}."
        )

    if gmm_fit_coord_bounds.shape[1] < 3 or gmm_fit_coord_bounds.shape[2] != 2:
        raise ValueError(
            "gmm_fit_coord_bounds must have shape (N, nd>=3, 2). "
            f"Got {gmm_fit_coord_bounds.shape}."
        )

    xmin = gmm_fit_coord_bounds[:, :, 0]
    xmax = gmm_fit_coord_bounds[:, :, 1]

    log_Mc_det = inv_flogit_bounds_np(samples[:, 0], xmin[:, 0], xmax[:, 0])
    q_det = inv_flogit_bounds_np(samples[:, 1], xmin[:, 1], xmax[:, 1])
    logd = inv_flogit_bounds_np(samples[:, 2], xmin[:, 2], xmax[:, 2])

    q_det = np.clip(q_det, 1.0e-12, 1.0 - 1.0e-12)

    return log_Mc_det, q_det, logd


def find_mass_redshift_bounds_np(
    z_from_dL_fn,
    wts_l_np,
    mus_l_np,
    cho_covs_l_np,
    H0_range,
    Om_range,
    w0_range,
    Xi0_range,
    nXi0_range,
    N,
    nd,
    dLinj,
    m1inj,
    m2inj,
    *,
    param,
    integrate_dc,
    sampling_GW,
    trials=1000,
    is_observed=False,
    rng=None,
    q_mbound_low=0.05,
    q_mbound_up=0.05,
    gmm_fit_transform_mode="legacy",
    gmm_fit_coord_bounds=None,
    gmm_fit_coord_names=None,
    DP_bounds_source="samples",
):
    if DP_bounds_source not in ("samples", "event_means"):
        raise ValueError(
            f"DP_bounds_source must be 'samples' or 'event_means', "
            f"got {DP_bounds_source!r}."
        )

    if rng is None:
        rng = np.random.default_rng()

    qlow_logMc, qlow_logitq, qup_logMc, qup_logitq = _parse_q_mbounds_np(q_mbound_low, q_mbound_up)

    #     _log_setup("Search for min and max mass scale with min and max posterior samples values")
    # else:


    H0_max, H0_min = H0_range[1], H0_range[0]
    Om_max, Om_min = Om_range[1], Om_range[0]
    w0_max, w0_min = w0_range[1], w0_range[0]
    Xi0_max, Xi0_min = Xi0_range[1], Xi0_range[0]
    nXi0_max, nXi0_min = nXi0_range[1], nXi0_range[0]

    max_dL = np.max(dLinj)
    min_dL = np.min(dLinj)

    Mc_det, q_inj = Mcq_from_m1m2(m1inj, m2inj)
    Mc_det = np.asarray(Mc_det, dtype=np.float64)
    q_inj = np.asarray(q_inj, dtype=np.float64)
    q_inj = np.clip(q_inj, 1.0e-12, 1.0 - 1.0e-12)

    logit_q_inj = np.log(q_inj / (1.0 - q_inj))
    lq_min_inj = float(np.nanmin(logit_q_inj))
    lq_max_inj = float(np.nanmax(logit_q_inj))


    lMc_max_inj = 0.0
    lMc_min_inj = 1.0e10

    m1_max_inj = 0.0
    m1_min_inj = 1.0e10

    m2_max_inj = 0.0
    m2_min_inj = 1.0e10

    for H0_ in (H0_max, H0_min):
        for Om_ in (Om_min, Om_max):
            for w0_ in (w0_min, w0_max):
                for Xi0_ in (Xi0_min, Xi0_max):
                    for nXi0_ in (nXi0_min, nXi0_max):

                        if not is_observed:
                            zinj = z_from_dL_fn(
                                np.squeeze(dLinj),
                                float(H0_),
                                float(Om_),
                                float(w0_),
                                float(Xi0_),
                                float(nXi0_),
                                param=param,
                                integrate_dc=integrate_dc,
                            )
                        else:
                            zinj = np.squeeze(dLinj)




                        if not is_observed:
                            log_Mc_src = np.log(Mc_det) - np.log1p(zinj)
                            m1inj_src = m1inj / (1.0 + zinj)
                            m2inj_src = m2inj / (1.0 + zinj)
                        else:
                            log_Mc_src = np.log(Mc_det)
                            m1inj_src = m1inj
                            m2inj_src = m2inj

                        log_Mc_src_min_ = np.min(log_Mc_src)
                        log_Mc_src_max_ = np.max(log_Mc_src)

                        if log_Mc_src_max_ > lMc_max_inj:
                            lMc_max_inj = log_Mc_src_max_
                        if log_Mc_src_min_ < lMc_min_inj:
                            lMc_min_inj = log_Mc_src_min_

                        m1_src_min_ = np.min(m1inj_src)
                        m1_src_max_ = np.max(m1inj_src)
                        m2_src_min_ = np.min(m2inj_src)
                        m2_src_max_ = np.max(m2inj_src)

                        if m1_src_max_ > m1_max_inj:
                            m1_max_inj = m1_src_max_
                        if m1_src_min_ < m1_min_inj:
                            m1_min_inj = m1_src_min_

                        if m2_src_max_ > m2_max_inj:
                            m2_max_inj = m2_src_max_
                        if m2_src_min_ < m2_min_inj:
                            m2_min_inj = m2_src_min_

    _log_setup("min, max injection distance: %s, %s Gpc" % (min_dL, max_dL))

    if not is_observed:
        _log_setup("min, max injection log(Mc_src): %s, %s " % (lMc_min_inj, lMc_max_inj))
        _log_setup("min, max injection m1: %s, %s " % (m1_min_inj, m1_max_inj))
        _log_setup("min, max injection m2: %s, %s " % (m2_min_inj, m2_max_inj))
    else:
        _log_setup("min, max injection log(Mc_det): %s, %s " % (lMc_min_inj, lMc_max_inj))

    _log_setup("min, max injection logit(q): %s, %s " % (lq_min_inj, lq_max_inj))
    _log_setup("Finding data redshift and mass range...")

    lMc_maxs, lMc_mins = [], []
    lqs_maxs, lqs_mins = [], []

    lqs_diffs, lMc_diffs = ([], [])
    m1_diffs, m2_diffs = [], []

    m1_maxs, m2_maxs = [], []
    m1_mins, m2_mins = [], []

    # For PE-resolution-based L_small floors.
    # Each entry is one trial containing one draw per event.
    res_logMc_trials = []
    res_logitq_trials = []

    _log_setup("H0_range used in prior search will  be %s" % str(H0_range))
    _log_setup("Om_range used in prior search will  be %s" % str(Om_range))

    for it in range(trials):

        Xwhite = rng.standard_normal(size=(N, nd))
        samples = sample_from_per_event_gmm_np(
            wts_l_np,
            mus_l_np,
            cho_covs_l_np,
            Xwhite,
            rng=rng,
        )

        if it == 0:
            _log_setup("\nDEBUG first trial raw GMM samples in find_mass_redshift_bounds_np")
            _log_setup("samples shape:", samples.shape)

            for kk in range(samples.shape[1]):
                _log_setup(
                    "  raw samples coord %d percentiles:" % kk,
                    np.percentile(
                        samples[:, kk],
                        [0, 5, 10, 25, 50, 75, 90, 95, 100],
                    ),
                )

            _log_setup("\nDEBUG mus_l_np coordinate percentiles")
            for kk in range(mus_l_np.shape[-1]):
                _log_setup(
                    "  mus coord %d:" % kk,
                    np.percentile(
                        mus_l_np[..., kk].ravel(),
                        [0, 5, 10, 25, 50, 75, 90, 95, 100],
                    ),
                )

            _log_setup("\nDEBUG cho_covs_l_np diagonal coordinate percentiles")
            diag = np.diagonal(cho_covs_l_np, axis1=-2, axis2=-1)

            for kk in range(diag.shape[-1]):
                _log_setup(
                    "  chol diag coord %d:" % kk,
                    np.percentile(
                        diag[..., kk].ravel(),
                        [0, 5, 10, 25, 50, 75, 90, 95, 100],
                    ),
                )

        log_Mc_det, q_det, logd = _decode_gauss_bound_finder_samples_np(
            samples,
            gmm_fit_transform_mode=gmm_fit_transform_mode,
            gmm_fit_coord_bounds=gmm_fit_coord_bounds,
            gmm_fit_coord_names=gmm_fit_coord_names,
        )

        d_nodes = np.exp(logd)

        H0 = rng.uniform(*H0_range) if H0_range[0] != H0_range[1] else H0_range[0]
        Om = rng.uniform(*Om_range) if Om_range[0] != Om_range[1] else Om_range[0]
        w0 = rng.uniform(*w0_range) if w0_range[0] != w0_range[1] else w0_range[0]
        Xi0 = rng.uniform(*Xi0_range) if Xi0_range[0] != Xi0_range[1] else Xi0_range[0]
        nXi0 = rng.uniform(*nXi0_range) if nXi0_range[0] != nXi0_range[1] else nXi0_range[0]

        if not is_observed:
            z_nodes = z_from_dL_fn(
                d_nodes,
                float(H0),
                float(Om),
                float(w0),
                float(Xi0),
                float(nXi0),
                param=param,
                integrate_dc=integrate_dc,
            )
        else:
            z_nodes = d_nodes

        z_data = np.asarray(z_nodes, dtype=np.float64)
        logz_data = np.log1p(z_data)

        log_Mc_src = log_Mc_det - logz_data if not is_observed else log_Mc_det
        log_Mc_src = np.asarray(log_Mc_src, dtype=np.float64)

        q = np.clip(np.asarray(q_det, dtype=np.float64), 1.0e-12, 1.0 - 1.0e-12)
        logit_q = np.log(q / (1.0 - q))

        res_logMc_trials.append(np.asarray(log_Mc_src, dtype=np.float64).copy())
        res_logitq_trials.append(np.asarray(logit_q, dtype=np.float64).copy())

        m1_src, m2_src = m1m2_from_Mcq(np.exp(log_Mc_src), q)
        m1_src = np.asarray(m1_src, dtype=np.float64)
        m2_src = np.asarray(m2_src, dtype=np.float64)

        # Sample-based coordinate bounds from this trial.
        lMc_data_min, lMc_data_max = _bounds_from_values_np(
            log_Mc_src,
            qlow_logMc,
            qup_logMc,
        )
        
        lq_data_min, lq_data_max = _bounds_from_values_np(
            logit_q,
            qlow_logitq,
            qup_logitq,
        )
        
        # Physical mass bounds remain extrema.
        m1_data_min = float(np.nanmin(m1_src))
        m1_data_max = float(np.nanmax(m1_src))
        m2_data_min = float(np.nanmin(m2_src))
        m2_data_max = float(np.nanmax(m2_src))

        lMc_maxs.append(lMc_data_max)
        lMc_mins.append(lMc_data_min)

        lqs_maxs.append(lq_data_max)
        lqs_mins.append(lq_data_min)


        m1_mins.append(m1_data_min)
        m1_maxs.append(m1_data_max)

        m2_mins.append(m2_data_min)
        m2_maxs.append(m2_data_max)

        tol = 1.0e-12

        logMc_sorted = np.sort(np.asarray(log_Mc_src, dtype=np.float64).copy())
        logitq_sorted = np.sort(np.asarray(logit_q, dtype=np.float64).copy())
        m1_sorted = np.sort(np.asarray(m1_src, dtype=np.float64).copy())
        m2_sorted = np.sort(np.asarray(m2_src, dtype=np.float64).copy())

        logMc_sorted = logMc_sorted[np.insert(np.diff(logMc_sorted) > tol, 0, True)]
        logitq_sorted = logitq_sorted[np.insert(np.diff(logitq_sorted) > tol, 0, True)]
        m1_sorted = m1_sorted[np.insert(np.diff(m1_sorted) > tol, 0, True)]
        m2_sorted = m2_sorted[np.insert(np.diff(m2_sorted) > tol, 0, True)]


        dMc_pos = np.diff(logMc_sorted)
        dMc_pos = dMc_pos[dMc_pos > tol]

        dq_pos = np.diff(logitq_sorted)
        dq_pos = dq_pos[dq_pos > tol]

        dm1_pos = np.diff(m1_sorted)
        dm1_pos = dm1_pos[dm1_pos > tol]

        dm2_pos = np.diff(m2_sorted)
        dm2_pos = dm2_pos[dm2_pos > tol]

        if dMc_pos.size > 0:
            lMc_diffs.append(np.min(dMc_pos))
        if dq_pos.size > 0:
            lqs_diffs.append(np.min(dq_pos))
        if dm1_pos.size > 0:
            m1_diffs.append(np.min(dm1_pos))
        if dm2_pos.size > 0:
            m2_diffs.append(np.min(dm2_pos))

    # Default sample-based final bounds.

    lMc_max_data = max(lMc_maxs)
    lMc_min_data = min(lMc_mins)

    lq_max_data = max(lqs_maxs)
    lq_min_data = min(lqs_mins)

    m1_max_data = max(m1_maxs)
    m1_min_data = min(m1_mins)

    m2_max_data = max(m2_maxs)
    m2_min_data = min(m2_mins)

    if len(lMc_diffs) == 0:
        raise ValueError("No positive logMc spacings found while estimating bounds.")
    if len(lqs_diffs) == 0:
        raise ValueError("No positive logit(q) spacings found while estimating bounds.")
    if len(m1_diffs) == 0:
        raise ValueError("No positive m1 spacings found while estimating bounds.")
    if len(m2_diffs) == 0:
        raise ValueError("No positive m2 spacings found while estimating bounds.")

    lMc_diff = np.mean(lMc_diffs)
    lq_diff = np.mean(lqs_diffs)
    m1_diff = np.mean(m1_diffs)
    m2_diff = np.mean(m2_diffs)

    # Convert trial-major arrays (trials, N_events) -> event-major (N_events, trials).
    res_logMc_by_event = np.asarray(res_logMc_trials, dtype=np.float64).T
    res_logitq_by_event = np.asarray(res_logitq_trials, dtype=np.float64).T

    # Optional event-mean bounds for DP/DPUC means.
    # This overwrites only the transformed-coordinate DP mean bounds.
    # It does not overwrite physical m1/m2 support bounds.
    if DP_bounds_source == "event_means":
        event_logMc_mean = np.nanmean(res_logMc_by_event, axis=1)
        event_logitq_mean = np.nanmean(res_logitq_by_event, axis=1)

        lMc_min_data, lMc_max_data = _bounds_from_values_np(
            event_logMc_mean,
            qlow_logMc,
            qup_logMc,
        )
        
        lq_min_data, lq_max_data = _bounds_from_values_np(
            event_logitq_mean,
            qlow_logitq,
            qup_logitq,
        )
        

        _log_setup("DP-GMM bounds computed from event posterior means.")
        _log_setup("event-mean logMc bounds: %s, %s" % (lMc_min_data, lMc_max_data))
        _log_setup("event-mean logit(q) bounds: %s, %s" % (lq_min_data, lq_max_data))
    else:
        _log_setup("DP-GMM bounds computed from posterior samples.")

    L1_res_p25_fac05, L2_res_p25_fac05, Lres_summary_default = (
        estimate_L_small_from_transformed_event_resolution_np(
            res_logMc_by_event,
            res_logitq_by_event,
            percentile=25.0,
            factor=0.5,
            min_samples=max(10, int(0.1 * trials)),
        )
    )

    if not is_observed:
        _log_setup("min, max data log(Mc)_src: %s, %s " % (lMc_min_data, lMc_max_data))
        _log_setup("min, max data logit(q): %s, %s " % (lq_min_data, lq_max_data))
        _log_setup("min, max data m1: %s, %s " % (m1_min_data, m1_max_data))
        _log_setup("min, max data m2: %s, %s " % (m2_min_data, m2_max_data))
        _log_setup("min log(Mc)_src scale: %s " % (lMc_diff))
        _log_setup("min logit(q) scale: %s " % (lq_diff))
        _log_setup("min m1 src scale: %s " % (m1_diff))
        _log_setup("min m2 src scale: %s " % (m2_diff))
    else:
        _log_setup("min, max data log(Mc_D): %s, %s " % (lMc_min_data, lMc_max_data))
        _log_setup("min, max data logit(q): %s, %s " % (lq_min_data, lq_max_data))
        _log_setup("min, max data m1_D: %s, %s " % (m1_min_data, m1_max_data))
        _log_setup("min, max data m2_D: %s, %s " % (m2_min_data, m2_max_data))
        _log_setup("min log(Mc_D) scale: %s " % (lMc_diff))
        _log_setup("min logit(q) scale: %s " % (lq_diff))
        _log_setup("min m1_D scale: %s " % (m1_diff))
        _log_setup("min m2_D scale: %s " % (m2_diff))

    return dict(lMc_min_data=lMc_min_data, lMc_max_data=lMc_max_data, lq_min_data=lq_min_data, lq_max_data=lq_max_data, lMc_diff=lMc_diff, lq_diff=lq_diff, m1_diff=m1_diff, m2_diff=m2_diff, lMc_min_inj=lMc_min_inj, lMc_max_inj=lMc_max_inj, lq_max_inj=lq_max_inj, lq_min_inj=lq_min_inj, m1_min_inj=m1_min_inj, m2_min_inj=m2_min_inj, m1_max_inj=m1_max_inj, m2_max_inj=m2_max_inj, m1_min_data=m1_min_data, m1_max_data=m1_max_data, m2_min_data=m2_min_data, m2_max_data=m2_max_data, lMc_res_p25_fac05=L1_res_p25_fac05, lq_res_p25_fac05=L2_res_p25_fac05, Lres_summary_default=Lres_summary_default, res_logMc_by_event=res_logMc_by_event, res_logitq_by_event=res_logitq_by_event)


# ---------------------------------------------------------------------
# Utilities
# 
# ---------------------------------------------------------------------


def logit(bk, p):
    return bk.log(p) - bk.log(1. - p)


def m1m2_from_Mcq(Mc, q):
    m1 = Mc * (1.0 + q) ** (1.0 / 5.0) / q ** (3.0 / 5.0)
    m2 = q * m1
    return m1, m2


def inv_logit(p):
    return 1.0 / (1.0 + jnp.exp(-p))


def inv_flogit(p):
    ep = jnp.exp(p)
    return (ep - 1.0) / (1.0 + ep)



def safe_logsumexp_jax(a, axis=0):
    finite = jnp.isfinite(a)
    all_bad = jnp.all(~finite, axis=axis, keepdims=True)
    a_safe = jnp.where(finite, a, -jnp.inf)
    m = jnp.max(jnp.where(all_bad, 0.0, a_safe), axis=axis, keepdims=True)
    s = jnp.sum(jnp.exp(a_safe - m), axis=axis, keepdims=True)
    out = m + jnp.log(s)
    out = jnp.where(all_bad, -jnp.inf, out)
    return jnp.squeeze(out, axis=axis)

def logsumexp(bk, x, axis=None):
    """
    Backend-agnostic logsumexp.
    Assumes bk has max, exp, log, sum.
    """
    m = bk.max(x, axis=axis, keepdims=True)
    s = bk.sum(bk.exp(x - m), axis=axis, keepdims=True)
    out = m + bk.log(s)
    if axis is None:
        return out.reshape(())
    return bk.squeeze(out, axis=axis)


def _logsumexp_np(x: np.ndarray) -> float:
    """Stable logsumexp for 1D numpy arrays."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))

def logsumexp2_unsafe(bk, a, b):
    """Stable log(exp(a)+exp(b)) for two terms."""
    m = bk.maximum(a, b)
    return m + bk.log(bk.exp(a - m) + bk.exp(b - m))


def logsumexp2(bk, a, b):
    both_neg_inf = bk.logical_and(
        bk.eq(a, -jnp.inf),
        bk.eq(b, -jnp.inf)
    )

    m = bk.maximum(a, b)

    # safe shifts: replace -inf with large negative number BEFORE subtraction
    a_safe = bk.where(bk.eq(a, -jnp.inf), -1e30, a)
    b_safe = bk.where(bk.eq(b, -jnp.inf), -1e30, b)

    m_safe = bk.maximum(a_safe, b_safe)

    out = m_safe + bk.log(
        bk.exp(a_safe - m_safe) + bk.exp(b_safe - m_safe)
    )

    return bk.where(both_neg_inf, -jnp.inf, out)


def logaddexp(bk, a, b):
    """Stable logaddexp using logsumexp2."""
    return logsumexp2(bk, a, b)

def sigmoid(bk, x, x0, s, eps=1e-12, clip=1e-15):
    s_pos = bk.maximum(s, eps)
    t = (x - x0) / s_pos
    y = 0.5 * (bk.tanh(0.5 * t) + 1.0)
    if clip is not None:
        y = bk.clip(y, clip, 1.0 - clip)
    return y


def log_sigmoid(bk, x, m, sig):
    return bk.log(sigmoid(bk, x, m, sig))


def safe_sigmoid(bk, x, x0, eps):
    return sigmoid(bk, x, x0, eps, clip=1e-15)


# def logdiffexp(bk, a, b, *, eps=1e-16):
#     """
#     Stable log(exp(a) - exp(b)) elementwise.
#     Returns -inf where b >= a.
#     """
#     delta = bk.minimum(b - a, 0.0)     # <= 0
#     ed = bk.exp(delta)                # in [0,1]

#     out = a + bk.log1p(-bk.minimum(ed, 1.0 - eps))
#     return bk.where(b < a, out, -np.inf)

def logdiffexp_nosafe(bk, a, b):
    # returns log(exp(a) - exp(b)); -inf if b >= a
    return bk.where(
        a > b,
        a + bk.log1p(-bk.exp(b - a)),
        -np.inf,
    )


def logdiffexp(bk, a, b, eps=1e-16):
    delta = bk.minimum(b - a, 0.0)           # <= 0
    ed = bk.exp(delta)                        # in (0,1]
    ed = bk.minimum(ed, 1.0 - eps)
    out = a + bk.log1p(-ed)
    return bk.where(b < a, out, -np.inf)


def log_power_gate(bk, x, low, high, p=16.0, eps=1e-12, clip=1e-15):
    """
    Smooth log gate on positive x:

      low cutoff:  -log(1 + (low/x)^p)
      high cutoff: -log(1 + (x/high)^p)

    Returns log gate <= 0.
    """

    logx = bk.log(bk.maximum(x, eps))

    t_low  = p * (bk.log(low)  - logx)
    t_high = p * (logx - bk.log(high))

    log_gate = -bk.logaddexp(0.0, t_low) - bk.logaddexp(0.0, t_high)

    if clip is not None:
        log_gate = bk.maximum(log_gate, bk.log(clip))

    return log_gate

# ---------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------

def attrapzvec(bk, y, x, *, axis=-1):
    """
    Backend-agnostic trapezoid integral along `axis`.

    Equivalent to:
      ∫ y(x) dx ≈ sum_i 0.5 * (x[i+1]-x[i]) * (y[i+1]+y[i])

    Gradients:
      - w.r.t y: correct
      - w.r.t x: caller should pass stop_grad(x) if x is constant

    Example: norm = attrapzvec(bk, integrand, bk.stop_grad(z))
    """
    d = bk.diff(x, axis=axis)

    # broadcast d if x is 1D and y is ND
    if getattr(x, "ndim", None) == 1 and getattr(y, "ndim", None) != 1:
        shape = [1] * y.ndim
        shape[axis] = d.shape[0]
        d = bk.reshape(d, shape)

    nd = y.ndim
    sl1 = [slice(None)] * nd
    sl2 = [slice(None)] * nd
    sl1[axis] = slice(1, None)
    sl2[axis] = slice(None, -1)

    return bk.sum(d * (y[tuple(sl1)] + y[tuple(sl2)]) * 0.5, axis=axis)


def atcumtrapz(bk, y, x, *, axis=-1):
    """
    Backend-agnostic cumulative trapezoid integral.

    Returns:
      cumulative integral with shape shortened by 1 along `axis`
      (same convention as scipy.integrate.cumtrapz without `initial`)

    Example: cdf  = atcumtrapz(bk, p2, bk.stop_grad(xx))
    """
    d = bk.diff(x, axis=axis)

    if getattr(x, "ndim", None) == 1 and getattr(y, "ndim", None) != 1:
        shape = [1] * y.ndim
        shape[axis] = d.shape[0]
        d = bk.reshape(d, shape)

    nd = y.ndim
    sl1 = [slice(None)] * nd
    sl2 = [slice(None)] * nd
    sl1[axis] = slice(1, None)
    sl2[axis] = slice(None, -1)

    pieces = d * (y[tuple(sl1)] + y[tuple(sl2)]) * 0.5
    return bk.cumsum(pieces, axis=axis)






# ---------------------------------------------------------------------
# Interpolation
# 
# ---------------------------------------------------------------------

def atinterp(bk, x, xs, ys):

  idxs = bk.searchsorted(xs, x, side='left')
  idxs = bk.clip(idxs, 1, xs.shape[0] - 1) # out of index case

  xl = xs[idxs-1]
  yl = ys[idxs-1]
  xh = xs[idxs]
  yh = ys[idxs]

  r = (x-xl)/(xh-xl)

  return r*yh + (1.0-r)*yl


def atinterp_uniform(bk, x, *args, eps=1e-12, side="left"):
    # New style: (xp, fp)
    if len(args) == 2:
        xp, fp_const = args
        n = xp.shape[0]
        x0 = xp[0]
        dx = xp[1] - xp[0]
    # Legacy style: (x0, x1, nU, fp)
    elif len(args) == 4:
        x0, x1, nU, fp_const = args
        n = nU
        dx = (x1 - x0) / bk.maximum(nU - 1, 1)
    else:
        raise TypeError("atinterp_uniform expects (x, xp, fp) or (x, x0, x1, nU, fp)")

    dx = bk.maximum(dx, eps)
    t = (x - x0) / dx
    t = bk.clip(t, 0.0, (n - 1) * 1.0)

    if side == "right":
        j = bk.ceil(t) - 1.0
    else:
        j = bk.floor(t)

    if hasattr(bk, "asarray"):
        j = bk.asarray(j, dtype="int32")
    else:
        j = j.astype("int32")
    j = bk.clip(j, 0, n - 2)

    xl = x0 + j * dx
    yl = fp_const[j]
    yh = fp_const[j + 1]
    r = (x - xl) / dx
    r = bk.clip(r, 0.0, 1.0)
    return (1.0 - r) * yl + r * yh




def _interp_indices_nonuniform_safe(bk, x, x_grid):
    """
    Robust index+weight for non-uniform 1D interpolation.

    Returns:
      j  in [1, N-1]
      r  in [0, 1]
    such that:
      xL = x_grid[j-1], xR = x_grid[j]
      y(x) ~ (1-r)*y[j-1] + r*y[j]
    """
    N = x_grid.shape[0]

    # clip x into grid domain (avoid out-of-bounds indices)
    x_clip = bk.clip(x, x_grid[0], x_grid[-1])

    # searchsorted gives insertion index in [0..N]
    j = bk.searchsorted(x_grid, x_clip, side="left")

    # clamp to valid interpolation interval [1..N-1]
    j = bk.clip(j, 1, N - 1)

    xL = x_grid[j - 1]
    xR = x_grid[j]
    denom = bk.maximum(xR - xL, 1e-30)

    r = (x_clip - xL) / denom
    r = bk.clip(r, 0.0, 1.0)

    return j, r



def interp_1d_nonuniform_multiY(bk, x, x_grid, Y, side="left", eps=1e-30):
    """
    Simple multi-Y nonuniform 1D linear interpolation.

    Inputs:
      x      : (...,)
      x_grid : (N,)
      Y      : (K, N)

    Output:
      out    : (K, ...)   (same as your Op)
    """
    N = x_grid.shape[0]

    # clip x into grid domain
    x_clip = bk.clip(x, x_grid[0], x_grid[-1])

    # indices of right/left bracket
    j = bk.searchsorted(x_grid, x_clip, side=side)
    j = bk.clip(j, 1, N - 1)

    xL = x_grid[j - 1]
    xR = x_grid[j]
    denom = bk.maximum(xR - xL, eps)
    r = (x_clip - xL) / denom
    r = bk.clip(r, 0.0, 1.0)

    # gather Y at j-1 and j for all K
    jm1 = (j - 1)[None, ...]  # (1, ...)
    j0  = j[None, ...]        # (1, ...)

    # prefer take_along_axis if backend has it
    if hasattr(bk, "take_along_axis"):
        YL = bk.take_along_axis(Y, jm1, axis=1)  # (K, ...)
        YR = bk.take_along_axis(Y, j0,  axis=1)  # (K, ...)
    else:
        # works for numpy-like backends
        YL = Y[:, (j - 1)]
        YR = Y[:, j]

    out = (1.0 - r)[None, ...] * YL + r[None, ...] * YR
    return out





# ---------------------------------------------------------------------
# JAX grid utilities for numerical normalization
# ---------------------------------------------------------------------

def trapz_weights_1d_jax(x):
    """Return trapezoidal integration weights for a sorted 1D JAX grid.

    The weights integrate functions sampled at x in the measure dx.
    x must be one-dimensional and have at least two points.
    """
    x = jnp.asarray(x)
    dx = jnp.diff(x)
    return jnp.concatenate([0.5 * dx[:1], 0.5 * (dx[:-1] + dx[1:]), 0.5 * dx[-1:]])


def sorted_clipped_nodes_jax(x, lo, hi):
    """Clip nodes to [lo, hi] and sort them without changing array shape.

    Duplicates are intentionally preserved; they contribute zero-width
    trapezoid intervals and keep the function JIT-shape stable.
    """
    return jnp.sort(jnp.clip(jnp.asarray(x), lo, hi))


# GP distance-reconstruction helpers from gp-pure-jax (cefafc0).

def reference_resolution_curve_debug():
    x_centers = np.array([
        0.25822883, 0.32575605, 0.39906972, 0.49655208, 0.58352095,
        0.65686376, 0.73981212, 0.82216291, 0.90026552, 0.98239349,
        1.06575534, 1.14405042, 1.21578127
    ], dtype=np.float64)

    r_centers = np.array([
        0.02370171, 0.03663001, 0.04111473, 0.06069169, 0.04841643,
        0.06055054, 0.07089649, 0.06296696, 0.07174308, 0.08220798,
        0.09366922, 0.09040657, 0.09191132
    ], dtype=np.float64)

    return x_centers, r_centers

def summarize_eventwise_redshift_cloud(z_event_samples):
    z_event_samples = np.asarray(z_event_samples, dtype=np.float64)
    z_event_samples = np.maximum(z_event_samples, 0.0)

    x_event_samples = np.log1p(z_event_samples)

    z_q16 = np.quantile(z_event_samples, 0.16, axis=1)
    z_q50 = np.quantile(z_event_samples, 0.50, axis=1)
    z_q84 = np.quantile(z_event_samples, 0.84, axis=1)

    x_q16 = np.quantile(x_event_samples, 0.16, axis=1)
    x_q50 = np.quantile(x_event_samples, 0.50, axis=1)
    x_q84 = np.quantile(x_event_samples, 0.84, axis=1)

    return dict(
        z_median=z_q50,
        z_q16=z_q16,
        z_q84=z_q84,
        z_w68=0.5 * (z_q84 - z_q16),
        x_median=x_q50,
        x_q16=x_q16,
        x_q84=x_q84,
        x_w68=0.5 * (x_q84 - x_q16),
        z_samples=z_event_samples,
        x_samples=x_event_samples,
        z_min_data=np.min(z_q16),
        z_max_data=np.max(z_q84),
        x_min_data=np.min(x_q16),
        x_max_data=np.max(x_q84),
    )

def estimate_resolution_curve_binned(x_evt, w_evt, nbins=15, q=0.10, min_per_bin=5):
    x_evt = np.asarray(x_evt, dtype=np.float64)
    w_evt = np.asarray(w_evt, dtype=np.float64)

    mask = np.isfinite(x_evt) & np.isfinite(w_evt) & (w_evt > 0)
    x_evt = x_evt[mask]
    w_evt = w_evt[mask]

    bins = np.linspace(x_evt.min(), x_evt.max(), int(nbins) + 1)

    xs, rs = [], []
    for i in range(int(nbins)):
        m = (x_evt >= bins[i]) & (x_evt < bins[i + 1])
        if np.sum(m) < min_per_bin:
            continue
        xs.append(np.mean(x_evt[m]))
        rs.append(np.quantile(w_evt[m], q))

    return np.asarray(xs, dtype=np.float64), np.asarray(rs, dtype=np.float64)

def sample_from_per_event_gmm(wts_l, mus_l, cho_covs_l, x, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    wts_l = np.asarray(wts_l, dtype=np.float64)
    mus_l = np.asarray(mus_l, dtype=np.float64)
    cho_covs_l = np.asarray(cho_covs_l, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    N, K = wts_l.shape

    u = rng.random(N)
    cdf = np.cumsum(wts_l, axis=1)
    ig = (u[:, None] <= cdf).argmax(axis=1).astype(int)

    idx = np.arange(N)
    mu_sel = mus_l[idx, ig, :]
    L_sel = cho_covs_l[idx, ig, :, :]

    return mu_sel + np.einsum("nij,nj->ni", L_sel, x)

def find_event_redshift_resolution(
    wts_l_np,
    mus_l_np,
    cho_covs_l_np,
    H0_range,
    Om_range,
    w0_range,
    Xi0_range,
    nXi0_range,
    N,
    nd,
    dLinj,
    z_from_dL_fn,
    sampling_GW,
    trials=1000,
    s0=0.10,
    rng=np.random.default_rng(123),
    return_diff=False,
):
    _ = (
        H0_range,
        Om_range,
        w0_range,
        Xi0_range,
        nXi0_range,
        sampling_GW,
        s0,
        return_diff,
    )

    H0_ref = 67.66
    Om_ref = 0.31
    w0_ref = -1.0
    Xi0_ref = 1.0
    nXi0_ref = 0.0

    dLinj = np.asarray(dLinj, dtype=np.float64).reshape(-1)
    z_inj = z_from_dL_fn(
        dLinj,
        float(H0_ref),
        float(Om_ref),
        float(w0_ref),
        float(Xi0_ref),
        float(nXi0_ref),
    )
    z_inj = np.asarray(z_inj, dtype=np.float64)

    z_event_samples = np.empty((int(N), int(trials)), dtype=np.float64)

    _log_setup(f"Drawing {trials} reference-cosmology redshift samples per event from GMM...")
    for j in range(int(trials)):
        Xwhite = rng.standard_normal((int(N), int(nd)))
        samples = sample_from_per_event_gmm(
            wts_l_np,
            mus_l_np,
            cho_covs_l_np,
            Xwhite,
            rng=rng,
        )
        d_nodes = np.exp(samples[:, 2])
        z_nodes = z_from_dL_fn(
            d_nodes,
            float(H0_ref),
            float(Om_ref),
            float(w0_ref),
            float(Xi0_ref),
            float(nXi0_ref),
        )
        z_event_samples[:, j] = np.asarray(z_nodes, dtype=np.float64)

    out = summarize_eventwise_redshift_cloud(z_event_samples)
    out["z_min_inj"] = np.min(z_inj)
    out["z_max_inj"] = np.max(z_inj)
    return out

def find_event_redshift_resolution_from_dL_samples(
    H0_range, Om_range, w0_range, Xi0_range, nXi0_range,
    dLinj,
    dL_samples,
    z_from_dL_fn,
    trials=1000,
    s0=0.10,
    rng=None,
    return_diff=False,
):
    _ = (H0_range, Om_range, w0_range, Xi0_range, nXi0_range, dLinj, s0, return_diff)

    if rng is None:
        rng = np.random.default_rng(123)

    H0_ref, Om_ref, w0_ref, Xi0_ref, nXi0_ref = 67.66, 0.31, -1.0, 1.0, 0.0

    dL_samples = np.asarray(dL_samples, dtype=np.float64)
    n_events, n_samples = dL_samples.shape

    if trials is None or trials <= 0 or n_samples <= trials:
        dL_use = dL_samples
    else:
        idx = np.empty((n_events, int(trials)), dtype=np.int64)
        for i in range(n_events):
            idx[i] = rng.choice(n_samples, size=int(trials), replace=False)
        dL_use = dL_samples[np.arange(n_events)[:, None], idx]

    z_flat = z_from_dL_fn(
        dL_use.reshape(-1),
        H0_ref, Om_ref, w0_ref, Xi0_ref, nXi0_ref,
    )

    z_event_samples = np.asarray(z_flat, dtype=np.float64).reshape(dL_use.shape)
    return summarize_eventwise_redshift_cloud(z_event_samples)

def gibbs_kernel_matrix(bk, x, ell, eta, x_centers, r_centers, jitter=1e-4):
    x = bk.asarray(x)
    ell = bk.asarray(ell)
    eta = bk.asarray(eta)

    r_x = r_of_x_gibbs_resolution(bk, x, x_centers, r_centers)
    ell_x = ell * r_x

    lx = ell_x[:, None]
    lxp = ell_x[None, :]

    dx = x[:, None] - x[None, :]
    denom = lx**2 + lxp**2

    pref = bk.sqrt((2.0 * lx * lxp) / denom)
    expo = bk.exp(-(dx**2) / denom)

    K = eta**2 * pref * expo
    return K + jitter * bk.eye(x.shape[0])

def matern52_kernel_matrix(bk, x, ell, eta, jitter=1e-4):
    x = bk.asarray(x)
    ell = bk.asarray(ell)
    eta = bk.asarray(eta)

    dx = bk.abs(x[:, None] - x[None, :])
    r = bk.sqrt(5.0) * dx / ell

    K = eta**2 * (1.0 + r + r**2 / 3.0) * bk.exp(-r)
    return K + jitter * bk.eye(x.shape[0])

def r_of_x_gibbs_resolution(bk, x, x_centers, r_centers):
    x = bk.asarray(x)
    x_centers = bk.asarray(x_centers)
    r_centers = bk.asarray(r_centers)

    log_r_centers = bk.log(r_centers)

    x_min = x_centers[0]
    x_max = x_centers[-1]

    x_clip = bk.clip(x, x_min, x_max)
    log_r_interp = atinterp(bk, x_clip, x_centers, log_r_centers)

    log_r_left = log_r_centers[0]

    n_tail = min(10, int(x_centers.shape[0]))
    n_tail = max(n_tail, 2)
    slope_floor = 0.1

    x_tail = x_centers[-n_tail:]
    y_tail = log_r_centers[-n_tail:]

    w = bk.linspace(0.5, 1.5, n_tail)
    W = bk.sum(w)

    x_mean = bk.sum(w * x_tail) / W
    y_mean = bk.sum(w * y_tail) / W

    cov_xy = bk.sum(w * (x_tail - x_mean) * (y_tail - y_mean))
    var_x = bk.sum(w * (x_tail - x_mean) ** 2)

    slope = cov_xy / (var_x + 1e-12)
    slope = bk.maximum(slope, slope_floor)

    log_r_right = y_tail[-1] + slope * (x - x_tail[-1])

    log_r = bk.where(
        x < x_min,
        log_r_left,
        bk.where(x > x_max, log_r_right, log_r_interp),
    )

    return bk.exp(log_r)

def interp_r_asym_extrap_np_numpyro(x_centers, r_centers, x_eval, n_tail=10, slope_floor=0.1):
    import numpy as np

    x_centers = np.asarray(x_centers, dtype=np.float64)
    r_centers = np.asarray(r_centers, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)

    log_r = np.log(r_centers)

    log_r_eval = np.interp(
        np.clip(x_eval, x_centers.min(), x_centers.max()),
        x_centers,
        log_r,
    )

    # left: constant
    left = x_eval < x_centers[0]
    log_r_eval[left] = log_r[0]

    # right: weighted linear fit on last n_tail points
    right = x_eval > x_centers[-1]
    if np.any(right):
        n_tail = min(int(n_tail), len(x_centers))
        n_tail = max(n_tail, 2)

        x_tail = x_centers[-n_tail:]
        y_tail = log_r[-n_tail:]

        w = np.linspace(0.5, 1.5, n_tail)
        W = np.sum(w)

        x_mean = np.sum(w * x_tail) / W
        y_mean = np.sum(w * y_tail) / W

        cov_xy = np.sum(w * (x_tail - x_mean) * (y_tail - y_mean))
        var_x = np.sum(w * (x_tail - x_mean) ** 2)

        slope = cov_xy / (var_x + 1e-12)
        slope = max(slope, slope_floor)

        log_r_eval[right] = y_tail[-1] + slope * (x_eval[right] - x_tail[-1])

    return np.exp(log_r_eval)

def plot_resolution_curve_numpyro(
    x_centers,
    r_centers,
    x_evt=None,
    w_evt=None,
    fout=None,
    filename="gp_resolution_curve.png",
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    x_centers = np.asarray(x_centers, dtype=np.float64)
    r_centers = np.asarray(r_centers, dtype=np.float64)

    z_centers = np.expm1(x_centers)

    if x_evt is not None and w_evt is not None and len(x_evt) > 0:
        x_evt = np.asarray(x_evt, dtype=np.float64)
        w_evt = np.asarray(w_evt, dtype=np.float64)
        mask = np.isfinite(x_evt) & np.isfinite(w_evt) & (w_evt > 0)
        x_evt = x_evt[mask]
        w_evt = w_evt[mask]
    else:
        x_evt = np.asarray([], dtype=np.float64)
        w_evt = np.asarray([], dtype=np.float64)

    if len(x_evt) > 0:
        x_min_plot = min(float(x_evt.min()), float(x_centers.min()))
        x_max_plot = max(float(x_evt.max()), float(x_centers.max())) * 1.15
    else:
        x_min_plot = float(x_centers.min())
        x_max_plot = float(x_centers.max()) * 1.15

    x_plot = np.linspace(x_min_plot, x_max_plot, 500)
    r_plot = interp_r_asym_extrap_np_numpyro(
        x_centers,
        r_centers,
        x_plot,
        n_tail=10,
        slope_floor=0.1,
    )
    z_plot = np.expm1(x_plot)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    if len(x_evt) > 0:
        ax.scatter(
            np.expm1(x_evt),
            w_evt,
            s=24,
            alpha=0.45,
            label=r"Event widths",
        )
    else:
        _log_setup("No event widths to plot; plotting reference resolution only.")

    ax.plot(
        z_centers,
        r_centers,
        "o",
        ms=6,
        label=r"Binned resolution points",
    )

    ax.plot(
        z_plot,
        r_plot,
        lw=2.6,
        label=r"Interpolated / extrapolated $r(x)$",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"Resolution in $x=\log(1+z)$")
    ax.set_title("Data-driven redshift resolution curve")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    fig.tight_layout()

    if fout is not None:
        os.makedirs(fout, exist_ok=True)
        path = os.path.join(fout, filename)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        _log_setup(f"Saved GP resolution plot: {path}")
    else:
        return fig, ax
