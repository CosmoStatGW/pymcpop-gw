# jax_models.py
from __future__ import annotations

import json

import numpy as np
import jax

jax.config.update("jax_debug_nans", True)

import jax.numpy as jnp
from jax import lax

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints


import cosmology as cosmo
from backends import NPBackend, JAXBackend
from likelihood import LikDataGauss, MarginalSampleData, encode_dLprior_list, make_loglik_gauss, make_loglik_marginal_samples
from population import _make_pop_and_sel_core
from constants import PlanckFiducials, PLANCK15_H0, PLANCK15_OM, z_nodes_jax


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
    # GWData = (mus_s, cho_s, log_wts_l, mus_l, icovs_l, log_dets_l, cho_covs_l, Tobs_np, Nevs, allnames)
    mus_s, cho_s, log_wts_l, mus_l, icovs_l, log_dets_l, _, _Tobs_np, _Nevs, _allnames = GWData

    mus_s = np.asarray(mus_s, dtype=np.float64)
    cho_s = np.asarray(cho_s, dtype=np.float64)
    mus_l = np.asarray(mus_l, dtype=np.float64)
    icovs_l = np.asarray(icovs_l, dtype=np.float64)
    log_dets_l = np.asarray(log_dets_l, dtype=np.float64)
    log_wts_l = np.asarray(log_wts_l, dtype=np.float64)

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
        print("\nInjections shape: ninj, dLinj0.shape, m2inj0.shape, lpdinj0.shape, lp_incl0.shape")
        print(ninj, dLinj0.shape, m2inj0.shape, lpdinj0.shape, lp_incl0.shape )
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
    mask_flat = where_compute.reshape(n_tot)

    m1_flat = m1det.reshape(n_tot)[mask_flat]
    m2_flat = m2det.reshape(n_tot)[mask_flat]
    dL_flat = dL.reshape(n_tot)[mask_flat]
    prior_flat_raw = dL_prior.reshape(n_tot)[mask_flat]
    if np.all(prior_flat_raw > 0.0):
        log_pe_flat = np.log(np.clip(prior_flat_raw, 1e-300, np.inf))
    else:
        log_pe_flat = prior_flat_raw
        
    
    event_id_flat = np.repeat(np.arange(N, dtype=np.int32), S)[mask_flat]

    Nsamples_evt = np.asarray(allNsamples, dtype=np.float64)
    if Nsamples_evt.ndim == 0:
        Nsamples_evt = np.full((N,), float(Nsamples_evt), dtype=np.float64)
    if Nsamples_evt.shape != (N,):
        raise ValueError(f"allNsamples must be scalar or shape ({N},), got {Nsamples_evt.shape}")
    if not np.all(Nsamples_evt > 1):
        raise ValueError("Need at least two posterior samples per event to compute MC variance")

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

    print(f"[PE marginal] M={m1_flat.shape[0]}, chunk_pe={int(chunk_pe)}")

    return MarginalSampleData(
        m1det_pe=jnp.asarray(m1_flat, dtype=jnp.float64),
        m2det_pe=jnp.asarray(m2_flat, dtype=jnp.float64),
        dL_pe=jnp.asarray(dL_flat, dtype=jnp.float64),
        spins_pe=jnp.asarray(spins_flat, dtype=jnp.float64),
        log_PE_prior_pe=jnp.asarray(log_pe_flat, dtype=jnp.float64),
        event_id_pe=jnp.asarray(event_id_flat, dtype=jnp.int32),
        Nsamples_evt=jnp.asarray(Nsamples_evt, dtype=jnp.float64),

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
        interp_mass=int(interp_mass),
        integrate_dc=data.integrate_dc,

        # we are in pop_only=False branch
        pop_only=False,

        # you are not using var_u in the gauss likelihood path right now
        stop_grad_var_u=bool(stop_grad_var_u),
        return_var=True,
    )

    loglik = make_loglik_gauss(core, data)
    # optional: force compilation once (useful for debugging shapes)
    # _ = loglik(jnp.zeros((1,)), jnp.zeros((data.Nobs, data.mus_s.shape[1])))

    return core, loglik



def build_core_and_loglik_samples_marginal(
    data,
    *,
    chunk_inj=0,
    K_dp=30,
    DP_truncate=False,
    DP_m1_env=False,
    interp_mass=0,
    stop_grad_var_u=True,
    verbose=False,
    z_nodes=None,
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
        skip_sel=False,
        chunk_inj=int(chunk_inj),
        chunk_evt=int(data.chunk_pe),
        K_dp=int(K_dp),
        DP_truncate=bool(DP_truncate),
        DP_m1_env=bool(DP_m1_env),
        interp_mass=int(interp_mass),
        integrate_dc=data.integrate_dc,
        pop_only=True,
        stop_grad_var_u=bool(stop_grad_var_u),
        return_var=True,
    )

    print("[PE marginal] using event streaming:", int(data.chunk_pe) > 0)
    print("[PE marginal] chunk_pe:", int(data.chunk_pe))

    loglik = make_loglik_marginal_samples(core, data)
    return core, loglik



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


def bounded_sigmoid_logpositive(name: str, low: float, high: float, raw_sigma: float = 1.5):
    raw = numpyro.sample(f"{name}_raw", dist.Normal(0.0, raw_sigma))
    lowlog, highlog = np.log(low), np.log(high)
    logx = lowlog + (highlog - lowlog) * jax.nn.sigmoid(raw)
    numpyro.deterministic(f"{name}_log", logx)
    val = jnp.exp(logx)
    numpyro.deterministic(name, val)
    return val


    
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
                 L_small_3 = 0.1,
                 s_local = 0.5,
                 find_m_bounds = False,
                 q_mbound = 0.05,
                 alpha_inv_params = (1, 1),
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
               alpha_beta_prior='poly',
               dil_factor=1,
               use_log_alpha_beta=False ,
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
                 DP_truncate_low=False,
                 DP_m1_env = False,
                 detach_var = False,
                remove_spin_prior = False,
                r = 0,
                     vary_mb = True,
                     taper_kind = "sigmoid",
                taper_p = 12.0
                ):



    ################################################
    # Read in data and set dimensions
    ################################################

    
    ## GW data
    if not pop_only:
        # gw data are interpolants of single-event posteriors
        if sampling_GW=='gauss':
            # we sample single-event parameters from broad gaussian approximations of the posteriors
            mus_s, cho_s, log_wts_l, mus_l, icovs_l, log_dets_l, cho_covs_l, Tobs_np, Nevs, allnames = GWData
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
            
        if (spin_model=='default') or (spin_model=='default_gauss'):
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
    #Nevs_np = np.atleast_1d(Nevs)
    Nevs_np = np.atleast_1d(Nevs).astype(np.int32)
    N_total = int(np.sum(Nevs_np))


        
    if not pop_only:
        N = mus_l.shape[0]
        N_np = N 
        ngmm = mus_l.shape[1]
        ngmm_np = ngmm
        nd = mus_l.shape[2]
        nd_np = nd
        print('N:%s, max ngmm: %s, nd: %s '%(N_np, ngmm_np, nd_np))
        print('N evs is %s'%Nevs_np)
        print('Tobs is %s'%Tobs_np)
    else:
        N = m1det.shape[0] 
        N_np = N 
        Nsamples = m1det.shape[1]
        Nsamples_np = Nsamples 

        if Nsamplesuse !=-1 :
            if Nsamplesuse>Nsamples_np:
                raise ValueError("Must use less samples than those available.")
            print("Nsamples_np available is %s, but %s will be used"%(Nsamples_np, Nsamplesuse))
            
            m1det, m2det, d = m1det[:, :Nsamplesuse], m2det[:, :Nsamplesuse], d[:, :Nsamplesuse]
            dL_prior = dL_prior[:, :Nsamplesuse]
            spins = np.asarray([s[:, :Nsamplesuse] for s in spins ])
            if (spin_model=='default') or (spin_model=='default_gauss'):
               chi1, chi2, cost1, cost2 = chi1[:, :Nsamplesuse], chi2[:, :Nsamplesuse], cost1[:, :Nsamplesuse], cost2[:, :Nsamplesuse]

            allNsamples = Nsamplesuse

            Nsamples = m1det.shape[1]
            Nsamples_np = Nsamples 
            allNsamples_np = np.full( N, Nsamplesuse )

        else:
            allNsamples_np = allNsamples 
        
        assert np.all( allNsamples_np == Nsamples_np )
        print("N samples will be ")
        print(Nsamples_np)
        print('N:%s, n samples: %s '%(N_np, Nsamples_np))


        ### reshape

        if spin_model in ("default", "default_gauss"):
            spins = np.stack([chi1, chi2, cost1, cost2], axis=1)  # (N,4)

             
        logd = np.log(d)
        
        NsamplesTot = N*Nsamples

        print("Reshaping samples to %s"%NsamplesTot)
        
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

    
    print('Injections: :%s, '%(ninj_np))

    print('ninj: :%s, %s datasets,'%(Ndet_np, ndata_np))

    coords = {'event_index': event_index}

    

    if mass_model in ('DP', 'DPUC'):
        coords['component'] = np.arange(N_DP_comp_max_np, dtype=int)
        
        if rate_model in ('DPUC','DPUC-vol', 'DPUC-vol-MD'):
            ndim_GMM = 3
        else:
            ndim_GMM = 2

        print('GMM dimension is %s'%ndim_GMM)
        
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
        print('No values for parameters to fix passed. Default values will be used. If fixing parameters, check that the values are consistent. Values of fixed parameters:')
        print(PlanckFiducials)
        params_fix=PlanckFiducials


 
    ################################################
    # Build grids and other data-dependend quantities
    ################################################
    
    if ( find_z_bounds or (mass_model in ('DPUC', 'DP') and find_m_bounds) or mmin_inj!=-1 ):

        raise NotImplementedError()
          

    if z_grid_mode=='cst':

        print("Using pre-defined z grid with 1200 points")
        z_nodes = z_nodes_jax

    elif z_grid_mode=='man':

        print("Using custom z grid with %s points"%zres)
        z_nodes = make_z_grid_jax( zres )

    else:
        raise ValueError()

    print("Min, max redshift grid: %s, %s"%(z_nodes.min(), z_nodes.max()))


    
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
            
            print("Loading bilby pre-computed PE prior from distance for later interpolation")
            dat = np.load("dLgrid_gpc_bilby_prior_grid_O4a.npz")
            dLgrid_bilby_gpc = dat["dLgrid_gpc"]
            PE_prior_bilby_grid =  dat["prior_grid"]

        else:
            dLgrid_bilby_gpc = None
            PE_prior_bilby_grid = None
            

        if ( ( vol_in_prior or vol_in_prior_from_bilby) and (penorm_lims != []) ):
    
            print("Normalization of PE volume prior on distance required.")
                
      
            bkNP = NPBackend()
            
            Nchunks = len(Nevs_np)
            assert len(allnames) == Nchunks
            j = 0
            all_PE_log_norms = np.zeros(N, dtype=np.float64)
            for i in range(Nchunks):
                
                if  penorm_lims[i]=='none':
                    print("No normalization of PE prior on distance included for chunk %s"%i)
                    for key in allnames[i]:
                        all_PE_log_norms[j] = 0.
                        j+=1
                else:
                    with open( penorm_lims[i] , 'r') as fp:
                        plims_ = json.load(fp)
                    
                    print("Normalization of PE prior on distance for chunk %s loaded"%i)
                    
                    for key in allnames[i]:
                        try:
                            lims_ = plims_[key]
                        except:
                            raise ValueError("limits for %s not present"%key)    
                         
                        log_norm_PE_prior_, za, zb = cosmo.compute_log_norm_UniformSourceFrame(bkNP, lims_[0]/1000, lims_[1]/1000, 67.9, 0.3065, -1)
                        #print("event, dmin [Gpc], dmax [Gpc], zmin, zmax, log_norm")
                        #
                        
                        
                        print(key, lims_[0]/1000, lims_[1]/1000, za, zb, log_norm_PE_prior_)
                
                        all_PE_log_norms[j] = log_norm_PE_prior_
                        j+=1
                
                print("at the end of chunk %s, index j is %s"%(i,j))
    
            all_PE_log_norms = np.asarray(all_PE_log_norms)
        else:
            print("No normalization of PE volume prior on distance required.")
            all_PE_log_norms = np.zeros(Nevs_np.sum(), dtype=np.float64)

    
        if remove_spin_prior:
            print("Removing PE spin prior")
            amax = 0.99
            spinp= (1./amax)*(1./amax)*0.5*0.5  

            # all_PE_log_norms is later subtraceted to log_PE_prior, so we subtract log(spinp) here
            # this will result in adding log(spinp) to the log_PE_prior which is subtracted itself
            # to the likelihood. 
            all_PE_log_norms -= np.log(spinp)

             
        print("All PE log norms is ")
        print("Shape: %s"%all_PE_log_norms.shape)

 
        
    
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
        print("pack_data_gauss_popnot ok")
        print()
        core, loglik = build_core_and_loglik_gauss_popnot(
            data,
            chunk_inj=(0 if chunk_inj in (-1, None) else chunk_inj),
            K_dp=30,
            DP_truncate=False,
            DP_m1_env=DP_m1_env,
            interp_mass=interp_mass,
            stop_grad_var_u=True,
            skip_sel=False,
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

        core, loglik = build_core_and_loglik_samples_marginal(
            data,
            chunk_inj=(0 if chunk_inj in (-1, None) else chunk_inj),
            K_dp=30,
            DP_truncate=False,
            DP_m1_env=DP_m1_env,
            interp_mass=interp_mass,
            stop_grad_var_u=True,
            verbose=False,
            z_nodes=z_nodes,
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
        if rate_model in ("MD", "DPUC-vol-MD"):
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
        if mass_model == "DPLDP" or mass_model=='PLDP':
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
 

                if mass_model=='DPLDP':
                    
                    # reparam latents
                    # initvals are handled by init_strategy outside (init_to_value),
                    # so we just declare the sample sites.
                    
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
        

                else:
                    # single power law PLDP
        
                    a_diff = numpyro.deterministic("alpha_diff", 0. )

                    # deterministics
                    alpha1_ = numpyro.deterministic("alpha1", a_bar )
                    alpha2_ = numpyro.deterministic("alpha2", a_bar )

                    mb_ = numpyro.deterministic("mb", 35. )

                    delta_m2_ = numpyro.deterministic("delta_m2", delta_m1_)
                    m2_low_ = numpyro.deterministic("m2_low", m1_low_ )
                    
                
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

                if mass_model=='DPLDP':
                    
                    
                    alpha2_ = numpyro.sample("alpha2", dist.Uniform(priors["alpha2"][0], priors["alpha2"][1]))
                    mb_ = numpyro.sample("mb", dist.Uniform(priors["mb"][0], priors["mb"][1]))

                    u = numpyro.sample("u", dist.Uniform(0, 1))
                    m1_low_ = 3.0 + (10.0 - 3.0) * jnp.sqrt(u)
                    numpyro.deterministic("m1_low", m1_low_)
            
                    v = numpyro.sample("v", dist.Uniform(0, 1))
                    m2_low_ = 3.0 + v * (m1_low_ - 3.0)
                    numpyro.deterministic("m2_low", m2_low_)

                    delta_m2_ = numpyro.sample("delta_m2", dist.Uniform(priors["delta_m2"][0], priors["delta_m2"][1]))
                    
                else:
                    # single power law , PLDP
                    
                    alpha2_ = numpyro.deterministic("alpha2", alpha1_)
                    mb_ = numpyro.deterministic("mb", 35. )
                    delta_m2_ = numpyro.deterministic("delta_m2", delta_m1_)
                    m1_low_ = numpyro.sample("m1_low", dist.Uniform(priors["m1_low"][0], priors["m1_low"][1]))
                    m2_low_ = numpyro.deterministic("m2_low", m1_low_ )
                    

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
        #Lambda = jnp.asarray(Lambda_list, dtype=jnp.float64)
        Lambda = jnp.stack([jnp.asarray(z, dtype=jnp.float64) for z in Lambda_list])

    


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
            ll, log_lik_var_sg = loglik(Lambda, x, lR0=lR0)
        else:
            ll, log_lik_var_sg = loglik(Lambda, lR0=lR0)

        ## Uncomment for debugging
        # jax.debug.print("ll = {}", ll)
        # jax.debug.print("ll finite? {}", jnp.all(jnp.isfinite(jnp.asarray(ll))))


        #ll_safe = jnp.where(jnp.isfinite(ll), ll, -1e30)
        numpyro.factor("likelihood", ll)

        
        ## likelihood variance bound:
        if sel_smoothing=='sigmoid':
            gate_llv = jax.nn.log_sigmoid( (log_lik_var_min - log_lik_var_sg) / 0.01 )
        elif sel_smoothing=='none':
            ## hard version 
            gate_llv = jnp.where(log_lik_var_sg <= log_lik_var_min, 0.0, -1e30)
        elif sel_smoothing=='PL':
            # Heinzel et al 2025
            v = jnp.maximum(log_lik_var_sg - log_lik_var_min, 0.0)
            gate_llv = -100.0 * v**2
        elif sel_smoothing=='x30':
            # Callister Farr 2023
            x = log_lik_var_sg / log_lik_var_min
            gate_llv = -jnp.log1p(x**30)
        else:
            raise ValueError("sel_smoothing takes values sigmoid, PL, none, x30, got %s"%sel_smoothing)

        ## Uncomment for debugging
        #jax.debug.print("log_lik_var_sg = {}", log_lik_var_sg)
        #jax.debug.print("gate_llv = {}", gate_llv)
        #jax.debug.print("log_lik_var_sg finite? {}", jnp.isfinite(log_lik_var_sg))
        
        #gate_llv_safe = jnp.where(jnp.isfinite(gate_llv), gate_llv, -1e30)
        numpyro.factor("bound_log_lik_var", gate_llv)


        # optional for debugging/outputs
        numpyro.deterministic("loglik", ll)
        numpyro.deterministic("log_lik_var", log_lik_var_sg)
        
        
        
    return model_numpyro, data, core, loglik