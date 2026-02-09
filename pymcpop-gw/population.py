from __future__ import annotations

# backend-agnostic pieces
from cosmology import Xi_vanilla, Xi_polexp, Efun, log_ddL_dz as log_ddL_dz_bk
from rate_models import log_p_z_MD_unnorm as log_p_z_MD_unnorm_bk
from spin_models import logpdf_default_spin_gauss as logpdf_default_spin_gauss_bk
from mass_models import logpdf_DPLDP as logpdf_DPLDP_bk
from pytensor_utils import logdiffexp as logdiffexp_bk
from pytensor_utils import logsumexp as _logsumexp
from jax_utils import _interp_prepare_bk, _interp_apply_bk, _interp_apply_multi_bk, _interp_prepare_uniform_bk
from pytensor_utils import atinterp, atinterp_uniform

import jax.numpy as jnp
from jax import lax

try:
    import jax    
except Exception as e:
    print(e)
    raise ValueError()



def _zeros_like_tree(x):
    # works for arrays, scalars, and lists/tuples of arrays
    return jax.tree_util.tree_map(lambda a: jnp.zeros_like(a), x)



# ---------------------------------------------------------------------
#  p_pop
# ---------------------------------------------------------------------

def log_p_pop(
    bk,
    m1s,
    m2s,
    z,
    dL,
    spins,
    Lambda,
    *,
    rate_model,
    mass_model,
    spin_model,
    smoothing="LVK",
    simplex_repair=False,
    has_m2_break=False,
    norm_gauss="uplow",
    dc=None,
    Xi=None,
    E=None,
    log_ddL_dz_pre=None,
    param="vanilla",
    #interp_vals_mass=None,
    #interp_grids_mass=None,
    #is_observed=False,
    z_grid=None,
    verbose=False,
    linear_mass=False
):
    """
    Backend-agnostic log_p_pop_at,
    currently supporting ONLY:
      rate_model == "MD"
      spin_model == "default_gauss"
      mass_model == "DPLDP"
    """

    # Cosmology hyper-params
    H0, Om, w0, Xi0, n = Lambda[0], Lambda[1], Lambda[2], Lambda[3], Lambda[4]

    if Xi is None:
        if param == "vanilla":
            Xi = Xi_vanilla(bk, z, Xi0, n)
        elif param == "polexp":
            Xi = Xi_polexp(bk, z, Xi0, n)
        else:
            raise ValueError(f"Unknown param='{param}'")
    # If dc not provided, infer from dL and Xi(z)
    if dc is None:
        dc = dL / (1.0 + z) / Xi
        
    if E is None:
        E = Efun(bk, z, Om, w0)

    # -----------------------
    # rate model (MD only)
    # -----------------------
    if rate_model == "MD":
        gamma, kappa, zp = Lambda[5], Lambda[6], Lambda[7]
        lpz = log_p_z_MD_unnorm_bk(bk, z, gamma, kappa, zp, H0, Om, w0, dc=dc, E=E)
        istart = 8
        z_dpuc = None
    else:
        raise NotImplementedError("Only rate_model=='MD' is implemented in this rewrite.")

    # -----------------------
    # spin model (default_gauss only)
    # -----------------------
    if spin_model == "default_gauss":
        muChi = Lambda[istart + 0]
        sigmaChi = Lambda[istart + 1]
        zeta = Lambda[istart + 2]
        sigmat = Lambda[istart + 3]

        # expected spins layout: (chi1, chi2, cost1, cost2)
        lpspin = logpdf_default_spin_gauss_bk(bk, spins, (muChi, sigmaChi, zeta, sigmat))
        istart_spin = istart + 4
    else:
        raise NotImplementedError("Only spin_model=='default_gauss' is implemented in this rewrite.")

    # -----------------------
    # mass model (DPLDP only)
    # -----------------------
    if mass_model == "DPLDP":
        # 21 params
        x1  = Lambda[istart_spin +  0]; x2  = Lambda[istart_spin +  1]
        x3  = Lambda[istart_spin +  2]; x4  = Lambda[istart_spin +  3]
        x5  = Lambda[istart_spin +  4]; x6  = Lambda[istart_spin +  5]
        x7  = Lambda[istart_spin +  6]; x8  = Lambda[istart_spin +  7]
        x9  = Lambda[istart_spin +  8]; x10 = Lambda[istart_spin +  9]
        x11 = Lambda[istart_spin + 10]; x12 = Lambda[istart_spin + 11]
        x13 = Lambda[istart_spin + 12]; x14 = Lambda[istart_spin + 13]
        x15 = Lambda[istart_spin + 14]; x16 = Lambda[istart_spin + 15]
        x17 = Lambda[istart_spin + 16]; x18 = Lambda[istart_spin + 17]
        x19 = Lambda[istart_spin + 18]; x20 = Lambda[istart_spin + 19]
        x21 = Lambda[istart_spin + 20]

        lambdaBBHmass = (
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
            x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21
        )
        

        lpmass = logpdf_DPLDP_bk(
            bk,
            (m1s, m2s),
            lambdaBBHmass,
            force_m2_less_than_m1=False,
            has_m2_break=has_m2_break,
            smoothing=smoothing,
            interp_vals=None,
            interp_grids=None,
            norm=True,
            simplex_repair=simplex_repair,
            norm_gauss=norm_gauss,
        )
        
    else:
        raise NotImplementedError("Only mass_model=='DPLDP' is implemented in this rewrite.")

    # -----------------------
    # Jacobian term
    # -----------------------
    if log_ddL_dz_pre is None:
        # IMPORTANT: we already have dc, so cosmology.log_ddL_dz doesn't need x01,w01
        log_dthD_dth = log_ddL_dz_bk(
            bk, z, H0, Om, w0, Xi0, n, dc=dc, param=param, Xi=Xi, E=E, 
        )
    else:
        log_dthD_dth = log_ddL_dz_pre

    log_dthD_dth = log_dthD_dth + 2.0 * bk.log1p(z)

    # population log density
    lp = lpz - log_dthD_dth + lpmass + lpspin
    return lp



# ---------------------------------------------------------------------
#  standard sel. bias
# ---------------------------------------------------------------------


def sel_bias_with_uncertainty_legacy(
    bk,
    m1inj,
    m2inj,
    dLinj,
    spinsInj,
    log_p_draw,
    log_p_incl,
    dL_grid,
    dc_grid,
    log_ddL_dz_grid,
    Lambda,
    Ndraw,
    *,
    #zgrid,
    rate_model,
    mass_model,
    spin_model,
    smoothing="LVK",
    simplex_repair=False,
    has_m2_break=False,
    norm_gauss="uplow",
    param="vanilla",
    #interp_vals_mass=None,
    #interp_grids_mass=None,
    #is_observed=False,
    z_grid=None,
    verbose=False,
    subtract_log_p_incl=True,
    eps_interp=1e-12,
    side_interp="right",
    # -------- optional precomputed cosmology at injections --------
    zinj=None,
    dcinj=None,
    log_ddL_dz_inj=None,
    XiInj=None,
    Einj=None,
    linear_mass=False
):
    """
    Single canonical selection-bias function used by both forward and VJP.

    If zinj/dcinj/log_ddL_dz_inj are provided, internal inversions/interps are skipped.
    """

    # ---- if not precomputed, compute zinj, dcinj, log_ddL_dz_inj by interpolation ----
    if zinj is None:
        raise ValueError()
        # z(dL): xp=dL_grid, fp=zgrid
        i_dL, t_dL = _interp_prepare_bk(bk, dLinj, dL_grid, eps=eps_interp, side=side_interp)
        zinj = _interp_apply_bk(bk, i_dL, t_dL, z_grid)

    if (dcinj is None) or (log_ddL_dz_inj is None):
        raise ValueError()
        # dc(z), log_ddL_dz(z): xp=zgrid, fp=grids
        i_z, t_z = _interp_prepare_bk(bk, zinj, z_grid, eps=eps_interp, side=side_interp)
        if dcinj is None:
            dcinj = _interp_apply_bk(bk, i_z, t_z, dc_grid)
        if log_ddL_dz_inj is None:
            log_ddL_dz_inj = _interp_apply_bk(bk, i_z, t_z, log_ddL_dz_grid)

    onepz = 1.0 + zinj
    m1Src = m1inj / onepz
    m2Src = m2inj / onepz

    log_p_pop_vals = log_p_pop(
        bk,
        m1Src, m2Src, zinj, dLinj,
        spinsInj,
        Lambda,
        rate_model=rate_model,
        mass_model=mass_model,
        spin_model=spin_model,
        smoothing=smoothing,
        simplex_repair=simplex_repair,
        has_m2_break=has_m2_break,
        norm_gauss=norm_gauss,
        dc=dcinj,
        log_ddL_dz_pre=log_ddL_dz_inj,
        Xi=XiInj,
        E=Einj,
        param=param,
        #interp_vals_mass=interp_vals_mass,
        #interp_grids_mass=interp_grids_mass,
        #is_observed=is_observed,
        z_grid=z_grid,
        verbose=verbose,
        linear_mass=linear_mass
    )

    log_sel_b = log_p_pop_vals - log_p_draw
    if subtract_log_p_incl:
        log_sel_b = log_sel_b - log_p_incl

    # fast two-logsumexp reduction (matches your Op)
    x = log_sel_b
    m = bk.max(x)
    u = bk.exp(x - m)
    s1 = bk.sum(u)
    s2 = bk.sum(u * u)
    lse1 = m + bk.log(s1)
    lse2 = 2.0 * m + bk.log(s2)

    logN = bk.log(Ndraw)
    log_mu = lse1 - logN
    logs2  = lse2 - logN

    var_log_lik_u = logdiffexp_bk(bk, logs2 - 2.0 * log_mu, 1.0) - bk.log(Ndraw - 1.0)
    return log_mu, var_log_lik_u





# ---------------------------------------------------------------------
#  sel. bias with streaming 
# ---------------------------------------------------------------------


def sel_bias_with_uncertainty_streaming_vjp(
    bk,
    m1inj,
    m2inj,
    dLinj,
    spinsInj,
    log_p_draw,
    log_p_incl,
    dL_grid,
    dc_grid,
    log_ddL_dz_grid,
    Lambda,
    Ndraw,
    *,
    rate_model,
    mass_model,
    spin_model,
    smoothing="LVK",
    simplex_repair=False,
    has_m2_break=False,
    norm_gauss="uplow",
    param="vanilla",
    z_grid=None,
    verbose=False,
    eps_interp=1e-12,
    side_interp="right",
    # precomputed injection cosmology (required in your current usage)
    zinj,
    dcinj,
    log_ddL_dz_inj,
    XiInj,
    Einj,
    linear_mass=False,
    # new controls
    chunk_size: int = 65536,
):
    """
    Optimized selection term:
      - forward: streaming (one-pass) computation of log_mu and var_u
      - backward: custom VJP that differentiates ONLY log_mu (forces g_var_u=0)
      - chunked recomputation in backward to avoid huge tape/intermediates

    chunk_size:
      - >0 : process in chunks of this size
      - <=0: treat as single chunk (no chunking)
    """

    # ---- constants are treated as constants upstream already, but we keep them as args ----
    # NOTE: you said subtract_log_p_incl is always True, so we bake it in.

    B = int(chunk_size) if chunk_size and chunk_size > 0 else 0

    def _pad_to_multiple(x, n_pad, *, mode="edge"):
        # pad 1D or 2D arrays on axis=0 to length n_pad
        n = x.shape[0]
        pad = n_pad - n
        if pad == 0:
            return x
        if x.ndim == 1:
            if mode == "edge":
                tail = jnp.repeat(x[-1:], pad, axis=0)
            else:
                tail = jnp.zeros((pad,), dtype=x.dtype)
            return jnp.concatenate([x, tail], axis=0)
        elif x.ndim == 2:
            if mode == "edge":
                tail = jnp.repeat(x[-1:, :], pad, axis=0)
            else:
                tail = jnp.zeros((pad, x.shape[1]), dtype=x.dtype)
            return jnp.concatenate([x, tail], axis=0)
        else:
            raise ValueError("Unexpected ndim for padding")

    def _make_mask(n, n_pad):
        # True for real entries, False for padded
        return jnp.arange(n_pad) < n

    def _score_chunk(
        Lambda_,
        zinj_c,
        dcinj_c,
        logdd_c,
        Xi_c,
        E_c,
        m1_c,
        m2_c,
        dL_c,
        spins_c,
        lpd_c,
        lpi_c,
        mask_c,
    ):
        onepz = 1.0 + zinj_c
        m1Src = m1_c / onepz
        m2Src = m2_c / onepz

        lp_pop = log_p_pop(
            bk,
            m1Src, m2Src, zinj_c, dL_c,
            spins_c,
            Lambda_,
            rate_model=rate_model,
            mass_model=mass_model,
            spin_model=spin_model,
            smoothing=smoothing,
            simplex_repair=simplex_repair,
            has_m2_break=has_m2_break,
            norm_gauss=norm_gauss,
            dc=dcinj_c,
            log_ddL_dz_pre=logdd_c,
            Xi=Xi_c,
            E=E_c,
            param=param,
            z_grid=z_grid,
            verbose=verbose,
            linear_mass=linear_mass,
        )

        x = lp_pop - lpd_c - lpi_c
        # padded entries contribute nothing: set to -inf
        x = jnp.where(mask_c, x, -jnp.inf)
        return x  # (B,)

    @jax.custom_vjp
    def _sel_core(
        Lambda_,
        zinj_,
        dcinj_,
        logdd_,
        Xi_,
        E_,
        # big constant arrays (treated constant in bwd)
        m1_,
        m2_,
        dL_,
        spins_,
        lpd_,
        lpi_,
        Ndraw_,
    ):
        # forward uses helper fwd below
        log_mu_, var_u_ = _sel_fwd_only(
            Lambda_, zinj_, dcinj_, logdd_, Xi_, E_,
            m1_, m2_, dL_, spins_, lpd_, lpi_, Ndraw_
        )
        return log_mu_, var_u_

    def _sel_fwd_only(
        Lambda_,
        zinj_,
        dcinj_,
        logdd_,
        Xi_,
        E_,
        m1_,
        m2_,
        dL_,
        spins_,
        lpd_,
        lpi_,
        Ndraw_,
    ):
        n = m1_.shape[0]
        if B == 0:
            n_chunks = 1
            n_pad = n
            B_use = n
        else:
            n_chunks = (n + B - 1) // B
            n_pad = n_chunks * B
            B_use = B

        mask = _make_mask(n, n_pad)

        # pad everything to n_pad using edge padding (safe for math), mask kills padded
        m1p = _pad_to_multiple(m1_, n_pad, mode="edge")
        m2p = _pad_to_multiple(m2_, n_pad, mode="edge")
        dLp = _pad_to_multiple(dL_, n_pad, mode="edge")
        spinsp = _pad_to_multiple(spins_, n_pad, mode="edge")
        lpdp = _pad_to_multiple(lpd_, n_pad, mode="edge")
        lpip = _pad_to_multiple(lpi_, n_pad, mode="edge")

        zinjp = _pad_to_multiple(zinj_, n_pad, mode="edge")
        dcinjp = _pad_to_multiple(dcinj_, n_pad, mode="edge")
        logddp = _pad_to_multiple(logdd_, n_pad, mode="edge")
        Xip = _pad_to_multiple(Xi_, n_pad, mode="edge")
        Ep = _pad_to_multiple(E_, n_pad, mode="edge")

        # streaming accumulators: m, s1, s2
        init = (jnp.array(-jnp.inf, dtype=jnp.float64),
                jnp.array(0.0, dtype=jnp.float64),
                jnp.array(0.0, dtype=jnp.float64))

        def body(carry, k):
            m, s1, s2 = carry
            start = k * B_use

            z0 = jnp.array(0, dtype=start.dtype)   # or jnp.int32(0) if you standardize on int32

            m1c = lax.dynamic_slice(m1p, (start,), (B_use,))
            m2c = lax.dynamic_slice(m2p, (start,), (B_use,))
            dLc = lax.dynamic_slice(dLp, (start,), (B_use,))
            spc = lax.dynamic_slice(spinsp, (start, z0), (B_use, spinsp.shape[1]))
            lpdc = lax.dynamic_slice(lpdp, (start,), (B_use,))
            lpic = lax.dynamic_slice(lpip, (start,), (B_use,))

            zc = lax.dynamic_slice(zinjp, (start,), (B_use,))
            dcc = lax.dynamic_slice(dcinjp, (start,), (B_use,))
            logddc = lax.dynamic_slice(logddp, (start,), (B_use,))
            Xic = lax.dynamic_slice(Xip, (start,), (B_use,))
            Ec = lax.dynamic_slice(Ep, (start,), (B_use,))

            mc = lax.dynamic_slice(mask, (start,), (B_use,))

            x = _score_chunk(Lambda_, zc, dcc, logddc, Xic, Ec, m1c, m2c, dLc, spc, lpdc, lpic, mc)

            m_chunk = jnp.max(x)
            m_new = jnp.maximum(m, m_chunk)

            # rescale old sums if max increased
            scale1 = jnp.exp(m - m_new)
            scale2 = jnp.exp(2.0 * (m - m_new))

            u1 = jnp.exp(x - m_new)
            u2 = jnp.exp(2.0 * (x - m_new))

            s1_new = s1 * scale1 + jnp.sum(u1)
            s2_new = s2 * scale2 + jnp.sum(u2)

            return (m_new, s1_new, s2_new), None

        (m_fin, s1_fin, s2_fin), _ = lax.scan(body, init, jnp.arange(n_chunks, dtype=jnp.int32))

        lse1 = m_fin + jnp.log(s1_fin)
        lse2 = 2.0 * m_fin + jnp.log(s2_fin)

        logN = jnp.log(Ndraw_)
        log_mu = lse1 - logN
        logs2 = lse2 - logN

        # same formula you used (still forward-only for our hard-constraint plan)
        var_u = logdiffexp_bk(bk, logs2 - 2.0 * log_mu, 1.0) - jnp.log(Ndraw_ - 1.0)

        return log_mu, var_u

    def _sel_fwd(
        Lambda_,
        zinj_,
        dcinj_,
        logdd_,
        Xi_,
        E_,
        m1_,
        m2_,
        dL_,
        spins_,
        lpd_,
        lpi_,
        Ndraw_,
    ):
        log_mu, var_u = _sel_fwd_only(
            Lambda_, zinj_, dcinj_, logdd_, Xi_, E_,
            m1_, m2_, dL_, spins_, lpd_, lpi_, Ndraw_
        )

        # Save only what we need for backward:
        # we need lse1 for weights; easiest is to recompute (m,s1) cheaply? we already have log_mu + logN
        # lse1 = log_mu + logN
        lse1 = log_mu + jnp.log(Ndraw_)

        # also save n and chunking info implicitly via shapes/static B
        #res = (lse1, Ndraw_)
        res = (lse1, Ndraw_,
           Lambda_, zinj_, dcinj_, logdd_, Xi_, E_,
           m1_, m2_, dL_, spins_, lpd_, lpi_)
        primals = (log_mu, var_u)
        return primals, res

    def _sel_bwd(res, g):
        #(lse1, Ndraw_) = res
        (lse1, Ndraw_,
         Lambda_, zinj_, dcinj_, logdd_, Xi_, E_,
         m1_, m2_, dL_, spins_, lpd_, lpi_) = res
        
        g_log_mu, g_var_u = g

        # HARD CONSTRAINT PLAN: do not differentiate var_u
        g_var_u = jnp.array(0.0, dtype=jnp.float64)

        #n = m1inj.shape[0]
        n = m1_.shape[0]

        if B == 0:
            n_chunks = 1
            n_pad = n
            B_use = n
        else:
            n_chunks = (n + B - 1) // B
            n_pad = n_chunks * B
            B_use = B

        mask = _make_mask(n, n_pad)

        # pad inputs (same as forward)
        m1p = _pad_to_multiple(m1_, n_pad, mode="edge")
        m2p = _pad_to_multiple(m2_, n_pad, mode="edge")
        dLp = _pad_to_multiple(dL_, n_pad, mode="edge")
        spinsp = _pad_to_multiple(spins_, n_pad, mode="edge")
        lpdp = _pad_to_multiple(lpd_, n_pad, mode="edge")
        lpip = _pad_to_multiple(lpi_, n_pad, mode="edge")

        zinjp = _pad_to_multiple(zinj_, n_pad, mode="edge")
        dcinjp = _pad_to_multiple(dcinj_, n_pad, mode="edge")
        logddp = _pad_to_multiple(logdd_, n_pad, mode="edge")
        Xip = _pad_to_multiple(Xi_, n_pad, mode="edge")
        Ep = _pad_to_multiple(E_, n_pad, mode="edge")

        # accumulators: dLambda plus big grads for cosmology carrier arrays
        dLambda = jnp.zeros_like(Lambda_)
        dz = jnp.zeros((n_pad,), dtype=jnp.float64)
        ddc = jnp.zeros((n_pad,), dtype=jnp.float64)
        dlogdd = jnp.zeros((n_pad,), dtype=jnp.float64)
        dXi = jnp.zeros((n_pad,), dtype=jnp.float64)
        dE = jnp.zeros((n_pad,), dtype=jnp.float64)

        def body(carry, k):
            dLambda, dz, ddc, dlogdd, dXi, dE = carry
            start = k * B_use

            z0 = jnp.array(0, dtype=start.dtype)

            m1c = lax.dynamic_slice(m1p, (start,), (B_use,))
            m2c = lax.dynamic_slice(m2p, (start,), (B_use,))
            dLc = lax.dynamic_slice(dLp, (start,), (B_use,))
            spc = lax.dynamic_slice(spinsp, (start, z0), (B_use, spinsp.shape[1]))
            lpdc = lax.dynamic_slice(lpdp, (start,), (B_use,))
            lpic = lax.dynamic_slice(lpip, (start,), (B_use,))

            zc = lax.dynamic_slice(zinjp, (start,), (B_use,))
            dcc = lax.dynamic_slice(dcinjp, (start,), (B_use,))
            logddc = lax.dynamic_slice(logddp, (start,), (B_use,))
            Xic = lax.dynamic_slice(Xip, (start,), (B_use,))
            Ec = lax.dynamic_slice(Ep, (start,), (B_use,))

            mc = lax.dynamic_slice(mask, (start,), (B_use,))

            def score_wrapped(Lam_, z_, dc_, logdd_, Xi_, E_):
                return _score_chunk(Lam_, z_, dc_, logdd_, Xi_, E_, m1c, m2c, dLc, spc, lpdc, lpic, mc)

            # x = score_wrapped(Lambda_, zc, dcc, logddc, Xic, Ec)
            # w = jnp.exp(x - lse1)  # softmax weights (chunked)

            # cot = g_log_mu * w  # (B,)

            # # VJP wrt the *differentiated* args only
            # (x_out, pull) = jax.vjp(score_wrapped, Lambda, zc, dcc, logddc, Xic, Ec)
            # dLam_c, dz_c, ddc_c, dlogdd_c, dXi_c, dE_c = pull(cot)

            x, pull = jax.vjp(score_wrapped, Lambda, zc, dcc, logddc, Xic, Ec)  # x is the primal!
            w = jnp.exp(x - lse1)
            
            cot = g_log_mu * w
            dLam_c, dz_c, ddc_c, dlogdd_c, dXi_c, dE_c = pull(cot)


            dLambda = dLambda + dLam_c
            # dz = dz.at[start:start + B_use].set(dz_c)
            # ddc = ddc.at[start:start + B_use].set(ddc_c)
            # dlogdd = dlogdd.at[start:start + B_use].set(dlogdd_c)
            # dXi = dXi.at[start:start + B_use].set(dXi_c)
            # dE = dE.at[start:start + B_use].set(dE_c)

            dz     = lax.dynamic_update_slice(dz,     dz_c,     (start,))
            ddc    = lax.dynamic_update_slice(ddc,    ddc_c,    (start,))
            dlogdd = lax.dynamic_update_slice(dlogdd, dlogdd_c, (start,))
            dXi    = lax.dynamic_update_slice(dXi,    dXi_c,    (start,))
            dE     = lax.dynamic_update_slice(dE,     dE_c,     (start,))

            return (dLambda, dz, ddc, dlogdd, dXi, dE), None

        (dLambda, dz, ddc, dlogdd, dXi, dE), _ = lax.scan(
            body,
            (dLambda, dz, ddc, dlogdd, dXi, dE),
            jnp.arange(n_chunks, dtype=jnp.int32)
        )

        # strip padding
        dz = dz[:n]
        ddc = ddc[:n]
        dlogdd = dlogdd[:n]
        dXi = dXi[:n]
        dE = dE[:n]

        # Return grads for all args of _sel_core in order:
        # (Lambda, zinj, dcinj, logdd, Xi, E, m1, m2, dL, spins, lpd, lpi, Ndraw)
        zeros_m1 = jnp.zeros_like(m1_)
        zeros_m2 = jnp.zeros_like(m2_)
        zeros_dL = jnp.zeros_like(dL_)
        zeros_sp = jnp.zeros_like(spins_)
        zeros_lpd = jnp.zeros_like(lpd_)
        zeros_lpi = jnp.zeros_like(lpi_)
        zeros_N = jnp.zeros_like(jnp.asarray(Ndraw).reshape(()))

        return (dLambda, dz, ddc, dlogdd, dXi, dE,
                zeros_m1, zeros_m2, zeros_dL, zeros_sp, zeros_lpd, zeros_lpi, zeros_N)

    _sel_core.defvjp(_sel_fwd, _sel_bwd)

    # call the custom_vjp core
    log_mu, var_u = _sel_core(
        Lambda,
        zinj, dcinj, log_ddL_dz_inj, XiInj, Einj,
        m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl,
        jnp.asarray(Ndraw).reshape(()),
    )
    return log_mu, var_u





# ---------------------------------------------------------------------
#  sel bias wrapper
# ---------------------------------------------------------------------


def sel_bias_with_uncertainty(
    bk,
    m1inj,
    m2inj,
    dLinj,
    spinsInj,
    log_p_draw,
    log_p_incl,
    dL_grid,
    dc_grid,
    log_ddL_dz_grid,
    Lambda,
    Ndraw,
    *,
    rate_model,
    mass_model,
    spin_model,
    smoothing="LVK",
    simplex_repair=False,
    has_m2_break=False,
    norm_gauss="uplow",
    param="vanilla",
    z_grid=None,
    verbose=False,
    subtract_log_p_incl=True,
    eps_interp=1e-12,
    side_interp="right",
    zinj=None,
    dcinj=None,
    log_ddL_dz_inj=None,
    XiInj=None,
    Einj=None,
    linear_mass=False,
    # new flags
    use_streaming_vjp: bool = True,
    sel_chunk_size: int = 10*65536,
):
    # keep your invariant
    if subtract_log_p_incl is not True:
        raise ValueError("subtract_log_p_incl must be True in this configuration.")
    if zinj is None or dcinj is None or log_ddL_dz_inj is None or XiInj is None or Einj is None:
        raise ValueError("Optimized selection expects precomputed zinj/dcinj/log_ddL_dz_inj/XiInj/Einj.")

    if not use_streaming_vjp:
        return sel_bias_with_uncertainty_legacy(
            bk,
            m1inj, m2inj, dLinj, spinsInj,
            log_p_draw, log_p_incl,
            dL_grid, dc_grid, log_ddL_dz_grid,
            Lambda, Ndraw,
            rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
            smoothing=smoothing, simplex_repair=simplex_repair,
            has_m2_break=has_m2_break, norm_gauss=norm_gauss,
            param=param, z_grid=z_grid, verbose=verbose,
            subtract_log_p_incl=True,
            eps_interp=eps_interp, side_interp=side_interp,
            zinj=zinj, dcinj=dcinj, log_ddL_dz_inj=log_ddL_dz_inj,
            XiInj=XiInj, Einj=Einj,
            linear_mass=linear_mass
        )

    return sel_bias_with_uncertainty_streaming_vjp(
        bk,
        m1inj, m2inj, dLinj, spinsInj,
        log_p_draw, log_p_incl,
        dL_grid, dc_grid, log_ddL_dz_grid,
        Lambda, Ndraw,
        rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
        smoothing=smoothing, simplex_repair=simplex_repair,
        has_m2_break=has_m2_break, norm_gauss=norm_gauss,
        param=param, z_grid=z_grid, verbose=verbose,
        eps_interp=eps_interp, side_interp=side_interp,
        zinj=zinj, dcinj=dcinj, log_ddL_dz_inj=log_ddL_dz_inj,
        XiInj=XiInj, Einj=Einj,
        linear_mass=linear_mass,
        chunk_size=sel_chunk_size,
    )






# ---------------------------------------------------------------------
#  custom vjp for z(dL)
# ---------------------------------------------------------------------



def make_dL_to_z_cuvjp(*, bk, eps_interp=1e-12, side_interp="right"):
    """
    Fast custom VJP for z = z(dL) using the *dL-grid bracketing* only.
    Avoids a second z->grid searchsorted/interp in the backward.

    Signature matches your call:
      inv_dL_to_z(dL, dL_grid, zgrid, log_ddL_dz_grid, dL_dtheta_grid, Lambda) -> z

    Gradients:
      - g_dL exact for the chosen linear inversion: dz/ddL = (dz/dk)/(ddL/dk)
      - g_Lambda[:5] via implicit: dz/dtheta = -(dL_dtheta)/(ddL/dz)
      - no grads to grids (return zeros)
    """
    import jax
    import jax.numpy as jnp

    @jax.custom_vjp
    def inv_dL_to_z(dL, dL_grid, zgrid, log_ddL_dz_grid, dL_dtheta_grid, Lambda):
        # Use your standard interpolator in dL space
        return atinterp(bk, dL, dL_grid, zgrid, eps=eps_interp, side=side_interp)

    def fwd(dL, dL_grid, zgrid, log_ddL_dz_grid, dL_dtheta_grid, Lambda):
        # Build dL-space bracketing once
        # i, t satisfy: dL_grid[i] <= dL < dL_grid[i+1] (depending on side)
        i, t = _interp_prepare_bk(bk, dL, dL_grid, eps=eps_interp, side=side_interp)

        # z(dL) by applying same weights to zgrid (1-1 correspondence with dL_grid)
        z = _interp_apply_bk(bk, i, t, zgrid)

        # Build dz/ddL using slopes on the same bracket.
        # dz/ddL = (z[i+1]-z[i]) / (dL[i+1]-dL[i])
        # Need gathers at i and i+1
        i1 = i + 1

        z0  = jnp.take(zgrid,    i,  mode="clip")
        z1  = jnp.take(zgrid,    i1, mode="clip")
        dL0 = jnp.take(dL_grid,  i,  mode="clip")
        dL1 = jnp.take(dL_grid,  i1, mode="clip")

        dz = z1 - z0
        ddL = dL1 - dL0

        # Safe divide (monotonic dL_grid => ddL>0, but keep eps for robustness)
        dz_ddL = dz / (ddL + eps_interp)  # (N,)

        # Interpolate dL_dtheta_grid[:, i] in the SAME bracket using t
        # dL_dtheta_grid shape (5, Nz)
        dLth0 = jnp.take(dL_dtheta_grid, i,  axis=1, mode="clip")  # (5,N)
        dLth1 = jnp.take(dL_dtheta_grid, i1, axis=1, mode="clip")  # (5,N)
        dL_dtheta_at = (1.0 - t)[None, :] * dLth0 + t[None, :] * dLth1

        # dz/dtheta = -(dL/dtheta) / (ddL/dz)  and dz/ddL = 1/(ddL/dz)
        dz_dtheta = -dL_dtheta_at * dz_ddL[None, :]  # (5,N)

        # Save small things + small grids for zeros_like in bwd
        saved = (dz_ddL, dz_dtheta, dL_grid, zgrid, log_ddL_dz_grid, dL_dtheta_grid, Lambda)
        return z, saved

    def bwd(saved, g_z):
        dz_ddL, dz_dtheta, dL_grid, zgrid, log_ddL_dz_grid, dL_dtheta_grid, Lambda = saved

        # g_dL
        g_dL = g_z * dz_ddL

        # Only first 5 Lambda entries affected by inversion path
        g_theta5 = jnp.sum(dz_dtheta * g_z[None, :], axis=1)  # (5,)
        g_Lambda = jnp.zeros_like(Lambda)
        g_Lambda = g_Lambda.at[:5].set(g_theta5)

        # No grads for the grids (treated as constants in this op)
        g_dL_grid = jnp.zeros_like(dL_grid)
        g_zgrid = jnp.zeros_like(zgrid)
        g_logdd = jnp.zeros_like(log_ddL_dz_grid)
        g_dLdth = jnp.zeros_like(dL_dtheta_grid)

        return (g_dL, g_dL_grid, g_zgrid, g_logdd, g_dLdth, g_Lambda)

    inv_dL_to_z.defvjp(fwd, bwd)
    return inv_dL_to_z








def make_dL_to_z_cuvjp_uniform(*, bk, eps_interp=1e-12):
    """
    Custom VJP for z = atinterp_uniform(dL; dL_u, z_u).

    Inputs:
      dL: (N,)
      dL_u: (NdL,)    uniform in dL
      z_u:  (NdL,)
      zgrid: (Nz,)    uniform in z (only used to interpolate dL_dtheta_grid to z)
      dL_dtheta_grid: (5, Nz)
      Lambda: (33,)

    Output:
      z: (N,)

    Gradients:
      - g_dL exact for the chosen linear interpolation on (dL_u, z_u)
      - g_Lambda[:5] via implicit: dz/dtheta = -(dL_dtheta(z)) * dz/ddL
      - no grads to tables/grids
    """
    import jax
    import jax.numpy as jnp

    def _prep_uniform(x, x_u):
        # x_u must be 1D increasing, uniform spacing
        x0 = x_u[0]
        dx = x_u[1] - x_u[0]
        # fractional index
        r = (x - x0) / (dx + eps_interp)
        i = jnp.floor(r).astype(jnp.int32)
        # clip to valid [0, n-2]
        n = x_u.shape[0]
        i = jnp.clip(i, 0, n - 2)
        t = r - i.astype(r.dtype)
        # clip t for safety (can happen due to eps)
        t = jnp.clip(t, 0.0, 1.0)
        return i, t, dx

    def _apply_uniform(i, t, fp):
        fp0 = jnp.take(fp, i, mode="clip")
        fp1 = jnp.take(fp, i + 1, mode="clip")
        return (1.0 - t) * fp0 + t * fp1

    @jax.custom_vjp
    def inv_dL_to_z_uniform(dL, dL_u, z_u, zgrid, dL_dtheta_grid, Lambda):
        # your canonical forward for this branch
        return atinterp_uniform(bk, dL, dL_u, z_u, eps=eps_interp, side="right")

    def fwd(dL, dL_u, z_u, zgrid, dL_dtheta_grid, Lambda):
        # ---- invert on uniform dL table
        i, t, ddL = _prep_uniform(dL, dL_u)
        z = _apply_uniform(i, t, z_u)

        # dz/ddL from local slope in the same bracket
        z0 = jnp.take(z_u, i, mode="clip")
        z1 = jnp.take(z_u, i + 1, mode="clip")
        dz_ddL = (z1 - z0) / (ddL + eps_interp)  # (N,)

        # ---- interpolate dL_dtheta_grid to *z* (zgrid is uniform in this branch)
        # First: build dL_dtheta_u = dL_dtheta_grid evaluated at z_u (NdL ~ 4096 => cheap)
        iz, tz, _dz = _prep_uniform(z_u, zgrid)  # z_u is (NdL,)
        # gather (5, NdL)
        dLth0 = jnp.take(dL_dtheta_grid, iz, axis=1, mode="clip")
        dLth1 = jnp.take(dL_dtheta_grid, iz + 1, axis=1, mode="clip")
        dL_dtheta_u = (1.0 - tz)[None, :] * dLth0 + tz[None, :] * dLth1  # (5,NdL)

        # Then: interpolate dL_dtheta_u along the same dL bracket used for inversion
        dLth_u0 = jnp.take(dL_dtheta_u, i, axis=1, mode="clip")      # (5,N)
        dLth_u1 = jnp.take(dL_dtheta_u, i + 1, axis=1, mode="clip")  # (5,N)
        dL_dtheta_at = (1.0 - t)[None, :] * dLth_u0 + t[None, :] * dLth_u1  # (5,N)

        # Implicit: dz/dtheta = -(dL/dtheta) * dz/ddL
        dz_dtheta = -dL_dtheta_at * dz_ddL[None, :]  # (5,N)

        saved = (dz_ddL, dz_dtheta, dL_u, z_u, zgrid, dL_dtheta_grid, Lambda)
        return z, saved

    def bwd(saved, g_z):
        dz_ddL, dz_dtheta, dL_u, z_u, zgrid, dL_dtheta_grid, Lambda = saved

        g_dL = g_z * dz_ddL  # (N,)

        g_theta5 = jnp.sum(dz_dtheta * g_z[None, :], axis=1)  # (5,)
        g_Lambda = jnp.zeros_like(Lambda)
        g_Lambda = g_Lambda.at[:5].set(g_theta5)

        # No grads to tables/grids
        g_dL_u = jnp.zeros_like(dL_u)
        g_z_u = jnp.zeros_like(z_u)
        g_zgrid = jnp.zeros_like(zgrid)
        g_dL_dtheta_grid = jnp.zeros_like(dL_dtheta_grid)

        return (g_dL, g_dL_u, g_z_u, g_zgrid, g_dL_dtheta_grid, g_Lambda)

    inv_dL_to_z_uniform.defvjp(fwd, bwd)
    return inv_dL_to_z_uniform


