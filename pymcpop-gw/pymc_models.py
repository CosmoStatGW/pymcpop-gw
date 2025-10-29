#
#    Copyright (c) 2025 Michele Mancarella <mancarella@cpt.univ-mrs.fr>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import pytensor_tools as atools
import pytensor_utils as putils
import pytensor.tensor as at
import pytensor
import pymc as pm
import numpy as np
from pytensor.gradient import disconnected_grad as stop_grad
#from pymc.pytensorf import collect_default_updates
from pytensor import config

PLPeakO3params = {'H0': 67.66, 'Om':0.31, 'w0':-1, 'Xi0': 1, 'nXi0':0}


#####################################################
#####################################################


def log_p_pop_at(m1s, m2s, z, dL, spins, Lambda, rate_model, mass_model, spin_model, smoothing='LVK', has_m2_break=False, dc=None):


    ###################################
    # get parameters and compute log p_pop
    ####################################
    
    H0, Om, w0, Xi0, n = Lambda[:5] 

    if dc is None:
        Xi = atools.Xifun_at(z, Xi0, n)
        dc = dL/(1+z)/Xi #atools.dcfun_at(z, H0, Om, w0, interp=False)

    ##################################
    # redshift 
    
    if rate_model=='MD':
        
        gamma, kappa, zp = Lambda[5:8]
        lpz = atools.log_p_z_MD_unnorm(z, gamma, kappa, zp, H0, Om, w0, dc=dc )
        istart = 8
        
    elif rate_model=='PL':
        
        gamma = Lambda[5]
        lpz = atools.log_p_z_PL_unnorm(z, gamma, H0, Om, w0, dc=dc )
        istart = 6

    # ##################################
    # spin
    
    if spin_model=='chieffchip':
        
        muE, sigE, muP, sigP, rho = Lambda[istart:istart+5]
        chieff, chip = spins[0], spins[1]

        lpspin = atools.logpdf_multivariate_trunc_2D(  chieff, chip, muE, muP, sigE, sigP, rho,
                                                     at.as_tensor_variable(-1.), at.as_tensor_variable(1.), 
                                                     at.as_tensor_variable(0.), at.as_tensor_variable(1.) 
                                                    )

    elif spin_model=='chieffchip_uc':
        
        muE, sigE, muP, sigP = Lambda[istart:istart+4]
        chieff, chip = spins[0], spins[1]

        lpchie = atools.truncGausslowerupper_at_lpdf(chieff, muE, sigE, xmin=at.as_tensor_variable(-1), xmax=at.as_tensor_variable(1))
        lpchip = atools.truncGausslowerupper_at_lpdf(chip, muP, sigP, xmin=at.as_tensor_variable(0), xmax=at.as_tensor_variable(1))

        lpspin = lpchie+lpchip

    elif spin_model=='default':

        alphaChi, betaChi, zeta, sigmat = Lambda[istart:istart+4]
        lpspin = atools.logpdf_default_spin(spins, [alphaChi, betaChi, zeta, sigmat])
    
    elif spin_model=='default_gauss':
        muChi, sigmaChi, zeta, sigmat = Lambda[istart:istart+4]
        lpspin = atools.logpdf_default_spin_gauss(spins, [muChi, sigmaChi, zeta, sigmat])
   
    else:
        lpspin = at.zeros( z.shape )

    
    ###################################
    # mass

    ### BBH
    if mass_model=='PLPreg':
        
        lp, al, bb, dm, ml, mh, muM, sM = Lambda[-8:]
        lpmass = atools.logpdf_PLP_reg([m1s, m2s], [lp, al, bb, dm, ml, mh, muM, sM], smoothing=smoothing)

    elif mass_model=='DPLDP':
        
        lambdaBBHmass = Lambda[-20:]
        lpmass = atools.logpdf_DPLDP([m1s, m2s], lambdaBBHmass, force_m2_less_than_m1=False, has_m2_break=has_m2_break, smoothing=smoothing )
        
        
    ### BNS
    elif mass_model=='BNSgauss':
        muM, sM = Lambda[-2:]
        lpmass = atools.logpdf_gauss([m1s, m2s], [muM, sM] )
        
    elif mass_model=='BNSgaussCond':
        muM, sM = Lambda[-2:]
        lpmass = atools.logpdf_gauss_cond([m1s, m2s], [muM, sM] )

    ### Non - parametric
    elif mass_model=='DPUC':

        w, mu, sd, logw  = Lambda[-5:-1]
        Nmax=Lambda[-1]

        
        # Broadcast to (n_comp, n_obs)
        diff1 = m1s[None, :] - mu[0][:, None]
        diff2 = m2s[None, :] - mu[1][:, None]
        
        sd1 = sd[0][:, None]
        sd2 = sd[1][:, None]
        
        # Per-dimension log-Normal pdfs, broadcasted
        logp1 = -0.5 * (diff1**2 / (sd1**2)) - 0.5 * at.log(2 * atools.PI) - at.log(sd1)
        logp2 = -0.5 * (diff2**2 / (sd2**2)) - 0.5 * at.log(2 * atools.PI) - at.log(sd2)
        
        # Sum the two independent dimensions → (n_comp, n_obs)
        logp_components = logp1 + logp2
        
        # Mixture over components → (n_obs,)
        lpmass = at.logsumexp(logp_components + logw[:, None], axis=0)
    
    elif mass_model=='DP':

        alpha, beta, w, mu, fishers, ldets_inv, logw  = Lambda[-8:-1]
        Nmax=Lambda[-1]

        # 1) Pack observations into (N, 2)
        X = at.stack([m1s, m2s], axis=1)          # (N, 2)
        
        # 2) Differences to component means -> (K, N, 2)
        mu_k2 = mu.T                               # (K, 2)
        diff  = X[None, :, :] - mu_k2[:, None, :]  # (K, N, 2)
        
        # 3) Quadratic form (x-μ)^T Σ^{-1} (x-μ) for all (k, n)
        #    Using batched matmul; result tmp is (K, N, 2), then rowwise dot with diff
        tmp  = at.matmul(diff, fishers)            # (K, N, 2)
        quad = at.sum(diff * tmp, axis=2)          # (K, N)
        
        # 4) Component log-densities (MvN with precision)
        nd = 2
        logp_components = (
            -0.5 * quad
            - 0.5 * nd * at.log(2.0 * np.pi)
            + 0.5 * ldets_inv[:, None]
            + logw[:, None]
        )                                           # (K, N)
        
        # 5) Mixture over components -> per-observation log-lik
        lpmass = at.logsumexp(logp_components, axis=0)  # (N,)

    ###################################
    # jacobian
    
    log_ddL_dz = atools.log_ddL_dz(z, H0, Om, w0, Xi0, n, dc=dc)

    
    ###################################
    # return log pdf
    ####################################
    
    lp =  lpz - log_ddL_dz - 2*at.log1p(z) + lpmass + lpspin

    return lp


#####################################################

def sel_bias_with_uncertainty_at_loop(
    m1inj, m2inj, dLinj, spinsInj, log_p_draw, Lambda,
    Ndraw,
    rate_model, mass_model, spin_model,
    smoothing, has_m2_break,
    interp,
    dL_grid=None, z_grid=None,
    *,
    chunk_size=10_000,
    use_float32=False,
    N_inj_py=None,   # REQUIRED: Python int, e.g. int(m1inj_np.shape[0])
    scan_updates = False
):
    """
    Chunked, scan-free version of sel_bias_with_uncertainty_at_0.
    Returns (log_mu, Neff, var_log_lik_u) as float64.

    When use_float32=True, arrays stay fp32 but reductions/log-sums run in fp64.
    """
    if N_inj_py is None:
        raise ValueError("Pass N_inj_py=<python int>, e.g. int(m1inj_np.shape[0]).")

    out_dtype  = pytensor.config.floatX  # was "float64" #"float64"
    work_dtype = "float32" if use_float32 else str(getattr(m1inj, "dtype", out_dtype))
    


    def cast_work(x):
        return x if x is None else at.cast(x, work_dtype)

    m1inj      = cast_work(m1inj)
    m2inj      = cast_work(m2inj)
    dLinj      = cast_work(dLinj)
    log_p_draw = cast_work(log_p_draw)
    Lambda     = at.cast(Lambda, work_dtype)
    Ndraw_w    = at.cast(Ndraw, work_dtype)

    if (spin_model == 'default') or (spin_model == 'default_gauss'):
        spinsInj_sel = [cast_work(spinsInj[0]), cast_work(spinsInj[1]),
                        cast_work(spinsInj[2]), cast_work(spinsInj[3])]
    elif spin_model == 'none':
        spinsInj_sel = []
    else:
        spinsInj_sel = [] if (spinsInj is None) else [cast_work(s) for s in spinsInj]

    H0, Om, w0, Xi0, n = Lambda[:5]
    CHUNK = int(chunk_size)
    N_py  = int(N_inj_py)

    # Accumulators: float64 when use_float32, else work_dtype
    acc_dtype = out_dtype if use_float32 else work_dtype
    log_sum  = at.constant(-np.inf, dtype=acc_dtype)
    log_sum2 = at.constant(-np.inf, dtype=acc_dtype)

    dL_grid_w = cast_work(dL_grid)
    z_grid_w  = cast_work(z_grid)

    # stable logsumexp in float64
    def _logsumexp64(x64):
        m = at.max(x64)
        return m + at.log(at.sum(at.exp(x64 - m)))

    for start in range(0, N_py, CHUNK):
        stop = min(start + CHUNK, N_py)

        m1c   = m1inj[start:stop]
        m2c   = m2inj[start:stop]
        dLc   = dLinj[start:stop]
        lpd_c = log_p_draw[start:stop]
        spins_c = [s[start:stop] for s in spinsInj_sel] if len(spinsInj_sel) else []

        if dL_grid is None:
            zinj_c = atools.z_from_dL_at(dLc, H0, Om, w0, Xi0, n, interp=interp)
        else:
            if z_grid is None:
                raise ValueError('Pass z_grid if passing pre-computed dL_grid')
            zinj_c = atools.atinterp(dLc, dL_grid_w, z_grid_w)
        zinj_c = at.cast(zinj_c, work_dtype)

        m1Src = at.cast(m1c / (1 + zinj_c), work_dtype)
        m2Src = at.cast(m2c / (1 + zinj_c), work_dtype)

        if mass_model in ('DP', 'DPUC'):
            Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
            mass_1_use = at.cast(at.log(Mc_src_inj), work_dtype)
            mass_2_use = at.cast(atools.logitat(q_inj), work_dtype)
        else:
            mass_1_use = m1Src
            mass_2_use = m2Src

        log_p_pop_c = log_p_pop_at(
            mass_1_use, mass_2_use, zinj_c, dLc, spins_c, Lambda,
            rate_model, mass_model, spin_model,
            smoothing=smoothing, has_m2_break=has_m2_break
        )
        log_p_pop_c = at.cast(log_p_pop_c, work_dtype)

        if mass_model in ('DP', 'DPUC'):
            log_p_pop_c = log_p_pop_c + at.cast(
                -at.log(m2Src) - at.log(m1Src - m2Src) - at.log1p(zinj_c),
                work_dtype
            )

        if use_float32:
            # <-- KEY: do log-weights and reductions in float64 -->
            log_sel_b_c64 = at.cast(log_p_pop_c, out_dtype) - at.cast(lpd_c, out_dtype)
            log_sum  = at.logaddexp(log_sum,  _logsumexp64(     log_sel_b_c64))
            log_sum2 = at.logaddexp(log_sum2, _logsumexp64(2.0 * log_sel_b_c64))
        else:
            log_sel_b_c = at.cast(log_p_pop_c - lpd_c, work_dtype)
            log_sum  = at.cast(at.logaddexp(log_sum,  at.logsumexp(     log_sel_b_c)), work_dtype)
            log_sum2 = at.cast(at.logaddexp(log_sum2, at.logsumexp(2.0 * log_sel_b_c)), work_dtype)

    if use_float32:
        # Finish entirely in float64 then cast outputs
        Ndraw64 = at.cast(Ndraw, out_dtype)
        log_mu64 = log_sum  - at.log(Ndraw64)
        logs264  = log_sum2 - at.log(Ndraw64)
        logNeff64 = 2.0 * log_mu64 - logs264 + at.log(Ndraw64)
        delta64 = logs264 - 2.0 * log_mu64
        eps64   = at.as_tensor_variable(1e-6, dtype=out_dtype)
        delta64 = at.maximum(delta64, -eps64)
        var64   = atools.logdiffexp(delta64, 1.0) - at.log(Ndraw64 - 1.0)
        Neff64  = at.exp(logNeff64)

        return at.cast(log_mu64, out_dtype), at.cast(Neff64, out_dtype), at.cast(var64, out_dtype)
    else:
        log_mu = at.cast(log_sum  - at.log(Ndraw_w), work_dtype)
        logs2  = at.cast(log_sum2 - at.log(Ndraw_w), work_dtype)
        logNeff = at.cast(2.0 * log_mu - logs2 + at.log(Ndraw_w), work_dtype)
        var_log_lik_u = at.cast(atools.logdiffexp(logs2 - 2.0 * log_mu, 1.0) - at.log(Ndraw_w - 1.0), work_dtype)
        Neff = at.cast(at.exp(logNeff), work_dtype)
        return at.cast(log_mu, out_dtype), at.cast(Neff, out_dtype), at.cast(var_log_lik_u, out_dtype)


def _two_pass_logsumexp_stream(log_sel_b):
    """
    JAX-friendly streaming reductions with no dynamic slicing.
    Returns: logsumexp(x), logsumexp(2x), N
    """
    # 1) global max via a regular reduction (maps to jnp.max)
    m = at.max(log_sel_b)

    # 2) streaming sums via scan over the sequence values (no slicing)
    def sum_step(x_t, s1, s2, m):
        return (s1 + at.exp(x_t - m), s2 + at.exp(2.0 * x_t - 2.0 * m))

    init_s1 = at.as_tensor_variable(0.0, dtype=log_sel_b.dtype)
    init_s2 = at.as_tensor_variable(0.0, dtype=log_sel_b.dtype)

    (sum1_seq, sum2_seq), _ = pytensor.scan(
        fn=sum_step,
        sequences=[log_sel_b],        # <-- gives one scalar per step; no indexing
        outputs_info=[init_s1, init_s2],
        non_sequences=[m],
        strict=True,
        profile=True
    )

    S1 = sum1_seq[-1]
    S2 = sum2_seq[-1]

    logsumexp1 = m + at.log(S1)
    logsumexp2 = 2.0 * m + at.log(S2)
    N = log_sel_b.shape[0]
    return logsumexp1, logsumexp2, N


def sel_bias_with_uncertainty_at_scan(
    m1inj, m2inj, dLinj, spinsInj, log_p_draw, Lambda, Ndraw,
    rate_model, mass_model, spin_model, smoothing, has_m2_break, interp,
    dL_grid=None, z_grid=None, use_float32=False, **kwargs
):
    H0, Om, w0, Xi0, n  = Lambda[:5]

    if (spin_model == 'default') or (spin_model == 'default_gauss'):
        # Keep this tensor-y; avoid Python lists
        spinsInj_sel = [spinsInj[0], spinsInj[1], spinsInj[2], spinsInj[3]]
    elif spin_model == 'none':
        # empty tensor (won't be used downstream)
        spinsInj_sel = []

    if dL_grid is None:
        zinj = atools.z_from_dL_at(dLinj, H0, Om, w0, Xi0, n, interp=interp)
    else:
        if z_grid is None:
            raise ValueError('Pass z grid if passing pre-computed dL grid')
        zinj = atools.atinterp(dLinj, dL_grid, z_grid)

    m1Src = m1inj / (1 + zinj)
    m2Src = m2inj / (1 + zinj)

    if mass_model in ('DP', 'DPUC'):
        Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
        mass_1_use = at.log(Mc_src_inj)
        mass_2_use = atools.logitat(q_inj)
    else:
        mass_1_use = m1Src
        mass_2_use = m2Src

    log_p_pop = log_p_pop_at(
        mass_1_use, mass_2_use, zinj, dLinj, spinsInj_sel, Lambda,
        rate_model, mass_model, spin_model,
        smoothing=smoothing, has_m2_break=has_m2_break
    )

    if mass_model in ('DP', 'DPUC'):
        # remove jacobian m1, m2 --> log(Mc), logit(q)
        log_p_pop += (-at.log(m2Src) - at.log(m1Src - m2Src) - at.log1p(zinj))

    log_sel_b = log_p_pop - log_p_draw

    # JAX-safe streaming reductions (no dynamic start/stop slices)
    logsumexp1, logsumexp2, N = _two_pass_logsumexp_stream(log_sel_b)

    log_mu = logsumexp1 - at.log(Ndraw)
    logs2  = logsumexp2 - at.log(Ndraw)

    # Talbot & Golomb 2023
    logNeff = 2 * log_mu - logs2 + at.log(Ndraw)
    Neff = at.exp(logNeff)

    # variance of log-l per unit obs (Talbot & Golomb 2023)
    var_log_lik_u = atools.logdiffexp(logs2 - 2 * log_mu, 1.) - at.log(Ndraw - 1)

    return log_mu, Neff, var_log_lik_u


def _pad_to_multiple(x, B, pad_value):
    """
    Pad a 1-D tensor x to a multiple of the Python int B, returning:
      xB:      shape (n_batches, B)
      n_batches: int64 TensorVariable
      N:       original length (int64 TensorVariable)
      N_pad:   pad length (int64 TensorVariable)
    No Python slicing with symbolic indices.
    """
    # lengths as symbolic ints
    N = x.shape[0]
    # ceil div without floats: (N + B - 1) // B
    n_batches = (N + B - 1) // B
    N_pad = n_batches * B - N

    # make a pad vector of symbolic length N_pad
    # NOTE: shape must be a tuple; entries may be symbolic
    pad = at.full((N_pad,), at.as_tensor_variable(pad_value, dtype=x.dtype), dtype=x.dtype)

    x_pad = at.concatenate([x, pad], axis=0)

    # reshape to (n_batches, B); B is a Python int (static), first dim is symbolic
    xB = x_pad.reshape((n_batches, B))
    return xB, n_batches, N, N_pad



def _combine_logsumexp(m_s, s_s, m_c, s_c):
    # combine two log-sum-exp states: (m_s, s_s) with (m_c, s_c)
    m_new = at.maximum(m_s, m_c)
    s_new = s_s * at.exp(m_s - m_new) + s_c * at.exp(m_c - m_new)
    return m_new, s_new


# def sel_bias_with_uncertainty_at_0_batched(
#     m1inj, m2inj, dLinj, spinsInj, log_p_draw, Lambda, Ndraw,
#     rate_model, mass_model, spin_model, smoothing, has_m2_break, interp,
#     chunk_size=4096, dL_grid=None, z_grid=None, scan_updates=False, **kwargs
# ):
#     def _pad_to_multiple(x, k, pad_value):
#         N = x.shape[0]
#         n_chunks = (N + k - 1) // k
#         N_pad = n_chunks * k - N
#         pad = at.full((N_pad,), at.as_tensor_variable(pad_value, dtype=x.dtype), dtype=x.dtype)
#         x_pad = at.concatenate([x, pad], axis=0)
#         xK = x_pad.reshape((n_chunks, k))
#         return xK, n_chunks, N, N_pad

#     def _combine_logsumexp(m_s, s_s, m_c, s_c):
#         m_new = at.maximum(m_s, m_c)
#         s_new = s_s * at.exp(m_s - m_new) + s_c * at.exp(m_c - m_new)
#         return m_new, s_new

#     Lambda_t = at.as_tensor_variable(Lambda)
#     use_dp = mass_model in ("DP", "DPUC")
#     spin_is_default = spin_model in ("default", "default_gauss")

#     has_grid = dL_grid is not None
#     if has_grid:
#         dL_grid_t = at.as_tensor_variable(dL_grid)
#         z_grid_t  = at.as_tensor_variable(z_grid)

#     # Pad observed vectors (safe pads)
#     m1K, n_chunks, N, _ = _pad_to_multiple(m1inj,   chunk_size, 2.0)  # m1 > m2
#     m2K, _,        _, _ = _pad_to_multiple(m2inj,   chunk_size, 1.0)
#     dLK,  _,        _, _ = _pad_to_multiple(dLinj,   chunk_size, 1.0)
#     lpdK, _,        _, _ = _pad_to_multiple(log_p_draw, chunk_size, 0.0)

#     # If spins used, pad each component separately
#     if spin_is_default:
#         s1K,  _, _, _ = _pad_to_multiple(spinsInj[0], chunk_size, 0.0)
#         s2K,  _, _, _ = _pad_to_multiple(spinsInj[1], chunk_size, 0.0)
#         ct1K, _, _, _ = _pad_to_multiple(spinsInj[2], chunk_size, 1.0)
#         ct2K, _, _, _ = _pad_to_multiple(spinsInj[3], chunk_size, 1.0)

#     valid_mask = (at.arange(n_chunks * chunk_size) < N).reshape((n_chunks, chunk_size))
#     #NEG_BIG = at.constant(-1e30, dtype=m1inj.dtype)


#     # ---- scan body ----
#     if spin_is_default:
#         def batch_step(i, m_state, m2_state, s1_state, s2_state,
#                        m1K, m2K, dLK, lpdK, valid_mask, Lambda_t,
#                        s1K, s2K, ct1K, ct2K, *maybe_grids):

#             m1 = m1K[i]; m2 = m2K[i]; dL = dLK[i]; lpd = lpdK[i]; mask = valid_mask[i]
#             s1 = s1K[i]; s2 = s2K[i]; ct1 = ct1K[i]; ct2 = ct2K[i]
#             spins_use = [s1, s2, ct1, ct2]

#             if len(maybe_grids) == 2:
#                 dL_grid_t, z_grid_t = maybe_grids
#                 zinj = atools.atinterp(dL, dL_grid_t, z_grid_t)
#             else:
#                 H0, Om, w0, Xi0, n = Lambda_t[:5]
#                 zinj = atools.z_from_dL_at(dL, H0, Om, w0, Xi0, n, interp=interp)

#             one_plus_z = 1 + zinj
#             m1Src = m1 / one_plus_z
#             m2Src = m2 / one_plus_z

#             if use_dp:
#                 Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
#                 mass_1_use = at.log(Mc_src_inj)
#                 mass_2_use = atools.logitat(q_inj)
#             else:
#                 mass_1_use = m1Src
#                 mass_2_use = m2Src

#             log_p_pop = log_p_pop_at(
#                 mass_1_use, mass_2_use, zinj, dL, spins_use, Lambda_t,
#                 rate_model, mass_model, spin_model,
#                 smoothing=smoothing, has_m2_break=has_m2_break
#             )

#             if use_dp:
#                 eps = at.constant(1e-30, dtype=m1inj.dtype)
#                 log_p_pop += (-at.log(at.maximum(m2Src, eps))
#                               - at.log(at.maximum(m1Src - m2Src, eps))
#                               - at.log1p(zinj))

#             x = log_p_pop - lpd
#             x = at.where(mask, x, -np.inf)

#             m_chunk = at.max(x)
#             s1_chunk = at.exp(x - m_chunk).sum()
#             s2_chunk = at.exp(2.0 * x - 2.0 * m_chunk).sum()

#             m1_new, s1_new = _combine_logsumexp(m_state,  s1_state,  m_chunk,     s1_chunk)
#             m2_new, s2_new = _combine_logsumexp(m2_state, s2_state,  2.0*m_chunk, s2_chunk)
#             return m1_new, m2_new, s1_new, s2_new
#     else:
#         def batch_step(i, m_state, m2_state, s1_state, s2_state,
#                        m1K, m2K, dLK, lpdK, valid_mask, Lambda_t, *maybe_grids):

#             m1 = m1K[i]; m2 = m2K[i]; dL = dLK[i]; lpd = lpdK[i]; mask = valid_mask[i]
#             spins_use = []

#             if len(maybe_grids) == 2:
#                 dL_grid_t, z_grid_t = maybe_grids
#                 zinj = atools.atinterp(dL, dL_grid_t, z_grid_t)
#             else:
#                 H0, Om, w0, Xi0, n = Lambda_t[:5]
#                 zinj = atools.z_from_dL_at(dL, H0, Om, w0, Xi0, n, interp=interp)

#             one_plus_z = 1 + zinj
#             m1Src = m1 / one_plus_z
#             m2Src = m2 / one_plus_z

#             if use_dp:
#                 Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
#                 mass_1_use = at.log(Mc_src_inj)
#                 mass_2_use = atools.logitat(q_inj)
#             else:
#                 mass_1_use = m1Src
#                 mass_2_use = m2Src

#             log_p_pop = log_p_pop_at(
#                 mass_1_use, mass_2_use, zinj, dL, spins_use, Lambda_t,
#                 rate_model, mass_model, spin_model,
#                 smoothing=smoothing, has_m2_break=has_m2_break
#             )

#             if use_dp:
#                 eps = at.constant(1e-30, dtype=m1inj.dtype)
#                 log_p_pop += (-at.log(at.maximum(m2Src, eps))
#                               - at.log(at.maximum(m1Src - m2Src, eps))
#                               - at.log1p(zinj))

#             x = log_p_pop - lpd
#             x = at.where(mask, x, -np.inf)

#             m_chunk = at.max(x)
#             s1_chunk = at.exp(x - m_chunk).sum()
#             s2_chunk = at.exp(2.0 * x - 2.0 * m_chunk).sum()

#             m1_new, s1_new = _combine_logsumexp(m_state,  s1_state,  m_chunk,     s1_chunk)
#             m2_new, s2_new = _combine_logsumexp(m2_state, s2_state,  2.0*m_chunk, s2_chunk)
#             return m1_new, m2_new, s1_new, s2_new

#     # scan setup
#     idxs = at.arange(n_chunks, dtype="int64")
#     m_init  = at.as_tensor_variable(-at.inf, dtype=m1inj.dtype)
#     m2_init = at.as_tensor_variable(-at.inf, dtype=m1inj.dtype)
#     s1_init = at.as_tensor_variable(0.0, dtype=m1inj.dtype)
#     s2_init = at.as_tensor_variable(0.0, dtype=m1inj.dtype)

#     # Build non_sequences (tensors only). Pass spins as separate tensors if used.
#     nonseq = [m1K, m2K, dLK, lpdK, valid_mask, Lambda_t]
#     if spin_is_default:
#         nonseq += [s1K, s2K, ct1K, ct2K]
#     if has_grid:
#         nonseq += [dL_grid_t, z_grid_t]

#     (m_final, m2_final, s1_final, s2_final), _ = pytensor.scan(
#         fn=batch_step,
#         sequences=[idxs],
#         outputs_info=[m_init, m2_init, s1_init, s2_init],
#         non_sequences=nonseq,
#         strict=True,
#         profile=True
#     )

#     logsumexp1 = m_final[-1] + at.log(s1_final[-1])
#     logsumexp2 = m2_final[-1] + at.log(s2_final[-1])

#     log_mu = logsumexp1 - at.log(Ndraw)
#     logs2  = logsumexp2 - at.log(Ndraw)
#     logNeff = 2.0 * log_mu - logs2 + at.log(Ndraw)
#     Neff = at.exp(logNeff)
#     var_log_lik_u = atools.logdiffexp(logs2 - 2.0 * log_mu, 1.0) - at.log(Ndraw - 1)
#     return log_mu, Neff, var_log_lik_u

def sel_bias_with_uncertainty_at_0_batched(
    m1inj, m2inj, dLinj, spinsInj, log_p_draw, Lambda, Ndraw,
    rate_model, mass_model, spin_model, smoothing, has_m2_break, interp,
    chunk_size=4096, dL_grid=None, z_grid=None, scan_updates=False, **kwargs
):
    """
    Mixed-precision version:
      - If use_float32=True (kwarg), the *per-injection* compute inside each batch runs in fp32,
        but the reductions/accumulators stay in fp64 for numerical stability.
      - If use_float32=False, the fp64 path is IDENTICAL to your original (unchanged math).

    Returns: (log_mu, Neff, var_log_lik_u)
    """
    use_float32 = bool(kwargs.get("use_float32", False))
    reduce_dtype = pytensor.config.floatX  # was "float64"
    

    # ---------------- helpers (unchanged for fp64 path) ----------------
    def _pad_to_multiple(x, k, pad_value):
        N = x.shape[0]
        n_chunks = (N + k - 1) // k
        N_pad = n_chunks * k - N
        pad = at.full((N_pad,), at.as_tensor_variable(pad_value, dtype=x.dtype), dtype=x.dtype)
        x_pad = at.concatenate([x, pad], axis=0)
        xK = x_pad.reshape((n_chunks, k))
        return xK, n_chunks, N, N_pad

    def _combine_logsumexp(m_s, s_s, m_c, s_c):
        m_new = at.maximum(m_s, m_c)
        s_new = s_s * at.exp(m_s - m_new) + s_c * at.exp(m_c - m_new)
        return m_new, s_new

    Lambda_t = at.as_tensor_variable(Lambda)
    use_dp = mass_model in ("DP", "DPUC")
    spin_is_default = spin_model in ("default", "default_gauss")

    has_grid = dL_grid is not None
    if has_grid:
        dL_grid_t = at.as_tensor_variable(dL_grid)
        z_grid_t  = at.as_tensor_variable(z_grid)

    # ---------------- pad to fixed batch size ----------------
    m1K, n_chunks, N, _ = _pad_to_multiple(m1inj,   chunk_size, 2.0)  # m1 > m2
    m2K, _,        _, _ = _pad_to_multiple(m2inj,   chunk_size, 1.0)
    dLK,  _,        _, _ = _pad_to_multiple(dLinj,  chunk_size, 1.0)
    lpdK, _,        _, _ = _pad_to_multiple(log_p_draw, chunk_size, 0.0)

    if spin_is_default:
        s1K,  _, _, _ = _pad_to_multiple(spinsInj[0], chunk_size, 0.0)
        s2K,  _, _, _ = _pad_to_multiple(spinsInj[1], chunk_size, 0.0)
        ct1K, _, _, _ = _pad_to_multiple(spinsInj[2], chunk_size, 1.0)
        ct2K, _, _, _ = _pad_to_multiple(spinsInj[3], chunk_size, 1.0)

    valid_mask = (at.arange(n_chunks * chunk_size) < N).reshape((n_chunks, chunk_size))

    # dtype-constant helpers for the fp32 branch
    neg_inf32 = at.constant(-np.inf, dtype="float32")
    zero32    = at.constant(0.0,     dtype="float32")

    # ---------------- scan body ----------------
    if spin_is_default:
        def batch_step(i, m_state, m2_state, s1_state, s2_state,
                       m1K, m2K, dLK, lpdK, valid_mask, Lambda_t,
                       s1K, s2K, ct1K, ct2K, *maybe_grids):

            m1 = m1K[i]; m2 = m2K[i]; dL = dLK[i]; lpd = lpdK[i]; mask = valid_mask[i]
            s1 = s1K[i]; s2 = s2K[i]; ct1 = ct1K[i]; ct2 = ct2K[i]
            spins_use = [s1, s2, ct1, ct2]

            if len(maybe_grids) == 2:
                dL_grid_t, z_grid_t = maybe_grids
                zinj = atools.atinterp(dL, dL_grid_t, z_grid_t)
            else:
                H0, Om, w0, Xi0, n = Lambda_t[:5]
                zinj = atools.z_from_dL_at(dL, H0, Om, w0, Xi0, n, interp=interp)

            one_plus_z = 1 + zinj
            m1Src = m1 / one_plus_z
            m2Src = m2 / one_plus_z

            if use_dp:
                Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
                mass_1_use = at.log(Mc_src_inj)
                mass_2_use = atools.logitat(q_inj)
            else:
                mass_1_use = m1Src
                mass_2_use = m2Src

            log_p_pop = log_p_pop_at(
                mass_1_use, mass_2_use, zinj, dL, spins_use, Lambda_t,
                rate_model, mass_model, spin_model,
                smoothing=smoothing, has_m2_break=has_m2_break
            )

            if use_dp:
                eps = at.constant(1e-30, dtype=m1inj.dtype)
                log_p_pop += (-at.log(at.maximum(m2Src, eps))
                              - at.log(at.maximum(m1Src - m2Src, eps))
                              - at.log1p(zinj))

            # ---------- mixed-precision boundary ----------
            if use_float32:
                log_p_pop32 = at.cast(log_p_pop, "float32")
                lpd32       = at.cast(lpd,       "float32")
                x32 = log_p_pop32 - lpd32
                x32 = at.where(mask, x32, neg_inf32)

                # per-batch reductions in fp32
                has_valid = at.any(mask)
                m_chunk32  = at.switch(has_valid, at.max(x32), neg_inf32)
                s1_chunk32 = at.switch(has_valid, at.exp(x32 - m_chunk32).sum(), zero32)
                s2_chunk32 = at.switch(has_valid, at.exp(2.0 * (x32 - m_chunk32)).sum(), zero32)

                # promote only the reduction scalars to fp64
                m_chunk64  = at.cast(m_chunk32,  reduce_dtype)
                s1_chunk64 = at.cast(s1_chunk32, reduce_dtype)
                s2_chunk64 = at.cast(s2_chunk32, reduce_dtype)

                m1_new, s1_new = _combine_logsumexp(m_state,  s1_state,  m_chunk64,     s1_chunk64)
                m2_new, s2_new = _combine_logsumexp(m2_state, s2_state,  2.0*m_chunk64, s2_chunk64)
            else:
                # --------------- ORIGINAL fp64 path (exactly as before) ---------------
                x = log_p_pop - lpd
                x = at.where(mask, x, -np.inf)

                m_chunk = at.max(x)
                s1_chunk = at.exp(x - m_chunk).sum()
                s2_chunk = at.exp(2.0 * x - 2.0 * m_chunk).sum()

                m1_new, s1_new = _combine_logsumexp(m_state,  s1_state,  m_chunk,     s1_chunk)
                m2_new, s2_new = _combine_logsumexp(m2_state, s2_state,  2.0*m_chunk, s2_chunk)

            return m1_new, m2_new, s1_new, s2_new
    else:
        def batch_step(i, m_state, m2_state, s1_state, s2_state,
                       m1K, m2K, dLK, lpdK, valid_mask, Lambda_t, *maybe_grids):

            m1 = m1K[i]; m2 = m2K[i]; dL = dLK[i]; lpd = lpdK[i]; mask = valid_mask[i]
            spins_use = []

            if len(maybe_grids) == 2:
                dL_grid_t, z_grid_t = maybe_grids
                zinj = atools.atinterp(dL, dL_grid_t, z_grid_t)
            else:
                H0, Om, w0, Xi0, n = Lambda_t[:5]
                zinj = atools.z_from_dL_at(dL, H0, Om, w0, Xi0, n, interp=interp)

            one_plus_z = 1 + zinj
            m1Src = m1 / one_plus_z
            m2Src = m2 / one_plus_z

            if use_dp:
                Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
                mass_1_use = at.log(Mc_src_inj)
                mass_2_use = atools.logitat(q_inj)
            else:
                mass_1_use = m1Src
                mass_2_use = m2Src

            log_p_pop = log_p_pop_at(
                mass_1_use, mass_2_use, zinj, dL, spins_use, Lambda_t,
                rate_model, mass_model, spin_model,
                smoothing=smoothing, has_m2_break=has_m2_break
            )

            if use_dp:
                eps = at.constant(1e-30, dtype=m1inj.dtype)
                log_p_pop += (-at.log(at.maximum(m2Src, eps))
                              - at.log(at.maximum(m1Src - m2Src, eps))
                              - at.log1p(zinj))

            
            if use_float32:
                
                log_p_pop32 = at.cast(log_p_pop, "float32")
                lpd32       = at.cast(lpd,       "float32")
                x32 = log_p_pop32 - lpd32
                x32 = at.where(mask, x32, neg_inf32)

                has_valid = at.any(mask)
                m_chunk32  = at.switch(has_valid, at.max(x32), neg_inf32)
                s1_chunk32 = at.switch(has_valid, at.exp(x32 - m_chunk32).sum(), zero32)
                s2_chunk32 = at.switch(has_valid, at.exp(2.0 * (x32 - m_chunk32)).sum(), zero32)

                m_chunk64  = at.cast(m_chunk32,  reduce_dtype)
                s1_chunk64 = at.cast(s1_chunk32, reduce_dtype)
                s2_chunk64 = at.cast(s2_chunk32, reduce_dtype)

                m1_new, s1_new = _combine_logsumexp(m_state,  s1_state,  m_chunk64,     s1_chunk64)
                m2_new, s2_new = _combine_logsumexp(m2_state, s2_state,  2.0*m_chunk64, s2_chunk64)
            else:
                # --------------- ORIGINAL fp64 path (exactly as before) ---------------
                x = log_p_pop - lpd
                x = at.where(mask, x, -np.inf)

                m_chunk = at.max(x)
                s1_chunk = at.exp(x - m_chunk).sum()
                s2_chunk = at.exp(2.0 * x - 2.0 * m_chunk).sum()

                m1_new, s1_new = _combine_logsumexp(m_state,  s1_state,  m_chunk,     s1_chunk)
                m2_new, s2_new = _combine_logsumexp(m2_state, s2_state,  2.0*m_chunk, s2_chunk)

            return m1_new, m2_new, s1_new, s2_new

    # ---------------- scan setup ----------------
    idxs = at.arange(n_chunks)
    m_init  = at.as_tensor_variable(-at.inf, dtype=m1inj.dtype)
    m2_init = at.as_tensor_variable(-at.inf, dtype=m1inj.dtype)
    s1_init = at.as_tensor_variable(0.0, dtype=m1inj.dtype)
    s2_init = at.as_tensor_variable(0.0, dtype=m1inj.dtype)

    nonseq = [m1K, m2K, dLK, lpdK, valid_mask, Lambda_t]
    if spin_is_default:
        nonseq += [s1K, s2K, ct1K, ct2K]
    if has_grid:
        nonseq += [dL_grid_t, z_grid_t]

    (m_final, m2_final, s1_final, s2_final), _ = pytensor.scan(
        fn=batch_step,
        sequences=[idxs],
        outputs_info=[m_init, m2_init, s1_init, s2_init],
        non_sequences=nonseq,
        strict=True,
        profile=False  # was True; set False for normal runs
    )

    # ---------------- final reductions (fp64, unchanged) ----------------
    logsumexp1 = m_final[-1] + at.log(s1_final[-1])
    logsumexp2 = m2_final[-1] + at.log(s2_final[-1])

    Ndraw64 = at.as_tensor_variable(Ndraw).astype(reduce_dtype)

    log_mu  = logsumexp1 - at.log(Ndraw64)
    logs2   = logsumexp2 - at.log(Ndraw64)
    logNeff = 2.0 * log_mu - logs2 + at.log(Ndraw64)
    Neff    = at.exp(logNeff)
    var_log_lik_u = atools.logdiffexp(logs2 - 2.0 * log_mu, 1.0) - at.log(Ndraw64 - 1.0)

    return log_mu, Neff, var_log_lik_u


# def sel_bias_with_uncertainty_at_0(m1inj, m2inj, dLinj, spinsInj, log_p_draw, Lambda,  Ndraw, rate_model, mass_model, spin_model, smoothing, has_m2_break, interp, dL_grid=None, z_grid=None, **kwargs):


#     H0, Om, w0, Xi0, n  = Lambda[:5]

#     if (spin_model=='default') or (spin_model=='default_gauss'):
#         spinsInj_sel = [spinsInj[0], spinsInj[1], spinsInj[2], spinsInj[3]]
#     elif spin_model=='none':
#         spinsInj_sel = []
#     if dL_grid is None:
#         zinj = atools.z_from_dL_at(dLinj, H0, Om, w0, Xi0, n, interp=interp  )
#     else:
#         print('Inverting with interpolation of pre-computed grid for injections')
#         if z_grid is None:
#             raise ValueError('Pass z grid if passing pre-computed dL grid')
#         #zinj = atools.invert_monotone_binary_at(dLinj, dL_grid, z_grid)
#         zinj = atools.atinterp(dLinj, dL_grid, z_grid)
    

#     m1Src  = m1inj/(1+zinj)
#     m2Src  = m2inj/(1+zinj)

#     if mass_model in ('DP', 'DPUC'):
#         Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
#         log_Mc_src_inj = at.log(Mc_src_inj)
#         logit_q_inj = atools.logitat(q_inj)      
#         mass_1_use = log_Mc_src_inj
#         mass_2_use = logit_q_inj
#     else:
#         mass_1_use = m1Src
#         mass_2_use = m2Src

#     log_p_pop = log_p_pop_at(mass_1_use, mass_2_use, zinj, dLinj, spinsInj_sel, Lambda, rate_model, mass_model, spin_model, smoothing=smoothing, has_m2_break=has_m2_break)

#     if mass_model in ('DP', 'DPUC'):
#         # remove jacobian m1, m2 --> log(Mc), logit(q)
#         log_p_pop += (- at.log(m2Src) - at.log(m1Src-m2Src) - at.log1p(zinj) )

#     log_sel_b = log_p_pop-log_p_draw
  
    
#     log_mu = at.logsumexp(log_sel_b) - at.log(Ndraw)
    
#     logs2 = at.logsumexp(2.0*log_sel_b) - at.log(Ndraw)


#     #####################################
#     # This is N_eff as in Farr 2019
#     #####################################
#     ## way 1
#     #mu = at.exp(log_mu)
#     #muSq = mu*mu
#     #s2 = at.exp(  logs2 )
#     #sigmaSq = s2 - muSq/Ndraw
#     #Neff = muSq/sigmaSq

#     ## way 2
#     #print("sel_bias_at_vec logs2-2*log_mu " )
#     #print((logs2-2*log_mu).eval())
    
#     #logNeff = -atools.logdiffexp( logs2-2*log_mu, -at.log(Ndraw) )


#     #####################################
#     # This is N_eff as in Talbot Golomb 2023
#     # Difference between the two is ~1/N_draw , so negligible for large injection sets
#     #####################################

#     logNeff = 2*log_mu - logs2 + at.log(Ndraw)

#     #####################################
#     # This is variance of log l per unit obs as in Talbot Golomb 2023
#     #####################################

#     var_log_lik_u = atools.logdiffexp( logs2-2*log_mu, 1.) - at.log(Ndraw-1)

#     Neff = at.exp(logNeff)
    
    
#     return log_mu, Neff, var_log_lik_u


def sel_bias_with_uncertainty_at_0(
    m1inj, m2inj, dLinj, spinsInj, log_p_draw, Lambda, Ndraw,
    rate_model, mass_model, spin_model, smoothing, has_m2_break, interp,
    dL_grid=None, z_grid=None, **kwargs
):
    """
    Mixed-precision option:
      - If use_float32=True, heavy pointwise work is done in fp32 and the *final reductions*
        (max/exp-sum/log) are promoted to fp64 for stability.
      - If use_float32=False, the fp64 path is left EXACTLY as before.
    """
    use_float32 = bool(kwargs.get("use_float32", False))
    reduce_dtype = pytensor.config.floatX  # was "float64"
    print("Reduce dtype is %s"%reduce_dtype)

    H0, Om, w0, Xi0, n = Lambda[:5]

    if (spin_model == "default") or (spin_model == "default_gauss"):
        spinsInj_sel = [spinsInj[0], spinsInj[1], spinsInj[2], spinsInj[3]]
    elif spin_model == "none":
        spinsInj_sel = []

    if dL_grid is None:
        zinj = atools.z_from_dL_at(dLinj, H0, Om, w0, Xi0, n, interp=interp)
    else:
        print("Inverting with interpolation of pre-computed grid for injections")
        if z_grid is None:
            raise ValueError("Pass z grid if passing pre-computed dL grid")
        zinj = atools.atinterp(dLinj, dL_grid, z_grid)

    m1Src = m1inj / (1 + zinj)
    m2Src = m2inj / (1 + zinj)

    if mass_model in ("DP", "DPUC"):
        Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
        log_Mc_src_inj = at.log(Mc_src_inj)
        logit_q_inj = atools.logitat(q_inj)
        mass_1_use = log_Mc_src_inj
        mass_2_use = logit_q_inj
    else:
        mass_1_use = m1Src
        mass_2_use = m2Src

    log_p_pop = log_p_pop_at(
        mass_1_use, mass_2_use, zinj, dLinj, spinsInj_sel, Lambda,
        rate_model, mass_model, spin_model, smoothing=smoothing, has_m2_break=has_m2_break
    )

    if mass_model in ("DP", "DPUC"):
        # remove Jacobian m1,m2 -> log(Mc), logit(q)
        log_p_pop += (-at.log(m2Src) - at.log(m1Src - m2Src) - at.log1p(zinj))

    if use_float32:
        
        
        # ----- Mixed-precision path -----
        # Compute in fp32, then promote the *reduction scalars* to fp64.
        log_p_pop32 = at.cast(log_p_pop, "float32")
        lpd32       = at.cast(log_p_draw, "float32")
        x32         = log_p_pop32 - lpd32

        m32   = at.max(x32)
        s1_32 = at.exp(x32 - m32).sum()
        s2_32 = at.exp(2.0 * x32 - 2.0 * m32).sum()

        m64   = at.cast(m32,   reduce_dtype)
        s1_64 = at.cast(s1_32, reduce_dtype)
        s2_64 = at.cast(s2_32, reduce_dtype)

        logsumexp1 = m64 + at.log(s1_64)
        logsumexp2 = 2.0 * m64 + at.log(s2_64)

        # keep final scalars in fp64
        Ndraw64 = at.as_tensor_variable(Ndraw).astype(reduce_dtype)
        log_mu = logsumexp1 - at.log(Ndraw64)
        logs2  = logsumexp2 - at.log(Ndraw64)

    else:
        # ----- ORIGINAL fp64 path (EXACTLY as before) -----
        log_sel_b = log_p_pop - log_p_draw
        log_mu = at.logsumexp(log_sel_b) - at.log(Ndraw)
        logs2  = at.logsumexp(2.0 * log_sel_b) - at.log(Ndraw)

    # N_eff (Talbot & Golomb 2023 form)
    logNeff = 2 * log_mu - logs2 + at.log(Ndraw)

    # Variance of log-likelihood per unit obs (Talbot & Golomb 2023)
    var_log_lik_u = atools.logdiffexp(logs2 - 2 * log_mu, 1.0) - at.log(Ndraw - 1)

    Neff = at.exp(logNeff)
    return log_mu, Neff, var_log_lik_u
    

#####################################################



#####################################################
#####################################################


def make_model(  priors,
                 GWData,
                 InjData,
                 ivals={},
                 sampling_GW = 'gmm',
                 rate_model = 'MD',
                 mass_model = 'PLP',
                 smoothing='LVK',
                 has_m2_break = False,
                 spin_model = 'none',
                 spin_inj = 'none',
                 marginal_R0 = True,
                 dLprior = 'none',
                 fix_inj_len = False,
                 chunk_inj = -1,
                 chunk_reduce = False,
                 use_float32 = False,
                 sel_method='Tobs',
                 N_DP_comp_max = 20,
                 fix_H0 = True,
                fix_Om = True,
               fix_w0 = True,
                 fix_Xi0n = True,
               pade=False,
               zres='low',
               params_fix=None,
                 Neff_min=4,
                Neff_min_lik=1,
               log_lik_var_min=1,
                 use_sel_spin=True,
                 pop_only = False,
               N_successes_l=None,
               Nsamplesuse = -1,
               include_sel_uncertainty=False,
               sel_smoothing='poly',
               alpha_beta_prior='poly',
               dil_factor=1,
               use_log_alpha_beta=False ,
               allTobs=None,
                 use_updates=True,
                 inj_loop=False,
                 save_thetas=False
                ):

    ################################################
    # Read in data and set dimensions
    ################################################

    ## GW data
    if not pop_only:
        # gw data are interpolants of single-event posteriors
        if sampling_GW=='gauss':
            # we sample single-event parameters from broad gaussian approximations of the posteriors
            mus_s, cho_s, log_wts_l, mus_l, icovs_l, log_dets_l, Tobs, Nevs = GWData
        elif 'gmm' in sampling_GW or sampling_GW=='gumbel':
            # we sample single-event parameters from the actual single-event posteriors
            wts_l, mus_l, cho_covs_l, Tobs, Nevs = GWData
        else:
            raise ValueError('sampling_GW can be gmm, gmm_cat, gumbel,  gauss ')
            
        

    else:
        # gw data are single-event posterior samples
        # shape of each has to be n_events, n_samples
        m1det, m2det, d, spin_samples, Tobs, allNsamples, where_compute = GWData            

        if Nsamplesuse !=-1 :
            if Nsamplesuse>allNsamples:
                raise ValueError("Must use less samples than those available.")
            print("allNsamples availabe is %s, but %s will be used"%(allNsamples, Nsamplesuse))
            allNsamples =  Nsamplesuse   
            allNsamples_np = allNsamples #allNsamples.eval()
        
        if (spin_model=='default') or (spin_model=='default_gauss'):
           chi1, chi2, cost1, cost2 = spin_samples
        else:
            raise NotImplementedError()

    ## Injections data
    if spin_inj == 'none':
        dLinj, m1inj, m2inj, lpdinj, Ndraw, Ndet = InjData
    elif spin_inj == 'chieffchip':
        dLinj, m1inj, m2inj, chiefffInj, chipInj, lpdinj, Ndraw, Ndet = InjData
    elif (spin_inj == 'chi12xyz' or spin_inj == 'default'):
        if (spin_model=='default') or (spin_model=='default_gauss'):
            dLinj, m1inj, m2inj, chi1Inj, chi2Inj, cost1Inj, cost2Inj, lpdinj, Ndraw, Ndet = InjData
        elif spin_model == 'none':
            dLinj, m1inj, m2inj, lpdinj, Ndraw, Ndet = InjData

    
    Ndet_np = Ndet #Ndet.eval()
    N_DP_comp_max_np = N_DP_comp_max #N_DP_comp_max.eval()
    Nevs_np = Nevs #Nevs.eval()

    Tobs_np = Tobs #Tobs.eval()

        
    if not pop_only:
        N = mus_l.shape[0] # number of events in total
        N_np = N #N.eval()
        ngmm = mus_l.shape[1]
        ngmm_np = ngmm #ngmm.eval()
        nd = mus_l.shape[2]
        nd_np = nd #nd.eval()
        print('N:%s, max ngmm: %s, nd: %s '%(N_np, ngmm_np, nd_np))
        print('N evs is %s'%Nevs_np)
        print('Tobs is %s'%Tobs_np)
    else:
        N = m1det.shape[0] # number of events in total
        N_np = N #N.eval()
        Nsamples = m1det.shape[1]
        Nsamples_np = Nsamples #Nsamples.eval()
        print("N samples max will be ")
        print(Nsamples_np)
        print('N:%s, n samples: %s '%(N_np, allNsamples_np))




    
    event_index = np.arange(N_np, dtype=int)
    
    ndata = m1inj.shape[0] # number of observing runs to combine
    ndata_np = ndata #ndata.eval()
    ninj = m1inj.shape[1] # max number of injections
    ninj_np = ninj #ninj.eval()

    Ttot = np.sum(Tobs)

    
    print('Injections: :%s, '%(ninj_np))

    print('ninj: :%s, %s datasets,'%(Ndet_np, ndata_np))

    coords = {'event_index': event_index}

    

    if mass_model in ('DP', 'DPUC'):
        coords['component'] = np.arange(N_DP_comp_max_np, dtype=int)
        coords['GMMdimension'] = np.arange(2, dtype=int)
        coords['GMMdimension_1'] = np.arange(2, dtype=int)
        coords['GMMdimension_2'] = np.arange(2, dtype=int)
        p = 2*(2+1)//2  # packed length = 3 for n=2
        
        coords["packed_cholesky"] = np.arange(p)

    if pop_only:
        coords['nsamples'] = np.arange( Nsamples_np, dtype=int )
    else:
         coords['GWdimension'] = np.arange(nd_np, dtype=int)


    if params_fix is None:
        print('No values for parameters to fix passed. Default values will be used. If fixing parameters, check that the values are consistent. Values of fixed parameters:')
        print(PLPeakO3params)
        params_fix=PLPeakO3params
        
    ################################################
    # Build model
    ################################################
    
    with pm.Model(coords=coords) as model:


        if sampling_GW=='gauss' :
            # we sample single-event parameters from broad gaussian approximations of the posteriors
            mus_s, cho_s, log_wts_l, mus_l, icovs_l, log_dets_l = at.as_tensor_variable(mus_s), at.as_tensor_variable(cho_s), at.as_tensor_variable(log_wts_l), at.as_tensor_variable(mus_l), at.as_tensor_variable(icovs_l), at.as_tensor_variable(log_dets_l)
        elif 'gmm' in sampling_GW:
            # we sample single-event parameters from the actual single-event posteriors
            wts_l, mus_l, cho_covs_l = at.as_tensor_variable(wts_l), at.as_tensor_variable(mus_l), at.as_tensor_variable(cho_covs_l)

        ################################################
        # Cosmological parameters
        ################################################

        
        if fix_H0:
            H0_ =  params_fix['H0']
        else:
            H0_ =  pm.Uniform('H0', lower=priors['H0'][0], upper=priors['H0'][1], initval=ivals.get('H0'))
        
        if fix_Om:
            Om_ = params_fix['Om']
        else:
            Om_ = pm.Uniform('Om', lower=priors['Om'][0], upper=priors['Om'][1], initval=ivals.get('Om')) 

        if fix_w0:
            w0_ = -1.
        else:
            if pade:
                raise NotImplementedError("Pade appproximation with varying w0 not implemented yet. Use pade=False")
            w0_ =  pm.Uniform('w0', lower=priors['w0'][0], upper=priors['w0'][1], initval=ivals.get('w0'))
            
        
        if fix_Xi0n:
            Xi0_ =  1.
            nXi0_ = 0.
        else:
            Xi0_ =  pm.Uniform('Xi0', lower=priors['Xi0'][0], upper=priors['Xi0'][1], initval=ivals.get('Xi0'))
            nXi0_ = pm.Uniform('n', lower=priors['n'][0], upper=priors['n'][1], initval=ivals.get('nXi0')) 

        Lambda_ = [H0_, Om_, w0_, Xi0_, nXi0_]

        ################################################
        # Redshift evolution of merger rate
        ################################################
        
        if rate_model=='MD':
            
            print('Modeling evolution of merger rate with redshift with Madau-Dickinson profile')
            
            gamma_ = pm.Uniform('gamma', lower=priors['gamma'][0], upper=priors['gamma'][1], initval=ivals.get('gamma'))    
            kappa_ = pm.Uniform('kappa', lower=priors['kappa'][0], upper=priors['kappa'][1], initval=ivals.get('kappa'))
            zp_ = pm.Uniform('zp', lower=priors['zp'][0], upper=priors['zp'][1], initval=ivals.get('zp'))

            # gamma_ = atools.uniform_unconstrained("gamma",  priors['gamma'][0], priors['gamma'][1], init=ivals.get("gamma"))
            # kappa_ = atools.uniform_unconstrained("kappa",  priors['kappa'][0], priors['kappa'][1], init=ivals.get("kappa"))
            # zp_ = atools.uniform_unconstrained("zp",  priors['zp'][0], priors['zp'][1], init=ivals.get("zp"))
            
            Lambda_ += [gamma_, kappa_, zp_]

        elif rate_model=='PL':
            print('Modeling evolution of merger rate with a power law')
            gamma_ = pm.Uniform('gamma', lower=priors['gamma'][0], upper=priors['gamma'][1], initval=ivals.get('gamma'))

            Lambda_ += [gamma_]

        ################################################
        # Spin
        ################################################

        if spin_model == 'chieffchip':
            print('Modeling spin distribution with a gaussian in chieff-chip')
            muEff_ = pm.Uniform('muEff', lower=priors['muEff'][0], upper=priors['muEff'][1])
            sigEff_ = pm.Uniform('sigEff', lower=priors['sigEff'][0], upper=priors['sigEff'][1])
            muP_ = pm.Uniform('muP', lower=priors['muP'][0], upper=priors['muP'][1])
            sigP_ = pm.Uniform('sigP', lower=priors['sigP'][0], upper=priors['sigP'][1])
            rho_ = pm.Uniform('rho', lower=priors['rho'][0], upper=priors['rho'][1])

            Lambda_ += [muEff_, sigEff_, muP_, sigP_, rho_]

        elif spin_model=='chieffchip_uc':

            print('Modeling spin distribution with uncorrelated gaussians in chieff-chip')
            muEff_ = pm.Uniform('muEff', lower=priors['muEff'][0], upper=priors['muEff'][1])
            sigEff_ = pm.Uniform('sigEff', lower=priors['sigEff'][0], upper=priors['sigEff'][1])
            muP_ = pm.Uniform('muP', lower=priors['muP'][0], upper=priors['muP'][1])
            sigP_ = pm.Uniform('sigP', lower=priors['sigP'][0], upper=priors['sigP'][1])

            Lambda_ += [muEff_, sigEff_, muP_, sigP_]

        elif spin_model=='default':

            print('Modeling spin distribution with default spin model')

            if not use_log_alpha_beta:
                muChi_ = pm.Uniform('muChi', lower=priors['muChi'][0], upper=priors['muChi'][1])
                varChi_ = pm.Uniform('varChi', lower=priors['varChi'][0], upper=priors['varChi'][1])
                zeta_ = pm.Uniform('zeta', lower=priors['zeta'][0], upper=priors['zeta'][1])
                sigmat_ = pm.Uniform('sigmat', lower=priors['sigmat'][0], upper=priors['sigmat'][1])
    
                kappa_ = muChi_*(1-muChi_)/varChi_-1
    
                alphaChi_ = pm.Deterministic('alphaChi',  muChi_*kappa_ )
                betaChi_ = pm.Deterministic('betaChi',  (1-muChi_)*kappa_ )
                stdChi_ = pm.Deterministic('stdChi',  at.sqrt(varChi_) )
    
    
                Lambda_ += [alphaChi_, betaChi_, zeta_, sigmat_]
    
                # Bound alpha, beta > 1    
                
                if alpha_beta_prior=='poly':
                    print("Tapering prior on alpha_chi, beta_chi with polynomial smoothing")
                    _ = pm.Potential('bound_alphaChi', atools.log_f_smooth_poly(alphaChi_, 5e-4,  1 )  )
                    _ = pm.Potential('bound_betaChi', atools.log_f_smooth_poly(betaChi_, 5e-4,  1  ))
                elif alpha_beta_prior=='sigmoid':
                    print("Tapering prior on alpha_chi, beta_chi with sigmoid smoothing")
                    _ = pm.Potential('bound_alphaChi', atools.log_sigmoid(alphaChi_,  1+3e-04, 1e-04)  )
                    _ = pm.Potential('bound_betaChi', atools.log_sigmoid(betaChi_, 1+3e-04, 1e-04)  )
                else:
                    print("Putting prior on alpha_chi, beta_chi with hard cut")
                    _ = pm.Potential('bound_alphaChi', at.switch( at.le(alphaChi_, 1. ), -np.inf, at.as_tensor_variable(0.) ) )
                    _ = pm.Potential('bound_betaChi', at.switch( at.le(betaChi_, 1. ), -np.inf, 0.0 ) )
        
            else:
                # still to be tested. Might improve sampling/divergences
                print("Sampling in log(alpha-1), log(beta-1)")
                raise NotImplementedError()
                
        elif spin_model=='default_gauss':

            print('Modeling spin distribution with default spin model, gaussian distribution for magnitudes')

            muChi_ = pm.Uniform('muChi', lower=priors['muChi'][0], upper=priors['muChi'][1])
            sigmaChi_ = pm.Uniform('sigmaChi', lower=priors['sigmaChi'][0], upper=priors['sigmaChi'][1])
            
            zeta_ = pm.Uniform('zeta', lower=priors['zeta'][0], upper=priors['zeta'][1])
            sigmat_ = pm.Uniform('sigmat', lower=priors['sigmat'][0], upper=priors['sigmat'][1])

            Lambda_ += [muChi_, sigmaChi_, zeta_, sigmat_]
            
        else:
            print('No model of the spin distribution.')
                

            

        ################################################
        # Mass distribution
        ################################################
            
        if mass_model=='PLPreg':

            ### BBH
            
            # Power law + peak
            print('Modeling mass distribution with LVK Power Law + Peak with regularized edge')
            if smoothing=='LVK':
                print('Using LVK smoothing')
            elif smoothing=='poly':
                print('using differentiable polynomial smoothing')
            
            lamP_   = pm.Uniform("lambdaPeak", lower=priors["lambdaPeak"][0], upper=priors["lambdaPeak"][1], initval=ivals.get("lambdaPeak"))        
            alpha_  = pm.Uniform("alpha",      lower=priors["alpha"][0],      upper=priors["alpha"][1],      initval=ivals.get("alpha"))
            beta_   = pm.Uniform("beta",       lower=priors["beta"][0],       upper=priors["beta"][1],       initval=ivals.get("beta"))
            ml_     = pm.Uniform("ml",         lower=priors["ml"][0],         upper=priors["ml"][1],         initval=ivals.get("ml"))
            mh_     = pm.Uniform("mh",         lower=priors["mh"][0],         upper=priors["mh"][1],         initval=ivals.get("mh"))
            deltam_ = pm.Uniform("deltam",     lower=priors["deltam"][0],     upper=priors["deltam"][1],     initval=ivals.get("deltam"))
            muM_    = pm.Uniform("muMass",     lower=priors["muMass"][0],     upper=priors["muMass"][1],     initval=ivals.get("muMass"))
            sM_     = pm.Uniform("sigmaMass",  lower=priors["sigmaMass"][0],  upper=priors["sigmaMass"][1],  initval=ivals.get("sigmaMass"))

             #lamP_ = atools.uniform_unconstrained("lambdaPeak",  priors['lambdaPeak'][0], priors['lambdaPeak'][1], init=ivals.get("lambdaPeak"))
            # alpha_  = atools.uniform_unconstrained("alpha",     priors["alpha"][0],     priors["alpha"][1],     init=ivals.get("alpha"))
            # beta_   = atools.uniform_unconstrained("beta",      priors["beta"][0],      priors["beta"][1],      init=ivals.get("beta"))
            # ml_     = atools.uniform_unconstrained("ml",        priors["ml"][0],        priors["ml"][1],        init=ivals.get("ml"))
            # mh_     = atools.uniform_unconstrained("mh",        priors["mh"][0],        priors["mh"][1],        init=ivals.get("mh"))
            # deltam_ = atools.uniform_unconstrained("deltam",    priors["deltam"][0],    priors["deltam"][1],    init=ivals.get("deltam"))
            # muM_    = atools.uniform_unconstrained("muMass",    priors["muMass"][0],    priors["muMass"][1],    init=ivals.get("muMass"))
            # sM_     = atools.uniform_unconstrained("sigmaMass", priors["sigmaMass"][0], priors["sigmaMass"][1], init=ivals.get("sigmaMass"))

            Lambda_ += [lamP_, alpha_, beta_, deltam_, ml_, mh_, muM_, sM_ ]


        elif mass_model=='DPLDP':

            print('Modeling mass distribution with Double Power Law + Double Peak ')

            alpha1_   = pm.Uniform("alpha1",   lower=priors["alpha1"][0],   upper=priors["alpha1"][1],   initval=ivals.get("alpha1"))
            alpha2_   = pm.Uniform("alpha2",   lower=priors["alpha2"][0],   upper=priors["alpha2"][1],   initval=ivals.get("alpha2"))
            mb_       = pm.Uniform("mb",       lower=priors["mb"][0],       upper=priors["mb"][1],       initval=ivals.get("mb"))
            mu1_      = pm.Uniform("mu1",      lower=priors["mu1"][0],      upper=priors["mu1"][1],      initval=ivals.get("mu1"))
            sigma1_   = pm.Uniform("sigma1",   lower=priors["sigma1"][0],   upper=priors["sigma1"][1],   initval=ivals.get("sigma1"))
            mu2_      = pm.Uniform("mu2",      lower=priors["mu2"][0],      upper=priors["mu2"][1],      initval=ivals.get("mu2"))
            sigma2_   = pm.Uniform("sigma2",   lower=priors["sigma2"][0],   upper=priors["sigma2"][1],   initval=ivals.get("sigma2"))
            u         = pm.Uniform("u", 0, 1, initval=ivals.get("u"))
            m1_low_   = pm.Deterministic("m1_low", 3 + (10 - 3) * at.sqrt(u))
            v         = pm.Uniform("v", 0, 1, initval=ivals.get("v"))
            m2_low_   = pm.Deterministic("m2_low", 3 + v * (m1_low_ - 3))
            m_high_   = pm.Deterministic("m_high", at.as_tensor_variable(300.0)  )
            delta_m1_ = pm.Uniform("delta_m1", lower=priors["delta_m1"][0], upper=priors["delta_m1"][1], initval=ivals.get("delta_m1"))
            lambda_vec = pm.Dirichlet("lambda", a=np.array([1, 1, 1]), initval=ivals.get("lambda"))
            lambda0_  = pm.Deterministic("lambda0", lambda_vec[0])
            lambda1_  = pm.Deterministic("lambda1", lambda_vec[1])
            lambda2_  = pm.Deterministic("lambda2", lambda_vec[2])
            beta_     = pm.Uniform("beta",     lower=priors["beta"][0],     upper=priors["beta"][1],     initval=ivals.get("beta"))
            delta_m2_ = pm.Uniform("delta_m2", lower=priors["delta_m2"][0], upper=priors["delta_m2"][1], initval=ivals.get("delta_m2"))
            epsilon_  = pm.Deterministic("epsilon", at.as_tensor_variable(0.01))
            if has_m2_break:
                print("Including gap for secondary mass")
                m_g_     =  pm.Uniform("m_g", lower=priors["m_g"][0], upper=priors["m_g"][1], initval=ivals.get("m_g")) 
                w_g_     = pm.Uniform("w_g", lower=priors["w_g"][0], upper=priors["w_g"][1], initval=ivals.get("w_g")) 
                sig_g_l_ = at.as_tensor_variable(1e-02)
                sig_g_h_ = at.as_tensor_variable(1e-02)
            else:
                m_g_     = at.as_tensor_variable(45.)
                w_g_     = at.as_tensor_variable(70.)#.astype('float64')
                sig_g_l_ = at.as_tensor_variable(1e-02)
                sig_g_h_ = at.as_tensor_variable(1e-02)
            
            Lambda_ += [alpha1_, alpha2_, mb_, mu1_, sigma1_, mu2_, sigma2_, m1_low_, m_high_, delta_m1_, lambda0_, lambda1_, beta_, m2_low_, delta_m2_, epsilon_, m_g_, w_g_, sig_g_l_, sig_g_h_]

        ### BNS
        elif 'BNSgauss' in mass_model:

            if mass_model=='BNSgauss':
                # Uncorrelated gaussians
                print('Modeling mass distribution with uncorrelated gaussian distributions')
            elif mass_model=='BNSgaussCond':
                # Conditioned gaussians
                print('Modeling mass distribution with gaussian distributions with p(m1, m2) = p(m1) p(m2|m1) H(m1-m2)')
                
            muM_ = pm.Uniform('muMass', lower=priors['muMass'][0], upper=priors['muMass'][1])
            sM_ = pm.Uniform('sigmaMass', lower=priors['sigmaMass'][0], upper=priors['sigmaMass'][1] )  
            Lambda_ += [muM_, sM_ ]

        ### Non - parametric
        elif mass_model=='DPUC':
            print("Modeling mass distribution as Dirichelet Process. Max number of components: %s"%N_DP_comp_max)

            
            alpha = pm.Gamma("alpha", 1.0, 1.0)
            beta = pm.Beta("beta", 1.0, alpha, dims="component" )
            w = pm.Deterministic("w", atools.stick_breaking(beta), dims="component")
            logw = at.log(w)


            #### Sigma prior limits: 


            # Option 1: Fixes std from (\tau * \lambda ) parametrization
            # Here check how to choose the parameters of the Gamma priors!

            #---- global–local prior on SDs (uncorrelated) ----
            # controls local variability
            # a_lam1, b_lam1 = 0.6, 0.6
            # a_lam2, b_lam2 =   0.5, 0.5
            
            # # One global precision per dimension (shared across components)
            # a_tau1, b_tau1 =   24, 4  # for z1 = log(Mc)
            # a_tau2, b_tau2 =  4, 2 # for z2 = logit(q)
            # tau1 = pm.Gamma("tau1", a_tau1, b_tau1)
            # tau2 = pm.Gamma("tau2", a_tau2, b_tau2)
            
            # # Local precisions per component, per dimension
            # lam1 = pm.Gamma("lam1", a_lam1, b_lam1, dims="component")
            # lam2 = pm.Gamma("lam2", a_lam2, b_lam2, dims="component")
            
            # # Per-component, per-dimension standard deviations (independent axes)
            # sig1 = pm.Deterministic("sig1", 1.0 / at.sqrt(tau1 * lam1), dims="component")
            # sig2 = pm.Deterministic("sig2", 1.0 / at.sqrt(tau2 * lam2), dims="component")



            # mu_ln1, s_ln1 = -1.163789654413117, 0.36380796502993723
            # mu_ln2, s_ln2 = -1.1074484102105713, 0.48718303162186627
            # sig1_max, sig2_max = 0.7971738948530454, 2.0086525885321533

            # # In your PyMC model (uncorrelated shown; for correlated use as marginals):
            # sig1 = pm.Truncated("sig1", pm.LogNormal.dist(mu=mu_ln1, sigma=s_ln1),
            #                     lower=0.0, upper=sig1_max, dims="component")
            # sig2 = pm.Truncated("sig2", pm.LogNormal.dist(mu=mu_ln2, sigma=s_ln2),
            #                     lower=0.0, upper=sig2_max, dims="component")


            # sig1 = pm.Uniform("sig1", lower=0.01, upper=5, dims="component")
            # sig2 = pm.Uniform("sig2", lower=0.01, upper=10, dims="component")


            sig1 = pm.InverseGamma("sig1", alpha=4.72, beta=1.39, dims="component")
            sig2 = pm.InverseGamma("sig2", alpha=0.5, beta=0.4, dims="component")
            
            

            sd = pm.Deterministic("sig", at.stack([sig1, sig2], axis=0),  # (2,K)
                      dims=("GMMdimension", "component"))


            #### Mean prior limits:  remember that mu is log(Mc), logit(q).

            # Option 1 : sample mean of the gaussians from given prior
            # with this choice, the prior on the mean will be flat in log(Mc), logit(q).
        
            mu1 = pm.Uniform('mulMc', lower=1.13, upper=4.38, dims= ("component" ))
            mu2 = pm.Uniform('mulq', lower=-2.75, upper=9.37, dims= ("component" ))
            

            # z1_lo = 1.13
            # z1_hi = 4.38
            # mu1_mid = (z1_lo+z1_hi)/2
            # mu1_sd = (-z1_lo+z1_hi)/2
            # span1 = -z1_lo+z1_hi

            # z2_lo = -2.75
            # z2_hi = 9.37
            # mu2_mid = (z2_lo+z2_hi)/2
            # mu2_sd = (-z2_lo+z2_hi)/2
            # span2 = -z2_lo+z2_hi

            # mu1 = pm.TruncatedNormal("mu1", mu=mu1_mid, sigma=mu1_sd,
            #              lower=z1_lo-0.5*span1, upper=z1_hi+0.5*span1,
            #              dims="component")
            # mu2 = pm.TruncatedNormal("mu2", mu=mu2_mid, sigma=mu2_sd,
            #              lower=z2_lo-0.5*span2, upper=z2_hi+0.5*span2,
            #              dims="component")

            mu = pm.Deterministic("mu", at.stack([mu1, mu2], axis=0),  # (2,K)
                      dims=("GMMdimension", "component"))

            # Option 2: check ...

            Lambda_ += [ w, mu, sd, logw ]

            Lambda_ += [N_DP_comp_max]

        elif mass_model=='DP':

            alpha = pm.Gamma("alpha", 1.0, 1.0)
            beta = pm.Beta("beta", 1.0, alpha, dims="component" )
            w = pm.Deterministic("w", atools.stick_breaking(beta), dims="component")
            logw = at.log(w)


            mu1 = pm.Uniform('mulMc', lower=1.13, upper=4.38, dims= ("component" ))
            mu2 = pm.Uniform('mulq', lower=-2.75, upper=9.37, dims= ("component" ))

            mu = pm.Deterministic("mu", at.stack([mu1, mu2], axis=0),  # (2,K)
                      dims=("GMMdimension", "component"))


            ################################################
            # cholesky option 1 (slower)

            # ---- Per-component LKJ Cholesky (NO batching) ----
            # packed_list = []
            # L_list = []
            # for k in range(N_DP_comp_max):
            #     pk = pm.LKJCholeskyCov(f"chol_packed_{k}",
            #                            n=2, eta=2.0, sd_dist=pm.HalfNormal.dist(2.5),
            #                            compute_corr=False)     # returns length-3 vector
            #     Lk = pm.expand_packed_triangular(2, pk, lower=True)  # (2,2)
            #     packed_list.append(pk)
            #     L_list.append(Lk)
    
            # # Stack to tensors
            # chol_packed = at.stack(packed_list, axis=0)   # (K, 3)
            # L = at.stack(L_list, axis=0)                  # (K, 2, 2)
    
            # # Optional: save deterministics with coords
            # #pm.Deterministic("chol_packed_all", chol_packed, dims=("component",))
            # #pm.Deterministic("Sigma_chol", L, dims=("component","dim","dim"))
    
            # # Precompute Σ^{-1} and log|Σ|
            # invL = pm.math.matrix_inverse(L)                  # (K,2,2)
            # Fisher = at.matmul(at.swapaxes(invL, -1, -2), invL)
            # ldets_inv = 2.0 * (pm.math.log(L[:,0,0]) + pm.math.log(L[:,1,1]))  # (K,)



            ################################################
            # cholesky option 2

            # # ---- prior on SDs (uncorrelated) ----
            # # Per-component, per-dimension standard deviations (independent axes)

            sig1 = pm.InverseGamma("sig1", alpha=4.72, beta=1.39, dims="component")
            sig2 = pm.InverseGamma("sig2", alpha=0.5, beta=0.4, dims="component")

            #sig1 = pm.Uniform("sig1", lower=0.01, upper=3, dims="component")
            #sig2 = pm.Uniform("sig2", lower=0.01, upper=5, dims="component")

            # # ----- Correlation prior equivalent to LKJ(eta) in 2D -----
            eta = 1.0  # uninformative on correlations
            rho_u = pm.Beta("rho_u", alpha=eta, beta=eta, dims="component")   # (0,1)
            rho   = pm.Deterministic("rho", 2.0 * rho_u - 1.0, dims="component")  # (-1,1)


            # # Useful terms
            one_minus_r2 = 1.0 - rho**2
            sqrt1mr2     = at.sqrt(one_minus_r2)
            
            # ----- Cholesky of Σ (for reference / if you need solves) -----
            # Σ = [[s1^2, ρ s1 s2], [ρ s1 s2, s2^2]]
            # Cholesky L = diag([s1, s2]) @ [[1, 0], [ρ, sqrt(1-ρ^2)]]
            row0 = at.stack([sig1,               at.zeros_like(sig1)], axis=1)          # (K,2)
            row1 = at.stack([rho * sig2,         sig2 * sqrt1mr2     ], axis=1)          # (K,2)
            L    = at.stack([row0, row1], axis=1)     
            Cho_cov = pm.Deterministic("Cho_cov", L, dims=("component","GMMdimension","GMMdimension_1"))
            
            # ----- log |Σ^{-1}| (no inverses) -----
            # det Σ = s1^2 * s2^2 * (1 - ρ^2)
            # log |Σ^{-1}| = - log det Σ
            ldets_inv = pm.Deterministic(
                "ldets_inv",
                -2.0 * at.log(sig1) - 2.0 * at.log(sig2) - at.log(one_minus_r2),
                dims="component",
            )
            
            # ----- Precision Σ^{-1} in closed form (Fisher) -----
            # Σ^{-1} = 1 / [ (1-ρ^2) s1^2 s2^2 ] * [[ s2^2, -ρ s1 s2 ], [ -ρ s1 s2, s1^2 ]]
            den = one_minus_r2 * (sig1**2) * (sig2**2)
            F11 =  (sig2**2)            / den
            F22 =  (sig1**2)            / den
            F12 = -(rho * sig2 * sig1)    / den
            
            Fisher = pm.Deterministic( "Fisher", at.stack([
                at.stack([F11, F12], axis=1),
                at.stack([F12, F22], axis=1)
            ], axis=1), dims=("component","GMMdimension_1","GMMdimension_2"))  # shape: (K, 2, 2)

            
            ################################################

            Lambda_ += [ alpha, beta, w, mu, Fisher, ldets_inv, logw ]

            Lambda_+=[N_DP_comp_max]
            
        ################################################
        # If including total normalization of the rate, add it here
        ################################################
        
        if not marginal_R0:
            R0 = pm.Uniform('R0', lower=priors['R0'][0], upper=priors['R0'][1])
        else:
            R0 = at.as_tensor_variable(1.)    
        lR0 = at.log(R0)


        if zres=='low':
            print('Using z grid with 150 points')
            zgrid_ = atools.zGridGlobals_at_low
        elif zres=='high':
            print('Using z grid with 1000 points')
            zgrid_ = atools.zGridGlobals_at_high
        
        # One grid build to interpolate later
        dL_grid = atools.dLfun_at(zgrid_, H0_, Om_, w0_, Xi0_, nXi0_, interp=pade)

        if dLprior == 'dVdz':
            dVdz_grid = atools.log_dV_dz_at(zgrid_, 67.90, 0.3065, -1., dc=None )-at.log1p(zgrid_)



        if not pop_only:
            ################################################
            # Individual event mass and distance
            ################################################
    
            x = pm.Normal( 'x', mu=0, sigma=1, dims= ("event_index" , "GWdimension" ) )


            if 'gauss' not in sampling_GW:
                
                if 'gmm' in sampling_GW:
        
                    print('Sampling m1d, m2d, dL from GMM')
    
                    if sampling_GW=='gmm_cat':
                        ###################################
                        # categorical way
                        
                        ig = pm.Categorical('idx', p=wts_l, dims= "event_index" )
    
                    elif sampling_GW=='gmm':
                        ###################################
                        # continuous way
        
                        u_gmm = pm.Normal("u_gmm", 0.0, 1.0, dims= "event_index")
                        v_gmm = at.clip( atools.normal_cdf(u_gmm), 1e-9, 1.0 - 1e-9) 
    
                        cdf_w = at.cumsum(wts_l, axis=1)                                          
                        ig = pm.Deterministic('idx', (v_gmm[:, None] < cdf_w).argmax(axis=1), dims= "event_index" )             

                    
                    # Select means and Cholesky factors per batch
                    mu_selected = mus_l[ np.arange(N), ig, :]         # shape (N, D)
                    L_selected = cho_covs_l[ np.arange(N), ig, :, :]  # shape (N, D, D)
                     
                    # Batched matrix multiplication: (N, D, D) @ (N, D, 1) → (N, D, 1)
                    Lx = at.sum(L_selected * x[:, None, :], axis=2)  # → shape (N, D)

            
                else:
                    print('Sampling m1d, m2d, dL from gumbel soft assignment, tau=0.5')
                    
                    #tau = pm.MutableData("tau_gmm", 0.5)  # (note: if grads feel weak, raise to ~0.3–0.7)
                    tau=0.5
                    logits = at.log(at.clip(wts_l, 1e-12, 1.0))               # (N, K)
                    g = pm.Gumbel("gumbel", mu=0.0, beta=1.0, shape=wts_l.shape)  # (N, K)
                    y_soft = pm.math.softmax((logits + g) / tau, axis=1)      # (N, K)
                    
                    # hard label for inspection (unchanged)
                    ig = pm.Deterministic("idx", at.argmax(y_soft, axis=1), dims="event_index")  # (N,)
                    
                    # --- Straight-Through gate (hard forward, soft gradient) ---
                    # get K from your tensors (N, K, D)
                    K = mus_l.shape[1]
                    topk = at.argmax((logits + g) / tau, axis=1)                                     # (N,)
                    one_hot = at.eq(at.arange(K)[None, :], topk[:, None]).astype(y_soft.dtype)       # (N, K)
                    s_soft_hard = stop_grad(one_hot - y_soft) + y_soft                         # (N, K)

                    # --- Soft selection, but with ST gating in forward ---
                    # mu_selected: (N, D)
                    mu_selected = at.sum(mus_l * s_soft_hard[:, :, None], axis=1)
                    
                    # L_selected: (N, D, D)
                    L_selected = at.sum(cho_covs_l * s_soft_hard[:, :, None, None], axis=1)
                    
                    # Lx: (N, D)  [ (N,D,D) * (N,1,D) → (N,D,D); sum over last axis → (N,D) ]
                    Lx = at.sum(L_selected * x[:, None, :], axis=2)
                
                
                # Final transformed sample
                samples = mu_selected + Lx                # shape (N, D)
    
                
                log_Mc_det = samples[:,0]/dil_factor
                logit_q = samples[:,1]
                logd = samples[:,2]
                
    
                if (spin_model == 'chieffchip') or (spin_model == 'chieffchip_uc') :
        
                    chieff = atools.inv_flogitat(samples[:,3])
                    chip = atools.inv_logitat(samples[:,4])
        
                elif (spin_model == 'default') or (spin_model == 'default_gauss'):
                    # we have chi1, chi2, cost1, cost2
                    if save_thetas:
                        chi1 = pm.Deterministic('chi1', atools.inv_logitat(samples[:,3]))
                        chi2 = pm.Deterministic('chi2', atools.inv_logitat(samples[:,4]))
            
                        cost1 = pm.Deterministic('cost1', atools.inv_flogitat(samples[:,5]))
                        cost2 = pm.Deterministic('cost2', atools.inv_flogitat(samples[:,6]))
                    else:
                        chi1 = atools.inv_logitat(samples[:,3])
                        chi2 = atools.inv_logitat(samples[:,4])
                        cost1 =atools.inv_flogitat(samples[:,5])
                        cost2 =atools.inv_flogitat(samples[:,6])
                        
                else:
                    print("No spins computed")
            

            
            elif sampling_GW=='gauss' : # to be tested with spins
                
                print('Sampling log(Mc), logit(q), log(dL) from Gaussian approximant')

                # sample = mu + L @ x   (batched)
                samples = mus_s + at.matmul(cho_s, x[..., None])[..., 0]      # (N, d)
                
                # logp = log p(x) - log|L|
                # d = x.shape[1]
                log_px = -0.5 * at.sum(x**2, axis=1) - 0.5 * x.shape[1] * at.log(2.0 * np.pi)    # (N,)
                log_det_L = at.sum(at.log(at.diagonal(cho_s, axis1=1, axis2=2)), axis=1)  # (N,)
                pilik = log_px - log_det_L                                               # (N,)
                
                # unpack coordinates:
                log_Mc_det = samples[:, 0]
                logit_q    = samples[:, 1]
                logd       = samples[:, 2]
                

                if spin_model == 'none' :
                    
                    vals = at.stack([log_Mc_det, logit_q, logd ], axis=0)
                    # at.zeros(log_Mc_det.shape), at.zeros(log_Mc_det.shape), at.zeros(log_Mc_det.shape), at.zeros(log_Mc_det.shape) 

                elif spin_model == 'default' or spin_model == 'default_gauss':

                    chi1 = atools.inv_logitat(samples[:,3])
                    chi2 = atools.inv_logitat(samples[:,4])
        
                    cost1 = atools.inv_flogitat(samples[:,5])
                    cost2 = atools.inv_flogitat(samples[:,6])

                    vals = at.stack([log_Mc_det, logit_q, logd,  samples[:,3],  samples[:,4],  samples[:,5],  samples[:,6]], axis=0)


            

                # X as (N, d)
                X = vals.T                                   # (N, d)
                #print("X shape is %s"%(X[:, None, :].shape.eval()))
                #print("mus_l shape is %s"%(mus_l.shape.eval()))
                
                # Broadcast X against component-wise parameters
                # diff: (N, ngmm, d)
                diff = X[:, None, :] - mus_l                  # (N, 1, d) - (N, ngmm, d)
                
                # Quadratic form using precision F = Σ^{-1}
                # tmp = F @ diff[..., None]  -> (N, ngmm, d, 1) -> squeeze to (N, ngmm, d)
                tmp = at.matmul(icovs_l, diff[..., None])[..., 0]   # (N, ngmm, d)
                
                # r^T F r for each (obs, comp)
                quad = at.sum(diff * tmp, axis=-1)            # (N, ngmm)
                
                # Component logpdfs (Multivariate Normal)
                log_norm = -0.5 * vals.shape[0] * at.log(2.0 * np.pi)     # scalar
                logp_components = (
                    -0.5 * quad
                    + log_norm
                    - 0.5 * log_dets_l
                    + log_wts_l
                )                                             # (N, ngmm)
                
                # Mixture log-likelihood per observation: logsumexp over components
                gwl = at.logsumexp(logp_components, axis=1)   # (N,)
        
            
            else:
                raise NotImplementedError()


            Mc = at.exp(log_Mc_det)            
            q = atools.inv_logitat(logit_q)
            m1det, m2det = atools.m1m2_from_Mcq_at(Mc, q)

            if save_thetas:
                d = pm.Deterministic('dL', at.exp(logd) , dims="event_index")
        
                
                # Compute source-frame quantities. One redsfhit, mass1, mass2 for each event
                #zs = pm.Deterministic('z', atools.invert_monotone_binary_at(d, dL_grid, zgrid_), dims= "event_index" ) 
                zs = pm.Deterministic('z', atools.atinterp(d, dL_grid, zgrid_), dims= "event_index" ) 
                m1src = pm.Deterministic('m1src', m1det/(1+zs) , dims="event_index")
                m2src = pm.Deterministic('m2src', m2det/(1+zs) , dims="event_index") 
            else:
                d = at.exp(logd)
                zs =  atools.atinterp(d, dL_grid, zgrid_)
                m1src =  m1det/(1+zs)
                m2src =  m2det/(1+zs)

            
                
        else:
            # we are sampling the usual marginalise likelihood, with "only" pop parameters
            print('We are running inference only on population parameters.')


            # Compute source-frame quantities. One redsfhit, mass1, mass2 for each event
            # AND for each sample! 
            
            d_stacked  = at.flatten(d)
            #zs_stacked = atools.invert_monotone_binary_at(d_stacked, dL_grid, zgrid_)
            zs_stacked = atools.atinterp(d_stacked, dL_grid, zgrid_)
            
            zs = at.reshape( zs_stacked, (N, Nsamples) )
            m1src = m1det/(1+zs)
            m2src = m2det/(1+zs)
            
            logd = at.log(d)
        
        
        ################################################
        # Population prior
        ################################################

        
        if spin_model == 'chieffchip' or spin_model == 'chieffchip_uc' :

            spins = [ chieff, chip  ]

        elif (spin_model == 'default') or (spin_model == 'default_gauss'):

            spins = [chi1, chi2, cost1, cost2]

        elif spin_model == 'none':
            
            spins = []


        # Compute comoving distance - if gravity is modified, this is NOT d_L / (1+z) ! 
        Xi_ = atools.Xifun_at(zs, Xi0_, nXi0_)
        dc = d/(1+zs)/Xi_, 

        
        # Population prior of all events, without the term T_obs*R0
        if mass_model in ('DP', 'DPUC'):

            # dirichelet processs will be for log(Mc_src), logit(q) ...
            logMc_src =  log_Mc_det - at.log1p(zs)
            
            log_p_pop = log_p_pop_at( logMc_src, logit_q, zs, d, spins, Lambda_, rate_model, mass_model, spin_model,  dc=dc)
            # ... so remove a jacobian : p( m1, m2 ) = p( log(Mc), logit(q) ) * |J|
            log_p_pop -=  at.log(m2src) + at.log(m1src-m2src) + at.log1p(zs) 
            
        else:    
        
            log_p_pop = log_p_pop_at( m1src, m2src, zs, d, spins, Lambda_, rate_model, mass_model, spin_model, smoothing=smoothing, has_m2_break=has_m2_break, dc=dc)

        
        if dLprior=='dLsq':
            # Remove \pi(d)~dL^2 prior on distance 
            log_p_pop -= 2*logd
            print('Removing dL^2 prior')
        elif dLprior == 'dVdz':
            print('Removing prior proportional to 1/(1+z)*dV/dz with H0=67.90, Om=0.3065')
            lpi = atools.atinterp( zs, zgrid_, dVdz_grid )
            
            #atools.log_dV_dz_at(zs, 67.90, 0.3065, -1., dc=None )-at.log1p(zs)

            # The following is a hack.
            # When using GWTC data, O1-O2 do not have posteriors with dVdz prior, only dL^2
            # So I remove the dL^2 prior by hand on those
            # if not pop_only:
            #     # 1D case: shape (N,)
            #     lpi = at.concatenate([2 * logd[:10], lpi_[10:]], axis=0)
            # else:
            #     # 2D case: shape (N, Nsamples)
            #    lpi = at.concatenate([2 * logd[:10, :], lpi_[10:, :]], axis=0)
            
            log_p_pop -= lpi


        if not pop_only:
            if sampling_GW=='gauss' :
                # Add gw likelihood and correct for sampling prior pdf
                log_p_pop -= pilik
                log_p_pop += gwl
        
        # Put it all together
        if not pop_only:
            # just sum log likelihoods
            likelihood_val = at.sum( log_p_pop ) #pm.Deterministic("lik", at.sum( log_p_pop ) ) 
        else:
            # marginalise over single events parameters first
            # shape of p_pop is (hopefully) n_evs x n_samples
            # so average over second dimension
            
            # Compute only where there are samples
            log_p_pop_to_marg = log_p_pop[:, :allNsamples[0]]
            
            log_p_pop_marg = at.logsumexp( log_p_pop_to_marg, axis=1 ) - at.log(allNsamples)
            

            # then sum log likelihoods
            likelihood_val = at.sum( log_p_pop_marg ) #pm.Deterministic("lik", at.sum( log_p_pop_marg ) ) 

            # Check number of effective samples for computing MC integral 
            logs2 = at.logsumexp(2*log_p_pop_masked, axis=1) -2*at.log(allNsamples)
            
            Neff_lik =  pm.Deterministic('Neff_l', at.exp( 2.0*log_p_pop_marg - logs2) ) # this has len = n. of observations
            
            if Neff_min_lik>0:
                
                _ = pm.Potential("Neff_l_bound", at.sum( at.where( Neff_lik<Neff_min_lik*N, -np.inf, 0. ) ) )
                
                # see https://discourse.pymc.io/t/conditionally-reject-samples/3107
                # ind_sw_l = pm.Deterministic('ind_l', 1. * (Neff_lik<Neff_min_lik) )
                # ind_l = pm.Bernoulli('Neff_l_bound', ind_sw_l, observed=np.zeros(N_np), testval=np.zeros(N_np) )

            
            else:
                print("No bound on effective number of samples for individual event MC integrals")

        
        # add R0*Tobs if needed. 
        if not marginal_R0:
            print("Will not marginalise over R0.")
            # each term p_pop is multiplied by
            # R0*T_obs . So we get a factor (R0*T_obs)**N_i for every
            # observing run. R0 is the same for every run so I just have
            # (R0)**{\sum N_i} . For T_obs I have T_{obs,1}**N_1 * T_{obs,2}**N_2 * ...
            poiss_term = at.sum(Nevs*at.log(allTobs))+N*lR0
            likelihood_val += poiss_term
        else:
            print("Will marginalise over R0 with flat-in-log prior.")

        
        
        _ = pm.Potential("likelihood", likelihood_val ) 



        ################################################
        # Selection effect
        ################################################
        
        if sel_method=='skip':
            print('No selection bias!')
        else:
            # add sel effects    
            if ndata_np==1:
                # we passed a single injection set corresponding to multiple observing runs,
                # with injections already containing the correct weights
                print("Using selection effects from a single injection campaign")

                if use_sel_spin:
                    spin_model_name = spin_model
                    
                    if spin_model == 'chieffchip' or spin_model == 'chieffchip_uc' :
                        spinsInj = [ chiefffInj[0], chipInj[0] ]
                        
                    elif (spin_model == 'default') or (spin_model == 'default_gauss'):
                        spinsInj = [ chi1Inj[0], chi2Inj[0], cost1Inj[0], cost2Inj[0] ]
                        
                    else:
                        spinsInj = []

                else:
                    print("Spin distribution will not be used in the sel effect")
                    spinsInj = []
                    spin_model_name = 'none'

                if chunk_inj!=-1:
                    print('Using chunked version of sel. bias for memory efficiency.')
                    if inj_loop:
                        sel_bias_fun = sel_bias_with_uncertainty_at_loop
                        print("Using version with python loop")
                        print('Chunk size is %s'%chunk_inj)
                    else:
                        sel_bias_fun = sel_bias_with_uncertainty_at_0_batched 
                        #sel_bias_with_uncertainty_at_scan
                        print("Using version with pytensor scan in batches")
                        print('Chunk size is %s'%chunk_inj)
                        #print("use_float32 is %s"%use_float32)
                else:
                    if chunk_reduce:
                        print("Using chunked version for reduction of logsumexp")
                        sel_bias_fun = sel_bias_with_uncertainty_at_scan
                    else: 
                        print('Computing sel bias in one chunk')
                        sel_bias_fun = sel_bias_with_uncertainty_at_0


                
                log_mu_, Neff_, var_ll_u_ = sel_bias_fun( m1inj[0], m2inj[0], dLinj[0], spinsInj, lpdinj[0], 
                                                          Lambda_, 
                                                          Ndraw, 
                                                          rate_model, mass_model, spin_model_name, 
                                                          smoothing, 
                                                          has_m2_break, 
                                                          interp=pade, 
                                                          dL_grid=dL_grid, 
                                                          z_grid=zgrid_, 
                                                          chunk_size = chunk_inj, 
                                                          use_float32=use_float32, 
                                                          N_inj_py=ninj_np, 
                                                          scan_updates=use_updates,  
                                                        )
                
                if not marginal_R0:
                    # This is really the number of expected events 
                    sel_effect = -R0*Ttot*at.exp(log_mu_)
                else:
                    sel_effect = -N*log_mu_
    
            else:
                # we passed multiple injections set corresponding to multiple observing runs
                # they need to be properly combined
                # This is useful only if using older LVK injection sets,
                # Deprecated after GWTC-3 

                
                print("Combining selection effects from different injections campaigns")

                spin_model_name = spin_model
                if use_sel_spin:

                    if spin_model == 'chieffchip' or spin_model == 'chieffchip_uc':
                        # shapes: chi1Inj, chi2Inj -> (ndata, ninj)
                        # result: spinsInj -> (ndata, 2, ninj)
                        spinsInj = at.stack([chi1Inj, chi2Inj], axis=1)
                    
                    elif (spin_model == 'default') or (spin_model == 'default_gauss'):
                        # shapes: chi1Inj, chi2Inj, cost1Inj, cost2Inj -> (ndata, ninj)
                        # result: spinsInj -> (ndata, 4, ninj)
                        spinsInj = at.stack([chi1Inj, chi2Inj, cost1Inj, cost2Inj], axis=1)

                else:
                    spinsInj = at.ones( (ndata, 2, ninj) )
                    print("Spin distribution will not be used in the sel effect")
                    spin_model_name = 'none'
                    
                    
                
                if not fix_inj_len:
                    print("Loop over injections sets, dynamical slicing")
                    # This should improve efficiency. But it can give problems with pytensor.scan (?)

                    res_i, _ = pytensor.scan( lambda idata, m1inj_, m2inj_, dLinj_, spinsInj_, lpdinj_, L,  Ndraw_, Ndet_ : sel_bias_with_uncertainty_at( m1inj_[idata, : Ndet_[idata]], m2inj_[idata, : Ndet_[idata]], dLinj_[idata, :Ndet_[idata]],  spinsInj_[idata, :, :Ndet_[idata]], lpdinj_[idata, :Ndet_[idata]], L, Ndraw_[idata], rate_model, mass_model, spin_model_name, smoothing, has_m2_break, interp=pade, dL_grid=dL_grid, z_grid=zgrid_ ), 
                                          sequences = [ np.arange( ndata) ], 
                                          non_sequences = [m1inj, m2inj, dLinj, spinsInj, lpdinj, Lambda_,  Ndraw, Ndet],
                                            profile=True
                                            )
                    log_mu_vec = res_i[0]
                    Neff_ = at.sum(res_i[1])

                    
                else:
                    print("Loop over injections sets, no slicing")
                    # makes it jax-compatible (jax does not support dynamical slicing at the moment)
                    # Not true anymore after pymc v5.10 ? Check
                    res_i, _ = pytensor.scan( lambda idata, m1inj_, m2inj_, dLinj_, spinsInj_, lpdinj_, L,  Ndraw_ : sel_bias_with_uncertainty_at( m1inj_[idata ], m2inj_[idata ], dLinj_[idata], spinsInj_[idata],  lpdinj_[idata], L, Ndraw_[idata], rate_model, mass_model, spin_model, smoothing, has_m2_break, interp=pade, dL_grid=dL_grid, z_grid=zgrid_ ), 
                                      sequences = [ np.arange( ndata) ], 
                                      non_sequences = [m1inj, m2inj, dLinj, spinsInj, lpdinj,  Lambda_,  Ndraw] )

            
                    log_mu_ = res_i[0]
                    Neff_ = at.sum(res_i[1])
    

                
    
                if not marginal_R0:
                    # Sum number of expected events in the two observing runs
                    # p_pop does not contain R_0*Tobs . Add it here
                    sel_effect = -at.sum(at.exp(log_mu_+lR0+at.log(Tobs)))
                else:
                    if sel_method=='Tobs':
                        sel_effect = -N*at.logsumexp( at.log(Tobs/Ttot)+log_mu_ )
                        print('Using sel function with weighted obs time average. Obs times: %s'%str(Tobs))
                    elif sel_method=='Nevs':
                        # This is technically wrong, but I leave it here
                        # to check how large the error is when using the wrong expression
                        print('Using sel function with number of events')
                        sel_effect = -at.sum(Nevs*log_mu_)

            
            ################################################
            # Sel effect computed. Now exclude high-variance regions in the integral

            
            Neff = pm.Deterministic('Neff', Neff_ )

            if marginal_R0:
                log_lik_var = pm.Deterministic('log_lik_var', at.exp(var_ll_u_+2*at.log(N)) )
            else:
                log_lik_var = pm.Deterministic('log_lik_var', at.exp(  var_ll_u_+2*at.log( R0*Ttot ) + 2*log_mu_ ) )
            
     

            if ((Neff_min==0) and (log_lik_var_min==0)):
                print("No condition on number of effective points in MC integral for sel. effect")
                selection_bias =  sel_effect #pm.Deterministic("sel_bias", sel_effect )
            else:
                if log_lik_var_min==0:

                    # Thresholding on N_eff
                    print("MC integral for sel. effect thresholded on N_eff")
                    
                    if sel_smoothing=='sigmoid':
                        # smooth with sigmoid between Neff_min and Neff_min+1 x Nobs
                        # over a scale = Neff_min
                        # i.e. at Neff_min * Nobs the likelihood becomes smoothly -inf
                        selection_bias = atools.log_sigmoid(Neff, Neff_min*(N+1),  Neff_min)+sel_effect  #pm.Deterministic("sel_bias", atools.log_sigmoid(Neff, Neff_min*(N+1),  Neff_min)+sel_effect )
                    elif sel_smoothing=='poly':
                        # Polynomial smoothing
                        selection_bias =  atools.log_f_smooth_poly(Neff, N/2,  Neff_min*N-N/4)+sel_effect #pm.Deterministic("sel_bias", atools.log_f_smooth_poly(Neff, N/2,  Neff_min*N-N/4)+sel_effect ) 
                    else:
                        # Hard cut
                        
                        selection_bias = sel_effect #pm.Deterministic("sel_bias", sel_effect)                   
                        #ind_sw_sel = pm.Deterministic('ind_sel', 1. * (Neff<Neff_min*N ) )
                        #ind_sel = pm.Bernoulli('bound_Neff', ind_sw_sel, observed=np.zeros(1)  )
                        _ = pm.Potential("bound_Neff", at.switch(Neff >= Neff_min * N, 0.0, -np.inf))

                
                elif Neff_min==0:

                    # Thresholding on likelihood variance
                    print("MC integral for sel. effect thresholded on log lik. variance")
                    
                    if sel_smoothing=='sigmoid':
                        # smooth with sigmoid 
                        print("Tapering sel effect with sigmoid smoothing")
                        
                        selection_bias = sel_effect + atools.logdiffexp( at.log(1), atools.log_sigmoid(log_lik_var, log_lik_var_min*(1+0.002), 0.001 )) 

                    elif sel_smoothing=='poly':
                        print("Tapering sel effect with polynomial smoothing")
                        selection_bias = sel_effect + atools.logdiffexp( at.log(1), atools.log_f_smooth_poly(log_lik_var, 0.01,  log_lik_var_min*(1-0.005) ))  
 
                    elif sel_smoothing=='softplus':
                        print("Tapering sel effect with softplus")
                        # Slack (how sharp the corner is) and weight (penalty strength)
                        nu = at.as_tensor_variable(0.001)     # smaller = sharper transition
                        lam = at.as_tensor_variable(1e3)     # larger = stronger penalty
                        
                        excess  = (log_lik_var - log_lik_var_min) / nu
                        penalty = lam * at.softplus(excess)          # ≥ 0, ~0 if below threshold

                        selection_bias = sel_effect 
                        
                        # If log_lik_var is a vector, sum to get a scalar penalty:
                        pm.Potential("bound_log_lik_var", -at.sum(penalty))
                    else:
                        print("Tapering sel effect with hard cut")

                        selection_bias = sel_effect #pm.Deterministic("sel_bias", sel_effect)
                        # ind_sw_sel = pm.Deterministic('ind_sel', 1. * (log_lik_var>log_lik_var_min ) )
                        # ind_sel = pm.Bernoulli('bound_log_lik_var', ind_sw_sel, observed=np.zeros(1)  )
                        _ = pm.Potential("bound_log_lik_var", at.switch(log_lik_var <= log_lik_var_min, 0.0, -np.inf))

            
            _ = pm.Potential('selection_bias', selection_bias)

            if marginal_R0:
                if include_sel_uncertainty:
                    
                    
                    # from Farr 2019
                    # print("Including selection function uncertainty as in Farr 2019")
                    #sel_uncertainty = (3*N+N**2)/(2*Neff)

                    # from heinzel-Vitale 2025
                    print("Including selection function uncertainty as in Heinzel-Vitale 2025")
                    sel_uncertainty = - N*(N+1)/(2) * var_ll_u_
                    
                    _ = pm.Potential('selection_uncertainty', sel_uncertainty)
            

    return model

