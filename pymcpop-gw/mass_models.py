from __future__ import annotations

import numpy as np
from pytensor_utils import attrapzvec, atcumtrapz, logsumexp2, logaddexp, logdiffexp, sigmoid, log_sigmoid, safe_sigmoid, atinterp_uniform, logsumexp
from constants import _PI as PI
from constants import max_m, _tgrid_np
from pytensor_utils import atinterp
from jax.scipy.special import logsumexp as _logsumexp

try:
    import jax.numpy as jnp
except Exception:
    jnp = None
    
try:
    #from jax_utils import make_interp_pt_like #make_interp_pt_cached_dy as _make_interp_pt_cached_dy
    #_JAX_INTERP_PT = make_interp_pt_like(eps=1e-12, side="right")
    #_JAX_INTERP_PT_MULT = make_interp_pt_like_multiY(eps=1e-12, side="right")
    import jax
except Exception as e:
    print(e)
    raise ValueError()
    #_JAX_INTERP_PT = None




# Precompute a few sizes you will use
_LEGGAUSS_NP = {
    8:  np.polynomial.legendre.leggauss(8),
    16: np.polynomial.legendre.leggauss(16),
    32: np.polynomial.legendre.leggauss(32),
    64: np.polynomial.legendre.leggauss(64),
}

def leggauss_const(n: int):
    t_np, w_np = _LEGGAUSS_NP[int(n)]  # pure dict lookup, no mutation
    # Convert to JAX arrays (constants captured by XLA)
    t = jnp.asarray(t_np, dtype=jnp.float64)
    w = jnp.asarray(w_np, dtype=jnp.float64)
    return t, w


def grid_diagnostics(name, x):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    dx = np.diff(x)
    print(f"{name}: N={x.size}")
    print(f"  min dx = {dx.min():.3e}")
    print(f"  1% dx  = {np.quantile(dx, 0.01):.3e}")
    print(f"  median = {np.median(dx):.3e}")
    print(f"  max dx = {dx.max():.3e}")
    print(f"  any non-increasing? {np.any(dx <= 0)}")
    # near-duplicates relative to eps
    print(f"  dx < 1e-12: {np.sum(dx < 1e-12)}")
    print(f"  dx < 1e-10: {np.sum(dx < 1e-10)}")
    
# ---------------------------------------------------------------------
# Gaussians
# ---------------------------------------------------------------------



def truncGausslowerupper_at_lpdf_safe(
    bk,
    x,
    loc,
    scale,
    xmin=0.0,
    xmax=1.0,
    eps_scale=1e-12,
    eps_Z=1e-300,
    truncate=False,
):
    scale_pos = bk.maximum(scale, eps_scale)

    za = (xmin - loc) / scale_pos
    zb = (xmax - loc) / scale_pos

    # these must exist somewhere (paste next if not)
    logPhia = _log_ndtr(bk, za)
    logPhib = _log_ndtr(bk, zb)

    hi = bk.maximum(logPhib, logPhia)
    lo = bk.minimum(logPhib, logPhia)

    logZ = logdiffexp(bk, hi, lo)
    logZ = bk.maximum(logZ, bk.log(eps_Z))

    z = (x - loc) / scale_pos
    logp = -bk.log(scale_pos) - 0.5 * bk.log(2.0 * PI) - 0.5 * z**2 - logZ

    if truncate:
        in_bounds = (x >= xmin) & (x <= xmax)
        return bk.where(in_bounds, logp, -np.inf)
    else:
        return logp


def _log_ndtr(bk, z):
    """
    log Phi(z) using erf. Good enough for most parameter ranges.
    (If you need extreme-tail stability, can swap in a better approximation.)
    """
    sqrt2 = bk.sqrt(2.0)
    Phi = 0.5 * (1.0 + bk.erf(z / sqrt2))
    # guard against log(0)
    Phi = bk.clip(Phi, 1e-300, np.inf)
    return bk.log(Phi)
    

# ---------------------------------------------------------------------
# Smoothings
# ---------------------------------------------------------------------

def logS_PLP(bk, m, deltam, ml, eps=1e-12):
    t = (m - ml) / bk.maximum(deltam, eps)
    t = bk.clip(t, eps, 1.0 - eps)
    S = t * t * (3.0 - 2.0 * t)
    return bk.log(bk.clip(S, eps, 1.0))


def logS_PLP_LVK(bk, m, deltam, ml):
    maskL = m <= ml
    maskU = m >= (ml + deltam)
    maskM = ~(maskL | maskU)

    s = bk.where(maskL, -np.inf, 0.0)

    mid = bk.log(1.0 / (1.0 + bk.exp(deltam / (m - ml) + deltam / (m - ml - deltam))))
    s1 = bk.where(maskM, mid, s)

    return s1


# ---------------------------------------------------------------------
# Secondary mass
# ---------------------------------------------------------------------


def logpdfm2_PLP_noreg(
    bk,
    m,
    beta,
    deltam,
    ml,
    *,
    m_g=45.0,
    w_g=80.0,
    sig_g_low=5.0,
    sig_g_high=5.0,
    has_m2_break=False,
    smoothing="LVK",
):
    if smoothing == "LVK":
        lS = logS_PLP_LVK(bk, m, deltam, ml)
    else:
        lS = logS_PLP(bk, m, deltam, ml)

    lpdfval = beta * bk.log(m) + lS

    if not has_m2_break:
        return lpdfval

    # Two edges as in your legacy code:
    left_edge  = 1.0 - safe_sigmoid(bk, m, m_g, sig_g_low)
    right_edge = safe_sigmoid(bk, m, m_g + w_g, sig_g_high)

    # Your original: mask = log(left_edge + right_edge)
    mask = bk.log(left_edge + right_edge)

    return mask + lpdfval


def logpdfm2_PLP_reg(
    bk,
    m,
    beta,
    deltam,
    ml,
    *,
    sig_l=0.05,
    m_g=45.0,
    w_g=80.0,
    sig_g_low=5.0,
    sig_g_high=5.0,
    has_m2_break=False,
    smoothing="LVK",
):
    return (
        logpdfm2_PLP_noreg(
            bk,
            m,
            beta,
            deltam,
            ml,
            m_g=m_g,
            w_g=w_g,
            sig_g_low=sig_g_low,
            sig_g_high=sig_g_high,
            has_m2_break=has_m2_break,
            smoothing=smoothing,
        )
        + log_sigmoid(bk, m, ml, sig_l)
    )



# ---------------------------------------------------------------------
# Double Power Law plus Double Peak (DPLDP)
# ---------------------------------------------------------------------


    
def logpdfm1_DPLDP(
    bk,
    m1,
    alpha1,
    alpha2,
    mb,
    mu1,
    sigma1,
    mu2,
    sigma2,
    m1_low,
    m_high,
    delta_m1,
    lambda0,
    lambda1,
    lambda2,
    epsilon,
    smoothing="LVK",
    simplex_repair=False,
    eps_w=1e-15,
    sl=0.1,
    sh=1.,
    norm_gauss="uplow",
):
    if not simplex_repair:
        log_lambda0 = bk.log(lambda0)
        log_lambda1 = bk.log(lambda1)
        log_lambda2 = bk.log(lambda2)
    else:
        # Simplex repair (keep your logic; just bk ops)
        lam0 = bk.clip(lambda0, eps_w, 1.0 - eps_w)
        lam1 = bk.clip(lambda1, eps_w, 1.0 - eps_w)
        lam2_raw = 1.0 - lam0 - lam1

        # bk.softplus exists? If not, paste a bk-softplus utility later.
        # For now assume you have it or will add it to backends.
        lam2 = eps_w + bk.log1p(bk.exp(lam2_raw - eps_w))

        denom = lam0 + lam1 + lam2
        lam0 = lam0 / denom
        lam1 = lam1 / denom
        lam2 = lam2 / denom

        log_lambda0 = bk.log(lam0)
        log_lambda1 = bk.log(lam1)
        log_lambda2 = bk.log(lam2)

    log_ppl = log_broken_power_law_DPLDP_pdf(
        bk, m1, alpha1, alpha2, mb, m1_low, m_high, epsilon=epsilon
    )

    if norm_gauss == "uplow":
        log_pnorm1 = truncGausslowerupper_at_lpdf_safe(bk, m1, mu1, sigma1, xmin=m1_low, xmax=m_high)
        log_pnorm2 = truncGausslowerupper_at_lpdf_safe(bk, m1, mu2, sigma2, xmin=m1_low, xmax=m_high)

    elif norm_gauss == "low-once":
        log_pnorm1 = truncGausslower_at_lpdf_safe(bk, m1, mu1, sigma1, xmin=m1_low)
        log_pnorm2 = -0.5 * ((m1 - mu2) / sigma2) ** 2 - bk.log(sigma2) - 0.5 * bk.log(2.0 * PI)

    elif norm_gauss == "none":
        log_pnorm1 = -0.5 * ((m1 - mu1) / sigma1) ** 2 - bk.log(sigma1) - 0.5 * bk.log(2.0 * PI)
        log_pnorm2 = -0.5 * ((m1 - mu2) / sigma2) ** 2 - bk.log(sigma2) - 0.5 * bk.log(2.0 * PI)

    else:
        raise ValueError("norm_gauss can be uplow, low-once, or none (others not implemented here)")

    if smoothing == "LVK":
        log_S = logS_PLP_LVK(bk, m1, delta_m1, m1_low)
    else:
        log_S = logS_PLP(bk, m1, delta_m1, m1_low)

    term0 = log_lambda0 + log_ppl
    term1 = log_lambda1 + log_pnorm1
    term2 = log_lambda2 + log_pnorm2

    log_mix = logsumexp2(bk, logsumexp2(bk, term0, term1), term2)

    # gate: log(sigmoid_low) + log(1 - sigmoid_high)
    log_gate = log_sigmoid(bk, m1, m1_low, sl) + bk.log1p(-safe_sigmoid(bk, m1, m_high, sh))

    return log_S + log_mix + log_gate


def log_broken_power_law_DPLDP_pdf(
    bk,
    m1,
    alpha1,
    alpha2,
    mb,
    m1_low,
    m_high,
    sh=1.,
    sl=0.1,
    epsilon=0.1,
    eps=1e-12,
    eps_w=1e-12,
    t_floor=1e-12,
):
    mb_pos = bk.maximum(mb, eps)
    m1_pos = bk.maximum(m1, eps)

    log_N = log_broken_pl_norm_DPLDP(bk, alpha1, alpha2, mb_pos, m1_low, m_high, eps=eps, t_floor=t_floor)

    log_m1_over_mb = bk.log(m1_pos / mb_pos)
    log_val1 = -alpha1 * log_m1_over_mb
    log_val2 = -alpha2 * log_m1_over_mb

    w = safe_sigmoid(bk, -m1_pos, -mb_pos, epsilon)
    w = bk.clip(w, eps_w, 1.0 - eps_w)

    log_w = bk.log(w)
    log_1mw = bk.log1p(-w)

    log_mix_val = logaddexp(bk, log_w + log_val1, log_1mw + log_val2)
    return log_mix_val - log_N


def log_broken_pl_norm_DPLDP(bk, alpha1, alpha2, mb, m1_low, m_high, eps=1e-12, t_floor=1e-12):
    mb_pos = bk.maximum(mb, eps)

    u_low = bk.maximum(m1_low / mb_pos, eps)
    u_high = bk.maximum(m_high / mb_pos, u_low * (1.0 + 1e-12))

    one = 1.0

    # NOTE: bk.switch not in our minimal backend yet; use bk.where with scalars/tensors.
    logI1 = bk.where(
        u_low < one,
        log_norm_truncated_pl_num_alpha1_safe(bk, alpha1, u_low, one, eps=eps, t_floor=t_floor),
        -np.inf,
    )
    logI2 = bk.where(
        u_high > one,
        log_norm_truncated_pl_num_alpha1_safe(bk, alpha2, one, u_high, eps=eps, t_floor=t_floor),
        -np.inf,
    )

    return bk.log(mb_pos) + logaddexp(bk, logI1, logI2)


def log_norm_truncated_pl_num_alpha1_safe(bk, alpha, mmin, mmax, eps=1e-12, t_floor=1e-12):
    mmin_c = bk.clip(mmin, eps, np.inf)
    mmax_c = bk.clip(mmax, eps, np.inf)
    mmax_c = bk.maximum(mmax_c, mmin_c * (1.0 + 1e-12))

    t = 1.0 - alpha
    # avoid exactly zero
    t_safe = bk.where(bk.abs(t) < t_floor, bk.where(t >= 0.0, t_floor, -t_floor), t)

    b = bk.log(mmin_c)
    delta = bk.log(mmax_c) - b

    # need expm1; if backend doesn’t expose expm1, we can do exp(x)-1 safely later.
    # For now assume your backends have expm1 or you’ll add it.
    return t_safe * b + bk.log(bk.abs(bk.expm1(t_safe * delta))) - bk.log(bk.abs(t_safe))




def logpdf_DPLDP(
    bk,
    theta,
    lambdaBBHmass,
    force_m2_less_than_m1=False,
    has_m2_break=False,
    smoothing="LVK",
    resC=100,
    resN=500,
    interp_vals=None,
    interp_grids=None,
    norm=True,
    simplex_repair=False,
    norm_gauss="uplow",
):
    m1, m2 = theta
    (
        alpha1,
        alpha2,
        mb,
        mu1,
        sigma1,
        mu2,
        sigma2,
        m1_low,
        m_high,
        delta_m1,
        lambda0,
        lambda1,
        lambda2,
        beta,
        m2_low,
        delta_m2,
        epsilon,
        m_g,
        w_g,
        sig_g_low,
        sig_g_high,
    ) = lambdaBBHmass

    lpdfm1 = logpdfm1_DPLDP(
        bk,
        m1,
        alpha1,
        alpha2,
        mb,
        mu1,
        sigma1,
        mu2,
        sigma2,
        m1_low,
        m_high,
        delta_m1,
        lambda0,
        lambda1,
        lambda2,
        epsilon,
        smoothing=smoothing,
        simplex_repair=simplex_repair,
        norm_gauss=norm_gauss,
    )

    lpdfm2 = logpdfm2_PLP_reg(
        bk,
        m2,
        beta,
        delta_m2,
        m2_low,
        m_g=m_g,
        w_g=w_g,
        sig_g_low=sig_g_low,
        sig_g_high=sig_g_high,
        has_m2_break=has_m2_break,
        smoothing=smoothing,
    )

    lC = logC_DPLDP(
        bk,
        m1,
        beta,
        delta_m2,
        m2_low,
        m_g=m_g,
        w_g=w_g,
        sig_g_low=sig_g_low,
        sig_g_high=sig_g_high,
        has_m2_break=has_m2_break,
        res=resC,
        smoothing=smoothing,
    )

    if norm:
        # ln = logNorm_DPLDP(
        #     bk,
        #     alpha1,
        #     alpha2,
        #     mb,
        #     mu1,
        #     sigma1,
        #     mu2,
        #     sigma2,
        #     m1_low,
        #     m_high,
        #     delta_m1,
        #     lambda0,
        #     lambda1,
        #     lambda2,
        #     epsilon,
        #     smoothing=smoothing,
        #     res=resN,
        #     simplex_repair=simplex_repair,
        #     norm_gauss=norm_gauss,
        #)
 

        ln  = logNorm_m1_DPLDP_GL(
                bk,
                alpha1, alpha2, mb,
                mu1, sigma1, mu2, sigma2,
                m1_low, m_high, delta_m1,
                lambda0, lambda1, lambda2,
                epsilon,
                smoothing=smoothing,
                simplex_repair=simplex_repair,
                #sl=sl, sh=sh,
                norm_gauss=norm_gauss,
                n_gl=16,   
            )
        
    else:
        ln = 0.0

    lpdf = lpdfm1 + lpdfm2 - lC - ln

    if force_m2_less_than_m1:
        ok = bk.all(bk.asarray([(m2 <= m1), (m2 > 0.0), (m1 > 0.0)]), axis=0) if hasattr(bk, "all") else ((m2 <= m1) & (m2 > 0.0) & (m1 > 0.0))
        # (above keeps numpy happy; for ATBackend, & works fine too)
        return bk.where(ok, lpdf, -np.inf)
    else:
        return lpdf


def logC_DPLDP(
    bk,
    m,
    beta,
    deltam,
    m2_low,
    m_g=45,
    w_g=80,
    sig_g_low=5,
    sig_g_high=5,
    has_m2_break=False,
    res=500,
    smoothing="LVK",
):
    if res != 500:
        _tgrid = bk.linspace(0.0, 1.0, res)
    else:
        _tgrid = _tgrid_np #_get_t_grid()  # keep your existing helper

    xx = m2_low + (max_m - m2_low) * _tgrid  # max_m must exist (as before)

    l2 = logpdfm2_PLP_reg(
        bk,
        xx,
        beta,
        deltam,
        m2_low,
        m_g=m_g,
        w_g=w_g,
        sig_g_low=sig_g_low,
        sig_g_high=sig_g_high,
        has_m2_break=has_m2_break,
        smoothing=smoothing,
    )

    a = bk.max(l2)
    p2 = bk.exp(l2 - a)

    # legacy: cdf = atcumtrapz(p2, xx)
    cdf = atcumtrapz(bk, p2, xx) #bk.stop_grad(xx))  
    cdf = bk.clip(cdf, 1e-300, np.inf)

    x0 = xx[1]
    x1 = xx[-1]
    nU = xx.shape[0] - 1

    # log(cdf_scaled) + a gives log(cdf_original)
    itr = atinterp_uniform(bk, m, x0, x1, nU, bk.log(cdf) + a)
    return itr


# ---- Fast replacements for logNorm_DPLDP and logC_DPLDP using GL ----

def logNorm_m1_DPLDP_GL(
    bk,
    alpha1,
    alpha2,
    mb,
    mu1,
    sigma1,
    mu2,
    sigma2,
    m1_low,
    m_high,
    delta_m1,
    lambda0,
    lambda1,
    lambda2,
    epsilon,
    *,
    smoothing="LVK",
    simplex_repair=False,
    eps_w=1e-15,
    sl=0.1,
    sh=1,
    norm_gauss="uplow",
    n_gl=32,
):
    """
    logZ = log ∫_{m1_low}^{m_high} exp(logpdfm1_DPLDP(..., m1)) dm1
    computed by Gauss–Legendre on a small number of intervals.

    This keeps your EXACT model (soft break epsilon + gates + smoothing choice).
    """
    #m1_low = jnp.asarray(m1_low, dtype=jnp.float64)
    #m_high = jnp.asarray(m_high, dtype=jnp.float64)
    #mb     = jnp.asarray(mb, dtype=jnp.float64)

    # fixed split point for taper end
    m_taper_end = jnp.minimum(m1_low + delta_m1, m_high)

    # clamp break into [m_taper_end, m_high] so segments are ordered
    mb_c = jnp.clip(mb, m_taper_end, m_high)

    def _logf(x):
        return logpdfm1_DPLDP(
            bk,
            x,
            alpha1, alpha2, mb,
            mu1, sigma1, mu2, sigma2,
            m1_low, m_high, delta_m1,
            lambda0, lambda1, lambda2,
            epsilon,
            smoothing=smoothing,
            simplex_repair=simplex_repair,
            eps_w=eps_w,
            sl=sl,
            sh=sh,
            norm_gauss=norm_gauss,
        )

    # Segment 1: [m1_low, m_taper_end]
    logI1 = _log_integral_gl_1d(_logf, m1_low, m_taper_end, n=n_gl)

    # Segment 2: [m_taper_end, mb_c]
    logI2 = _log_integral_gl_1d(_logf, m_taper_end, mb_c, n=n_gl)

    # Segment 3: [mb_c, m_high]
    logI3 = _log_integral_gl_1d(_logf, mb_c, m_high, n=n_gl)

    # Total = log(exp(logI1)+exp(logI2)+exp(logI3))
    logZ = _logaddexp(_logaddexp(logI1, logI2), logI3)
    return logZ



def _log_integral_gl_1d(logf, a, b, *, n=32):
    """
    Compute log ∫_a^b exp(logf(x)) dx using n-point Gauss–Legendre.
    Works with JAX tracers for a,b.

    logf: callable x -> log density (broadcastable over x)
    a,b: scalars (can be JAX tracers)
    """
    #a = jnp.asarray(a, dtype=jnp.float64)
    #b = jnp.asarray(b, dtype=jnp.float64)

    # ensure ordering
    a0 = jnp.minimum(a, b)
    b0 = jnp.maximum(a, b)

    # if interval is empty, return -inf
    length = b0 - a0
    empty = length <= 0.0

    t, w = leggauss_const(int(n))  # constants

    # map nodes to [a0, b0]
    mid  = 0.5 * (a0 + b0)
    half = 0.5 * (b0 - a0)
    x = mid + half * t

    # log-integrand at nodes
    lv = logf(x)

    # include weights + Jacobian
    logw = jnp.log(w) + jnp.log(half)
    out = _logsumexp(lv + logw)

    return jnp.where(empty, -jnp.inf, out)


def _logaddexp(a, b):
    return jnp.logaddexp(a, b)


def logNorm_DPLDP(
    bk,
    alpha1,
    alpha2,
    mb,
    mu1,
    sigma1,
    mu2,
    sigma2,
    m1_low,
    m_high,
    delta_m1,
    lambda0,
    lambda1,
    lambda2,
    epsilon,
    res=500,
    smoothing="LVK",
    simplex_repair=False,
    eps_int=1e-300,
    norm_gauss="uplow",
):
    """
    Overflow-safe log normalization:
      log ∫ exp(logpdfm1_DPLDP(ms)) dms
    using max-subtraction.
    """
    if res != 500:
        _tgrid = bk.linspace(0.0, 1.0, res)
    else:
        _tgrid = _tgrid_np #_get_t_grid()

    ms = m1_low + (m_high - m1_low) * _tgrid

    lpdf = logpdfm1_DPLDP(
        bk,
        ms,
        alpha1,
        alpha2,
        mb,
        mu1,
        sigma1,
        mu2,
        sigma2,
        m1_low,
        m_high,
        delta_m1,
        lambda0,
        lambda1,
        lambda2,
        epsilon,
        smoothing=smoothing,
        simplex_repair=simplex_repair,
        norm_gauss=norm_gauss,
    )

    a = bk.max(lpdf)
    ps = bk.exp(lpdf - a)
    integ = attrapzvec(bk, ps, ms) #bk.stop_grad(ms))  # your backend-agnostic wrapper
    integ = bk.clip(integ, eps_int, np.inf)

    return a + bk.log(integ)





def precompute_DPLDP_mass_interp(
    bk,
    m1_grid,
    m2_grid,
    lambdaBBHmass,
    *,
    smoothing="LVK",
    simplex_repair=False,
    has_m2_break=False,
    norm_gauss="uplow",
    eps_cdf=1e-300,
    eps_interp=1e-30,
    side_interp="right",
):
    """
    Precompute 1D mass tables ONCE, then reuse via interpolation:

      interp_vals_mass  = (lp_m1_grid, lp_m2_grid, lC_of_m1_grid, ln_m1)
      interp_grids_mass = (m1_grid, m2_grid)

    Notes:
      - m1_grid, m2_grid are treated as fixed geometry (typically stop_grad upstream).
      - Gradients flow through mass parameters via the computed *values* on the grids.
      - lC(m1) is built from the CDF of p(m2) integrated on m2_grid.
    """
    (
        alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2,
        m1_low, m_high, delta_m1,
        lambda0, lambda1, lambda2,
        beta, m2_low, delta_m2,
        epsilon, m_g, w_g, sig_g_low, sig_g_high
    ) = lambdaBBHmass

    # log p(m1) on m1_grid
    lp_m1_grid = logpdfm1_DPLDP(
        bk,
        m1_grid,
        alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2,
        m1_low, m_high, delta_m1,
        lambda0, lambda1, lambda2,
        epsilon,
        smoothing=smoothing,
        simplex_repair=simplex_repair,
        norm_gauss=norm_gauss,
    )

    # log p(m2) on m2_grid
    lp_m2_grid = logpdfm2_PLP_reg(
        bk,
        m2_grid,
        beta,
        delta_m2,
        m2_low,
        m_g=m_g,
        w_g=w_g,
        sig_g_low=sig_g_low,
        sig_g_high=sig_g_high,
        has_m2_break=has_m2_break,
        smoothing=smoothing,
    )

    # ---- CDF over m2 (on m2_grid[1:]) ----
    # Use max-shift for stability
    a2 = bk.max(lp_m2_grid)
    p2 = bk.exp(lp_m2_grid - a2)

    cdf_m2 = atcumtrapz(bk, p2, m2_grid) #) bk.stop_grad(m2_grid))
    cdf_m2 = bk.clip(cdf_m2, eps_cdf, np.inf)

    m2_cdf_grid = m2_grid[1:]
    logcdf_m2 = bk.log(cdf_m2) + a2

    # Evaluate log C(m1) on m1_grid by interpolating logcdf(m2) at m2=m1
    mcap = bk.clip(m1_grid, m2_cdf_grid[0], m2_cdf_grid[-1])

    use_jax = (
        #_JAX_INTERP_PT is not None
         jnp is not None
        and type(mcap).__module__.startswith("jax")
    )

    if use_jax:
        lC_of_m1 = atinterp(bk, mcap, m2_cdf_grid, logcdf_m2) #jax.lax.stop_gradient(m2_cdf_grid)
    else:
        raise ValueError()
        # backend-agnostic fallback (same semantics as your sel_bias interpolation helpers)
        idx = np.searchsorted(np.asarray(m2_cdf_grid), np.asarray(mcap), side=side_interp)
        idx = np.clip(idx, 1, np.asarray(m2_cdf_grid).shape[0] - 1)
        x0 = m2_cdf_grid[idx - 1]
        x1 = m2_cdf_grid[idx]
        y0 = logcdf_m2[idx - 1]
        y1 = logcdf_m2[idx]
        denom = bk.maximum(x1 - x0, eps_interp)
        r = (mcap - x0) / denom
        lC_of_m1 = (1.0 - r) * y0 + r * y1

    # ---- normalization ln = log ∫ exp(lp_m1) dm1 ----
    lp_max = bk.max(lp_m1_grid)
    p_shift = bk.exp(lp_m1_grid - lp_max)
    I = attrapzvec(bk, p_shift, m1_grid) #bk.stop_grad(m1_grid))
    I = bk.clip(I, 1e-300, np.inf)
    ln_m1 = bk.log(I) + lp_max

    return (lp_m1_grid, lp_m2_grid, lC_of_m1, ln_m1)



def build_m1_grid_DPLDP_bk(
    bk,
    # --- hyperparameters (defaults are plausible; override in your model) ---
    alpha1=2.0, alpha2=4.0, mb=35.0,
    mu1=10.0, sigma1=3.0, mu2=35.0, sigma2=6.0,
    m1_low=3.0, m_high=300.0,
    delta_m1=9.0,
    # --- resolution controls ---
    n_peak=2500,
    n_tail_low=400,
    n_tail_high=400,
    frac_gauss1=0.2,
    frac_gauss2=0.2,
    k_sigma_gauss=3.0,
    k_sigma_band=4.0,
    n_taper=10,
    n_taper_eff=200,  # kept for API parity (not used here)
    # --- numerics / validation ---
    eps=1e-4,
    ramp_step=1e-6,
    validate=False,   # only validates when result is a NumPy array
):
    """
    Backend-agnostic version of your PyTensor build_m1_grid_DPLDP.

    Same structure/semantics:
      - stop_grad through grid geometry
      - fixed-length tails
      - gaussian windows with fallbacks if degenerate
      - combine -> clip -> sort -> tiny ramp for strict monotonicity

    Notes:
      - alpha1/alpha2 are accepted for signature parity but not used in geometry.
      - validate=True only checks if the output is a NumPy ndarray.
    """

    # ---- detach hyperparameters for grid geometry (no grad through geometry) ----
    mb_sg       = bk.stop_grad(mb)
    mu1_sg      = bk.stop_grad(mu1)
    sigma1_sg   = bk.stop_grad(sigma1)
    mu2_sg      = bk.stop_grad(mu2)
    sigma2_sg   = bk.stop_grad(sigma2)
    m1_low_sg   = bk.stop_grad(m1_low)
    m_high_sg   = bk.stop_grad(m_high)
    delta_m1_sg = bk.stop_grad(delta_m1)

    # gentle boundary offset (avoid exact endpoints)
    xmin = m1_low_sg + eps
    xmax = m_high_sg - eps
    span = bk.maximum(xmax - xmin, 1e-6)

    # ------------------------------------------------------------
    # 0) Taper grid: clustered near xmin
    # ------------------------------------------------------------
    taper_hi = bk.clip(xmin + bk.maximum(delta_m1_sg, 1e-6), xmin, xmax)
    taper_w  = bk.maximum(taper_hi - xmin, 1e-6)

    if n_taper > 1:
        eps_t = 1e-4
        u = bk.linspace(0.0, 1.0, int(n_taper))  # [0,1]
        t = bk.exp(bk.log(eps_t) * (1.0 - u))    # eps_t -> 1
        t = (t - eps_t) / (1.0 - eps_t)          # -> [0,1]
        m1_taper = xmin + taper_w * t
    else:
        m1_taper = bk.zeros((0,))

    # ------------------------------------------------------------
    # 1) Gaussian windows (clip to support)
    # ------------------------------------------------------------
    k_g = k_sigma_gauss
    k_b = k_sigma_band  # kept for parity; band envelope uses raw mins/maxs below

    g1_min_raw = mu1_sg - k_g * bk.abs(sigma1_sg)
    g1_max_raw = mu1_sg + k_g * bk.abs(sigma1_sg)
    g2_min_raw = mu2_sg - k_g * bk.abs(sigma2_sg)
    g2_max_raw = mu2_sg + k_g * bk.abs(sigma2_sg)

    g1_min = bk.clip(g1_min_raw, xmin, xmax)
    g1_max = bk.clip(g1_max_raw, xmin, xmax)
    g2_min = bk.clip(g2_min_raw, xmin, xmax)
    g2_max = bk.clip(g2_max_raw, xmin, xmax)

    tiny = 1e-6 * span
    g1_width = g1_max - g1_min
    g2_width = g2_max - g2_min

    has_g1 = bk.gt(g1_width, tiny)
    has_g2 = bk.gt(g2_width, tiny)

    # ------------------------------------------------------------
    # 2) Envelope "interesting band" over peaks + mb
    # ------------------------------------------------------------
    peak_min_raw = bk.minimum(g1_min_raw, g2_min_raw)
    peak_min_raw = bk.minimum(peak_min_raw, mb_sg)

    peak_max_raw = bk.maximum(g1_max_raw, g2_max_raw)
    peak_max_raw = bk.maximum(peak_max_raw, mb_sg)

    band_min = bk.clip(peak_min_raw, xmin, xmax)
    band_max = bk.clip(peak_max_raw, xmin, xmax)

    band_width = bk.maximum(band_max - band_min, tiny)

    # ------------------------------------------------------------
    # 3) Split n_peak between Gaussians + mid band (Python ints)
    # ------------------------------------------------------------
    n_g1 = int(n_peak * frac_gauss1)
    n_g2 = int(n_peak * frac_gauss2)
    if n_g1 < 0: n_g1 = 0
    if n_g2 < 0: n_g2 = 0
    if n_g1 + n_g2 > n_peak and (n_g1 + n_g2) > 0:
        scale = n_peak / n_g1 + n_g2
        n_g1 = int(round(n_g1 * scale))
        n_g2 = int(round(n_g2 * scale))
    n_mid = max(n_peak - n_g1 - n_g2, 0)

    # ------------------------------------------------------------
    # 4) Low tail: start AFTER taper, fixed length
    # ------------------------------------------------------------
    if n_tail_low > 0:
        denom_low = n_tail_low + 1
        t_low = (bk.arange(int(n_tail_low)) + 1.0) / denom_low  # in (0,1)

        low_start = taper_hi
        low_width = band_min - low_start

        fallback_w = bk.maximum(taper_w / bk.maximum(n_taper, 1), 1e-3)

        tail_good = low_start + low_width * t_low
        tail_fallback = low_start + fallback_w * t_low

        m1_low_tail = bk.switch(bk.gt(low_width, 0), tail_good, tail_fallback)
    else:
        m1_low_tail = bk.zeros((0,))

    # ------------------------------------------------------------
    # 5) Gaussian 1 segment (fallback if degenerate)
    # ------------------------------------------------------------
    if n_g1 > 0:
        denom_g1 = max(n_g1 - 1, 1)
        t_g1 = bk.arange(int(n_g1)) / denom_g1
        m1_g1 = g1_min + g1_width * t_g1

        fallback_width = 1e-8 * span
        g1_center = 0.5 * (g1_min + g1_max)
        g1_center = bk.clip(g1_center, xmin + fallback_width, xmax - fallback_width)
        fallback_g1 = g1_center + fallback_width * (t_g1 - 0.5)

        m1_g1 = bk.switch(has_g1, m1_g1, fallback_g1)
    else:
        m1_g1 = bk.zeros((0,))

    # ------------------------------------------------------------
    # 6) Gaussian 2 segment (fallback if degenerate)
    # ------------------------------------------------------------
    if n_g2 > 0:
        denom_g2 = max(n_g2 - 1, 1)
        t_g2 = bk.arange(int(n_g2)) / denom_g2
        m1_g2 = g2_min + g2_width * t_g2

        fallback_width = 1e-8 * span
        g2_center = 0.5 * (g2_min + g2_max)
        g2_center = bk.clip(g2_center, xmin + fallback_width, xmax - fallback_width)
        fallback_g2 = g2_center + fallback_width * (t_g2 - 0.5)

        m1_g2 = bk.switch(has_g2, m1_g2, fallback_g2)
    else:
        m1_g2 = bk.zeros((0,))

    # ------------------------------------------------------------
    # 7) Mid band segment
    # ------------------------------------------------------------
    if n_mid > 0:
        denom_mid = max(n_mid - 1, 1)
        t_mid = bk.arange(int(n_mid)) / denom_mid
        m1_mid = band_min + band_width * t_mid
    else:
        m1_mid = bk.zeros((0,))

    # ------------------------------------------------------------
    # 8) High tail: endpoint excluded (avoid exact xmax)
    # ------------------------------------------------------------
    if n_tail_high > 0:
        denom_high = max(n_tail_high, 1)  # not (n_tail_high - 1)
        t_high = bk.arange(int(n_tail_high)) / denom_high  # in [0,1)
        m1_high_tail = band_max + (xmax - band_max) * t_high
    else:
        m1_high_tail = bk.zeros((0,))

    # ------------------------------------------------------------
    # Combine -> clip -> sort -> tiny ramp
    # ------------------------------------------------------------
    m1_grid_raw = bk.concatenate(
        [m1_taper, m1_low_tail, m1_g1, m1_g2, m1_mid, m1_high_tail],
        axis=0,
    )

    m1_grid_clipped = bk.clip(m1_grid_raw, xmin, xmax)
    m1_grid_sorted = bk.sort(m1_grid_clipped)

    ramp = ramp_step * bk.arange(m1_grid_sorted.shape[0], dtype=getattr(m1_grid_sorted, "dtype", None))
    m1_grid_strict = bk.clip(m1_grid_sorted + ramp, xmin, xmax)

    # Optional validation (only for NumPy arrays)
    if validate:
        import numpy as _np
        if isinstance(m1_grid_strict, _np.ndarray):
            if not _np.all(_np.isfinite(m1_grid_strict)):
                raise ValueError("m1_grid contains non-finite values")
            if _np.any(_np.diff(m1_grid_strict) <= 0):
                raise ValueError("m1_grid is not strictly increasing")
            # “unique-ish” check
            tol = 10.0 * _np.finfo(m1_grid_strict.dtype).eps * max(1.0, _np.max(m1_grid_strict - _np.min(m1_grid_strict)))
            if _np.any(_np.diff(m1_grid_strict) <= tol):
                raise ValueError("m1_grid has (near-)duplicates")

    return m1_grid_strict



def build_m2_grid_bk(
    bk,
    *,
    m2_low=3.0,
    m2_high=300.0,
    delta_m2=8.0,
    # resolution controls
    n_total=500,
    n_taper=100,
    # numerics
    eps_m=1e-5,
    eps_t=1e-4,
    ramp_step=1e-12,
    validate=False,  # only validates when result is a NumPy array
):
    """
    Backend-agnostic translation of your PyTensor m2 grid:

      m2_lo        = m2_low + eps_m
      m2_taper_hi  = m2_lo + max(delta_m2, 1e-6)

      seg1: log-ramp clustered near m2_lo over [m2_lo, m2_taper_hi] with n_taper points
      seg2: linear from m2_taper_hi to m2_high with (n_total - n_taper) points

      grid = concat(seg1[:-1], seg2)

    Guarantees (as in your PT version + tiny ramp):
      - inside (m2_low, m2_high] up to eps_m shift
      - non-decreasing then forced strictly increasing with tiny ramp
    """

    n_total = int(n_total)
    n_taper = int(n_taper)
    if n_total < 2:
        raise ValueError("n_total must be >= 2")
    if n_taper < 2:
        raise ValueError("n_taper must be >= 2")
    if n_taper >= n_total:
        raise ValueError("n_taper must be < n_total")

    # detach geometry if m2_low/delta_m2 are symbolic in a backend that supports stop_grad
    m2_low_sg   = bk.stop_grad(m2_low)
    delta_m2_sg = bk.stop_grad(delta_m2)

    m2_lo = m2_low_sg + eps_m
    m2_taper_hi = m2_lo + bk.maximum(delta_m2_sg, 1e-6)

    m2_taper_hi = bk.minimum(m2_taper_hi, m2_high - eps_m)
    m2_taper_hi = bk.maximum(m2_taper_hi, m2_lo + 1e-6)

    # seg1: clustered in [m2_lo, m2_taper_hi]
    u1 = bk.linspace(0.0, 1.0, n_taper)
    t = bk.exp(bk.log(eps_t) * (1.0 - u1))      # eps_t -> 1
    t = (t - eps_t) / (1.0 - eps_t)      # -> [0,1]
    seg1 = m2_lo + (m2_taper_hi - m2_lo) * t

    # seg2: linear to m2_high
    n2 = n_total - n_taper
    u2 = bk.linspace(0.0, 1.0, n2)
    seg2 = m2_taper_hi + ( m2_high - m2_taper_hi) * u2

    # concat without duplicating the join point
    # (backend-agnostic slice: use Python slicing on the tensor)
    grid = bk.concatenate([seg1[:-1], seg2], axis=0)

    # enforce monotonic + strict (tiny ramp)
    # if your backend has cummax / maximum_accumulate, use it; else rely on ramp only
    try:
        grid = bk.maximum_accumulate(grid)
    except Exception:
        pass

    ramp = ramp_step * bk.arange(grid.shape[0], dtype=getattr(grid, "dtype", None))
    grid = grid + ramp

    # Optional validation (only for NumPy arrays)
    if validate:
        import numpy as _np
        if isinstance(grid, _np.ndarray):
            if not _np.all(_np.isfinite(grid)):
                raise ValueError("m2_grid contains non-finite values")
            if _np.any(_np.diff(grid) <= 0):
                raise ValueError("m2_grid is not strictly increasing")

    return grid


    
# def build_m1_grid_DPLDP_np(
#     *,
#     m1_low: float = 3.0,
#     m1_high: float = 350.0,
#     eps: float = 1e-4,
#     dtype=np.float64,
#     # ---- total points control ----
#     n_total: int = 1200,
#     # ---- segment boundaries (tweakable) ----
#     rise_hi: float = 10.0,          # low-mass rise capture (m1_low..~10)
#     peak1_lo: float = 5.0,          # peak around ~10 window
#     peak1_hi: float = 15.0,
#     bg_lo: float = 10.0,            # background PL (10..break)
#     bg_hi: float = 45.0,
#     break_lo: float = 25.0,         # break / peak2 window (~30-40)
#     break_hi: float = 50.0,
#     tail_lo: float = 45.0,          # tail start
#     tail_hi: float | None = None,   # defaults to m1_high
#     # ---- how to split n_total across segments (normalized) ----
#     # You can pass your own weights; they will be normalized.
#     weights: tuple[float, float, float, float, float] = (0.18, 0.20, 0.14, 0.28, 0.20),
#     # ---- spacing controls ----
#     rise_cluster: bool = True,
#     rise_eps_t: float = 1e-4,       # smaller => more clustering near m1_low
#     tail_cluster: bool = True,
#     tail_eps_t: float = 2e-3,       # smaller => more clustering near tail_lo
# ) -> np.ndarray:
#     """
#     Fixed non-uniform m1 grid with a single knob n_total.

#     Segments (in order):
#       1) rise:   [m1_low, rise_hi] (optionally clustered near m1_low)
#       2) peak1:  [peak1_lo, peak1_hi]
#       3) bg:     [bg_lo, bg_hi]
#       4) break:  [break_lo, break_hi]
#       5) tail:   [tail_lo, tail_hi] (optionally clustered near tail_lo)

#     n_total is allocated across segments via 'weights' (normalized).
#     """
#     m1_low = float(m1_low)
#     m1_high = float(m1_high)
#     if m1_high <= m1_low:
#         raise ValueError("m1_high must be > m1_low")

#     n_total = int(n_total)
#     if n_total < 10:
#         raise ValueError("n_total must be >= 10 (practically)")

#     xmin = m1_low + float(eps)
#     xmax = m1_high - float(eps)
#     if xmax <= xmin:
#         raise ValueError("m1_high - m1_low too small after eps")

#     if tail_hi is None:
#         tail_hi = xmax
#     else:
#         tail_hi = min(float(tail_hi), xmax)

#     # ---- allocate point counts ----
#     if len(weights) != 5:
#         raise ValueError("weights must have length 5 (rise, peak1, bg, break, tail)")
#     w = np.asarray(weights, dtype=np.float64)
#     if np.any(w < 0):
#         raise ValueError("weights must be non-negative")
#     s = float(w.sum())
#     if s <= 0:
#         raise ValueError("weights must sum to > 0")
#     w = w / s

#     raw = w * n_total
#     n = np.floor(raw).astype(int)
#     # distribute remainder by largest fractional parts
#     rem = n_total - int(n.sum())
#     if rem > 0:
#         frac = raw - np.floor(raw)
#         order = np.argsort(-frac)
#         n[order[:rem]] += 1

#     # guarantee at least 2 points in segments that are valid
#     # (we’ll later drop invalid segments automatically)
#     n = np.maximum(n, 2)

#     n_rise, n_peak1, n_bg, n_break, n_tail = map(int, n.tolist())

#     # ---- helpers ----
#     def _seg(lo, hi):
#         lo = max(float(lo), xmin)
#         hi = min(float(hi), xmax)
#         if hi <= lo:
#             return None
#         return lo, hi

#     def _lin(lo, hi, npts):
#         npts = int(npts)
#         if npts <= 0:
#             return np.zeros((0,), dtype=dtype)
#         if npts == 1:
#             return np.array([(lo + hi) * 0.5], dtype=dtype)
#         return np.linspace(lo, hi, npts, dtype=dtype)

#     def _log_ramp(lo, hi, npts, eps_t):
#         npts = int(npts)
#         if npts <= 0:
#             return np.zeros((0,), dtype=dtype)
#         if npts == 1:
#             return np.array([(lo + hi) * 0.5], dtype=dtype)
#         u = np.linspace(0.0, 1.0, npts, dtype=dtype)
#         eps_t = float(eps_t)
#         t = np.exp(np.log(eps_t) * (1.0 - u))  # eps_t -> 1
#         t = (t - eps_t) / (1.0 - eps_t)        # -> [0,1]
#         return (lo + (hi - lo) * t).astype(dtype, copy=False)

#     # ---- build segments ----
#     segs = []

#     s = _seg(xmin, rise_hi)
#     if s is not None:
#         segs.append(_log_ramp(s[0], s[1], n_rise, rise_eps_t) if rise_cluster else _lin(s[0], s[1], n_rise))

#     s = _seg(peak1_lo, peak1_hi)
#     if s is not None:
#         segs.append(_lin(s[0], s[1], n_peak1))

#     s = _seg(bg_lo, bg_hi)
#     if s is not None:
#         segs.append(_lin(s[0], s[1], n_bg))

#     s = _seg(break_lo, break_hi)
#     if s is not None:
#         segs.append(_lin(s[0], s[1], n_break))

#     s = _seg(tail_lo, tail_hi)
#     if s is not None:
#         segs.append(_log_ramp(s[0], s[1], n_tail, tail_eps_t) if tail_cluster else _lin(s[0], s[1], n_tail))

#     if not segs:
#         raise ValueError("No valid segments after clamping; check boundaries.")

#     grid = np.concatenate(segs).astype(dtype, copy=False)
#     grid = np.clip(grid, xmin, xmax)
#     grid = np.sort(grid)

#     # ---- deduplicate (tolerance) ----
#     tol = 10.0 * np.finfo(dtype).eps * max(1.0, xmax - xmin)
#     keep = np.ones_like(grid, dtype=bool)
#     keep[1:] = (grid[1:] - grid[:-1]) > tol
#     grid = grid[keep]

#     # ---- enforce strict monotonicity (tiny ramp) ----
#     ramp = np.linspace(0.0, 1e-12, grid.size, dtype=dtype)
#     grid = np.clip(grid + ramp, xmin, xmax)

#     return grid



# def build_m2_grid_np(
#     *,
#     m2_low: float,
#     m2_high: float = 300.0,
#     eps_m: float = 1e-5,
#     # taper region: [m2_low+eps, m2_low+eps+delta_m2_taper]
#     delta_m2_taper: float = 5.0,
#     n_taper: int = 100,
#     n_total: int = 500,
#     eps_t: float = 1e-4,
#     dtype=np.float64,
# ) -> np.ndarray:
#     """
#     Fixed non-uniform m2 grid:
#       - clustered (log-ramp) near m2_low over a taper width delta_m2_taper
#       - then linear out to m2_high

#     This mirrors your PyTensor construction but with fixed numeric params.
#     """
#     if n_total < 2:
#         raise ValueError("n_total must be >= 2")
#     if n_taper < 2:
#         raise ValueError("n_taper must be >= 2")
#     if n_taper >= n_total:
#         raise ValueError("n_taper must be < n_total")
#     if delta_m2_taper <= 0:
#         raise ValueError("delta_m2_taper must be > 0")
#     if m2_high <= m2_low:
#         raise ValueError("m2_high must be > m2_low")

#     m2_lo = float(m2_low) + float(eps_m)
#     m2_taper_hi = m2_lo + max(float(delta_m2_taper), 1e-6)

#     # segment 1: clustered in [m2_lo, m2_taper_hi]
#     u1 = np.linspace(0.0, 1.0, int(n_taper), dtype=dtype)
#     t = np.exp(np.log(float(eps_t)) * (1.0 - u1))  # eps_t -> 1
#     t = (t - float(eps_t)) / (1.0 - float(eps_t))  # -> [0, 1]
#     seg1 = m2_lo + (m2_taper_hi - m2_lo) * t

#     # segment 2: linear in [m2_taper_hi, m2_high]
#     n2 = int(n_total) - int(n_taper)
#     u2 = np.linspace(0.0, 1.0, n2, dtype=dtype)
#     seg2 = m2_taper_hi + (float(m2_high) - m2_taper_hi) * u2

#     # avoid duplicate at the join
#     grid = np.concatenate([seg1[:-1], seg2]).astype(dtype, copy=False)

#     # enforce strictly increasing (tiny ramp)
#     grid = np.maximum.accumulate(grid)
#     ramp = np.linspace(0.0, 1e-12, grid.size, dtype=dtype)
#     grid = grid + ramp

#     return grid