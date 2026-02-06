from __future__ import annotations

import numpy as np
from pytensor_utils import attrapzvec, atcumtrapz, logsumexp2, logaddexp, logdiffexp, sigmoid, log_sigmoid, safe_sigmoid, atinterp_uniform
from constants import _PI as PI
from constants import max_m, _tgrid_np

try:
    import jax.numpy as jnp
except Exception:
    jnp = None
    
try:
    from jax_utils import make_interp_pt_cached_dy as _make_interp_pt_cached_dy
    _JAX_INTERP_PT = _make_interp_pt_cached_dy(eps=1e-30, side="right")
except Exception:
    _JAX_INTERP_PT = None

    
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
    sl=0.05,
    sh=0.05,
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
    sh=0.05,
    sl=0.05,
    epsilon=0.01,
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
        ln = logNorm_DPLDP(
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
            smoothing=smoothing,
            res=resN,
            simplex_repair=simplex_repair,
            norm_gauss=norm_gauss,
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
    cdf = atcumtrapz(bk, p2, bk.stop_grad(xx))  # keep as-is; if you want bk version later, we’ll refactor
    cdf = bk.clip(cdf, 1e-300, np.inf)

    x0 = xx[1]
    x1 = xx[-1]
    nU = xx.shape[0] - 1

    # log(cdf_scaled) + a gives log(cdf_original)
    itr = atinterp_uniform(bk, m, x0, x1, nU, bk.log(cdf) + a)
    return itr


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
    integ = attrapzvec(bk, ps, bk.stop_grad(ms))  # your backend-agnostic wrapper
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

    cdf_m2 = atcumtrapz(bk, p2, bk.stop_grad(m2_grid))
    cdf_m2 = bk.clip(cdf_m2, eps_cdf, np.inf)

    m2_cdf_grid = m2_grid[1:]
    logcdf_m2 = bk.log(cdf_m2) + a2

    # Evaluate log C(m1) on m1_grid by interpolating logcdf(m2) at m2=m1
    mcap = bk.clip(m1_grid, m2_cdf_grid[0], m2_cdf_grid[-1])

    # use_jax = (
    #     _JAX_INTERP_PT is not None
    #     and jnp is not None
    #     and type(mcap).__module__.startswith("jax")
    # )

    #if use_jax:
    lC_of_m1 = _JAX_INTERP_PT(mcap, m2_cdf_grid, logcdf_m2)
    # else:
    #     # backend-agnostic fallback (same semantics as your sel_bias interpolation helpers)
    #     idx = np.searchsorted(np.asarray(m2_cdf_grid), np.asarray(mcap), side=side_interp)
    #     idx = np.clip(idx, 1, np.asarray(m2_cdf_grid).shape[0] - 1)
    #     x0 = m2_cdf_grid[idx - 1]
    #     x1 = m2_cdf_grid[idx]
    #     y0 = logcdf_m2[idx - 1]
    #     y1 = logcdf_m2[idx]
    #     denom = bk.maximum(x1 - x0, eps_interp)
    #     r = (mcap - x0) / denom
    #     lC_of_m1 = (1.0 - r) * y0 + r * y1

    # ---- normalization ln = log ∫ exp(lp_m1) dm1 ----
    lp_max = bk.max(lp_m1_grid)
    p_shift = bk.exp(lp_m1_grid - lp_max)
    I = attrapzvec(bk, p_shift, bk.stop_grad(m1_grid))
    I = bk.clip(I, 1e-300, np.inf)
    ln_m1 = bk.log(I) + lp_max

    return (lp_m1_grid, lp_m2_grid, lC_of_m1, ln_m1)



def build_m1_grid_DPLDP_np(
    *,
    m1_low: float = 3.0,
    m1_high: float = 350.0,
    eps: float = 1e-4,
    dtype=np.float64,
    # ---- total points control ----
    n_total: int = 1200,
    # ---- segment boundaries (tweakable) ----
    rise_hi: float = 10.0,          # low-mass rise capture (m1_low..~10)
    peak1_lo: float = 5.0,          # peak around ~10 window
    peak1_hi: float = 15.0,
    bg_lo: float = 10.0,            # background PL (10..break)
    bg_hi: float = 45.0,
    break_lo: float = 25.0,         # break / peak2 window (~30-40)
    break_hi: float = 50.0,
    tail_lo: float = 45.0,          # tail start
    tail_hi: float | None = None,   # defaults to m1_high
    # ---- how to split n_total across segments (normalized) ----
    # You can pass your own weights; they will be normalized.
    weights: tuple[float, float, float, float, float] = (0.18, 0.20, 0.14, 0.28, 0.20),
    # ---- spacing controls ----
    rise_cluster: bool = True,
    rise_eps_t: float = 1e-4,       # smaller => more clustering near m1_low
    tail_cluster: bool = True,
    tail_eps_t: float = 2e-3,       # smaller => more clustering near tail_lo
) -> np.ndarray:
    """
    Fixed non-uniform m1 grid with a single knob n_total.

    Segments (in order):
      1) rise:   [m1_low, rise_hi] (optionally clustered near m1_low)
      2) peak1:  [peak1_lo, peak1_hi]
      3) bg:     [bg_lo, bg_hi]
      4) break:  [break_lo, break_hi]
      5) tail:   [tail_lo, tail_hi] (optionally clustered near tail_lo)

    n_total is allocated across segments via 'weights' (normalized).
    """
    m1_low = float(m1_low)
    m1_high = float(m1_high)
    if m1_high <= m1_low:
        raise ValueError("m1_high must be > m1_low")

    n_total = int(n_total)
    if n_total < 10:
        raise ValueError("n_total must be >= 10 (practically)")

    xmin = m1_low + float(eps)
    xmax = m1_high - float(eps)
    if xmax <= xmin:
        raise ValueError("m1_high - m1_low too small after eps")

    if tail_hi is None:
        tail_hi = xmax
    else:
        tail_hi = min(float(tail_hi), xmax)

    # ---- allocate point counts ----
    if len(weights) != 5:
        raise ValueError("weights must have length 5 (rise, peak1, bg, break, tail)")
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    s = float(w.sum())
    if s <= 0:
        raise ValueError("weights must sum to > 0")
    w = w / s

    raw = w * n_total
    n = np.floor(raw).astype(int)
    # distribute remainder by largest fractional parts
    rem = n_total - int(n.sum())
    if rem > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(-frac)
        n[order[:rem]] += 1

    # guarantee at least 2 points in segments that are valid
    # (we’ll later drop invalid segments automatically)
    n = np.maximum(n, 2)

    n_rise, n_peak1, n_bg, n_break, n_tail = map(int, n.tolist())

    # ---- helpers ----
    def _seg(lo, hi):
        lo = max(float(lo), xmin)
        hi = min(float(hi), xmax)
        if hi <= lo:
            return None
        return lo, hi

    def _lin(lo, hi, npts):
        npts = int(npts)
        if npts <= 0:
            return np.zeros((0,), dtype=dtype)
        if npts == 1:
            return np.array([(lo + hi) * 0.5], dtype=dtype)
        return np.linspace(lo, hi, npts, dtype=dtype)

    def _log_ramp(lo, hi, npts, eps_t):
        npts = int(npts)
        if npts <= 0:
            return np.zeros((0,), dtype=dtype)
        if npts == 1:
            return np.array([(lo + hi) * 0.5], dtype=dtype)
        u = np.linspace(0.0, 1.0, npts, dtype=dtype)
        eps_t = float(eps_t)
        t = np.exp(np.log(eps_t) * (1.0 - u))  # eps_t -> 1
        t = (t - eps_t) / (1.0 - eps_t)        # -> [0,1]
        return (lo + (hi - lo) * t).astype(dtype, copy=False)

    # ---- build segments ----
    segs = []

    s = _seg(xmin, rise_hi)
    if s is not None:
        segs.append(_log_ramp(s[0], s[1], n_rise, rise_eps_t) if rise_cluster else _lin(s[0], s[1], n_rise))

    s = _seg(peak1_lo, peak1_hi)
    if s is not None:
        segs.append(_lin(s[0], s[1], n_peak1))

    s = _seg(bg_lo, bg_hi)
    if s is not None:
        segs.append(_lin(s[0], s[1], n_bg))

    s = _seg(break_lo, break_hi)
    if s is not None:
        segs.append(_lin(s[0], s[1], n_break))

    s = _seg(tail_lo, tail_hi)
    if s is not None:
        segs.append(_log_ramp(s[0], s[1], n_tail, tail_eps_t) if tail_cluster else _lin(s[0], s[1], n_tail))

    if not segs:
        raise ValueError("No valid segments after clamping; check boundaries.")

    grid = np.concatenate(segs).astype(dtype, copy=False)
    grid = np.clip(grid, xmin, xmax)
    grid = np.sort(grid)

    # ---- deduplicate (tolerance) ----
    tol = 10.0 * np.finfo(dtype).eps * max(1.0, xmax - xmin)
    keep = np.ones_like(grid, dtype=bool)
    keep[1:] = (grid[1:] - grid[:-1]) > tol
    grid = grid[keep]

    # ---- enforce strict monotonicity (tiny ramp) ----
    ramp = np.linspace(0.0, 1e-12, grid.size, dtype=dtype)
    grid = np.clip(grid + ramp, xmin, xmax)

    return grid



def build_m2_grid_np(
    *,
    m2_low: float,
    m2_high: float = 300.0,
    eps_m: float = 1e-5,
    # taper region: [m2_low+eps, m2_low+eps+delta_m2_taper]
    delta_m2_taper: float = 5.0,
    n_taper: int = 100,
    n_total: int = 500,
    eps_t: float = 1e-4,
    dtype=np.float64,
) -> np.ndarray:
    """
    Fixed non-uniform m2 grid:
      - clustered (log-ramp) near m2_low over a taper width delta_m2_taper
      - then linear out to m2_high

    This mirrors your PyTensor construction but with fixed numeric params.
    """
    if n_total < 2:
        raise ValueError("n_total must be >= 2")
    if n_taper < 2:
        raise ValueError("n_taper must be >= 2")
    if n_taper >= n_total:
        raise ValueError("n_taper must be < n_total")
    if delta_m2_taper <= 0:
        raise ValueError("delta_m2_taper must be > 0")
    if m2_high <= m2_low:
        raise ValueError("m2_high must be > m2_low")

    m2_lo = float(m2_low) + float(eps_m)
    m2_taper_hi = m2_lo + max(float(delta_m2_taper), 1e-6)

    # segment 1: clustered in [m2_lo, m2_taper_hi]
    u1 = np.linspace(0.0, 1.0, int(n_taper), dtype=dtype)
    t = np.exp(np.log(float(eps_t)) * (1.0 - u1))  # eps_t -> 1
    t = (t - float(eps_t)) / (1.0 - float(eps_t))  # -> [0, 1]
    seg1 = m2_lo + (m2_taper_hi - m2_lo) * t

    # segment 2: linear in [m2_taper_hi, m2_high]
    n2 = int(n_total) - int(n_taper)
    u2 = np.linspace(0.0, 1.0, n2, dtype=dtype)
    seg2 = m2_taper_hi + (float(m2_high) - m2_taper_hi) * u2

    # avoid duplicate at the join
    grid = np.concatenate([seg1[:-1], seg2]).astype(dtype, copy=False)

    # enforce strictly increasing (tiny ramp)
    grid = np.maximum.accumulate(grid)
    ramp = np.linspace(0.0, 1e-12, grid.size, dtype=dtype)
    grid = grid + ramp

    return grid