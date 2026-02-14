#
#    Copyright (c) 2025 Michele Mancarella <mancarella@cpt.univ-mrs.fr>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

C_LIGHT_KM_S = 299_792.458  # km/s

import numpy as onp
import jax.numpy as np
import pytensor
import pytensor.tensor as at



# -------------------------
# Taylor & Padé (NumPy)
# -------------------------
def _binomial_coefficients(*, xp=onp):
    return xp.array(
        [
            1,
            -1 / 2,
            3 / 8,
            -5 / 16,
            35 / 128,
            -63 / 256,
            231 / 1024,
            -429 / 2048,
            6435 / 32768,
            -12155 / 65536,
            46189 / 262144,
            -88179 / 524288,
            676039 / 4194304,
            -1300075 / 8388608,
            5014575 / 33554432,
            -9694845 / 67108864,
            300540195 / 2147483648,
        ],
        dtype=float,
    )

def flat_wcdm_taylor_expansion(w0: float, zpower: int = 0, *, xp=onp):
    # F(x) = 2 * sum C(-1/2, n) x^n / (1 - 2k + 6|w0| n + 3 max(w0,0))
    w0 = float(w0)
    k = int(zpower)
    n = xp.arange(0, 17, dtype=float)
    denom = 1.0 - 2.0 * k + 6.0 * abs(w0) * n + 3.0 * max(w0, 0.0)
    return _binomial_coefficients(xp=xp) / denom

def pade(c: onp.ndarray, m: int, n: int, *, xp=onp):
    """Return (p, q) high->low so that P_m/Q_n ~ sum c_k x^k."""
    c = xp.asarray(c, dtype=float)
    assert c.ndim == 1 and c.size >= (m + n + 1)
    rhs = -c[m + 1 : m + n + 1]
    A = xp.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            #A[i, j] = c[m + i - j]
            A = _set(A, (i, j), c[m + i - j])
    q_tail = xp.linalg.solve(A, rhs)
    q = xp.concatenate([xp.array([1.0]), q_tail])  # q0=1
    p = xp.zeros(m + 1, dtype=float)
    for k in range(m + 1):
        s = 0.0
        for j in range(min(k, n) + 1):
            s += q[j] * c[k - j]
        #p[k] = s
        p = set1(p, k, s)
    return p[::-1], q[::-1]  # high->low for polyval


def flat_wcdm_pade_coefficients(w0: float = -1.0, zpower: int = 0, *, xp=onp):
    coeffs = flat_wcdm_taylor_expansion(w0, zpower=zpower, xp=xp)
    # Use [7/7] approximant like wcosmo (from 17 Taylor coeffs)
    m = n = 7
    p, q = pade(coeffs, m, n, xp=xp)
    return xp.asarray(p, dtype=float), xp.asarray(q, dtype=float)

# -------------------------
#  helpers
# -------------------------

def _set(A, idx, val):
    # Works for JAX arrays (has .at) and NumPy arrays (no .at)
    if hasattr(A, "at"):
        return A.at[idx].set(val)
    A[idx] = val
    return A

def set1(x, k, v):
    if hasattr(x, "at"):   # JAX
        return x.at[k].set(v)
    x[k] = v               # NumPy
    return x



def _polyval_at_numpy_coeffs(coeffs, x, xp=onp):
    """
    Horner eval with Python/NumPy coeffs (high->low). Returns a PyTensor.
    """
    cs = onp.asarray(coeffs, dtype="float64").ravel()
    if cs.size == 0:
        return xp.asarray(0.0)
    y = xp.asarray(cs[0])
    for c in cs[1:]:
        y = y * x + float(c)
    return y

def indefinite_integral_pade(z, Om0, w0: float = -1.0, zpower: int = 0, p=None, q=None, xp=onp):
    """
    Padé indefinite integral I(z) for (1+z)^k / E(z) in flat wCDM.
    Pass p, q as NumPy arrays (high->low). No scan, no tensor coeffs.
    """
    if p is None or q is None:
        raise ValueError("Provide Padé coefficients p, q (NumPy/list, high->low).")

    z   = xp.asarray(z)
    Om0 = xp.asarray(Om0)
    w0t = xp.asarray(w0)
    k   = xp.asarray(zpower)

    # Use sign, not deprecated sgn
    sign     = xp.sign(w0t)
    abs_sign = xp.abs(sign)
    what     = (w0t + xp.abs(w0t)) / 2.0  # max(w0, 0)

    gamma = xp.pow(Om0, sign - abs_sign) * xp.pow(1.0 - Om0, -sign - abs_sign)
    gamma = xp.pow(gamma, 0.25)

    normalization = -2.0 * gamma * xp.pow(1.0 + z, k - 0.5 - 1.5 * what)
    x = xp.pow(Om0 / (1.0 - Om0), sign) * xp.pow(1.0 + z, -3.0 * xp.abs(w0t))

    num = _polyval_at_numpy_coeffs(p, x)
    den = _polyval_at_numpy_coeffs(q, x)
    ratio = num / den

    # If w0 == 0 → abs_sign == 0 → ratio**0 == 1
    return normalization * xp.pow(ratio, abs_sign)

def comoving_distance_pade(z, H0, Om0, w0: float = -1.0, p=None, q=None, xp=onp):
    """
    d_C(z) ≈ (c/H0) * [I(z) - I(0)] with k=0.
    """
    I_z = indefinite_integral_pade(z, Om0=Om0, w0=w0, zpower=0, p=p, q=q, xp=xp)
    I_0 = indefinite_integral_pade(0.0, Om0=Om0, w0=w0, zpower=0, p=p, q=q, xp=xp)
    return (C_LIGHT_KM_S / H0 ) * (I_z - I_0)*1e-03


