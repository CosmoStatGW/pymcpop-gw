from __future__ import annotations

from typing import Tuple
import numpy as np

import pytensor.tensor as at
from pytensor.graph.op import Op, Apply
from pytensor_utils import _interp1d_from_sorted_x, _logsumexp_np

import rate_models
import spin_models
import mass_models


import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jax_logsumexp
from jax import lax
import jax.scipy as jsp


from backends import JAXBackend
from population import log_p_pop

from pytensor.gradient import DisconnectedType






# ---------------------------------------------------------------------
#    given precomputed grids: dL_grid(z), dc_grid(z), log_ddL_dz_grid(z)
#    evaluate at an array of dL: return z(dL), dc(z), log_ddL_dz(z)


class CosmoFromDLGridsOp(Op):
    """
    Inputs:
      dL : (N,)

    Stored constants:
      zgrid           : (M,)
      dL_grid(zgrid)  : (M,)  (must be strictly monotonic increasing for inverse)
      dc_grid(zgrid)  : (M,)
      log_ddL_dz_grid : (M,)

    Outputs:
      z(dL)           : (N,)
      dc(z(dL))       : (N,)
      log_ddL_dz(z)   : (N,)

    This matches your current usage:
      zs = atinterp(d, dL_grid, zgrid_)                    # inverse
      log_ddL_dz = atinterp(zs, zgrid_, log_ddL_dz_grid)   # forward
      dc = atinterp(zs, zgrid_, dc_grid)                   # forward

    but does it in ONE Apply node (and therefore one place doing searchsorted),
    which is exactly what we want before we fuse SelectionOp.
    """

    itypes = [at.dvector]
    otypes = [at.dvector, at.dvector, at.dvector]

    def __init__(self, zgrid, dL_grid, dc_grid, log_ddL_dz_grid, *, clip=True):
        super().__init__()
        self.zgrid = np.asarray(zgrid, dtype="float64")
        self.dL_grid = np.asarray(dL_grid, dtype="float64")
        self.dc_grid = np.asarray(dc_grid, dtype="float64")
        self.log_ddL_dz_grid = np.asarray(log_ddL_dz_grid, dtype="float64")
        self.clip = bool(clip)

        # Strongly recommended sanity check (fail fast)
        if self.zgrid.ndim != 1:
            raise ValueError("zgrid must be 1D")
        if self.dL_grid.shape != self.zgrid.shape:
            raise ValueError("dL_grid shape must match zgrid")
        if self.dc_grid.shape != self.zgrid.shape:
            raise ValueError("dc_grid shape must match zgrid")
        if self.log_ddL_dz_grid.shape != self.zgrid.shape:
            raise ValueError("log_ddL_dz_grid shape must match zgrid")
        if not np.all(np.diff(self.dL_grid) > 0):
            raise ValueError("dL_grid must be strictly increasing for inverse interpolation")

    def make_node(self, dL):
        dL = at.as_tensor_variable(dL)
        if dL.ndim != 1:
            dL = dL.flatten()
        return Apply(self, [dL], [at.dvector(), at.dvector(), at.dvector()])

    def perform(self, node, inputs, outputs):
        (dL,) = inputs
        z_out, dc_out, log_out = outputs

        # 1) inverse: z(dL) using x = dL_grid, y = zgrid
        z = _interp1d_from_sorted_x(dL, self.dL_grid, self.zgrid, clip=self.clip)

        # 2) forward: dc(z) and log_ddL_dz(z) using x = zgrid, y = grids
        dc = _interp1d_from_sorted_x(z, self.zgrid, self.dc_grid, clip=self.clip)
        logv = _interp1d_from_sorted_x(z, self.zgrid, self.log_ddL_dz_grid, clip=self.clip)

        z_out[0] = z.astype("float64")
        dc_out[0] = dc.astype("float64")
        log_out[0] = logv.astype("float64")

    def grad(self, inputs, output_grads):
        # For now: disconnected.
        #
        # This Op is intended to be used inside the *SelectionOp* (NumPy/JAX),
        # so PyTensor gradients here are not needed.
        #
        # If later you decide to use it directly in-graph and need d/d(params),
        # we can implement a proper grad (but I'd rather keep it out of the graph).
        return [at.zeros_like(inputs[0])]


def cosmo_from_dL_grids(
    dL,
    *,
    zgrid,
    dL_grid,
    dc_grid,
    log_ddL_dz_grid,
    clip=True,
) -> Tuple[at.TensorVariable, at.TensorVariable, at.TensorVariable]:
    """
    Convenience wrapper so can do:

      zs, dcs, log_ddL_dz = atools.cosmo_from_dL_grids(
          d, zgrid=zgrid_, dL_grid=dL_grid, dc_grid=dc_grid, log_ddL_dz_grid=log_ddL_dz_grid
      )

    without directly instantiating the Op.
    """
    op = CosmoFromDLGridsOp(zgrid, dL_grid, dc_grid, log_ddL_dz_grid, clip=clip)
    return op(dL)





# ---------------------------------------------------------------------
#
# population function
# 
# ---------------------------------------------------------------------




def _logdiffexp_jax(a, b, eps=1e-16):
    """
    Stable log(exp(a) - exp(b)) elementwise.
    Returns -inf where b >= a.
    """
    delta = jnp.minimum(b - a, 0.0)  # <= 0
    ed = jnp.exp(delta)
    out = a + jnp.log1p(-jnp.minimum(ed, 1.0 - eps))
    return jnp.where(b < a, out, -jnp.inf)


class _LogPPopJAXVJPOp(Op):
    """
    Internal Op: computes VJP for log_p_pop.

    Inputs: same as LogPPopJAXOp + g_out (upstream grad wrt output, shape (N,))
    Outputs: grads for each tensor input, in same order as LogPPopJAXOp inputs.
    """

    itypes = [
        at.dvector,  # m1s
        at.dvector,  # m2s
        at.dvector,  # z
        at.dvector,  # dL
        at.dmatrix,  # spins
        at.dvector,  # Lambda
        at.dvector,  # dc
        at.dvector,  # log_ddL_dz_pre
        at.dvector,  # g_out (N,)
    ]
    otypes = [
        at.dvector,  # dm1
        at.dvector,  # dm2
        at.dvector,  # dz
        at.dvector,  # ddL
        at.dmatrix,  # dspins
        at.dvector,  # dLambda
        at.dvector,  # ddc
        at.dvector,  # dlog_ddL_dz_pre
    ]

    def __init__(
        self,
        *,
        rate_model,
        mass_model,
        spin_model,
        smoothing="LVK",
        simplex_repair=False,
        has_m2_break=False,
        norm_gauss="uplow",
        param="vanilla",
        interp_vals_mass=None,
        interp_grids_mass=None,
        is_observed=False,
        z_grid=None,
        verbose=False,
    ):
        super().__init__()

        # statics
        self.rate_model = rate_model
        self.mass_model = mass_model
        self.spin_model = spin_model
        self.smoothing = smoothing
        self.simplex_repair = bool(simplex_repair)
        self.has_m2_break = bool(has_m2_break)
        self.norm_gauss = norm_gauss
        self.param = param
        self.is_observed = bool(is_observed)
        self.verbose = bool(verbose)

        # optional static arrays (kept outside Op inputs by design)
        self.interp_vals_mass = None if interp_vals_mass is None else np.asarray(interp_vals_mass)
        if interp_grids_mass is None:
            self.interp_grids_mass = None
        else:
            self.interp_grids_mass = tuple(np.asarray(g) for g in interp_grids_mass)
        self.z_grid = None if z_grid is None else np.asarray(z_grid)

        # build jitted vjp
        self._jax_vjp = self._build_jax_vjp()

    def _build_jax_vjp(self):
        rate_model = self.rate_model
        mass_model = self.mass_model
        spin_model = self.spin_model
        smoothing = self.smoothing
        simplex_repair = self.simplex_repair
        has_m2_break = self.has_m2_break
        norm_gauss = self.norm_gauss
        param = self.param
        is_observed = self.is_observed
        verbose = self.verbose

        interp_vals_mass_jax = None if self.interp_vals_mass is None else jnp.asarray(self.interp_vals_mass)
        if self.interp_grids_mass is None:
            interp_grids_mass_jax = None
        else:
            interp_grids_mass_jax = tuple(jnp.asarray(g) for g in self.interp_grids_mass)

        z_grid_jax = None if self.z_grid is None else jnp.asarray(self.z_grid)

        spins_unpack = lambda s: spin_models._spins_as_list(s, spin_model)
        bk = JAXBackend()
        
        def _f(m1s, m2s, z, dL, spins, Lambda, dc, log_ddL_dz_pre):
            
            return log_p_pop(
                bk,
                m1s,
                m2s,
                z,
                dL,
                spins_unpack(spins),
                Lambda,
                rate_model=rate_model,
                mass_model=mass_model,
                spin_model=spin_model,
                smoothing=smoothing,
                simplex_repair=simplex_repair,
                has_m2_break=has_m2_break,
                norm_gauss=norm_gauss,
                dc=dc,
                log_ddL_dz_pre=log_ddL_dz_pre,
                param=param,
                interp_vals_mass=interp_vals_mass_jax,
                interp_grids_mass=interp_grids_mass_jax,
                is_observed=is_observed,
                z_grid=z_grid_jax,
                verbose=verbose,
            )

        # VJP: given upstream g_out (shape (N,)), return grads for each arg
        def _vjp(m1s, m2s, z, dL, spins, Lambda, dc, log_ddL_dz_pre, g_out):
            y, pullback = jax.vjp(_f, m1s, m2s, z, dL, spins, Lambda, dc, log_ddL_dz_pre)
            grads = pullback(g_out)  # tuple of grads matching inputs
            return grads

        return jax.jit(_vjp)

    def make_node(self, m1s, m2s, z, dL, spins, Lambda, dc, log_ddL_dz_pre, g_out):
        inputs = list(
            map(
                at.as_tensor_variable,
                [m1s, m2s, z, dL, spins, Lambda, dc, log_ddL_dz_pre, g_out],
            )
        )
        outs = [
            at.dvector(),  # dm1
            at.dvector(),  # dm2
            at.dvector(),  # dz
            at.dvector(),  # ddL
            at.dmatrix(),  # dspins
            at.dvector(),  # dLambda
            at.dvector(),  # ddc
            at.dvector(),  # dlog
        ]
        return Apply(self, inputs, outs)

    def perform(self, node, inputs, outputs):
        m1s, m2s, z, dL, spins, Lambda, dc, log_ddL_dz_pre, g_out = inputs
        grads = self._jax_vjp(
            _to_jnp(m1s),
            _to_jnp(m2s),
            _to_jnp(z),
            _to_jnp(dL),
            _to_jnp(spins),
            _to_jnp(Lambda),
            _to_jnp(dc),
            _to_jnp(log_ddL_dz_pre),
            _to_jnp(g_out),
        )

        # write grads to outputs
        for out_i, g in zip(outputs, grads):
            out_i[0] = np.asarray(g, dtype="float64")


class LogPPopJAXOp(Op):
    """
    JAX-backed Op for population.log_p_pop with gradients.

    Tensor inputs (all required):
      m1s, m2s, z, dL: (N,)
      spins: (N, nspin)
      Lambda: (P,)
      dc: (N,)
      log_ddL_dz_pre: (N,)

    All keyword args from population.log_p_pop are forwarded, with the *string/bool*
    config stored as statics in __init__ (so JAX compiles once per config).
    """

    itypes = [
        at.dvector,  # m1s
        at.dvector,  # m2s
        at.dvector,  # z
        at.dvector,  # dL
        at.dmatrix,  # spins
        at.dvector,  # Lambda
        at.dvector,  # dc
        at.dvector,  # log_ddL_dz_pre
    ]
    otypes = [at.dvector]  # lp (N,)

    def __init__(
        self,
        *,
        rate_model,
        mass_model,
        spin_model,
        smoothing="LVK",
        simplex_repair=False,
        has_m2_break=False,
        norm_gauss="uplow",
        param="vanilla",
        interp_vals_mass=None,
        interp_grids_mass=None,
        is_observed=False,
        z_grid=None,
        verbose=False,
    ):
        super().__init__()

        # store statics
        self.rate_model = rate_model
        self.mass_model = mass_model
        self.spin_model = spin_model
        self.smoothing = smoothing
        self.simplex_repair = bool(simplex_repair)
        self.has_m2_break = bool(has_m2_break)
        self.norm_gauss = norm_gauss
        self.param = param
        self.is_observed = bool(is_observed)
        self.verbose = bool(verbose)

        self.interp_vals_mass = interp_vals_mass
        self.interp_grids_mass = interp_grids_mass
        self.z_grid = z_grid

        self._jax_fwd = self._build_jax_fwd()

        # one VJP op instance with same statics
        self._vjp_op = _LogPPopJAXVJPOp(
            rate_model=rate_model,
            mass_model=mass_model,
            spin_model=spin_model,
            smoothing=smoothing,
            simplex_repair=simplex_repair,
            has_m2_break=has_m2_break,
            norm_gauss=norm_gauss,
            param=param,
            interp_vals_mass=interp_vals_mass,
            interp_grids_mass=interp_grids_mass,
            is_observed=is_observed,
            z_grid=z_grid,
            verbose=verbose,
        )

    def _build_jax_fwd(self):
        rate_model = self.rate_model
        mass_model = self.mass_model
        spin_model = self.spin_model
        smoothing = self.smoothing
        simplex_repair = self.simplex_repair
        has_m2_break = self.has_m2_break
        norm_gauss = self.norm_gauss
        param = self.param
        is_observed = self.is_observed
        verbose = self.verbose

        interp_vals_mass_jax = None if self.interp_vals_mass is None else jnp.asarray(self.interp_vals_mass)
        if self.interp_grids_mass is None:
            interp_grids_mass_jax = None
        else:
            interp_grids_mass_jax = tuple(jnp.asarray(g) for g in self.interp_grids_mass)

        z_grid_jax = None if self.z_grid is None else jnp.asarray(self.z_grid)
        spins_unpack = lambda s: spin_models._spins_as_list(s, spin_model)
        bk = JAXBackend()
        
        def _f(m1s, m2s, z, dL, spins, Lambda, dc, log_ddL_dz_pre):
            
            return log_p_pop(
                bk,
                m1s,
                m2s,
                z,
                dL,
                spins_unpack(spins),
                Lambda,
                rate_model=rate_model,
                mass_model=mass_model,
                spin_model=spin_model,
                smoothing=smoothing,
                simplex_repair=simplex_repair,
                has_m2_break=has_m2_break,
                norm_gauss=norm_gauss,
                dc=dc,
                log_ddL_dz_pre=log_ddL_dz_pre,
                param=param,
                interp_vals_mass=interp_vals_mass_jax,
                interp_grids_mass=interp_grids_mass_jax,
                is_observed=is_observed,
                z_grid=z_grid_jax,
                verbose=verbose,
            )

        return jax.jit(_f)

    def make_node(self, m1s, m2s, z, dL, spins, Lambda, dc, log_ddL_dz_pre):
        inputs = list(map(at.as_tensor_variable, [m1s, m2s, z, dL, spins, Lambda, dc, log_ddL_dz_pre]))
        return Apply(self, inputs, [at.dvector()])

    def perform(self, node, inputs, outputs):
        m1s, m2s, z, dL, spins, Lambda, dc, log_ddL_dz_pre = inputs
        val = self._jax_fwd(
            _to_jnp(m1s),
            _to_jnp(m2s),
            _to_jnp(z),
            _to_jnp(dL),
            _to_jnp(spins),
            _to_jnp(Lambda),
            _to_jnp(dc),
            _to_jnp(log_ddL_dz_pre),
        )
        outputs[0][0] = np.asarray(val, dtype="float64")

    def grad(self, inputs, output_grads):
        (g_out,) = output_grads  # (N,)

        m1s, m2s, z, dL, spins, Lambda, dc, log_ddL_dz_pre = inputs

        dm1, dm2, dz, ddL, dspins, dLambda, ddc, dlog = self._vjp_op(
            m1s, m2s, z, dL, spins, Lambda, dc, log_ddL_dz_pre, g_out
        )
        return [dm1, dm2, dz, ddL, dspins, dLambda, ddc, dlog]



# ---------------------------------------------------------------------
# streaming logsumexp utilities (JAX)
# ---------------------------------------------------------------------

def _update_logsumexp_pair(carry, xi):
    m, s1, s2 = carry
    new_m = jnp.maximum(m, xi)

    scale1 = jnp.exp(m - new_m)
    scale2 = jnp.exp(2.0 * (m - new_m))

    s1 = s1 * scale1 + jnp.exp(xi - new_m)
    s2 = s2 * scale2 + jnp.exp(2.0 * (xi - new_m))
    return (new_m, s1, s2), None


def _two_logsumexp_streaming(x):
    """
    Returns (lse1, lse2) where:
      lse1 = logsumexp(x)
      lse2 = logsumexp(2*x)
    computed in one pass via lax.scan.
    """
    x = jnp.asarray(x)
    n = x.shape[0]

    # If n==0 shouldn't happen in your use-case, but keep it safe:
    def _n0():
        return -jnp.inf, -jnp.inf

    def _n1():
        m0 = x[0]
        return m0, 2.0 * m0

    def _n2p():
        m0 = x[0]
        s1_0 = jnp.array(1.0, dtype=x.dtype)
        s2_0 = jnp.array(1.0, dtype=x.dtype)
        (m, s1, s2), _ = lax.scan(_update_logsumexp_pair, (m0, s1_0, s2_0), x[1:])
        lse1 = m + jnp.log(s1)
        lse2 = 2.0 * m + jnp.log(s2)
        return lse1, lse2

    return lax.cond(
        n == 0, _n0,
        lambda: lax.cond(n == 1, _n1, _n2p)
    )


# ---------------------------------------------------------------------
# interpolation utilities (JAX)
# ---------------------------------------------------------------------

def _interp_prepare_jax(x, xp, eps=1e-12, side="right"):
    idx = jnp.searchsorted(xp, x, side=side)
    idx = jnp.clip(idx, 1, xp.shape[0] - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    denom = jnp.maximum(x1 - x0, eps)
    t = (x - x0) / denom
    return idx, t


def _interp_apply_jax(idx, t, fp):
    y0 = fp[idx - 1]
    y1 = fp[idx]
    return (1.0 - t) * y0 + t * y1


# ---------------------------------------------------------------------
# pytensor glue utils
# ---------------------------------------------------------------------

def _connected_g(g):
    if isinstance(getattr(g, "type", None), DisconnectedType):
        return at.as_tensor_variable(np.array(0.0, dtype="float64"))
    return g


def _to_jnp(x):
    # your existing helper likely does this already; keep yours if you have one
    return jnp.asarray(x)


# ---------------------------------------------------------------------
# VJP Op: returns grads only wrt small cosmology/state vectors
# ---------------------------------------------------------------------

class _SelectionBiasJAXVJPOp(Op):
    """
    Inputs:
      m1inj, m2inj, dLinj: (Ninj,)  FIXED data (detector-frame masses)
      spinsInj: (Ninj, nspin) FIXED
      log_p_draw: (Ninj,) FIXED
      log_p_incl: (Ninj,) FIXED
      dL_grid, dc_grid, log_ddL_dz_grid: (M,)  (cosmo-derived grids)
      Lambda: (P,)  (population params, incl cosmo params if you pack them here)
      Ndraw: scalar FIXED (no grad)
      g_log_mu: scalar
      g_var: scalar

    Outputs:
      ddL_grid, ddc_grid, dlog_ddL_dz_grid, dLambda
    """

    itypes = [
        at.dvector,  # m1inj
        at.dvector,  # m2inj
        at.dvector,  # dLinj
        at.dmatrix,  # spinsInj
        at.dvector,  # log_p_draw
        at.dvector,  # log_p_incl
        at.dvector,  # dL_grid
        at.dvector,  # dc_grid
        at.dvector,  # log_ddL_dz_grid
        at.dvector,  # Lambda
        at.dscalar,  # Ndraw
        at.dscalar,  # g_log_mu
        at.dscalar,  # g_var
    ]

    otypes = [
        at.dvector,  # ddL_grid
        at.dvector,  # ddc_grid
        at.dvector,  # dlog_ddL_dz_grid
        at.dvector,  # dLambda
    ]

    def __init__(
        self,
        *,
        zgrid,
        rate_model,
        mass_model,
        spin_model,
        smoothing="LVK",
        simplex_repair=False,
        has_m2_break=False,
        norm_gauss="uplow",
        param="vanilla",
        interp_vals_mass=None,
        interp_grids_mass=None,
        is_observed=False,
        z_grid=None,
        verbose=False,
        subtract_log_p_incl=True,
        eps_interp=1e-12,
        side_interp="right",
    ):
        super().__init__()

        self.zgrid = np.asarray(zgrid, dtype="float64")
        self.eps_interp = float(eps_interp)
        self.side_interp = str(side_interp)

        self.rate_model = rate_model
        self.mass_model = mass_model
        self.spin_model = spin_model
        self.smoothing = smoothing
        self.simplex_repair = bool(simplex_repair)
        self.has_m2_break = bool(has_m2_break)
        self.norm_gauss = norm_gauss
        self.param = param
        self.is_observed = bool(is_observed)
        self.verbose = bool(verbose)
        self.subtract_log_p_incl = bool(subtract_log_p_incl)

        self.interp_vals_mass = None if interp_vals_mass is None else np.asarray(interp_vals_mass)
        if interp_grids_mass is None:
            self.interp_grids_mass = None
        else:
            self.interp_grids_mass = tuple(np.asarray(g) for g in interp_grids_mass)

        self.z_grid = None if z_grid is None else np.asarray(z_grid)

        self._jax_vjp = self._build_jax_vjp()
        self._cached = None  # large arrays cached on device

    def _build_jax_vjp(self):
        rate_model = self.rate_model
        mass_model = self.mass_model
        spin_model = self.spin_model
        smoothing = self.smoothing
        simplex_repair = self.simplex_repair
        has_m2_break = self.has_m2_break
        norm_gauss = self.norm_gauss
        param = self.param
        is_observed = self.is_observed
        verbose = self.verbose
        subtract_log_p_incl = self.subtract_log_p_incl

        eps_interp = self.eps_interp
        side_interp = self.side_interp

        zgrid_jax = jnp.asarray(self.zgrid)

        interp_vals_mass_jax = None if self.interp_vals_mass is None else jnp.asarray(self.interp_vals_mass)
        if self.interp_grids_mass is None:
            interp_grids_mass_jax = None
        else:
            interp_grids_mass_jax = tuple(jnp.asarray(g) for g in self.interp_grids_mass)

        z_grid_jax = None if self.z_grid is None else jnp.asarray(self.z_grid)

        if not (rate_model == "MD" and mass_model == "DPLDP" and spin_model == "default_gauss"):
            raise NotImplementedError(
                "SelectionBiasJAXOp currently implemented only for rate_model='MD', mass_model='DPLDP', spin_model='default_gauss'."
            )

        bk = JAXBackend()
        spins_unpack = lambda s: spin_models._spins_as_list(s, spin_model)

        def _sel(
            m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl,
            dL_grid, dc_grid, log_ddL_dz_grid, Lambda, Ndraw
        ):
            Ndraw = jnp.asarray(Ndraw).reshape(())

            # z(dL): xp=dL_grid (diff), fp=zgrid (static)
            i_dL, t_dL = _interp_prepare_jax(dLinj, dL_grid, eps=eps_interp, side=side_interp)
            zinj = _interp_apply_jax(i_dL, t_dL, zgrid_jax)

            # dc(z), log_ddL_dz(z): xp=zgrid static, fp=(diff grids)
            i_z, t_z = _interp_prepare_jax(zinj, zgrid_jax, eps=eps_interp, side=side_interp)
            dcinj = _interp_apply_jax(i_z, t_z, dc_grid)
            log_ddL_dz_inj = _interp_apply_jax(i_z, t_z, log_ddL_dz_grid)

            # source-frame masses depend on cosmology through zinj
            onepz = 1.0 + zinj
            m1Src = m1inj / onepz
            m2Src = m2inj / onepz

            log_p_pop_vals = log_p_pop(
                bk,
                m1Src, m2Src, zinj, dLinj,
                spins_unpack(spinsInj),
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
                param=param,
                interp_vals_mass=interp_vals_mass_jax,
                interp_grids_mass=interp_grids_mass_jax,
                is_observed=is_observed,
                z_grid=z_grid_jax,
                verbose=verbose,
            )

            log_sel_b = log_p_pop_vals - log_p_draw
            if subtract_log_p_incl:
                log_sel_b = log_sel_b - log_p_incl

            logN = jnp.log(Ndraw)
            
            #lse1, lse2 = _two_logsumexp_streaming(log_sel_b)
            #log_mu = lse1 - logN
            #logs2  = lse2 - logN
            
            # lse1 = jsp.special.logsumexp(log_sel_b)
            # lse2 = jsp.special.logsumexp( 2.0 * log_sel_b)

            x = log_sel_b
            m = jnp.max(x)
            #s1 = jnp.sum(jnp.exp(x - m))
            #s2 = jnp.sum(jnp.exp(2.0 * (x - m)))
            u  = jnp.exp(x - m)
            s1 = jnp.sum(u)
            s2 = jnp.sum(u * u)   # same as sum(exp(2*(x-m)))
            
            lse1 = m + jnp.log(s1)
            lse2 = 2.0 * m + jnp.log(s2)
            
            log_mu = lse1 - logN
            logs2  = lse2 - logN

            var_log_lik_u = _logdiffexp_jax(logs2 - 2.0 * log_mu, 1.0) - jnp.log(Ndraw - 1.0)
            return jnp.asarray(log_mu).reshape(()), jnp.asarray(var_log_lik_u).reshape(())

        def _vjp(
            m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl,
            dL_grid, dc_grid, log_ddL_dz_grid, Lambda, Ndraw,
            g_log_mu, g_var
        ):
            # Only differentiate wrt the small stuff:
            def _sel_params(dL_grid, dc_grid, log_ddL_dz_grid, Lambda):
                return _sel(
                    m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl,
                    dL_grid, dc_grid, log_ddL_dz_grid, Lambda, Ndraw
                )

            (_, _), pullback = jax.vjp(_sel_params, dL_grid, dc_grid, log_ddL_dz_grid, Lambda)
            ddL_grid, ddc_grid, dlog_ddL_dz_grid, dLambda = pullback((g_log_mu, g_var))
            return ddL_grid, ddc_grid, dlog_ddL_dz_grid, dLambda

        return jax.jit(_vjp)

    def make_node(self, *inputs):
        inputs = list(map(at.as_tensor_variable, inputs))
        outs = [at.dvector(), at.dvector(), at.dvector(), at.dvector()]
        return Apply(self, inputs, outs)

    def perform(self, node, inputs, outputs):
        (
            m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl,
            dL_grid, dc_grid, log_ddL_dz_grid, Lambda, Ndraw,
            g_log_mu, g_var
        ) = inputs

        if self._cached is None:
            self._cached = (
                jax.device_put(_to_jnp(m1inj)),
                jax.device_put(_to_jnp(m2inj)),
                jax.device_put(_to_jnp(dLinj)),
                jax.device_put(_to_jnp(spinsInj)),
                jax.device_put(_to_jnp(log_p_draw)),
                jax.device_put(_to_jnp(log_p_incl)),
                jax.device_put(_to_jnp(Ndraw).reshape(())),
            )

        m1inj_j, m2inj_j, dLinj_j, spins_j, lpd_j, lpi_j, Ndraw_j = self._cached

        grads = self._jax_vjp(
            m1inj_j, m2inj_j, dLinj_j, spins_j, lpd_j, lpi_j,
            jax.device_put(_to_jnp(dL_grid)),
            jax.device_put(_to_jnp(dc_grid)),
            jax.device_put(_to_jnp(log_ddL_dz_grid)),
            jax.device_put(_to_jnp(Lambda)),
            Ndraw_j,
            jax.device_put(_to_jnp(g_log_mu).reshape(())),
            jax.device_put(_to_jnp(g_var).reshape(())),
        )

        for out_i, g in zip(outputs, grads):
            out_i[0] = np.asarray(g, dtype="float64")


# ---------------------------------------------------------------------
# main Op (forward + grad glue)
# ---------------------------------------------------------------------

class SelectionBiasJAXOp(Op):
    itypes = [
        at.dvector,  # m1inj
        at.dvector,  # m2inj
        at.dvector,  # dLinj
        at.dmatrix,  # spinsInj
        at.dvector,  # log_p_draw
        at.dvector,  # log_p_incl
        at.dvector,  # dL_grid
        at.dvector,  # dc_grid
        at.dvector,  # log_ddL_dz_grid
        at.dvector,  # Lambda
        at.dscalar,  # Ndraw
    ]
    otypes = [at.dscalar, at.dscalar]  # (log_mu, var_u)

    def __init__(self, *, zgrid, **kwargs):
        super().__init__()
        self.zgrid = np.asarray(zgrid, dtype="float64")
        self.kwargs = dict(kwargs)

        self._jax_fwd = self._build_jax_fwd()
        self._vjp_op = _SelectionBiasJAXVJPOp(zgrid=self.zgrid, **self.kwargs)
        self._cached = None

    def _build_jax_fwd(self):
        # same config extraction as your code
        rate_model = self.kwargs["rate_model"]
        mass_model = self.kwargs["mass_model"]
        spin_model = self.kwargs["spin_model"]
        smoothing = self.kwargs.get("smoothing", "LVK")
        simplex_repair = bool(self.kwargs.get("simplex_repair", False))
        has_m2_break = bool(self.kwargs.get("has_m2_break", False))
        norm_gauss = self.kwargs.get("norm_gauss", "uplow")
        param = self.kwargs.get("param", "vanilla")
        is_observed = bool(self.kwargs.get("is_observed", False))
        verbose = bool(self.kwargs.get("verbose", False))
        subtract_log_p_incl = bool(self.kwargs.get("subtract_log_p_incl", True))
        eps_interp = float(self.kwargs.get("eps_interp", 1e-12))
        side_interp = str(self.kwargs.get("side_interp", "right"))

        interp_vals_mass = self.kwargs.get("interp_vals_mass", None)
        interp_grids_mass = self.kwargs.get("interp_grids_mass", None)
        z_grid = self.kwargs.get("z_grid", None)

        zgrid_jax = jnp.asarray(self.zgrid)

        interp_vals_mass_jax = None if interp_vals_mass is None else jnp.asarray(interp_vals_mass)
        if interp_grids_mass is None:
            interp_grids_mass_jax = None
        else:
            interp_grids_mass_jax = tuple(jnp.asarray(g) for g in interp_grids_mass)

        z_grid_jax = None if z_grid is None else jnp.asarray(z_grid)

        if not (rate_model == "MD" and mass_model == "DPLDP" and spin_model == "default_gauss"):
            raise NotImplementedError(
                "SelectionBiasJAXOp currently implemented only for rate_model='MD', mass_model='DPLDP', spin_model='default_gauss'."
            )

        bk = JAXBackend()
        spins_unpack = lambda s: spin_models._spins_as_list(s, spin_model)

        # This one expects *all* inputs (we'll call it only once we have cached arrays)
        def _f(
            m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl,
            dL_grid, dc_grid, log_ddL_dz_grid, Lambda, Ndraw
        ):
            Ndraw = jnp.asarray(Ndraw).reshape(())
    
            i_dL, t_dL = _interp_prepare_jax(dLinj, dL_grid, eps=eps_interp, side=side_interp)
            zinj = _interp_apply_jax(i_dL, t_dL, zgrid_jax)
    
            i_z, t_z = _interp_prepare_jax(zinj, zgrid_jax, eps=eps_interp, side=side_interp)
            dcinj = _interp_apply_jax(i_z, t_z, dc_grid)
            log_ddL_dz_inj = _interp_apply_jax(i_z, t_z, log_ddL_dz_grid)
    
            onepz = 1.0 + zinj
            m1Src = m1inj / onepz
            m2Src = m2inj / onepz
    
            log_p_pop_vals = log_p_pop(
                bk,
                m1Src, m2Src, zinj, dLinj,
                spins_unpack(spinsInj),
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
                param=param,
                interp_vals_mass=interp_vals_mass_jax,
                interp_grids_mass=interp_grids_mass_jax,
                is_observed=is_observed,
                z_grid=z_grid_jax,
                verbose=verbose,
            )
    
            log_sel_b = log_p_pop_vals - log_p_draw
            if subtract_log_p_incl:
                log_sel_b = log_sel_b - log_p_incl
    
            # USE the fast reduction form here (don’t scan)
            x = log_sel_b
            m = jnp.max(x)
            u = jnp.exp(x - m)
            s1 = jnp.sum(u)
            s2 = jnp.sum(u * u)
            lse1 = m + jnp.log(s1)
            lse2 = 2.0 * m + jnp.log(s2)
    
            logN = jnp.log(Ndraw)
            log_mu = lse1 - logN
            logs2  = lse2 - logN
    
            var_log_lik_u = _logdiffexp_jax(logs2 - 2.0 * log_mu, 1.0) - jnp.log(Ndraw - 1.0)
            return jnp.asarray(log_mu).reshape(()), jnp.asarray(var_log_lik_u).reshape(())

        return jax.jit(_f)


        
    def make_node(self, *inputs):
        inputs = list(map(at.as_tensor_variable, inputs))
        return Apply(self, inputs, [at.dscalar(), at.dscalar()])

    def perform(self, node, inputs, outputs):
        (
            m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl,
            dL_grid, dc_grid, log_ddL_dz_grid, Lambda, Ndraw
        ) = inputs
    
        # Cache fixed injection arrays once
        if self._cached is None:
            self._cached = (
                jax.device_put(_to_jnp(m1inj)),
                jax.device_put(_to_jnp(m2inj)),
                jax.device_put(_to_jnp(dLinj)),
                jax.device_put(_to_jnp(spinsInj)),
                jax.device_put(_to_jnp(log_p_draw)),
                jax.device_put(_to_jnp(log_p_incl)),
            )
    
        m1inj_j, m2inj_j, dLinj_j, spins_j, lpd_j, lpi_j = self._cached
    
        # Only "small" changing arrays each call
        log_mu, var_u = self._jax_fwd(
            m1inj_j, m2inj_j, dLinj_j, spins_j, lpd_j, lpi_j,
            jax.device_put(_to_jnp(dL_grid)),
            jax.device_put(_to_jnp(dc_grid)),
            jax.device_put(_to_jnp(log_ddL_dz_grid)),
            jax.device_put(_to_jnp(Lambda)),
            jax.device_put(_to_jnp(Ndraw).reshape(())),
        )
    
        outputs[0][0] = np.asarray(log_mu, dtype="float64")
        outputs[1][0] = np.asarray(var_u, dtype="float64")


    def grad(self, inputs, output_grads):
        g_log_mu, g_var = output_grads
        g_log_mu = _connected_g(g_log_mu)
        g_var = _connected_g(g_var)

        (
            m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl,
            dL_grid, dc_grid, log_ddL_dz_grid, Lambda, Ndraw
        ) = inputs

        ddL_grid, ddc_grid, dlog_ddL_dz_grid, dLambda = self._vjp_op(
            m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl,
            dL_grid, dc_grid, log_ddL_dz_grid, Lambda, Ndraw,
            g_log_mu, g_var
        )

        # Return zeros of correct shape for "no grad" inputs
        z_m1 = at.zeros_like(m1inj)
        z_m2 = at.zeros_like(m2inj)
        z_dL = at.zeros_like(dLinj)
        z_sp = at.zeros_like(spinsInj)
        z_lpd = at.zeros_like(log_p_draw)
        z_lpi = at.zeros_like(log_p_incl)
        z_Ndraw = at.zeros_like(Ndraw)

        return [
            z_m1, z_m2, z_dL, z_sp, z_lpd, z_lpi,
            ddL_grid, ddc_grid, dlog_ddL_dz_grid, dLambda,
            z_Ndraw,
        ]


    def reset_cache(self):
        self._cached = None