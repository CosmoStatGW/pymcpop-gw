from __future__ import annotations

from typing import Tuple
import numpy as np
import hashlib

import pytensor.tensor as at
from pytensor.graph.op import Op, Apply
from pytensor_utils import _interp1d_from_sorted_x, _logsumexp_np

import rate_models
import spin_models
import mass_models
from cosmology import dcfun_quad, dLfun, log_ddL_dz


import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jax_logsumexp
from jax import lax
import jax.scipy as jsp


from backends import JAXBackend
from population import log_p_pop, sel_bias_with_uncertainty

from pytensor.gradient import DisconnectedType, grad_not_implemented


from constants import _x01_np as x01
from constants import _w01_np as w01


def _hash_arr(x):
    x = np.asarray(x, dtype=np.float64, order="C")
    return hashlib.blake2b(x.view(np.uint8), digest_size=16).digest()


# ---------------------------------------------------------------------
#    given precomputed grids: dL_grid(z), dc_grid(z), log_ddL_dz_grid(z)
#    evaluate at an array of dL: return z(dL), dc(z), log_ddL_dz(z)


class _CosmoFromDLJAXVJPOp(Op):
    """
    VJP Op for CosmoFromDLJAXOp.

    Inputs:
      dL      : (N,)
      Lambda  : (P,)  (cosmo is Lambda[:5])
      gz,gdc,glog : (N,) upstream grads for outputs

    Outputs:
      ddL     : (N,)
      dLambda : (P,)
    """

    itypes = [at.dvector, at.dvector, at.dvector, at.dvector, at.dvector]
    otypes = [at.dvector, at.dvector]

    def __init__(
        self,
        *,
        zgrid,
        x01,
        w01,
        param="vanilla",
        eps_interp=1e-12,
        side_interp="right",
    ):
        super().__init__()
        self.zgrid = np.asarray(zgrid, dtype="float64")
        self.x01 = np.asarray(x01, dtype="float64")
        self.w01 = np.asarray(w01, dtype="float64")
        self.param = str(param)
        self.eps_interp = float(eps_interp)
        self.side_interp = str(side_interp)

        self._jax_vjp = self._build_jax_vjp()

    def _build_jax_vjp(self):
        bk = JAXBackend()
        zgrid = jnp.asarray(self.zgrid)
        x01 = jnp.asarray(self.x01)
        w01 = jnp.asarray(self.w01)
        param = self.param
        eps_interp = self.eps_interp
        side_interp = self.side_interp

        z_from_dL_interp = make_z_from_dL_interp(bk, param=self.param)


        # -------- full cosmo map f(dL, Lambda) -> (z, dc, log_ddL_dz) --------
        def f(dL, Lambda):
            theta5 = Lambda[:5]

            H0, Om, w0, Xi0, nXi0 = theta5

            # Build grids once (fast + used for dc/log interpolation)
            dc_grid = dcfun_quad(bk, zgrid, H0, Om, w0, x01, w01)
            log_ddL_dz_grid = log_ddL_dz(bk, zgrid, H0, Om, w0, Xi0, nXi0, dc=dc_grid, x01=x01, w01=w01, param=param)
            dL_grid = dLfun(bk, zgrid, H0, Om, w0, Xi0, nXi0, dc=dc_grid, param=param, x01=x01, w01=w01)

            # z(dL) with implicit-diff gradients
            z = z_from_dL_interp(dL, theta5, zgrid, dL_grid, x01, w01)

            # interpolate dc(z) and log_ddL_dz(z)
            iz, tz = _interp_prepare_jax(z, zgrid,  eps=eps_interp, side=side_interp )
            dc = _interp_apply_jax(iz, tz, dc_grid)
            log_ddL_dz_val = _interp_apply_jax(iz, tz, log_ddL_dz_grid)

            return z, dc, log_ddL_dz_val

        # VJP wrt (dL, Lambda)
        def vjp(dL, Lambda, gz, gdc, glog):
            (_, _, _), pull = jax.vjp(f, dL, Lambda)
            gdL, gLam = pull((gz, gdc, glog))
            return gdL, gLam

        return jax.jit(vjp)

    def make_node(self, dL, Lambda, gz, gdc, glog):
        dL = at.as_tensor_variable(dL)
        Lambda = at.as_tensor_variable(Lambda)
        gz = at.as_tensor_variable(gz)
        gdc = at.as_tensor_variable(gdc)
        glog = at.as_tensor_variable(glog)
        return Apply(self, [dL, Lambda, gz, gdc, glog], [at.dvector(), at.dvector()])

    def perform(self, node, inputs, outputs):
        dL, Lambda, gz, gdc, glog = inputs
        gdL, gLam = self._jax_vjp(_to_jnp(dL), _to_jnp(Lambda), _to_jnp(gz), _to_jnp(gdc), _to_jnp(glog))
        outputs[0][0] = np.asarray(gdL, dtype="float64")
        outputs[1][0] = np.asarray(gLam, dtype="float64")


class CosmoFromDLJAXOp(Op):
    """
    JAX-backed event-side cosmology:

      (z, dc, log_ddL_dz) = f(dL, Lambda)

    - dL: (N,)
    - Lambda: (P,) with cosmo in Lambda[:5] = (H0, Om, w0, Xi0, nXi0)

    Gradients:
      correct w.r.t. dL and Lambda (cosmo via implicit diff of the inversion).
    """

    itypes = [at.dvector, at.dvector]
    otypes = [at.dvector, at.dvector, at.dvector]

    def __init__(
        self,
        *,
        zgrid,
        x01,
        w01,
        param="vanilla",
        eps_interp=1e-12,
        side_interp="right",
    ):
        super().__init__()
        self.zgrid = np.asarray(zgrid, dtype="float64")
        self.x01 = np.asarray(x01, dtype="float64")
        self.w01 = np.asarray(w01, dtype="float64")
        self.param = str(param)
        self.eps_interp = float(eps_interp)
        self.side_interp = str(side_interp)

        self._jax_fwd = self._build_jax_fwd()
        self._vjp_op = _CosmoFromDLJAXVJPOp(
            zgrid=self.zgrid,
            x01=self.x01,
            w01=self.w01,
            param=self.param,
            eps_interp=self.eps_interp,
            side_interp=self.side_interp,
        )

    def _build_jax_fwd(self):
        bk = JAXBackend()
        zgrid = jnp.asarray(self.zgrid)
        x01 = jnp.asarray(self.x01)
        w01 = jnp.asarray(self.w01)
        param = self.param
        eps_interp = self.eps_interp
        side_interp = self.side_interp

        z_from_dL_interp = make_z_from_dL_interp(bk, param=param)
        

        def f(dL, Lambda):
            theta5 = Lambda[:5]
            H0, Om, w0, Xi0, nXi0 = theta5

            dc_grid = dcfun_quad(bk, zgrid, H0, Om, w0, x01, w01)
            log_ddL_dz_grid = log_ddL_dz(bk, zgrid, H0, Om, w0, Xi0, nXi0, dc=dc_grid, x01=x01, w01=w01, param=param)
            dL_grid = dLfun(bk, zgrid, H0, Om, w0, Xi0, nXi0, dc=dc_grid, param=param, x01=x01, w01=w01)

            z = z_from_dL_interp(dL, theta5, zgrid, dL_grid, x01, w01)

            iz, tz = _interp_prepare_jax(z, zgrid, eps=eps_interp, side=side_interp)
            dc = _interp_apply_jax(iz, tz, dc_grid)
            log_ddL_dz_val = _interp_apply_jax(iz, tz, log_ddL_dz_grid)

            return z, dc, log_ddL_dz_val

        return jax.jit(f)

    def make_node(self, dL, Lambda):
        dL = at.as_tensor_variable(dL)
        Lambda = at.as_tensor_variable(Lambda)
        return Apply(self, [dL, Lambda], [at.dvector(), at.dvector(), at.dvector()])

    def perform(self, node, inputs, outputs):
        dL, Lambda = inputs
        z, dc, log_ddL_dz_val = self._jax_fwd(_to_jnp(dL), _to_jnp(Lambda))
        outputs[0][0] = np.asarray(z, dtype="float64")
        outputs[1][0] = np.asarray(dc, dtype="float64")
        outputs[2][0] = np.asarray(log_ddL_dz_val, dtype="float64")

    def grad(self, inputs, output_grads):
        dL, Lambda = inputs
        gz, gdc, glog = output_grads
        gz   = _as_vec_like(gz, dL)
        gdc  = _as_vec_like(gdc, dL)
        glog = _as_vec_like(glog, dL)


        gdL, gLam = self._vjp_op(dL, Lambda, gz, gdc, glog)
        return [gdL, gLam]



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
        return [grad_not_implemented(self, 0, inputs[0])]


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

        z_grid_jax = None if self.z_grid is None else jnp.asarray(self.z_grid, dtype="float64")

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

        g_out = _as_vec_like(g_out, m1s)   # since output is (N,)

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

# def _connected_g(g):
#     if isinstance(getattr(g, "type", None), DisconnectedType):
#         return at.as_tensor_variable(np.array(0.0, dtype="float64"))
#     return g
# def _connected_g(g):
#     return at.as_tensor_variable(0.0, dtype="float64") if isinstance(g.type, DisconnectedType) else g

def _connected_g(g):
    t = getattr(g, "type", None)
    return at.as_tensor_variable(0.0, dtype="float64") if isinstance(t, DisconnectedType) else g

def _to_jnp(x):
    # your existing helper likely does this already; keep yours if you have one
    return jnp.asarray(x, dtype=jnp.float64)


def _as_vec_like(g, like):
    # if disconnected, g is a scalar 0.0; if connected, it's already a vector
    g = _connected_g(g)
    return at.broadcast_to(g, like.shape)

def _as_scalar(g):
    g = _connected_g(g)
    return at.as_tensor_variable(g).reshape(())  # force scalar



# ----------------------------
# VJP Op (Selection)
# ----------------------------

class _SelectionBiasJAXVJPOp(Op):
    """
    VJP for selection bias.

    Inputs:
      m1inj, m2inj, dLinj: (Ninj,)
      spinsInj: (Ninj, nspin)
      log_p_draw: (Ninj,)
      log_p_incl: (Ninj,)
      Lambda: (P,)  with cosmology params in Lambda[:5] = (H0, Om, w0, Xi0, nXi0)
      Ndraw: scalar
      g_log_mu: scalar
      g_var: scalar

    Outputs:
      dLambda: (P,)
    """

    itypes = [
        at.dvector,  # m1inj
        at.dvector,  # m2inj
        at.dvector,  # dLinj
        at.dmatrix,  # spinsInj
        at.dvector,  # log_p_draw
        at.dvector,  # log_p_incl
        at.dvector,  # Lambda
        at.dscalar,  # Ndraw
        at.dscalar,  # g_log_mu
        at.dscalar,  # g_var
    ]
    otypes = [at.dvector]  # dLambda

    def __init__(
        self,
        *,
        zgrid,
        x01,
        w01,
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
        self.x01 = np.asarray(x01, dtype="float64")
        self.w01 = np.asarray(w01, dtype="float64")

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

        self.eps_interp = float(eps_interp)
        self.side_interp = str(side_interp)

        self.interp_vals_mass = None if interp_vals_mass is None else np.asarray(interp_vals_mass)
        if interp_grids_mass is None:
            self.interp_grids_mass = None
        else:
            self.interp_grids_mass = tuple(np.asarray(g) for g in interp_grids_mass)

        self.z_grid = None if z_grid is None else np.asarray(z_grid, dtype="float64")

        self._jax_vjp = self._build_jax_vjp()
        self._cached = None  # cache big fixed arrays on device

    def _build_jax_vjp(self):
        # static arrays
        zgrid_jax = jnp.asarray(self.zgrid)
        x01_jax = jnp.asarray(self.x01)
        w01_jax = jnp.asarray(self.w01)

        # config
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

        interp_vals_mass_jax = None if self.interp_vals_mass is None else jnp.asarray(self.interp_vals_mass)
        if self.interp_grids_mass is None:
            interp_grids_mass_jax = None
        else:
            interp_grids_mass_jax = tuple(jnp.asarray(g) for g in self.interp_grids_mass)

        z_grid_jax = None if self.z_grid is None else jnp.asarray(self.z_grid)

        bk = JAXBackend()
        spins_unpack = lambda s: spin_models._spins_as_list(s, spin_model)

        def _sel_from_Lambda(m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl, Lambda, Ndraw):
            # Cosmology params are Lambda[:5]
            H0, Om, w0, Xi0, nXi0 = Lambda[0], Lambda[1], Lambda[2], Lambda[3], Lambda[4]

            # Build grids INSIDE (so gradients wrt cosmology params flow via chain rule)
            dc_grid = dcfun_quad(bk, zgrid_jax, H0, Om, w0, x01_jax, w01_jax)
            dL_grid = dLfun(
                bk, zgrid_jax, H0, Om, w0, Xi0, nXi0,
                x01=x01_jax, w01=w01_jax, dc=dc_grid, param=param
            )
            log_ddL_dz_grid = log_ddL_dz(
                bk, zgrid_jax, H0, Om, w0, Xi0, nXi0,
                dc=dc_grid, x01=x01_jax, w01=w01_jax, param=param
            )

            return sel_bias_with_uncertainty(
                bk,
                m1inj, m2inj, dLinj,
                spins_unpack(spinsInj),
                log_p_draw, log_p_incl,
                dL_grid, dc_grid, log_ddL_dz_grid,
                Lambda, Ndraw,
                zgrid=zgrid_jax,
                rate_model=rate_model,
                mass_model=mass_model,
                spin_model=spin_model,
                smoothing=smoothing,
                simplex_repair=simplex_repair,
                has_m2_break=has_m2_break,
                norm_gauss=norm_gauss,
                param=param,
                interp_vals_mass=interp_vals_mass_jax,
                interp_grids_mass=interp_grids_mass_jax,
                is_observed=is_observed,
                z_grid=z_grid_jax,
                verbose=verbose,
                subtract_log_p_incl=subtract_log_p_incl,
                eps_interp=eps_interp,
                side_interp=side_interp,
            )

        def _vjp(m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl, Lambda, Ndraw, g_log_mu, g_var):
            Ndraw = jnp.asarray(Ndraw).reshape(())
            g_log_mu = jnp.asarray(g_log_mu, dtype=jnp.float64).reshape(())
            g_var = jnp.asarray(g_var, dtype=jnp.float64).reshape(())

            def _f(Lambda_):
                return _sel_from_Lambda(m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl, Lambda_, Ndraw)

            (_, _), pullback = jax.vjp(_f, Lambda)
            (dLambda,) = pullback((g_log_mu, g_var))
            return dLambda

        return jax.jit(_vjp)

    def make_node(self, *inputs):
        inputs = list(map(at.as_tensor_variable, inputs))
        return Apply(self, inputs, [at.dvector()])

    def perform(self, node, inputs, outputs):
        (m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl, Lambda, Ndraw, g_log_mu, g_var) = inputs

        # cache large fixed arrays once
        if self._cached is None:
            self._cached = (
                jax.device_put(_to_jnp(m1inj)),
                jax.device_put(_to_jnp(m2inj)),
                jax.device_put(_to_jnp(dLinj)),
                jax.device_put(_to_jnp(spinsInj)),
                jax.device_put(_to_jnp(log_p_draw)),
                jax.device_put(_to_jnp(log_p_incl)),
                jax.device_put(_to_jnp(Ndraw).reshape(()))
            )
        m1inj_j, m2inj_j, dLinj_j, spins_j, lpd_j, lpi_j, Ndraw_j = self._cached

        dLambda = self._jax_vjp(
            m1inj_j, m2inj_j, dLinj_j, spins_j, lpd_j, lpi_j,
            jax.device_put(_to_jnp(Lambda)),
            Ndraw_j,
            jax.device_put(_to_jnp(g_log_mu).reshape(())),
            jax.device_put(_to_jnp(g_var).reshape(())),
        )
        outputs[0][0] = np.asarray(dLambda, dtype="float64")


# ----------------------------
# Main Selection Op
# ----------------------------

class SelectionBiasJAXOp(Op):
    """
    Forward returns (log_mu, var_u).

    Grad returns only dLambda. All other inputs are treated as fixed data.
    Lambda[:5] are cosmology params: (H0, Om, w0, Xi0, nXi0).
    """

    itypes = [
        at.dvector,  # m1inj
        at.dvector,  # m2inj
        at.dvector,  # dLinj
        at.dmatrix,  # spinsInj
        at.dvector,  # log_p_draw
        at.dvector,  # log_p_incl
        at.dvector,  # Lambda
        at.dscalar,  # Ndraw
    ]
    otypes = [at.dscalar, at.dscalar]  # (log_mu, var_u)

    def __init__(self, *, zgrid, x01, w01, **kwargs):
        super().__init__()
        self.zgrid = np.asarray(zgrid, dtype="float64")
        self.x01 = np.asarray(x01, dtype="float64")
        self.w01 = np.asarray(w01, dtype="float64")
        self.kwargs = dict(kwargs)

        self._jax_fwd = self._build_jax_fwd()
        self._vjp_op = _SelectionBiasJAXVJPOp(zgrid=self.zgrid, x01=self.x01, w01=self.w01, **self.kwargs)
        self._cached = None

    def _build_jax_fwd(self):
        zgrid_jax = jnp.asarray(self.zgrid)
        x01_jax = jnp.asarray(self.x01)
        w01_jax = jnp.asarray(self.w01)

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

        interp_vals_mass_jax = None if interp_vals_mass is None else jnp.asarray(interp_vals_mass)
        if interp_grids_mass is None:
            interp_grids_mass_jax = None
        else:
            interp_grids_mass_jax = tuple(jnp.asarray(g) for g in interp_grids_mass)

        z_grid_jax = None if z_grid is None else jnp.asarray(z_grid)

        bk = JAXBackend()
        spins_unpack = lambda s: spin_models._spins_as_list(s, spin_model)

        def _f(m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl, Lambda, Ndraw):
            Ndraw = jnp.asarray(Ndraw).reshape(())

            H0, Om, w0, Xi0, nXi0 = Lambda[0], Lambda[1], Lambda[2], Lambda[3], Lambda[4]

            dc_grid = dcfun_quad(bk, zgrid_jax, H0, Om, w0, x01_jax, w01_jax)
            dL_grid = dLfun(
                bk, zgrid_jax, H0, Om, w0, Xi0, nXi0,
                x01=x01_jax, w01=w01_jax, dc=dc_grid, param=param
            )

            
            log_ddL_dz_grid = log_ddL_dz(
                bk, zgrid_jax, H0, Om, w0, Xi0, nXi0,
                dc=dc_grid, x01=x01_jax, w01=w01_jax, param=param
            )

            log_mu, var_u = sel_bias_with_uncertainty(
                bk,
                m1inj, m2inj, dLinj,
                spins_unpack(spinsInj),
                log_p_draw, log_p_incl,
                dL_grid, dc_grid, log_ddL_dz_grid,
                Lambda, Ndraw,
                zgrid=zgrid_jax,
                rate_model=rate_model,
                mass_model=mass_model,
                spin_model=spin_model,
                smoothing=smoothing,
                simplex_repair=simplex_repair,
                has_m2_break=has_m2_break,
                norm_gauss=norm_gauss,
                param=param,
                interp_vals_mass=interp_vals_mass_jax,
                interp_grids_mass=interp_grids_mass_jax,
                is_observed=is_observed,
                z_grid=z_grid_jax,
                verbose=verbose,
                subtract_log_p_incl=subtract_log_p_incl,
                eps_interp=eps_interp,
                side_interp=side_interp,
            )
            return jnp.asarray(log_mu).reshape(()), jnp.asarray(var_u).reshape(())

        return jax.jit(_f)

    def make_node(self, *inputs):
        inputs = list(map(at.as_tensor_variable, inputs))
        return Apply(self, inputs, [at.dscalar(), at.dscalar()])

    def perform(self, node, inputs, outputs):
        (m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl, Lambda, Ndraw) = inputs

        if self._cached is None:
            self._cached = (
                jax.device_put(_to_jnp(m1inj)),
                jax.device_put(_to_jnp(m2inj)),
                jax.device_put(_to_jnp(dLinj)),
                jax.device_put(_to_jnp(spinsInj)),
                jax.device_put(_to_jnp(log_p_draw)),
                jax.device_put(_to_jnp(log_p_incl)),
                jax.device_put(_to_jnp(Ndraw).reshape(())) ,
            )
        m1inj_j, m2inj_j, dLinj_j, spins_j, lpd_j, lpi_j, Ndraw_j = self._cached

        log_mu, var_u = self._jax_fwd(
            m1inj_j, m2inj_j, dLinj_j, spins_j, lpd_j, lpi_j,
            jax.device_put(_to_jnp(Lambda)),
            Ndraw_j,
        )
        outputs[0][0] = np.asarray(log_mu, dtype="float64")
        outputs[1][0] = np.asarray(var_u, dtype="float64")

    def grad(self, inputs, output_grads):
        g_log_mu, g_var = output_grads
        g_log_mu = _as_scalar(g_log_mu)
        g_var =  _as_scalar(g_var)

        (m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl, Lambda, Ndraw) = inputs

        dLambda = self._vjp_op(
            m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl,
            Lambda, Ndraw, g_log_mu, g_var
        )

        # Treat all non-Lambda inputs as fixed data
        z_m1 = at.zeros_like(m1inj, dtype="float64")
        z_m2 = at.zeros_like(m2inj, dtype="float64")
        z_dL = at.zeros_like(dLinj, dtype="float64")
        z_sp = at.zeros_like(spinsInj, dtype="float64")
        z_lpd = at.zeros_like(log_p_draw, dtype="float64")
        z_lpi = at.zeros_like(log_p_incl, dtype="float64")
        z_Ndraw = at.zeros_like(Ndraw, dtype="float64")

        return [z_m1, z_m2, z_dL, z_sp, z_lpd, z_lpi, dLambda, z_Ndraw]

    def reset_cache(self):
        self._cached = None






class _PopAndSelJAXVJPOp(Op):
    itypes = [
        at.dvector, at.dvector, at.dvector, at.dmatrix,  # evt
        at.dvector, at.dvector, at.dvector, at.dmatrix, at.dvector, at.dvector,  # inj
        at.dvector, at.dscalar,  # Lambda, Ndraw
        at.dvector, at.dscalar, at.dscalar,  # cotangents
    ]
    otypes = [at.dvector, at.dvector, at.dvector, at.dmatrix, at.dvector]

    def __init__(self, *, zgrid, x01, w01, rate_model, mass_model, spin_model,
                 smoothing="LVK", simplex_repair=False, has_m2_break=False, norm_gauss="uplow",
                 param="vanilla", interp_vals_mass=None, interp_grids_mass=None,
                 is_observed=False, z_grid=None, verbose=False,
                 subtract_log_p_incl=True, eps_interp=1e-12, side_interp="right"):
        super().__init__()
        self.zgrid = np.asarray(zgrid, dtype="float64")
        self.x01 = np.asarray(x01, dtype="float64")
        self.w01 = np.asarray(w01, dtype="float64")

        self.rate_model = rate_model
        self.mass_model = mass_model
        self.spin_model = spin_model
        self.smoothing = smoothing
        self.simplex_repair = bool(simplex_repair)
        self.has_m2_break = bool(has_m2_break)
        self.norm_gauss = norm_gauss
        self.param = str(param)
        self.is_observed = bool(is_observed)
        self.verbose = bool(verbose)
        self.subtract_log_p_incl = bool(subtract_log_p_incl)
        self.eps_interp = float(eps_interp)
        self.side_interp = str(side_interp)

        self.interp_vals_mass = None if interp_vals_mass is None else np.asarray(interp_vals_mass)
        self.interp_grids_mass = None if interp_grids_mass is None else tuple(np.asarray(g) for g in interp_grids_mass)
        self.z_grid = None if z_grid is None else np.asarray(z_grid)

        self._cached_inj = None
        self._jax_vjp = self._build_jax_vjp()

    def _build_jax_vjp(self):
        bk = JAXBackend()
        z_from_dL_interp = make_z_from_dL_interp(bk, param=self.param)

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

        interp_vals_mass_jax = None if self.interp_vals_mass is None else _to_jnp(self.interp_vals_mass)
        interp_grids_mass_jax = None if self.interp_grids_mass is None else tuple(_to_jnp(g) for g in self.interp_grids_mass)
        pop_z_grid = None if self.z_grid is None else jnp.asarray(self.z_grid, dtype=jnp.float64)

        cosmo_zgrid = jnp.asarray(self.zgrid, dtype=jnp.float64)
        x01_jax = jnp.asarray(self.x01, dtype=jnp.float64)
        w01_jax = jnp.asarray(self.w01, dtype=jnp.float64)

        spins_unpack_evt = lambda s: spin_models._spins_as_list(s, spin_model)
        spins_unpack_inj = lambda s: spin_models._spins_as_list(s, spin_model)

        def _f(m1det, m2det, dLdet, spins_evt,
               m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
               Lambda, Ndraw):

            Ndraw = Ndraw.reshape(())
            theta5 = Lambda[:5]
            H0, Om, w0, Xi0, nXi0 = theta5

            dc_grid = dcfun_quad(bk, cosmo_zgrid, H0, Om, w0, x01_jax, w01_jax)
            dL_grid = dLfun(bk, cosmo_zgrid, H0, Om, w0, Xi0, nXi0, dc=dc_grid, param=param, x01=x01_jax, w01=w01_jax)
            log_ddL_dz_grid = log_ddL_dz(bk, cosmo_zgrid, H0, Om, w0, Xi0, nXi0, dc=dc_grid, x01=x01_jax, w01=w01_jax, param=param)

            # events
            z_evt = z_from_dL_interp(dLdet, theta5, cosmo_zgrid, dL_grid, x01_jax, w01_jax)
            iz, tz = _interp_prepare_jax(z_evt, cosmo_zgrid, eps=eps_interp, side=side_interp)
            dc_evt = _interp_apply_jax(iz, tz, dc_grid)
            log_ddL_dz_evt = _interp_apply_jax(iz, tz, log_ddL_dz_grid)

            onepz = 1.0 + z_evt
            m1src = m1det / onepz
            m2src = m2det / onepz

            logp_pop_evt = log_p_pop(
                bk, m1src, m2src, z_evt, dLdet,
                spins_unpack_evt(spins_evt), Lambda,
                rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
                smoothing=smoothing, simplex_repair=simplex_repair,
                has_m2_break=has_m2_break, norm_gauss=norm_gauss,
                dc=dc_evt, log_ddL_dz_pre=log_ddL_dz_evt,
                param=param,
                interp_vals_mass=interp_vals_mass_jax,
                interp_grids_mass=interp_grids_mass_jax,
                is_observed=is_observed, z_grid=pop_z_grid, verbose=verbose,
            )

            # injections (precompute and pass)
            z_inj = z_from_dL_interp(dLinj, theta5, cosmo_zgrid, dL_grid, x01_jax, w01_jax)
            iz2, tz2 = _interp_prepare_jax(z_inj, cosmo_zgrid, eps=eps_interp, side=side_interp)
            dc_inj = _interp_apply_jax(iz2, tz2, dc_grid)
            log_ddL_dz_inj = _interp_apply_jax(iz2, tz2, log_ddL_dz_grid)

            log_mu, var_u = sel_bias_with_uncertainty(
                bk,
                m1inj, m2inj, dLinj,
                spins_unpack_inj(spins_inj),
                log_p_draw, log_p_incl,
                dL_grid, dc_grid, log_ddL_dz_grid,
                Lambda, Ndraw,
                zgrid=cosmo_zgrid,
                rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
                smoothing=smoothing, simplex_repair=simplex_repair,
                has_m2_break=has_m2_break, norm_gauss=norm_gauss,
                param=param,
                interp_vals_mass=interp_vals_mass_jax,
                interp_grids_mass=interp_grids_mass_jax,
                is_observed=is_observed, z_grid=pop_z_grid, verbose=verbose,
                subtract_log_p_incl=subtract_log_p_incl,
                eps_interp=eps_interp, side_interp=side_interp,
                zinj=z_inj, dcinj=dc_inj, log_ddL_dz_inj=log_ddL_dz_inj,
            )

            return logp_pop_evt, jnp.asarray(log_mu, jnp.float64).reshape(()), jnp.asarray(var_u, jnp.float64).reshape(())

        def _vjp(m1det, m2det, dLdet, spins_evt,
                 m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
                 Lambda, Ndraw,
                 g_logp_pop, g_log_mu, g_var_u):

            g_logp_pop = jnp.reshape(g_logp_pop, m1det.shape)
            g_log_mu = jnp.reshape(g_log_mu, ())
            g_var_u = jnp.reshape(g_var_u, ())

            (_, _, _), pull = jax.vjp(
                _f,
                m1det, m2det, dLdet, spins_evt,
                m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
                Lambda, Ndraw
            )
            grads = pull((g_logp_pop, g_log_mu, g_var_u))
            return grads[0], grads[1], grads[2], grads[3], grads[10]

        return jax.jit(_vjp)

    def make_node(self, *inputs):
        inputs = list(map(at.as_tensor_variable, inputs))
        outs = [at.dvector(), at.dvector(), at.dvector(), at.dmatrix(), at.dvector()]
        return Apply(self, inputs, outs)

    def perform(self, node, inputs, outputs):
        (m1det, m2det, dLdet, spins_evt,
         m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
         Lambda, Ndraw,
         g_logp_pop, g_log_mu, g_var_u) = inputs

        if self._cached_inj is None:
            self._cached_inj = (
                jax.device_put(_to_jnp(m1inj)),
                jax.device_put(_to_jnp(m2inj)),
                jax.device_put(_to_jnp(dLinj)),
                jax.device_put(_to_jnp(spins_inj)),
                jax.device_put(_to_jnp(log_p_draw)),
                jax.device_put(_to_jnp(log_p_incl)),
                jax.device_put(float(np.asarray(Ndraw).reshape(()))),
            )
        m1inj_j, m2inj_j, dLinj_j, spins_inj_j, lpd_j, lpi_j, Ndraw_j = self._cached_inj

        g_logp_pop = np.asarray(g_logp_pop, dtype=np.float64)
        if g_logp_pop.ndim == 0:
            g_logp_pop = np.zeros_like(m1det, dtype=np.float64)
        else:
            g_logp_pop = g_logp_pop.reshape(m1det.shape)
        g_log_mu = np.asarray(g_log_mu, dtype=np.float64).reshape(())
        g_var_u = np.asarray(g_var_u, dtype=np.float64).reshape(())

        dm1det, dm2det, ddLdet, dsp_evt, dLam = self._jax_vjp(
            jax.device_put(_to_jnp(m1det)),
            jax.device_put(_to_jnp(m2det)),
            jax.device_put(_to_jnp(dLdet)),
            jax.device_put(_to_jnp(spins_evt)),
            m1inj_j, m2inj_j, dLinj_j, spins_inj_j, lpd_j, lpi_j,
            jax.device_put(_to_jnp(Lambda)),
            Ndraw_j,
            jax.device_put(_to_jnp(g_logp_pop)),
            jax.device_put(_to_jnp(g_log_mu)),
            jax.device_put(_to_jnp(g_var_u)),
        )

        outputs[0][0] = np.asarray(dm1det, dtype="float64")
        outputs[1][0] = np.asarray(dm2det, dtype="float64")
        outputs[2][0] = np.asarray(ddLdet, dtype="float64")
        outputs[3][0] = np.asarray(dsp_evt, dtype="float64")
        outputs[4][0] = np.asarray(dLam, dtype="float64")


class PopAndSelJAXOp(Op):
    itypes = [
        at.dvector, at.dvector, at.dvector, at.dmatrix,
        at.dvector, at.dvector, at.dvector, at.dmatrix, at.dvector, at.dvector,
        at.dvector, at.dscalar,
    ]
    otypes = [at.dvector, at.dscalar, at.dscalar]

    def __init__(self, *, zgrid, x01, w01, rate_model, mass_model, spin_model,
                 smoothing="LVK", simplex_repair=False, has_m2_break=False, norm_gauss="uplow",
                 param="vanilla", interp_vals_mass=None, interp_grids_mass=None,
                 is_observed=False, z_grid=None, verbose=False,
                 subtract_log_p_incl=True, eps_interp=1e-12, side_interp="right"):
        super().__init__()
        self.zgrid = np.asarray(zgrid, dtype="float64")
        self.x01 = np.asarray(x01, dtype="float64")
        self.w01 = np.asarray(w01, dtype="float64")

        self.kw = dict(
            zgrid=self.zgrid, x01=self.x01, w01=self.w01,
            rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
            smoothing=smoothing, simplex_repair=simplex_repair, has_m2_break=has_m2_break,
            norm_gauss=norm_gauss, param=param,
            interp_vals_mass=interp_vals_mass, interp_grids_mass=interp_grids_mass,
            is_observed=is_observed, z_grid=z_grid, verbose=verbose,
            subtract_log_p_incl=subtract_log_p_incl,
            eps_interp=eps_interp, side_interp=side_interp,
        )

        self._jax_fwd = self._build_jax_fwd()
        self._cached_inj = None
        self._vjp_op = _PopAndSelJAXVJPOp(**self.kw)

    def _build_jax_fwd(self):
        bk = JAXBackend()
        z_from_dL_interp = make_z_from_dL_interp(bk, param=self.kw["param"])

        zgrid = jnp.asarray(self.zgrid, dtype=jnp.float64)
        x01   = jnp.asarray(self.x01,   dtype=jnp.float64)
        w01   = jnp.asarray(self.w01,   dtype=jnp.float64)

        rate_model = self.kw["rate_model"]
        mass_model = self.kw["mass_model"]
        spin_model = self.kw["spin_model"]
        smoothing  = self.kw["smoothing"]
        simplex_repair = self.kw["simplex_repair"]
        has_m2_break   = self.kw["has_m2_break"]
        norm_gauss = self.kw["norm_gauss"]
        param      = self.kw["param"]
        is_observed = self.kw["is_observed"]
        verbose     = self.kw["verbose"]
        subtract_log_p_incl = self.kw["subtract_log_p_incl"]
        eps_interp  = self.kw["eps_interp"]
        side_interp = self.kw["side_interp"]

        interp_vals_mass = self.kw["interp_vals_mass"]
        interp_grids_mass = self.kw["interp_grids_mass"]
        z_grid = self.kw["z_grid"]

        interp_vals_mass_jax = None if interp_vals_mass is None else jnp.asarray(interp_vals_mass, dtype=jnp.float64)
        if interp_grids_mass is None:
            interp_grids_mass_jax = None
        else:
            interp_grids_mass_jax = tuple(jnp.asarray(g, dtype=jnp.float64) for g in interp_grids_mass)
        z_grid_jax = None if z_grid is None else jnp.asarray(z_grid, dtype=jnp.float64)

        spins_unpack_evt = lambda s: spin_models._spins_as_list(s, spin_model)
        spins_unpack_inj = lambda s: spin_models._spins_as_list(s, spin_model)

        def _f(
            m1det, m2det, dLdet, spins_evt,
            m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
            Lambda, Ndraw
        ):
            theta5 = Lambda[:5]
            H0, Om, w0, Xi0, nXi0 = theta5

            dc_grid = dcfun_quad(bk, zgrid, H0, Om, w0, x01, w01)
            dL_grid = dLfun(bk, zgrid, H0, Om, w0, Xi0, nXi0, dc=dc_grid, param=param, x01=x01, w01=w01)
            log_ddL_dz_grid = log_ddL_dz(bk, zgrid, H0, Om, w0, Xi0, nXi0, dc=dc_grid, x01=x01, w01=w01, param=param)

            # event
            z_evt = z_from_dL_interp(dLdet, theta5, zgrid, dL_grid, x01, w01)
            iz, tz = _interp_prepare_jax(z_evt, zgrid, eps=eps_interp, side=side_interp)
            dc_evt = _interp_apply_jax(iz, tz, dc_grid)
            log_ddL_dz_evt = _interp_apply_jax(iz, tz, log_ddL_dz_grid)

            onepz = 1.0 + z_evt
            m1src = m1det / onepz
            m2src = m2det / onepz

            logp_pop_evt = log_p_pop(
                bk,
                m1src, m2src, z_evt, dLdet,
                spins_unpack_evt(spins_evt),
                Lambda,
                rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
                smoothing=smoothing, simplex_repair=simplex_repair, has_m2_break=has_m2_break,
                norm_gauss=norm_gauss,
                dc=dc_evt, log_ddL_dz_pre=log_ddL_dz_evt,
                param=param,
                interp_vals_mass=interp_vals_mass_jax,
                interp_grids_mass=interp_grids_mass_jax,
                is_observed=is_observed,
                z_grid=z_grid_jax,
                verbose=verbose,
            )

            # inj (precompute cosmology pieces; passed into sel_bias)
            z_inj = z_from_dL_interp(dLinj, theta5, zgrid, dL_grid, x01, w01)
            iz2, tz2 = _interp_prepare_jax(z_inj, zgrid, eps=eps_interp, side=side_interp)
            dc_inj = _interp_apply_jax(iz2, tz2, dc_grid)
            log_ddL_dz_inj = _interp_apply_jax(iz2, tz2, log_ddL_dz_grid)

            log_mu, var_u = sel_bias_with_uncertainty(
                bk,
                m1inj, m2inj, dLinj,
                spins_unpack_inj(spins_inj),
                log_p_draw, log_p_incl,
                dL_grid, dc_grid, log_ddL_dz_grid,
                Lambda, Ndraw,
                zgrid=zgrid,
                rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
                smoothing=smoothing, simplex_repair=simplex_repair, has_m2_break=has_m2_break,
                norm_gauss=norm_gauss,
                param=param,
                interp_vals_mass=interp_vals_mass_jax,
                interp_grids_mass=interp_grids_mass_jax,
                is_observed=is_observed,
                z_grid=z_grid_jax,
                verbose=verbose,
                subtract_log_p_incl=subtract_log_p_incl,
                eps_interp=eps_interp,
                side_interp=side_interp,
                zinj=z_inj, dcinj=dc_inj, log_ddL_dz_inj=log_ddL_dz_inj,
            )

            return logp_pop_evt, jnp.asarray(log_mu, dtype=jnp.float64).reshape(()), jnp.asarray(var_u, dtype=jnp.float64).reshape(())

        return jax.jit(_f)

    def make_node(self, *inputs):
        inputs = list(map(at.as_tensor_variable, inputs))
        return Apply(self, inputs, [at.dvector(), at.dscalar(), at.dscalar()])

    def perform(self, node, inputs, outputs):
        (m1det, m2det, dLdet, spins_evt,
         m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
         Lambda, Ndraw) = inputs

        if self._cached_inj is None:
            self._cached_inj = (
                jax.device_put(_to_jnp(m1inj)),
                jax.device_put(_to_jnp(m2inj)),
                jax.device_put(_to_jnp(dLinj)),
                jax.device_put(_to_jnp(spins_inj)),
                jax.device_put(_to_jnp(log_p_draw)),
                jax.device_put(_to_jnp(log_p_incl)),
                jax.device_put(float(np.asarray(Ndraw).reshape(()))),
            )
        m1inj_j, m2inj_j, dLinj_j, spins_inj_j, lpd_j, lpi_j, Ndraw_j = self._cached_inj

        y1, y2, y3 = self._jax_fwd(
            jax.device_put(_to_jnp(m1det)),
            jax.device_put(_to_jnp(m2det)),
            jax.device_put(_to_jnp(dLdet)),
            jax.device_put(_to_jnp(spins_evt)),
            m1inj_j, m2inj_j, dLinj_j, spins_inj_j, lpd_j, lpi_j,
            jax.device_put(_to_jnp(Lambda)),
            Ndraw_j,
        )

        outputs[0][0] = np.asarray(y1, dtype="float64")
        outputs[1][0] = np.asarray(y2, dtype="float64")
        outputs[2][0] = np.asarray(y3, dtype="float64")

    def grad(self, inputs, output_grads):
        g_logp_pop, g_log_mu, g_var_u = output_grads
        (m1det, m2det, dLdet, spins_evt,
         m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
         Lambda, Ndraw) = inputs
    
        g_logp_pop = _as_vec_like(g_logp_pop, m1det)
        g_log_mu   = _as_scalar(g_log_mu)
        g_var_u    = _as_scalar(g_var_u)
    
        dm1det, dm2det, ddLdet, dspins_evt, dLambda = self._vjp_op(
            m1det, m2det, dLdet, spins_evt,
            m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
            Lambda, Ndraw,
            g_logp_pop, g_log_mu, g_var_u
        )
    
        # no grads for inj arrays, log_p_draw/log_p_incl, Ndraw
        z_m1inj = at.zeros_like(m1inj, dtype="float64")
        z_m2inj = at.zeros_like(m2inj, dtype="float64")
        z_dLinj = at.zeros_like(dLinj, dtype="float64")
        z_spinj = at.zeros_like(spins_inj, dtype="float64")
        z_lpd   = at.zeros_like(log_p_draw, dtype="float64")
        z_lpi   = at.zeros_like(log_p_incl, dtype="float64")
        z_Ndraw = at.zeros_like(Ndraw, dtype="float64")
    
        return [
            dm1det, dm2det, ddLdet, dspins_evt,
            z_m1inj, z_m2inj, z_dLinj, z_spinj, z_lpd, z_lpi,
            dLambda, z_Ndraw
        ]




def _interp1d_sorted(x, xp, fp, *, eps=1e-18):
    x = jnp.clip(x, xp[0], xp[-1])
    idx = jnp.searchsorted(xp, x, side="right")
    idx = jnp.clip(idx, 1, xp.shape[0] - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]

    t = (x - x0) / jnp.maximum(x1 - x0, eps)
    return (1.0 - t) * y0 + t * y1


# def make_z_from_dL_interp(bk, *, param="vanilla", newton_steps=2, z_eps=1e-12, max_step=0.2):
#     """
#     Invert dL -> z with:
#       - initial guess by inverting the grid (interp on (dL_grid -> zgrid))
#       - a few damped Newton steps on dLfun(z, theta)=dL
#       - backward: implicit-function gradient (includes theta gradients)

#     max_step controls the Newton step limiter in units of (1+z):
#       step <- clip(step, -max_step*(1+z), +max_step*(1+z))
#     """
#     param = str(param)
#     newton_steps = int(newton_steps)

#     # ---- pure primal implementation (NO custom_vjp calls inside) ----
#     def _z_impl(dL, theta5, zgrid, dL_grid, x01, w01):
#         # initial guess from monotonic grid inverse
#         z = _interp1d_sorted(dL, dL_grid, zgrid)
#         z = jnp.clip(z, zgrid[0] + z_eps, zgrid[-1] - z_eps)

#         H0, Om, w0, Xi0, nXi0 = theta5

#         def body(z, _):
#             dc = dcfun_quad(bk, z, H0, Om, w0, x01, w01)
#             dL_here = dLfun(bk, z, H0, Om, w0, Xi0, nXi0,
#                             dc=dc, x01=x01, w01=w01, param=param)
#             f = dL_here - dL  # want f=0

#             log_dd = log_ddL_dz(bk, z, H0, Om, w0, Xi0, nXi0,
#                                 dc=dc, x01=x01, w01=w01, param=param)
#             dd = jnp.exp(log_dd)

#             step = f / jnp.maximum(dd, 1e-300)

#             # ---- step limiter to avoid Newton explosions ----
#             lim = max_step * (1.0 + z)
#             step = jnp.clip(step, -lim, lim)

#             z_new = z - step
#             z_new = jnp.clip(z_new, zgrid[0] + z_eps, zgrid[-1] - z_eps)
#             return z_new, None

#         if newton_steps > 0:
#             z, _ = jax.lax.scan(body, z, xs=None, length=newton_steps)
#         return z

#     @jax.custom_vjp
#     def z_from_dL_interp(dL, theta5, zgrid, dL_grid, x01, w01):
#         return _z_impl(dL, theta5, zgrid, dL_grid, x01, w01)

#     def _z_fwd(dL, theta5, zgrid, dL_grid, x01, w01):
#         z = _z_impl(dL, theta5, zgrid, dL_grid, x01, w01)
#         return z, (z, theta5, x01, w01)

#     def _z_bwd(res, g_z):
#         z, theta5, x01, w01 = res
#         H0, Om, w0, Xi0, nXi0 = theta5

#         dc = dcfun_quad(bk, z, H0, Om, w0, x01, w01)
#         log_dd = log_ddL_dz(bk, z, H0, Om, w0, Xi0, nXi0,
#                             dc=dc, x01=x01, w01=w01, param=param)
#         inv = jnp.exp(-log_dd)  # 1 / (ddL/dz)

#         # dz/ddL = 1/(ddL/dz)
#         g_dL = g_z * inv

#         # dz/dtheta = - (∂_theta dL)/(∂_z dL)
#         v = -(g_z * inv)

#         def dL_at_theta(theta):
#             H0_, Om_, w0_, Xi0_, nXi0_ = theta
#             dc_ = dcfun_quad(bk, z, H0_, Om_, w0_, x01, w01)
#             return dLfun(bk, z, H0_, Om_, w0_, Xi0_, nXi0_,
#                          dc=dc_, x01=x01, w01=w01, param=param)

#         _, pull = jax.vjp(dL_at_theta, theta5)
#         (g_theta5,) = pull(v)

#         # grads for (dL, theta5, zgrid, dL_grid, x01, w01)
#         return (g_dL, g_theta5, None, None, None, None)

#     z_from_dL_interp.defvjp(_z_fwd, _z_bwd)
#     return z_from_dL_interp
        

# def _interp1d_sorted(x, xp, fp, *, eps=1e-18):
#     """
#     Linear interpolation fp(xp) -> y(x), with xp strictly increasing.
#     Clamps x to [xp[0], xp[-1]].
#     """
#     #x  = _as64(x)
#     #xp = _as64(xp)
#     #fp = _as64(fp)

#     x = jnp.clip(x, xp[0], xp[-1])
#     idx = jnp.searchsorted(xp, x, side="right")
#     idx = jnp.clip(idx, 1, xp.shape[0] - 1)

#     x0 = xp[idx - 1]
#     x1 = xp[idx]
#     y0 = fp[idx - 1]
#     y1 = fp[idx]

#     t = (x - x0) / jnp.maximum(x1 - x0, eps)
#     return (1.0 - t) * y0 + t * y1


# def make_z_from_dL_interp(bk, *, param="vanilla"):
#     param = str(param)

#     @jax.custom_vjp
#     def z_from_dL_interp(dL, theta5, zgrid, dL_grid, x01, w01):
#         return _interp1d_sorted(dL, dL_grid, zgrid)

#     def _z_fwd(dL, theta5, zgrid, dL_grid, x01, w01):
#         z = _interp1d_sorted(dL,dL_grid,zgrid)
#         return z, (z, theta5, x01, w01)  # no param cached

#     def _z_bwd(res, g_z):
#         z, theta5, x01, w01 = res
#         #g_z = _as64(g_z)
#         H0, Om, w0, Xi0, nXi0 = theta5

#         dc =   dcfun_quad(bk, z, H0, Om, w0, x01, w01)
#         log_dd =   log_ddL_dz(
#             bk, z, H0, Om, w0, Xi0, nXi0,
#             dc=dc, x01=x01, w01=w01, param=param,   # <- param from closure
#         )
#         #inv = 1.0 / jnp.exp(log_dd)
#         inv = jnp.exp(-log_dd)

#         g_dL = g_z * inv
#         v = -(g_z * inv)

#         def dL_vec(theta):
#             H0_, Om_, w0_, Xi0_, nXi0_ = theta
#             dc_ =   dcfun_quad(bk, z, H0_, Om_, w0_, x01, w01)
#             return   dLfun(
#                 bk, z, H0_, Om_, w0_, Xi0_, nXi0_,
#                 dc=dc_, x01=x01, w01=w01, param=param,  # <- closure
#             )

#         _, pull = jax.vjp(dL_vec, theta5)
#         (g_theta5,) = pull(v)

#         return (g_dL, g_theta5, None, None, None, None)

#     z_from_dL_interp.defvjp(_z_fwd, _z_bwd)
#     return z_from_dL_interp

 
# def make_z_from_dL_interp(bk, *, eps=1e-12, side="right", param="vanilla"):
#     """
#     Invert dL->z by linear interpolation on the (theta-dependent) table:
#         (dL_grid(theta), zgrid)  with dL_grid strictly increasing.

#     Forward: z = interp(dL; xp=dL_grid, fp=zgrid)
#     Backward: exact VJP of that piecewise-linear map w.r.t.:
#         - dL (query)
#         - dL_grid (knot positions)
#     This gives cosmology gradients through dL_grid(theta).

#     Safeguards:
#       - denom clamped by eps
#       - gradients are masked to 0 when dL is outside table range (clipped)
#       - gradients are masked to 0 when denom is too small
#     """

#     side = str(side)

#     def _interp_forward(dL, dL_grid, zgrid):
#         n = dL_grid.shape[0]

#         in_range = (dL >= dL_grid[0]) & (dL <= dL_grid[-1])

#         dL_c = jnp.clip(dL, dL_grid[0], dL_grid[-1])
#         idx = jnp.searchsorted(dL_grid, dL_c, side=side)
#         idx = jnp.clip(idx, 1, n - 1)

#         dl_lo = dL_grid[idx - 1]
#         dl_hi = dL_grid[idx]
#         z_lo  = zgrid[idx - 1]
#         z_hi  = zgrid[idx]

#         denom = dl_hi - dl_lo
#         safe = denom > eps
#         denom_s = jnp.where(safe, denom, 1.0)

#         t = (dL_c - dl_lo) / denom_s
#         t = jnp.where(safe, t, 0.0)

#         z = z_lo + t * (z_hi - z_lo)
#         return z, (idx, dL_c, in_range, safe, dl_lo, dl_hi, z_lo, z_hi, denom_s, dL_grid.shape)

#     @jax.custom_vjp
#     def z_from_dL_grid(dL, theta5, zgrid, dL_grid, x01, w01):
#         # IMPORTANT: must return ONLY z (NOT (z, res))
#         z, _ = _interp_forward(dL, dL_grid, zgrid)
#         return z

#     def fwd(dL, theta5, zgrid, dL_grid, x01, w01):
#         z, res = _interp_forward(dL, dL_grid, zgrid)
#         # cache res only; theta5/x01/w01 not needed here
#         return z, res

#     def bwd(res, g_z):
#         idx, dL_c, in_range, safe, dl_lo, dl_hi, z_lo, z_hi, denom_s, dL_grid_shape = res

#         dz = (z_hi - z_lo)

#         # ---- grad wrt query dL ----
#         # ∂z/∂dL = dz / denom
#         g_dL = g_z * (dz / denom_s)
#         g_dL = jnp.where(in_range & safe, g_dL, 0.0)

#         # ---- grad wrt knot locations dL_grid ----
#         # For active interval only:
#         # ∂z/∂dl_lo = dz * (dL - dl_hi) / denom^2
#         # ∂z/∂dl_hi = dz * (dl_lo - dL) / denom^2
#         denom2 = denom_s * denom_s
#         g_dl_lo = g_z * dz * (dL_c - dl_hi) / denom2
#         g_dl_hi = g_z * dz * (dl_lo - dL_c) / denom2
#         g_dl_lo = jnp.where(in_range & safe, g_dl_lo, 0.0)
#         g_dl_hi = jnp.where(in_range & safe, g_dl_hi, 0.0)

#         g_dL_grid = jnp.zeros(dL_grid_shape, dtype=g_z.dtype)
#         g_dL_grid = g_dL_grid.at[idx - 1].add(g_dl_lo)
#         g_dL_grid = g_dL_grid.at[idx].add(g_dl_hi)

#         # grads for (dL, theta5, zgrid, dL_grid, x01, w01)
#         return (g_dL, None, None, g_dL_grid, None, None)

#     z_from_dL_grid.defvjp(fwd, bwd)
#     return z_from_dL_grid




def make_z_from_dL_interp(bk, *, eps=1e-12, side="right", param="vanilla"):
    """
    Match PyTensor 'standard' atinterp semantics:

      - forward: piecewise-linear interpolation with idx clipped
                (=> extrapolates outside grid, does NOT clamp x)
      - backward: VJP of the same piecewise-linear map wrt:
            * dL (query)
            * dL_grid (knot locations)
        with denom = max(dl_hi - dl_lo, eps) (NO 'safe' mask, NO in_range mask)

    This is the closest JAX analogue to:
        idx = stop_grad(clip(searchsorted(xs,x),1,n-1))
        denom = max(xh-xl, eps)
        y = (1-r)*yl + r*yh
    """

    side = str(side)

    def _forward(dL, dL_grid, zgrid):
        n = dL_grid.shape[0]
        idx = jnp.searchsorted(dL_grid, dL, side=side)
        idx = jnp.clip(idx, 1, n - 1)

        dl_lo = dL_grid[idx - 1]
        dl_hi = dL_grid[idx]
        z_lo  = zgrid[idx - 1]
        z_hi  = zgrid[idx]

        denom = jnp.maximum(dl_hi - dl_lo, eps)
        r = (dL - dl_lo) / denom
        z = (1.0 - r) * z_lo + r * z_hi

        # cache everything needed for backward
        return z, (idx, dL, dl_lo, dl_hi, z_lo, z_hi, denom, dL_grid.shape)

    @jax.custom_vjp
    def z_from_dL_grid(dL, theta5, zgrid, dL_grid, x01, w01):
        z, _ = _forward(dL, dL_grid, zgrid)
        return z

    def fwd(dL, theta5, zgrid, dL_grid, x01, w01):
        z, res = _forward(dL, dL_grid, zgrid)
        return z, res

    def bwd(res, g_z):
        idx, dL, dl_lo, dl_hi, z_lo, z_hi, denom, dL_grid_shape = res

        dz = (z_hi - z_lo)

        # dz/ddL = dz/denom
        g_dL = g_z * dz / denom

        # grads wrt knot positions (matches your NumPy/PT derivation)
        denom2 = denom * denom
        g_dl_lo = g_z * dz * (dL - dl_hi) / denom2
        g_dl_hi = g_z * dz * (dl_lo - dL) / denom2

        g_dL_grid = jnp.zeros(dL_grid_shape, dtype=g_z.dtype)
        g_dL_grid = g_dL_grid.at[idx - 1].add(g_dl_lo)
        g_dL_grid = g_dL_grid.at[idx].add(g_dl_hi)

        # grads for (dL, theta5, zgrid, dL_grid, x01, w01)
        return (g_dL, None, None, g_dL_grid, None, None)

    z_from_dL_grid.defvjp(fwd, bwd)
    return z_from_dL_grid