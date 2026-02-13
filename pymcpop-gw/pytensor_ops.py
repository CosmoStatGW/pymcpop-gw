from __future__ import annotations
from functools import partial

from typing import Tuple
import numpy as np

import pytensor.tensor as at
from pytensor.graph.op import Op, Apply

from constants import c_light
import rate_models
import spin_models
import mass_models
from cosmology import dcfun_quad, dLfun, log_ddL_dz, Efun, Xi_vanilla, Xi_polexp, z_from_dL
from pytensor_utils import atinterp, make_dL_to_z_table, atinterp_uniform


import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jax_logsumexp
from jax import lax
import jax.scipy as jsp


from backends import JAXBackend
from population import log_p_pop, sel_bias_with_uncertainty, make_dL_to_z_cuvjp, make_dL_to_z_cuvjp_uniform
from jax_utils import _interp_prepare_bk, _interp_apply_bk, _interp_prepare_uniform_bk, _interp_apply_multi_bk

from pytensor.gradient import DisconnectedType, grad_not_implemented



# ---------------------------------------------------------------------
#   utils
# ---------------------------------------------------------------------


def _to_device(x):
    """Device-put only if needed.

    We assume dtype is already correct (float64) as per user setup.
    """
    # jax.Array is the public base class in newer JAX.
    if isinstance(x, jax.Array):
        return x
    # Backwards/older JAX: ArrayImpl
    try:
        import jaxlib
        if isinstance(x, jaxlib.xla_extension.ArrayImpl):
            return x
    except Exception:
        pass
    return jax.device_put(x)




def _connected_g(g):
    t = getattr(g, "type", None)
    return at.as_tensor_variable(0.0, dtype="float64") if isinstance(t, DisconnectedType) else g


def _as_vec_like(g, like):
    # if disconnected, g is a scalar 0.0; if connected, it's already a vector
    g = _connected_g(g)
    return at.broadcast_to(g, like.shape)

def _as_scalar(g):
    g = _connected_g(g)
    return at.as_tensor_variable(g).reshape(())  # force scalar

 




# ---------------------------------------------------------------------
#  core function
# ---------------------------------------------------------------------


def _make_pop_and_sel_core(
    *,
    bk,
    zgrid,
    rate_model,
    mass_model,
    spin_model,
    smoothing,
    simplex_repair,
    has_m2_break,
    norm_gauss,
    param,
    verbose,
    subtract_log_p_incl,
    skip_sel=False,
    chunk_inj=0,
    K_dp: int = 30, 
    DP_truncate=False,
    DP_m1_env=False
):
    """Build the single source of truth JAX core function.

    Returns a pure function:
        (evt arrays..., inj arrays..., Lambda, Ndraw) -> (logp_pop_evt, log_mu, var_u)
    """


    def _f(
        m1det, m2det, dLdet, spins_evt,
        m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
        Lambda, Ndraw,
    ):
        theta5 = Lambda[:5]
        H0, Om, w0, Xi0, nXi0 = theta5

        m1inj = lax.stop_gradient(m1inj)
        m2inj = lax.stop_gradient(m2inj)
        dLinj = lax.stop_gradient(dLinj)
        spins_inj = lax.stop_gradient(spins_inj)
        log_p_draw = lax.stop_gradient(log_p_draw)
        log_p_incl = lax.stop_gradient(log_p_incl)
        Ndraw = lax.stop_gradient(Ndraw)

  
        ##################################################

        z_evt = z_from_dL(bk, dLdet, H0=H0, Om=Om, w0=w0, Xi0=Xi0, nXi0=nXi0, z_nodes = zgrid, d_nodes = None) 
        
        onepz = 1.0 + z_evt
        m1src = m1det / onepz
        m2src = m2det / onepz

        ##################################################

        logp_pop_evt = log_p_pop(
            bk,
            m1src, m2src, z_evt, dLdet, spin_models._spins_as_list(spins_evt, spin_model), 
            Lambda,
            rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
                smoothing=smoothing, simplex_repair=simplex_repair,
                has_m2_break=has_m2_break, norm_gauss=norm_gauss,
                dc=None, log_ddL_dz_pre=None,
                Xi=None, E=None, 
                param=param,
                z_grid=zgrid, 
                verbose=verbose,
            K_dp=K_dp, 
            DP_truncate=DP_truncate,
            DP_m1_env=DP_m1_env
        )

        if skip_sel:
             return (
            logp_pop_evt,
            jnp.asarray(0., dtype=jnp.float64).reshape(()),
            jnp.asarray(0., dtype=jnp.float64).reshape(()),
        )
            
            
        ##################################################
        
        log_mu, var_u = sel_bias_with_uncertainty(
            bk,
            m1inj, m2inj, dLinj,
            spins_inj,
            log_p_draw, log_p_incl,
            Lambda, Ndraw,
            rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
            smoothing=smoothing, simplex_repair=simplex_repair,
            has_m2_break=has_m2_break, norm_gauss=norm_gauss,
            param=param, 
            z_grid=zgrid, 
            verbose=verbose,
            subtract_log_p_incl=subtract_log_p_incl,
            use_streaming_vjp= bool(chunk_inj>0),          # <--- enable optimized backward
            sel_chunk_size=chunk_inj,            # <--- tune
            K_dp=K_dp,
            DP_truncate=DP_truncate,
            DP_m1_env=DP_m1_env
        )


        return (
            logp_pop_evt,
            jnp.asarray(log_mu, dtype=jnp.float64).reshape(()),
            jnp.asarray(var_u, dtype=jnp.float64).reshape(()),
        )

    return _f




# ---------------------------------------------------------------------
#  Gradient Op
# ---------------------------------------------------------------------


class _PopAndSelJAXVJPOp(Op):
    itypes = [
        at.dvector, at.dvector, at.dvector, at.dmatrix,  # evt
        at.dvector, at.dvector, at.dvector, at.dmatrix, at.dvector, at.dvector,  # inj
        at.dvector, at.dscalar,  # Lambda, Ndraw
        at.dvector, at.dscalar, at.dscalar,  # cotangents
    ]
    otypes = [at.dvector, at.dvector, at.dvector, at.dmatrix, at.dvector]

    def __init__(self, *, zgrid,  rate_model, mass_model, spin_model,
                 smoothing="LVK", simplex_repair=False, has_m2_break=False, norm_gauss="uplow",
                 param="vanilla", 
                 verbose=False,
                 subtract_log_p_incl=False, 
                skip_sel=False,
                 K_dp : int = 30,
                 DP_truncate=False,
                 DP_m1_env=False
                ):
        super().__init__()
        self.zgrid = zgrid 

        self.rate_model = rate_model
        self.mass_model = mass_model
        self.spin_model = spin_model
        self.smoothing = smoothing
        self.simplex_repair = bool(simplex_repair)
        self.has_m2_break = bool(has_m2_break)
        self.norm_gauss = norm_gauss
        self.param = str(param)
        self.verbose = bool(verbose)
        self.subtract_log_p_incl = bool(subtract_log_p_incl)
        self.skip_sel = skip_sel
        self.K_dp = int(K_dp)
        self.DP_truncate = DP_truncate
        self.DP_m1_env = DP_m1_env

        self._cached_inj = None
        self._jax_vjp = self._build_jax_vjp()

    def _build_jax_vjp(self):
        bk = JAXBackend()

        cosmo_zgrid = jnp.asarray(self.zgrid, dtype=jnp.float64)
 

        core_f = _make_pop_and_sel_core(
            bk=bk,
            zgrid=cosmo_zgrid,
            rate_model=self.rate_model,
            mass_model=self.mass_model,
            spin_model=self.spin_model,
            smoothing=self.smoothing,
            
            simplex_repair=self.simplex_repair,
            has_m2_break=self.has_m2_break,
            norm_gauss=self.norm_gauss,
            param=self.param,
            verbose=self.verbose,
            subtract_log_p_incl= self.subtract_log_p_incl,
            skip_sel = self.skip_sel,
            K_dp = self.K_dp,
            DP_truncate= self.DP_truncate,
            DP_m1_env = self.DP_m1_env
        )



        def _vjp(m1det, m2det, dLdet, spins_evt,
                 m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
                 Lambda, Ndraw,
                 g_logp_pop, g_log_mu, g_var_u):

            g_logp_pop = jnp.reshape(g_logp_pop, m1det.shape)
            g_log_mu = jnp.reshape(g_log_mu, ())
            g_var_u = jnp.reshape(g_var_u, ())

            if self.skip_sel:
                (_, _, _), pull = jax.vjp(
                core_f,
                m1det, m2det, dLdet, spins_evt,
                m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
                Lambda, Ndraw
                )
                zeros = jnp.zeros((), dtype=jnp.float64)
                grads = pull((g_logp_pop, zeros, zeros))
                return grads[0], grads[1], grads[2], grads[3], grads[10]

            (_, _, _), pull = jax.vjp(
                core_f,
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

        parent = getattr(self, "_parent_op", None)

        # Cache injections once (shared with parent op when available)
        inj_cache = None
        if parent is not None and getattr(parent, "_cached_inj", None) is not None:
            inj_cache = parent._cached_inj
            self._cached_inj = inj_cache
        if inj_cache is None:
            if self._cached_inj is None:
                self._cached_inj = (
                    _to_device(m1inj),
                    _to_device(m2inj),
                    _to_device(dLinj),
                    _to_device(spins_inj),
                    _to_device(log_p_draw),
                    _to_device(log_p_incl),
                    _to_device(float(np.asarray(Ndraw).reshape(()))),
                )
                if parent is not None:
                    parent._cached_inj = self._cached_inj
            inj_cache = self._cached_inj

        m1inj_j, m2inj_j, dLinj_j, spins_inj_j, lpd_j, lpi_j, Ndraw_j = inj_cache

        # Reuse last-call device-put args from the forward pass when possible
        m1det_j = m2det_j = dLdet_j = spins_evt_j = Lambda_j = None
        if parent is not None and getattr(parent, "_last_call", None) is not None:
            key_cached, args_cached = parent._last_call
            key_now = (
                id(m1det), getattr(m1det, "shape", None),
                id(m2det), getattr(m2det, "shape", None),
                id(dLdet), getattr(dLdet, "shape", None),
                id(spins_evt), getattr(spins_evt, "shape", None),
                id(Lambda), getattr(Lambda, "shape", None),
            )
            if key_now == key_cached:
                m1det_j, m2det_j, dLdet_j, spins_evt_j, Lambda_j = args_cached

        if m1det_j is None:
            m1det_j = _to_device(m1det)
            m2det_j = _to_device(m2det)
            dLdet_j = _to_device(dLdet)
            spins_evt_j = _to_device(spins_evt)
            Lambda_j = _to_device(Lambda)

        dm1det, dm2det, ddLdet, dspins_evt, dLambda = self._jax_vjp(
            m1det_j, m2det_j, dLdet_j, spins_evt_j,
            m1inj_j, m2inj_j, dLinj_j, spins_inj_j, lpd_j, lpi_j,
            Lambda_j, Ndraw_j,
            _to_device(g_logp_pop), _to_device(g_log_mu), _to_device(g_var_u),
        )

        outputs[0][0] = np.asarray(dm1det, dtype="float64")
        outputs[1][0] = np.asarray(dm2det, dtype="float64")
        outputs[2][0] = np.asarray(ddLdet, dtype="float64")
        outputs[3][0] = np.asarray(dspins_evt, dtype="float64")
        outputs[4][0] = np.asarray(dLambda, dtype="float64")





# ---------------------------------------------------------------------
#  forward Op
# ---------------------------------------------------------------------

class PopAndSelJAXOp(Op):
    itypes = [
        at.dvector, at.dvector, at.dvector, at.dmatrix,  # evt
        at.dvector, at.dvector, at.dvector, at.dmatrix, at.dvector, at.dvector,  # inj
        at.dvector, at.dscalar,  # Lambda, Ndraw
    ]
    otypes = [at.dvector, at.dscalar, at.dscalar]

    def __init__(self, *, zgrid, rate_model, mass_model, spin_model,
                 smoothing="LVK", simplex_repair=False, has_m2_break=False, norm_gauss="uplow",
                 param="vanilla",
                 interp_mass=0,
                 verbose=False,
                 subtract_log_p_incl=False, 
                skip_sel=False,
                 chunk_inj=0,
                 K_dp: int = 30,
                 DP_truncate = False,
                 DP_m1_env = False
                ):
        super().__init__()

        self.zgrid = jnp.asarray(zgrid, dtype="float64")
 
        self.rate_model = rate_model
        self.mass_model = mass_model
        self.spin_model = spin_model
        self.smoothing = smoothing
        self.simplex_repair = bool(simplex_repair)
        self.has_m2_break = bool(has_m2_break)
        self.norm_gauss = norm_gauss
        self.param = param
        self.verbose = bool(verbose)
        self.subtract_log_p_incl = bool(subtract_log_p_incl)
        self.skip_sel=skip_sel
        self.chunk_inj = chunk_inj
        self.K_dp = int(K_dp)

        
        self.DP_truncate=DP_truncate
        self.DP_m1_env=DP_m1_env

        # Backend (needed by cosmology/mass grid builders)
        self._bk = JAXBackend()

        # One-time device cache (shared with vjp op)
        self._cached_inj = None

        # Last-call cache to reuse per-call device args between fwd and vjp
        # Stores: (key, (m1det_j, m2det_j, dLdet_j, spins_evt_j, Lambda_j))
        self._last_call = None

        # Build and jit core forward function
        self._jax_fwd = self._build_jax_fwd()

        # Build vjp op (shares caches via _parent_op pointer)
        self._vjp_op = _PopAndSelJAXVJPOp(
            zgrid=self.zgrid, 
            rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
            smoothing=smoothing, simplex_repair=simplex_repair, has_m2_break=has_m2_break,
            norm_gauss=norm_gauss, param=param, 
            verbose=verbose, subtract_log_p_incl=subtract_log_p_incl,
            skip_sel=self.skip_sel,
            K_dp = self.K_dp,
            DP_truncate=self.DP_truncate,
            DP_m1_env=self.DP_m1_env
        )
        self._vjp_op._parent_op = self



    def _build_jax_fwd(self):
       
        full_f = _make_pop_and_sel_core(
            bk=self._bk,
            zgrid=self.zgrid,
            rate_model=self.rate_model,
            mass_model=self.mass_model,
            spin_model=self.spin_model,
            smoothing=self.smoothing,
            simplex_repair=self.simplex_repair,
            has_m2_break=self.has_m2_break,
            norm_gauss=self.norm_gauss,
            param=self.param,
            verbose=self.verbose,
            subtract_log_p_incl=self.subtract_log_p_incl,
            chunk_inj=self.chunk_inj,
            K_dp = self.K_dp,
            DP_truncate=self.DP_truncate,
            DP_m1_env=self.DP_m1_env,
            
            skip_sel=self.skip_sel
        )
        return jax.jit(full_f)


    
    def make_node(self, *inputs):
            inputs = list(map(at.as_tensor_variable, inputs))
            return Apply(self, inputs, [at.dvector(), at.dscalar(), at.dscalar()])

    def perform(self, node, inputs, outputs):
            (m1det, m2det, dLdet, spins_evt,
             m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
             Lambda, Ndraw) = inputs


            # Cache injections (and constant Ndraw) once, and share with the VJP op
            if self._cached_inj is None:
                self._cached_inj = (
                    _to_device(m1inj),
                    _to_device(m2inj),
                    _to_device(dLinj),
                    _to_device(spins_inj),
                    _to_device(log_p_draw),
                    _to_device(log_p_incl),
                    _to_device(float(np.asarray(Ndraw).reshape(()))),
                )
                self._vjp_op._cached_inj = self._cached_inj

            m1inj_j, m2inj_j, dLinj_j, spins_inj_j, lpd_j, lpi_j, Ndraw_j = self._cached_inj

            # Device-put per-call inputs only if needed; also keep a last-call cache
            m1det_j = _to_device(m1det)
            m2det_j = _to_device(m2det)
            dLdet_j = _to_device(dLdet)
            spins_evt_j = _to_device(spins_evt)
            Lambda_j = _to_device(Lambda)

            key = (
                id(m1det), getattr(m1det, "shape", None),
                id(m2det), getattr(m2det, "shape", None),
                id(dLdet), getattr(dLdet, "shape", None),
                id(spins_evt), getattr(spins_evt, "shape", None),
                id(Lambda), getattr(Lambda, "shape", None),
            )
            self._last_call = (key, (m1det_j, m2det_j, dLdet_j, spins_evt_j, Lambda_j))


            y1, y2, y3 = self._jax_fwd(
            m1det_j, m2det_j, dLdet_j, spins_evt_j,
            m1inj_j, m2inj_j, dLinj_j, spins_inj_j, lpd_j, lpi_j,
            Lambda_j, Ndraw_j,
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


            if self.skip_sel:
                g_log_mu = at.zeros_like(Ndraw, dtype="float64")
                g_var_u = at.zeros_like(Ndraw, dtype="float64")
                

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





