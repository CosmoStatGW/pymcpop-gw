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
from cosmology import dcfun_quad, dLfun, log_ddL_dz, Efun, Xi_vanilla, Xi_polexp
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

    


def _extract_lambdaBBHmass(Lambda, *, rate_model, spin_model, mass_model):
    if rate_model == "MD":
        istart = 8
    else:
        raise NotImplementedError

    if spin_model == "default_gauss":
        istart_spin = istart + 4
    else:
        raise NotImplementedError

    if mass_model == "DPLDP":
        s = istart_spin
        # keep as a tuple for JAX (static container, dynamic leaves)
        return tuple(Lambda[s + k] for k in range(21))

    raise NotImplementedError




# ---------------------------------------------------------------------
#  core function
# ---------------------------------------------------------------------


def _make_pop_and_sel_core(
    *,
    bk,
    zgrid,
    x01,
    w01,
    rate_model,
    mass_model,
    spin_model,
    smoothing,
    simplex_repair,
    has_m2_break,
    norm_gauss,
    param,
    #is_observed,
    verbose,
    subtract_log_p_incl,
    eps_interp,
    side_interp,
    #interp_mass,
    #has_interp_mass,
    #has_mass_grids,
    #mass_grids_jax,
    linear_mass,
    linear_z,
    use_cuvjp=False,
    skip_sel=False,
    chunk_inj=0
):
    """Build the single source of truth JAX core function.

    Returns a pure function:
        (evt arrays..., inj arrays..., Lambda, Ndraw) -> (logp_pop_evt, log_mu, var_u)
    """

    if use_cuvjp:
        if not linear_z:
            inv_dL_to_z = make_dL_to_z_cuvjp(bk=bk, eps_interp=eps_interp, side_interp=side_interp)
        else:
            inv_dL_to_z = make_dL_to_z_cuvjp_uniform(bk=bk, eps_interp=eps_interp)

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


        E_grid = Efun(bk, zgrid, Om, w0)
        if param == "vanilla":
            Xi_grid = Xi_vanilla(bk, zgrid, Xi0, nXi0)
        elif param == "polexp":
            Xi_grid = Xi_polexp(bk, zgrid, Xi0, nXi0)

        dc_grid = dcfun_quad(bk, zgrid, H0, Om, w0, x01, w01)
        dL_grid = dLfun(bk, zgrid, H0, Om, w0, Xi0, nXi0, dc=dc_grid, Xi=Xi_grid, param=param, x01=x01, w01=w01)
        log_ddL_dz_grid = log_ddL_dz(bk, zgrid, H0, Om, w0, Xi0, nXi0, dc=dc_grid, E=E_grid, Xi=Xi_grid,  x01=x01, w01=w01, param=param)

        if use_cuvjp:
            # ------------------------------------------------------------------
            # Derivative grids for implicit differentiation of z(dL; theta)
            # theta = (H0, Om, w0, Xi0, nXi0)
            # We build d(dc)/dOm, d(dc)/dw0 using the same Gauss–Legendre quadrature
            # used in dcfun_quad, but differentiating the integrand 1/E.
            # ------------------------------------------------------------------
    
            # quadrature nodes on [0,z]: z_nodes has shape (Nz, nquad)
            z_nodes = zgrid[..., None] * x01
            onepz_nodes = 1.0 + z_nodes
    
            # S = Om*(1+z)^3 + (1-Om)*(1+z)^{3(1+w0)}
            a3 = onepz_nodes**3.0
            a3w = onepz_nodes ** (3.0 * (1.0 + w0))
            S = Om * a3 + (1.0 - Om) * a3w
    
            # invE = 1/sqrt(S)
            sqrtS = bk.sqrt(S)
            invE = 1.0 / sqrtS
    
            # d/dtheta(invE) = -0.5 * S^{-3/2} * dS/dtheta
            invS32 = 1.0 / (S * sqrtS)  # S^{-3/2}
    
            dS_dOm = a3 - a3w
            # d/dw0 a^{3(1+w0)} = a^{3(1+w0)} * 3*log(a)
            dS_dw0 = (1.0 - Om) * a3w * (3.0 * bk.log(onepz_nodes))
    
            dinvE_dOm = -0.5 * invS32 * dS_dOm
            dinvE_dw0 = -0.5 * invS32 * dS_dw0
    
            # Integrate derivatives: I_theta(z) = sum_j w_j * dinvE_dtheta(z*x_j)
            I_dOm = bk.sum(w01 * dinvE_dOm, axis=-1)   # shape (Nz,)
            I_dw0 = bk.sum(w01 * dinvE_dw0, axis=-1)   # shape (Nz,)
    
            # dc(z) = (c/H0)*z*I * 1e-03  -> derivative wrt Om,w0 uses z*I_dtheta
            pref_dc = (c_light / H0) * zgrid * 1e-03
            dc_dOm_grid = pref_dc * I_dOm
            dc_dw0_grid = pref_dc * I_dw0
    
            # Now build d(dL)/dtheta on the grid.
            onepz = 1.0 + zgrid
    
            # dL_dH0: because dc ∝ 1/H0 and Xi,E don't depend on H0
            dL_dH0_grid = -dL_grid / H0
    
            # dL_dOm, dL_dw0 from dc derivatives
            dL_dOm_grid = Xi_grid * onepz * dc_dOm_grid
            dL_dw0_grid = Xi_grid * onepz * dc_dw0_grid

            # Xi derivatives depend on parameterization
            if param == "vanilla":
                # Xi(z) = Xi0 + (1-Xi0)*(1+z)^(-n)
                onepz_mn = onepz ** (-nXi0)
                dXi_dXi0 = 1.0 - onepz_mn
                dXi_dn   = -(1.0 - Xi0) * onepz_mn * bk.log1p(zgrid)
    
                dL_dXi0_grid = onepz * dc_grid * dXi_dXi0
                dL_dn_grid   = onepz * dc_grid * dXi_dn
            else:
                # TODO: add analytic dXi/dXi0 and dXi/dn for polexp
                raise NotImplementedError("dL_dtheta_grid currently implemented only for param='vanilla'.")
    
            # Stack: shape (5, Nz) in the order (H0, Om, w0, Xi0, n)
            dL_dtheta_grid = bk.stack(
                [dL_dH0_grid, dL_dOm_grid, dL_dw0_grid, dL_dXi0_grid, dL_dn_grid],
                axis=0
            )

        ##################################################
        
        lambdaBBHmass = _extract_lambdaBBHmass(
            Lambda,
            rate_model=rate_model,
            spin_model=spin_model,
            mass_model=mass_model,
        )

      
        ##################################################
        
        spins_evt_list = spin_models._spins_as_list(spins_evt, spin_model)
        #spins_inj_list = spin_models._spins_as_list(spins_inj, spin_model)
        spins_inj_list = spins_inj


        if not linear_z:

            if use_cuvjp:
                z_evt = inv_dL_to_z(dLdet, dL_grid, zgrid, log_ddL_dz_grid, dL_dtheta_grid, Lambda)
                z_inj = inv_dL_to_z(dLinj, dL_grid, zgrid, log_ddL_dz_grid, dL_dtheta_grid, Lambda)
            else:
                z_inj = atinterp(bk, dLinj, dL_grid, zgrid)
                z_evt = atinterp(bk, dLdet, dL_grid, zgrid)

             
            # Reuse a single searchsorted for any z->grid interpolations we need downstream

            # prepare indices once
            idx_z_evt, t_z_evt = _interp_prepare_bk(bk, z_evt, zgrid)
            idx_z_inj, t_z_inj = _interp_prepare_bk(bk, z_inj, zgrid)
            
            # build stacked fps ONCE
            fps4 = bk.stack([dc_grid, log_ddL_dz_grid, E_grid, Xi_grid], axis=0)  # (4, Nz)
            
            # interpolate for events
            vals_evt = _interp_apply_multi_bk(bk, idx_z_evt, t_z_evt, fps4)
            dc_evt, log_ddL_dz_evt, E_evt, Xi_evt = vals_evt[0], vals_evt[1], vals_evt[2], vals_evt[3]
            
            # interpolate for injections (reuse fps4)
            vals_inj = _interp_apply_multi_bk(bk, idx_z_inj, t_z_inj, fps4)
            dc_inj, log_ddL_dz_inj, E_inj, Xi_inj = vals_inj[0], vals_inj[1], vals_inj[2], vals_inj[3]



        else:
            ##### Uniform grids, no searchsorted
            dL_u, z_u = make_dL_to_z_table(
                bk,
                dL_grid,
                zgrid,
                NdL=4096,
                eps=eps_interp,
                side=side_interp,
                logspace=False,     
            )
            
            # 2) use *uniform* interp for the million-point calls (no searchsorted)
            if use_cuvjp:
                z_evt = inv_dL_to_z(dLdet, dL_u, z_u, zgrid, dL_dtheta_grid, Lambda)
                z_inj = inv_dL_to_z(dLinj, dL_u, z_u, zgrid, dL_dtheta_grid, Lambda)
            else: 
                z_inj = atinterp_uniform(bk, dLinj, dL_u, z_u)
                z_evt = atinterp_uniform(bk, dLdet, dL_u, z_u)


            # Uniform grids, no searchsorted
            idx_z_evt, t_z_evt = _interp_prepare_uniform_bk(bk, z_evt, zgrid)
            
            fps_evt = bk.stack([dc_grid, log_ddL_dz_grid, E_grid, Xi_grid], axis=0)  # (4, Nz)
            vals_evt = _interp_apply_multi_bk(bk, idx_z_evt, t_z_evt, fps_evt)
            dc_evt, log_ddL_dz_evt, E_evt, Xi_evt = vals_evt[0], vals_evt[1], vals_evt[2], vals_evt[3]
            
            idx_z_inj, t_z_inj = _interp_prepare_uniform_bk(bk, z_inj, zgrid)
            
            # IMPORTANT: reuse the same stacked fps (don’t rebuild it)
            vals_inj = _interp_apply_multi_bk(bk, idx_z_inj, t_z_inj, fps_evt)
            dc_inj, log_ddL_dz_inj, E_inj, Xi_inj = vals_inj[0], vals_inj[1], vals_inj[2], vals_inj[3]
            
            
        ##################################################
 
        onepz = 1.0 + z_evt
        m1src = m1det / onepz
        m2src = m2det / onepz

        logp_pop_evt = log_p_pop(
            bk,
            m1src, m2src, z_evt, dLdet,
            spins_evt_list, Lambda,
            rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
                smoothing=smoothing, simplex_repair=simplex_repair,
                has_m2_break=has_m2_break, norm_gauss=norm_gauss,
                dc=dc_evt, log_ddL_dz_pre=log_ddL_dz_evt,
                Xi=Xi_evt, E=E_evt, 
                param=param,
                #interp_vals_mass=interp_vals_mass_jax,
                #interp_grids_mass=interp_grids_mass_jax,
                #is_observed=is_observed, 
                z_grid=zgrid, 
                verbose=verbose,
                linear_mass=linear_mass
        )
        
        if skip_sel:
             return (
            logp_pop_evt,
            jnp.asarray(0., dtype=jnp.float64).reshape(()),
            jnp.asarray(0., dtype=jnp.float64).reshape(()),
        )
            

        # log_mu, var_u = sel_bias_with_uncertainty(
        #         bk,
        #             m1inj, m2inj, dLinj,
        #             spins_inj_list,
        #             log_p_draw, log_p_incl,
        #             dL_grid, dc_grid, log_ddL_dz_grid,
        #             Lambda, Ndraw,
        #             rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
        #             smoothing=smoothing, simplex_repair=simplex_repair,
        #             has_m2_break=has_m2_break, norm_gauss=norm_gauss,
        #             param=param,
        #             #interp_vals_mass=interp_vals_mass_jax,
        #             #interp_grids_mass=interp_grids_mass_jax,
        #             #is_observed=is_observed, 
        #             z_grid=zgrid, verbose=verbose,
        #             subtract_log_p_incl=subtract_log_p_incl,
        #             eps_interp=eps_interp, side_interp=side_interp,
        #             zinj=z_inj, dcinj=dc_inj, log_ddL_dz_inj=log_ddL_dz_inj,
        #             Einj=E_inj, XiInj=Xi_inj,
        #             linear_mass=linear_mass
        #     )

        log_mu, var_u = sel_bias_with_uncertainty(
            bk,
            m1inj, m2inj, dLinj,
            spins_inj_list,
            log_p_draw, log_p_incl,
            dL_grid, dc_grid, log_ddL_dz_grid,
            Lambda, Ndraw,
            rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
            smoothing=smoothing, simplex_repair=simplex_repair,
            has_m2_break=has_m2_break, norm_gauss=norm_gauss,
            param=param, z_grid=zgrid, verbose=verbose,
            subtract_log_p_incl=True,
            eps_interp=eps_interp, side_interp=side_interp,
            zinj=z_inj, dcinj=dc_inj, log_ddL_dz_inj=log_ddL_dz_inj,
            Einj=E_inj, XiInj=Xi_inj,
            linear_mass=linear_mass,
            use_streaming_vjp= bool(chunk_inj>0),          # <--- enable optimized backward
            sel_chunk_size=chunk_inj,            # <--- tune
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

    def __init__(self, *, zgrid, x01, w01, rate_model, mass_model, spin_model,
                 smoothing="LVK", simplex_repair=False, has_m2_break=False, norm_gauss="uplow",
                 param="vanilla", 
                 #interp_mass=0,
                 #is_observed=False, 
                 #mass_grids = None,
                 verbose=False,
                 linear_mass=False, linear_z=False,
                 subtract_log_p_incl=False, eps_interp=1e-12, side_interp="left",
                skip_sel=False
                ):
        super().__init__()
        self.zgrid = zgrid #jnp.asarray(zgrid, dtype="float64")
        self.x01 = x01 #jnp.asarray(x01, dtype="float64")
        self.w01 = w01 #jnp.asarray(w01, dtype="float64")

        self.rate_model = rate_model
        self.mass_model = mass_model
        self.spin_model = spin_model
        self.smoothing = smoothing
        self.simplex_repair = bool(simplex_repair)
        self.has_m2_break = bool(has_m2_break)
        self.norm_gauss = norm_gauss
        self.param = str(param)
        #self.is_observed = bool(is_observed)
        self.verbose = bool(verbose)
        self.subtract_log_p_incl = bool(subtract_log_p_incl)
        self.eps_interp = float(eps_interp)
        self.side_interp = str(side_interp)

        #self.interp_mass = interp_mass
        #self.has_interp_mass = bool(interp_mass>0)
        #self.has_mass_grids = bool(mass_grids is not None)
        #self.mass_grids = mass_grids
        self.linear_mass=linear_mass
        self.linear_z=linear_z
        self.skip_sel=skip_sel

        self._cached_inj = None
        self._jax_vjp = self._build_jax_vjp()

    def _build_jax_vjp(self):
        bk = JAXBackend()


        rate_model = self.rate_model
        mass_model = self.mass_model
        spin_model = self.spin_model
        smoothing = self.smoothing
        simplex_repair = self.simplex_repair
        has_m2_break = self.has_m2_break
        norm_gauss = self.norm_gauss
        param = self.param
        #is_observed = self.is_observed
        verbose = self.verbose
        subtract_log_p_incl = self.subtract_log_p_incl
        eps_interp = self.eps_interp
        side_interp = self.side_interp
        #interp_mass = self.interp_mass
        #has_interp_mass = self.has_interp_mass
        #has_mass_grids = self.has_mass_grids
        #mass_grids = self.mass_grids
        linear_mass=self.linear_mass
        linear_z=self.linear_z
        skip_sel = self.skip_sel

        
        cosmo_zgrid = jnp.asarray(self.zgrid, dtype=jnp.float64)
        x01_jax = jnp.asarray(self.x01, dtype=jnp.float64)
        w01_jax = jnp.asarray(self.w01, dtype=jnp.float64)


        core_f = _make_pop_and_sel_core(
            bk=bk,
            zgrid=cosmo_zgrid,
            x01=x01_jax,
            w01=w01_jax,
            rate_model=rate_model,
            mass_model=mass_model,
            spin_model=spin_model,
            smoothing=smoothing,
            simplex_repair=simplex_repair,
            has_m2_break=has_m2_break,
            norm_gauss=norm_gauss,
            param=param,
            #is_observed=is_observed,
            verbose=verbose,
            subtract_log_p_incl=subtract_log_p_incl,
            eps_interp=eps_interp,
            side_interp=side_interp,
            #interp_mass=interp_mass,
            #has_interp_mass=has_interp_mass,
            #has_mass_grids=has_mass_grids,
            #mass_grids_jax=mass_grids_jax,
            linear_mass=linear_mass,
            linear_z=linear_z,
            skip_sel=skip_sel
        )



        def _vjp(m1det, m2det, dLdet, spins_evt,
                 m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
                 Lambda, Ndraw,
                 g_logp_pop, g_log_mu, g_var_u):

            g_logp_pop = jnp.reshape(g_logp_pop, m1det.shape)
            g_log_mu = jnp.reshape(g_log_mu, ())
            g_var_u = jnp.reshape(g_var_u, ())

            if skip_sel:
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

    def __init__(self, *, zgrid, x01, w01, rate_model, mass_model, spin_model,
                 smoothing="LVK", simplex_repair=False, has_m2_break=False, norm_gauss="uplow",
                 param="vanilla",
                 interp_mass=0,
                 #is_observed=False,
                 #mass_grids=None,
                 verbose=False,
                 linear_mass=False,
                 linear_z=False,
                 subtract_log_p_incl=False, eps_interp=1e-12, side_interp="left",
                skip_sel=False,
                 chunk_inj=0
                ):
        super().__init__()

        self.zgrid = jnp.asarray(zgrid, dtype="float64")
        self.x01 = jnp.asarray(x01, dtype="float64")
        self.w01 = jnp.asarray(w01, dtype="float64")

        self.rate_model = rate_model
        self.mass_model = mass_model
        self.spin_model = spin_model
        self.smoothing = smoothing
        self.simplex_repair = bool(simplex_repair)
        self.has_m2_break = bool(has_m2_break)
        self.norm_gauss = norm_gauss
        self.param = param
        #self.interp_mass = int(interp_mass)
        #self.is_observed = bool(is_observed)
        #self.mass_grids = mass_grids
        self.verbose = bool(verbose)
        self.subtract_log_p_incl = bool(subtract_log_p_incl)
        self.eps_interp = float(eps_interp)
        self.side_interp = side_interp
        self.linear_mass=linear_mass
        self.linear_z=linear_z
        self.skip_sel=skip_sel
        self.chunk_inj = chunk_inj

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
            zgrid=self.zgrid, x01=self.x01, w01=self.w01,
            rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
            smoothing=smoothing, simplex_repair=simplex_repair, has_m2_break=has_m2_break,
            norm_gauss=norm_gauss, param=param, 
            #interp_mass=interp_mass,
            #is_observed=is_observed, mass_grids=mass_grids,
            verbose=verbose, subtract_log_p_incl=subtract_log_p_incl,
            eps_interp=eps_interp, side_interp=side_interp,
            linear_mass=self.linear_mass, linear_z=self.linear_z,
            skip_sel=self.skip_sel
        )
        self._vjp_op._parent_op = self



    def _build_jax_fwd(self):
       
        full_f = _make_pop_and_sel_core(
            bk=self._bk,
            zgrid=self.zgrid,
            x01=self.x01,
            w01=self.w01,
            rate_model=self.rate_model,
            mass_model=self.mass_model,
            spin_model=self.spin_model,
            smoothing=self.smoothing,
            simplex_repair=self.simplex_repair,
            has_m2_break=self.has_m2_break,
            norm_gauss=self.norm_gauss,
            param=self.param,
            #is_observed=self.is_observed,
            verbose=self.verbose,
            subtract_log_p_incl=self.subtract_log_p_incl,
            eps_interp=self.eps_interp,
            side_interp=self.side_interp,
            #interp_mass=self.interp_mass,
            #has_interp_mass=has_interp_mass,
            #has_mass_grids=has_mass_grids,
            #mass_grids_jax=mass_grids_jax,
            linear_mass=self.linear_mass,
            linear_z=self.linear_z,
            chunk_inj=self.chunk_inj
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




