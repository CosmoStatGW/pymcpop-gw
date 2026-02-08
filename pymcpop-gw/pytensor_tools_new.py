from __future__ import annotations

from typing import Tuple
import numpy as np

import pytensor.tensor as at
from pytensor.graph.op import Op, Apply

import rate_models
import spin_models
import mass_models
from cosmology import dcfun_quad, dLfun, log_ddL_dz
#from cosmology_jax import make_z_from_dL_interp
from pytensor_utils import atinterp


import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jax_logsumexp
from jax import lax
import jax.scipy as jsp


from backends import JAXBackend
from population import log_p_pop, sel_bias_with_uncertainty
from jax_utils import _interp_prepare_bk, _interp_apply_bk

from pytensor.gradient import DisconnectedType, grad_not_implemented



# ---------------------------------------------------------------------
# pytensor glue utils
# ---------------------------------------------------------------------


def _connected_g(g):
    t = getattr(g, "type", None)
    return at.as_tensor_variable(0.0, dtype="float64") if isinstance(t, DisconnectedType) else g



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


def _maybe_precompute_mass_tables(
    bk,
    lambdaBBHmass,        # <- pass sliced mass params, NOT full Lambda
    interp_grids_mass_jax,
    *,
    mass_model: str,
    smoothing: str,
    simplex_repair: bool,
    has_m2_break: bool,
    norm_gauss: str,
):
    if interp_grids_mass_jax is None:
        return None

    if mass_model == "DPLDP":
        m1_grid, m2_grid = interp_grids_mass_jax
        return mass_models.precompute_DPLDP_mass_interp(
            bk,
            m1_grid,
            m2_grid,
            lambdaBBHmass,
            smoothing=smoothing,
            simplex_repair=simplex_repair,
            has_m2_break=has_m2_break,
            norm_gauss=norm_gauss,
        )

    raise NotImplementedError


    



# def _make_pop_and_sel_core(
#     *,
#     bk,
#     zgrid,
#     x01,
#     w01,
#     rate_model,
#     mass_model,
#     spin_model,
#     smoothing,
#     simplex_repair,
#     has_m2_break,
#     norm_gauss,
#     param,
#     is_observed,
#     verbose,
#     subtract_log_p_incl,
#     eps_interp,
#     side_interp,
#     interp_mass,
#     has_interp_mass,
#     has_mass_grids,
#     mass_grids_jax,
# ):
#     """
#     Returns two pure JAX functions:

#       full_f(m1det,m2det,dLdet,spins_evt, m1inj,m2inj,dLinj,spins_inj,log_p_draw,log_p_incl, Lambda,Ndraw)
#           -> (logp_pop_evt, log_mu, var_u)

#       evt_f(m1det,m2det,dLdet,spins_evt, Lambda)
#           -> logp_pop_evt
#     """

#     def _common_from_Lambda(Lambda):
#         theta5 = Lambda[:5]
#         H0, Om, w0, Xi0, nXi0 = theta5

#         dc_grid = dcfun_quad(bk, zgrid, H0, Om, w0, x01, w01)
#         dL_grid = dLfun(
#             bk, zgrid, H0, Om, w0, Xi0, nXi0,
#             dc=dc_grid, param=param, x01=x01, w01=w01
#         )
#         log_ddL_dz_grid = log_ddL_dz(
#             bk, zgrid, H0, Om, w0, Xi0, nXi0,
#             dc=dc_grid, x01=x01, w01=w01, param=param
#         )

#         lambdaBBHmass = _extract_lambdaBBHmass(
#             Lambda,
#             rate_model=rate_model,
#             spin_model=spin_model,
#             mass_model=mass_model,
#         )

#         if has_interp_mass:
#             if not has_mass_grids:
#                 (
#                     alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1,
#                     lambda0, lambda1, lambda2, beta, m2_low, delta_m2, epsilon, m_g, w_g,
#                     sig_g_low, sig_g_high
#                 ) = lambdaBBHmass

#                 m1_grid = mass_models.build_m1_grid_DPLDP_bk(
#                     bk,
#                     alpha1=alpha1, alpha2=alpha2, mb=mb,
#                     mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2,
#                     m1_low=m1_low, m_high=m_high,
#                     delta_m1=delta_m1,
#                     n_peak=interp_mass,
#                     n_tail_low=interp_mass//3,
#                     n_tail_high=interp_mass//4,
#                     n_taper=interp_mass//2,
#                     frac_gauss1=0.4,
#                 )

#                 m2_grid = mass_models.build_m2_grid_bk(
#                     bk,
#                     m2_low=m2_low,
#                     m2_high=m_high,
#                     delta_m2=delta_m2,
#                     n_grid=interp_mass,
#                     n_tail_low=interp_mass//3,
#                     n_tail_high=interp_mass//4,
#                     n_taper=interp_mass//2,
#                 )
#                 interp_grids_mass = (m1_grid, m2_grid)
#             else:
#                 interp_grids_mass = mass_grids_jax

#             interp_vals_mass = _maybe_precompute_mass_tables(
#                 bk,
#                 lambdaBBHmass=lambdaBBHmass,
#                 interp_grids_mass_jax=interp_grids_mass,
#                 mass_model=mass_model,
#                 smoothing=smoothing,
#                 simplex_repair=simplex_repair,
#                 has_m2_break=has_m2_break,
#                 norm_gauss=norm_gauss,
#             )
#         else:
#             interp_grids_mass = None
#             interp_vals_mass = None

#         return dc_grid, dL_grid, log_ddL_dz_grid, interp_grids_mass, interp_vals_mass

#     def evt_f(m1det, m2det, dLdet, spins_evt, Lambda):
#         dc_grid, dL_grid, log_ddL_dz_grid, interp_grids_mass, interp_vals_mass = _common_from_Lambda(Lambda)

#         spins_evt_list = spin_models._spins_as_list(spins_evt, spin_model)

#         z_evt = atinterp(bk, dLdet, dL_grid, zgrid, eps=eps_interp, side=side_interp)

#         idx_z_evt, t_z_evt = _interp_prepare_bk(bk, z_evt, zgrid, eps=eps_interp, side=side_interp)
#         idx_z_evt = bk.stop_grad(idx_z_evt)
#         dc_evt = _interp_apply_bk(bk, idx_z_evt, t_z_evt, dc_grid)
#         log_ddL_dz_evt = _interp_apply_bk(bk, idx_z_evt, t_z_evt, log_ddL_dz_grid)

#         onepz = 1.0 + z_evt
#         m1src = m1det / onepz
#         m2src = m2det / onepz

#         logp_pop_evt = log_p_pop(
#             bk,
#             m1src, m2src, z_evt, dLdet,
#             spins_evt_list, Lambda,
#             rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
#             smoothing=smoothing, simplex_repair=simplex_repair,
#             has_m2_break=has_m2_break, norm_gauss=norm_gauss,
#             dc=dc_evt, log_ddL_dz_pre=log_ddL_dz_evt,
#             param=param,
#             interp_vals_mass=interp_vals_mass,
#             interp_grids_mass=interp_grids_mass,
#             is_observed=is_observed, z_grid=zgrid, verbose=verbose,
#         )
#         return logp_pop_evt

#     def full_f(
#         m1det, m2det, dLdet, spins_evt,
#         m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
#         Lambda, Ndraw,
#     ):
#         dc_grid, dL_grid, log_ddL_dz_grid, interp_grids_mass, interp_vals_mass = _common_from_Lambda(Lambda)

#         # --- event likelihood
#         spins_evt_list = spin_models._spins_as_list(spins_evt, spin_model)

#         z_evt = atinterp(bk, dLdet, dL_grid, zgrid, eps=eps_interp, side=side_interp)

#         idx_z_evt, t_z_evt = _interp_prepare_bk(bk, z_evt, zgrid, eps=eps_interp, side=side_interp)
#         idx_z_evt = bk.stop_grad(idx_z_evt)
#         dc_evt = _interp_apply_bk(bk, idx_z_evt, t_z_evt, dc_grid)
#         log_ddL_dz_evt = _interp_apply_bk(bk, idx_z_evt, t_z_evt, log_ddL_dz_grid)

#         onepz = 1.0 + z_evt
#         m1src = m1det / onepz
#         m2src = m2det / onepz

#         logp_pop_evt = log_p_pop(
#             bk,
#             m1src, m2src, z_evt, dLdet,
#             spins_evt_list, Lambda,
#             rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
#             smoothing=smoothing, simplex_repair=simplex_repair,
#             has_m2_break=has_m2_break, norm_gauss=norm_gauss,
#             dc=dc_evt, log_ddL_dz_pre=log_ddL_dz_evt,
#             param=param,
#             interp_vals_mass=interp_vals_mass,
#             interp_grids_mass=interp_grids_mass,
#             is_observed=is_observed, z_grid=zgrid, verbose=verbose,
#         )

#         # --- selection (injections)
#         spins_inj_list = spin_models._spins_as_list(spins_inj, spin_model)

#         z_inj = atinterp(bk, dLinj, dL_grid, zgrid, eps=eps_interp, side=side_interp)

#         idx_z_inj, t_z_inj = _interp_prepare_bk(bk, z_inj, zgrid, eps=eps_interp, side=side_interp)
#         idx_z_inj = bk.stop_grad(idx_z_inj)
#         dc_inj = _interp_apply_bk(bk, idx_z_inj, t_z_inj, dc_grid)
#         log_ddL_dz_inj = _interp_apply_bk(bk, idx_z_inj, t_z_inj, log_ddL_dz_grid)

#         log_mu, var_u = sel_bias_with_uncertainty(
#             bk,
#             m1inj, m2inj, dLinj,
#             spins_inj_list,
#             log_p_draw, log_p_incl,
#             dL_grid, dc_grid, log_ddL_dz_grid,
#             Lambda, Ndraw,
#             rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
#             smoothing=smoothing, simplex_repair=simplex_repair,
#             has_m2_break=has_m2_break, norm_gauss=norm_gauss,
#             param=param,
#             interp_vals_mass=interp_vals_mass,
#             interp_grids_mass=interp_grids_mass,
#             is_observed=is_observed, z_grid=zgrid, verbose=verbose,
#             subtract_log_p_incl=subtract_log_p_incl,
#             eps_interp=eps_interp, side_interp=side_interp,
#             zinj=z_inj, dcinj=dc_inj, log_ddL_dz_inj=log_ddL_dz_inj,
#         )

#         return (
#             logp_pop_evt,
#             jnp.asarray(log_mu, dtype=jnp.float64).reshape(()),
#             jnp.asarray(var_u, dtype=jnp.float64).reshape(()),
#         )

#     return full_f, evt_f



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
    is_observed,
    verbose,
    subtract_log_p_incl,
    eps_interp,
    side_interp,
    interp_mass,
    has_interp_mass,
    has_mass_grids,
    mass_grids_jax,
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

        dc_grid = dcfun_quad(bk, zgrid, H0, Om, w0, x01, w01)
        dL_grid = dLfun(bk, zgrid, H0, Om, w0, Xi0, nXi0, dc=dc_grid, param=param, x01=x01, w01=w01)
        log_ddL_dz_grid = log_ddL_dz(bk, zgrid, H0, Om, w0, Xi0, nXi0, dc=dc_grid, x01=x01, w01=w01, param=param)

        lambdaBBHmass = _extract_lambdaBBHmass(
            Lambda,
            rate_model=rate_model,
            spin_model=spin_model,
            mass_model=mass_model,
        )

        if has_interp_mass:
            if not has_mass_grids:
                alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, lambda2, beta, m2_low, delta_m2, epsilon, m_g, w_g, sig_g_low, sig_g_high = lambdaBBHmass

                m1_grid = mass_models.build_m1_grid_DPLDP_bk(
                    bk,
                    alpha1=alpha1, alpha2=alpha2, mb=mb,
                    mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2,
                    m1_low=m1_low, m_high=m_high,
                    delta_m1=delta_m1,
                    n_peak=interp_mass,
                    n_tail_low=interp_mass//3,
                    n_tail_high=interp_mass//4,
                    n_taper=interp_mass//2,
                    frac_gauss1=0.4,
                )

                m2_grid = mass_models.build_m2_grid_bk(
                    bk,
                    m2_low=m2_low,
                    m2_high=m_high,
                    delta_m2=delta_m2,
                    n_grid=interp_mass,
                    n_tail_low=interp_mass//3,
                    n_tail_high=interp_mass//4,
                    n_taper=interp_mass//2,
                )
                interp_grids_mass_jax = (m1_grid, m2_grid)
            else:
                # mass_grids are static: preconverted once outside and captured here
                interp_grids_mass_jax = mass_grids_jax
            
            interp_vals_mass_jax = _maybe_precompute_mass_tables(
                bk,
                lambdaBBHmass,
                interp_grids_mass_jax=interp_grids_mass_jax,
                mass_model=mass_model,
                smoothing=smoothing,
                simplex_repair=simplex_repair,
                has_m2_break=has_m2_break,
                norm_gauss=norm_gauss,
                #is_observed=is_observed,
            )
        else:
            interp_grids_mass_jax = None
            interp_vals_mass_jax = None

        # --- event likelihood
        spins_evt_list = spin_models._spins_as_list(spins_evt, spin_model)

        z_evt = atinterp(bk, dLdet, dL_grid, zgrid, eps=eps_interp, side=side_interp)
        #dc_evt = atinterp(bk, z_evt, zgrid, dc_grid, eps=eps_interp, side=side_interp)
        #log_ddL_dz_evt = atinterp(bk, z_evt, zgrid, log_ddL_dz_grid, eps=eps_interp, side=side_interp)
        # Reuse a single searchsorted for any z->grid interpolations we need downstream
        idx_z_evt, t_z_evt = _interp_prepare_bk(bk, z_evt, zgrid, eps=eps_interp, side=side_interp)
        #idx_z_evt = bk.stop_grad(idx_z_evt)
        dc_evt = _interp_apply_bk(bk, idx_z_evt, t_z_evt, dc_grid)
        log_ddL_dz_evt = _interp_apply_bk(bk, idx_z_evt, t_z_evt, log_ddL_dz_grid)


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
                param=param,
                interp_vals_mass=interp_vals_mass_jax,
                interp_grids_mass=interp_grids_mass_jax,
                is_observed=is_observed, z_grid=zgrid, verbose=verbose,
        )

        # --- selection (injections)
        spins_inj_list = spin_models._spins_as_list(spins_inj, spin_model)

        z_inj = atinterp(bk, dLinj, dL_grid, zgrid, eps=eps_interp, side=side_interp)
        #dc_inj = atinterp(bk, z_inj, zgrid, dc_grid, eps=eps_interp, side=side_interp)
        #log_ddL_dz_inj = atinterp(bk, z_inj, zgrid, log_ddL_dz_grid, eps=eps_interp, side=side_interp)
        idx_z_inj, t_z_inj = _interp_prepare_bk(bk, z_inj, zgrid, eps=eps_interp, side=side_interp)
        #idx_z_inj = bk.stop_grad(idx_z_inj)  # optional: affects grads, not values
        
        dc_inj = _interp_apply_bk(bk, idx_z_inj, t_z_inj, dc_grid)
        log_ddL_dz_inj = _interp_apply_bk(bk, idx_z_inj, t_z_inj, log_ddL_dz_grid)



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
                param=param,
                interp_vals_mass=interp_vals_mass_jax,
                interp_grids_mass=interp_grids_mass_jax,
                is_observed=is_observed, z_grid=zgrid, verbose=verbose,
                subtract_log_p_incl=subtract_log_p_incl,
                eps_interp=eps_interp, side_interp=side_interp,
                zinj=z_inj, dcinj=dc_inj, log_ddL_dz_inj=log_ddL_dz_inj,
        )

        return (
            logp_pop_evt,
            jnp.asarray(log_mu, dtype=jnp.float64).reshape(()),
            jnp.asarray(var_u, dtype=jnp.float64).reshape(()),
        )

    return _f







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
                 interp_mass=0,
                 is_observed=False, 
                 mass_grids = None,
                 verbose=False,
                 subtract_log_p_incl=False, eps_interp=1e-12, side_interp="right"):
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
        self.is_observed = bool(is_observed)
        self.verbose = bool(verbose)
        self.subtract_log_p_incl = bool(subtract_log_p_incl)
        self.eps_interp = float(eps_interp)
        self.side_interp = str(side_interp)

        self.interp_mass = interp_mass
        self.has_interp_mass = bool(interp_mass>0)
        self.has_mass_grids = bool(mass_grids is not None)
        self.mass_grids = mass_grids

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
        is_observed = self.is_observed
        verbose = self.verbose
        subtract_log_p_incl = self.subtract_log_p_incl
        eps_interp = self.eps_interp
        side_interp = self.side_interp
        interp_mass = self.interp_mass
        has_interp_mass = self.has_interp_mass
        has_mass_grids = self.has_mass_grids
        mass_grids = self.mass_grids

        
        cosmo_zgrid = jnp.asarray(self.zgrid, dtype=jnp.float64)
        
        x01_jax = jnp.asarray(self.x01, dtype=jnp.float64)
        w01_jax = jnp.asarray(self.w01, dtype=jnp.float64)



        # Pre-convert static mass grids once (if provided)
        mass_grids_jax = None
        if has_mass_grids:
            mg = mass_grids
            mass_grids_jax = (
                jnp.asarray(mg[0], dtype=jnp.float64),
                jnp.asarray(mg[1], dtype=jnp.float64),
            )

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
            is_observed=is_observed,
            verbose=verbose,
            subtract_log_p_incl=subtract_log_p_incl,
            eps_interp=eps_interp,
            side_interp=side_interp,
            interp_mass=interp_mass,
            has_interp_mass=has_interp_mass,
            has_mass_grids=has_mass_grids,
            mass_grids_jax=mass_grids_jax,
        )



        def _vjp(m1det, m2det, dLdet, spins_evt,
                 m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
                 Lambda, Ndraw,
                 g_logp_pop, g_log_mu, g_var_u):

            g_logp_pop = jnp.reshape(g_logp_pop, m1det.shape)
            g_log_mu = jnp.reshape(g_log_mu, ())
            g_var_u = jnp.reshape(g_var_u, ())

            (_, _, _), pull = jax.vjp(
                core_f,
                m1det, m2det, dLdet, spins_evt,
                m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
                Lambda, Ndraw
            )
            grads = pull((g_logp_pop, g_log_mu, g_var_u))
            return grads[0], grads[1], grads[2], grads[3], grads[10]

        return jax.jit(_vjp)

    # def _build_jax_vjp(self):
    #     bk = JAXBackend()
    
    #     rate_model = self.rate_model
    #     mass_model = self.mass_model
    #     spin_model = self.spin_model
    #     smoothing = self.smoothing
    #     simplex_repair = self.simplex_repair
    #     has_m2_break = self.has_m2_break
    #     norm_gauss = self.norm_gauss
    #     param = self.param
    #     is_observed = self.is_observed
    #     verbose = self.verbose
    #     subtract_log_p_incl = self.subtract_log_p_incl
    #     eps_interp = self.eps_interp
    #     side_interp = self.side_interp
    #     interp_mass = self.interp_mass
    #     has_interp_mass = self.has_interp_mass
    #     has_mass_grids = self.has_mass_grids
    #     mass_grids = self.mass_grids
    
    #     cosmo_zgrid = jnp.asarray(self.zgrid, dtype=jnp.float64)
    #     x01_jax = jnp.asarray(self.x01, dtype=jnp.float64)
    #     w01_jax = jnp.asarray(self.w01, dtype=jnp.float64)
    
    #     mass_grids_jax = None
    #     if has_mass_grids:
    #         mg = mass_grids
    #         mass_grids_jax = (
    #             jnp.asarray(mg[0], dtype=jnp.float64),
    #             jnp.asarray(mg[1], dtype=jnp.float64),
    #         )
    
    #     full_f, evt_f = _make_pop_and_sel_core(
    #         bk=bk,
    #         zgrid=cosmo_zgrid,
    #         x01=x01_jax,
    #         w01=w01_jax,
    #         rate_model=rate_model,
    #         mass_model=mass_model,
    #         spin_model=spin_model,
    #         smoothing=smoothing,
    #         simplex_repair=simplex_repair,
    #         has_m2_break=has_m2_break,
    #         norm_gauss=norm_gauss,
    #         param=param,
    #         is_observed=is_observed,
    #         verbose=verbose,
    #         subtract_log_p_incl=subtract_log_p_incl,
    #         eps_interp=eps_interp,
    #         side_interp=side_interp,
    #         interp_mass=interp_mass,
    #         has_interp_mass=has_interp_mass,
    #         has_mass_grids=has_mass_grids,
    #         mass_grids_jax=mass_grids_jax,
    #     )
    
    #     # -------- kernel A: event grads only (NO selection) --------
    #     def _evt_grads(m1det, m2det, dLdet, spins_evt, Lambda, g_logp_pop):
    #         g_logp_pop = jnp.reshape(g_logp_pop, m1det.shape)
    
    #         def phi_evt(m1det_, m2det_, dLdet_, spins_evt_):
    #             lp_evt = evt_f(m1det_, m2det_, dLdet_, spins_evt_, Lambda)
    #             return jnp.vdot(g_logp_pop, lp_evt)
    
    #         dm1, dm2, ddL, dspin = jax.grad(phi_evt, argnums=(0, 1, 2, 3))(m1det, m2det, dLdet, spins_evt)
    #         return dm1, dm2, ddL, dspin
    
    #     _evt_grads_jit = jax.jit(_evt_grads)
    
    #     # -------- kernel B: Lambda grad (event + selection) --------
    #     def _lambda_grad(
    #         m1det, m2det, dLdet, spins_evt,
    #         m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
    #         Lambda, Ndraw,
    #         g_logp_pop, g_log_mu, g_var_u,
    #     ):
    #         g_logp_pop = jnp.reshape(g_logp_pop, m1det.shape)
    #         g_log_mu = jnp.reshape(g_log_mu, ())
    #         g_var_u = jnp.reshape(g_var_u, ())
    
    #         # Treat data arrays as constants for Lambda differentiation
    #         m1det_c = lax.stop_gradient(m1det)
    #         m2det_c = lax.stop_gradient(m2det)
    #         dLdet_c = lax.stop_gradient(dLdet)
    #         spins_evt_c = lax.stop_gradient(spins_evt)
    
    #         m1inj_c = lax.stop_gradient(m1inj)
    #         m2inj_c = lax.stop_gradient(m2inj)
    #         dLinj_c = lax.stop_gradient(dLinj)
    #         spins_inj_c = lax.stop_gradient(spins_inj)
    #         lpd_c = lax.stop_gradient(log_p_draw)
    #         lpi_c = lax.stop_gradient(log_p_incl)
    #         Ndraw_c = lax.stop_gradient(Ndraw)
    
    #         def phi_L(Lambda_):
    #             lp_evt, log_mu, var_u = full_f(
    #                 m1det_c, m2det_c, dLdet_c, spins_evt_c,
    #                 m1inj_c, m2inj_c, dLinj_c, spins_inj_c, lpd_c, lpi_c,
    #                 Lambda_, Ndraw_c
    #             )
    #             return jnp.vdot(g_logp_pop, lp_evt) + g_log_mu * log_mu + g_var_u * var_u
    
    #         return jax.grad(phi_L)(Lambda)
    
    #     _lambda_grad_jit = jax.jit(_lambda_grad)
    
    #     # -------- final callable used by perform(): returns event grads + Lambda grad --------
    #     def _vjp(
    #         m1det, m2det, dLdet, spins_evt,
    #         m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
    #         Lambda, Ndraw,
    #         g_logp_pop, g_log_mu, g_var_u
    #     ):
    #         dm1, dm2, ddL, dspin = _evt_grads_jit(m1det, m2det, dLdet, spins_evt, Lambda, g_logp_pop)
    #         dLambda = _lambda_grad_jit(
    #             m1det, m2det, dLdet, spins_evt,
    #             m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
    #             Lambda, Ndraw,
    #             g_logp_pop, g_log_mu, g_var_u
    #         )
    #         return dm1, dm2, ddL, dspin, dLambda
    
    #     return jax.jit(_vjp)
    
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
                 is_observed=False,
                 mass_grids=None,
                 verbose=False,
                 subtract_log_p_incl=False, eps_interp=1e-12, side_interp="right"):
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
        self.interp_mass = int(interp_mass)
        self.is_observed = bool(is_observed)
        self.mass_grids = mass_grids
        self.verbose = bool(verbose)
        self.subtract_log_p_incl = bool(subtract_log_p_incl)
        self.eps_interp = float(eps_interp)
        self.side_interp = side_interp

        # Backend (needed by cosmology/mass grid builders)
        self._bk = JAXBackend()

        # One-time device cache (shared with vjp op)
        self._cached_inj = None

        # Last-call cache to reuse per-call device args between fwd and vjp
        # Stores: (key, (m1det_j, m2det_j, dLdet_j, spins_evt_j, Lambda_j))
        self._last_call = None

        # Pre-device-put static mass grids (if provided)
        self._mass_grids_jax = None
        has_mass_grids = mass_grids is not None
        if has_mass_grids:
            m1g, m2g = mass_grids
            self._mass_grids_jax = (_to_device(m1g), _to_device(m2g))

        # Build and jit core forward function
        self._jax_fwd = self._build_jax_fwd()

        # Build vjp op (shares caches via _parent_op pointer)
        self._vjp_op = _PopAndSelJAXVJPOp(
            zgrid=self.zgrid, x01=self.x01, w01=self.w01,
            rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
            smoothing=smoothing, simplex_repair=simplex_repair, has_m2_break=has_m2_break,
            norm_gauss=norm_gauss, param=param, interp_mass=interp_mass,
            is_observed=is_observed, mass_grids=mass_grids,
            verbose=verbose, subtract_log_p_incl=subtract_log_p_incl,
            eps_interp=eps_interp, side_interp=side_interp
        )
        self._vjp_op._parent_op = self

    def _build_jax_fwd(self):
        has_interp_mass = self.interp_mass > 0
        has_mass_grids = self.mass_grids is not None
        mass_grids_jax = self._mass_grids_jax

        core_f = _make_pop_and_sel_core(
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
            is_observed=self.is_observed,
            verbose=self.verbose,
            subtract_log_p_incl=self.subtract_log_p_incl,
            eps_interp=self.eps_interp,
            side_interp=self.side_interp,
            interp_mass=self.interp_mass,
            has_interp_mass=has_interp_mass,
            has_mass_grids=has_mass_grids,
            mass_grids_jax=mass_grids_jax,
        )
        return jax.jit(core_f)


    # def _build_jax_fwd(self):
    #     has_interp_mass = self.interp_mass > 0
    #     has_mass_grids = self.mass_grids is not None
    #     mass_grids_jax = self._mass_grids_jax
    
    #     full_f, _evt_f = _make_pop_and_sel_core(
    #         bk=self._bk,
    #         zgrid=self.zgrid,
    #         x01=self.x01,
    #         w01=self.w01,
    #         rate_model=self.rate_model,
    #         mass_model=self.mass_model,
    #         spin_model=self.spin_model,
    #         smoothing=self.smoothing,
    #         simplex_repair=self.simplex_repair,
    #         has_m2_break=self.has_m2_break,
    #         norm_gauss=self.norm_gauss,
    #         param=self.param,
    #         is_observed=self.is_observed,
    #         verbose=self.verbose,
    #         subtract_log_p_incl=self.subtract_log_p_incl,
    #         eps_interp=self.eps_interp,
    #         side_interp=self.side_interp,
    #         interp_mass=self.interp_mass,
    #         has_interp_mass=has_interp_mass,
    #         has_mass_grids=has_mass_grids,
    #         mass_grids_jax=mass_grids_jax,
    #     )
    #     return jax.jit(full_f)


    
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




