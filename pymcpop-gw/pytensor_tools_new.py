from __future__ import annotations

from typing import Tuple
import numpy as np
import hashlib

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

from pytensor.gradient import DisconnectedType, grad_not_implemented



# ---------------------------------------------------------------------
# pytensor glue utils
# ---------------------------------------------------------------------


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
                 #interp_vals_mass=None, 
                 interp_mass=0,
                 is_observed=False, 
                 #z_grid=None, 
                 verbose=False,
                 subtract_log_p_incl=False, eps_interp=1e-12, side_interp="right"):
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

        self.interp_mass = interp_mass
        self.has_interp_mass = bool(interp_mass>0)
        #self.z_grid = None if z_grid is None else np.asarray(z_grid)

        self._cached_inj = None
        self._jax_vjp = self._build_jax_vjp()

    def _build_jax_vjp(self):
        bk = JAXBackend()
        # z_from_dL_interp = make_z_from_dL_interp(
        #         bk,
        #         eps=self.eps_interp,
        #         side=self.side_interp,
        #         param=self.param,
        #     )

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

        
        #pop_z_grid = None if self.z_grid is None else jnp.asarray(self.z_grid, dtype=jnp.float64)

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


            lambdaBBHmass = _extract_lambdaBBHmass(
            Lambda,
            rate_model=rate_model,
            spin_model=spin_model,
            mass_model=mass_model,
            )

            

            if has_interp_mass:
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
                            # resolution controls
                            n_total=500,
                            n_taper=200,
                        )

    
                interp_grids_mass_jax = tuple( jnp.asarray(g, dtype=jnp.float64) for g in (m1_grid, m2_grid) )
            else:
                interp_grids_mass_jax = None
                
            interp_vals_mass_jax = _maybe_precompute_mass_tables(
                bk,
                lambdaBBHmass,
                interp_grids_mass_jax,
                mass_model=mass_model,
                smoothing=smoothing,
                simplex_repair=simplex_repair,
                has_m2_break=has_m2_break,
                norm_gauss=norm_gauss,
            )
   
            # events
            z_evt = atinterp(bk, dLdet, dL_grid, cosmo_zgrid) #z_from_dL_interp(dLdet, theta5, cosmo_zgrid, dL_grid, x01_jax, w01_jax)
            dc_evt = atinterp(bk, z_evt, cosmo_zgrid, dc_grid)
            log_ddL_dz_evt = atinterp(bk, z_evt, cosmo_zgrid, log_ddL_dz_grid)


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
                is_observed=is_observed, z_grid=cosmo_zgrid, verbose=verbose,
            )

            # injections (precompute and pass)
            z_inj = atinterp(bk, dLinj, dL_grid, cosmo_zgrid ) #z_from_dL_interp(dLinj, theta5, cosmo_zgrid, dL_grid, x01_jax, w01_jax)
            dc_inj = atinterp(bk, z_inj, cosmo_zgrid, dc_grid)
            log_ddL_dz_inj = atinterp(bk, z_inj, cosmo_zgrid, log_ddL_dz_grid)

            log_mu, var_u = sel_bias_with_uncertainty(
                bk,
                m1inj, m2inj, dLinj,
                spins_unpack_inj(spins_inj),
                log_p_draw, log_p_incl,
                dL_grid, dc_grid, log_ddL_dz_grid,
                Lambda, Ndraw,
                #zgrid=cosmo_zgrid,
                rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
                smoothing=smoothing, simplex_repair=simplex_repair,
                has_m2_break=has_m2_break, norm_gauss=norm_gauss,
                param=param,
                interp_vals_mass=interp_vals_mass_jax,
                interp_grids_mass=interp_grids_mass_jax,
                is_observed=is_observed, z_grid=cosmo_zgrid, verbose=verbose,
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
        
        
        #print("g_logp_pop ndim/shape:", np.asarray(g_logp_pop).ndim, np.asarray(g_logp_pop).shape)
        
        
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


# ----------------------------
#  Op (Pop + Selection)
# ----------------------------


        
class PopAndSelJAXOp(Op):
    itypes = [
        at.dvector, at.dvector, at.dvector, at.dmatrix,
        at.dvector, at.dvector, at.dvector, at.dmatrix, at.dvector, at.dvector,
        at.dvector, at.dscalar,
    ]
    otypes = [at.dvector, at.dscalar, at.dscalar]

    def __init__(self, *, zgrid, x01, w01, rate_model, mass_model, spin_model,
                 smoothing="LVK", simplex_repair=False, has_m2_break=False, norm_gauss="uplow",
                 param="vanilla", 
                 #interp_vals_mass=None, 
                 interp_mass=None,
                 is_observed=False, 
                 #z_grid=None, 
                 verbose=False,
                 subtract_log_p_incl=False, eps_interp=1e-12, side_interp="right"):
        super().__init__()
        self.zgrid = np.asarray(zgrid, dtype="float64")
        self.x01 = np.asarray(x01, dtype="float64")
        self.w01 = np.asarray(w01, dtype="float64")
        self.has_interp_mass=bool(interp_mass>0)

        self.kw = dict(
            zgrid=self.zgrid, x01=self.x01, w01=self.w01,
            rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
            smoothing=smoothing, simplex_repair=simplex_repair, has_m2_break=has_m2_break,
            norm_gauss=norm_gauss, param=param,
            #interp_vals_mass=interp_vals_mass, 
            interp_mass=interp_mass,
            #has_interp_mass=bool(interp_mass>0),
            is_observed=is_observed, 
            #z_grid=z_grid, 
            verbose=verbose,
            subtract_log_p_incl=subtract_log_p_incl,
            eps_interp=eps_interp, side_interp=side_interp,
        )

        self._jax_fwd = self._build_jax_fwd()
        self._cached_inj = None
        self._vjp_op = _PopAndSelJAXVJPOp(**self.kw)

    def _build_jax_fwd(self):
        bk = JAXBackend()
        # z_from_dL_interp = make_z_from_dL_interp(
        #     bk,
        #     eps=self.kw["eps_interp"],
        #     side=self.kw["side_interp"],
        #     param=self.kw["param"],
        # )

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

        interp_mass = self.kw["interp_mass"]
        has_interp_mass = self.has_interp_mass
        #z_grid = self.kw["z_grid"]

    
        

        #z_grid_jax = None if z_grid is None else jnp.asarray(z_grid, dtype=jnp.float64)

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

            lambdaBBHmass = _extract_lambdaBBHmass(
                Lambda,
                rate_model=rate_model,
                spin_model=spin_model,
                mass_model=mass_model,
            )

            if has_interp_mass:
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
                            # resolution controls
                            n_total=500,
                            n_taper=200,
                        )

                interp_grids_mass_jax = tuple( jnp.asarray(g, dtype=jnp.float64) for g in (m1_grid, m2_grid) )
            else:
                interp_grids_mass_jax = None
                

            interp_vals_mass_jax = _maybe_precompute_mass_tables(
                bk,
                lambdaBBHmass,
                interp_grids_mass_jax,
                mass_model=mass_model,
                smoothing=smoothing,
                simplex_repair=simplex_repair,
                has_m2_break=has_m2_break,
                norm_gauss=norm_gauss,
            )
    

            
            # event
            z_evt = atinterp(bk, dLdet, dL_grid, zgrid ) #z_from_dL_interp(dLdet, theta5, zgrid, dL_grid, x01, w01)
            dc_evt = atinterp(bk, z_evt, zgrid, dc_grid)
            log_ddL_dz_evt = atinterp(bk, z_evt, zgrid, log_ddL_dz_grid)


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
                z_grid=zgrid,
                verbose=verbose,
            )

            # inj (precompute cosmology pieces; passed into sel_bias)
            z_inj = atinterp(bk, dLinj, dL_grid, zgrid ) #z_from_dL_interp(dLinj, theta5, zgrid, dL_grid, x01, w01)
            dc_inj = atinterp(bk, z_inj, zgrid, dc_grid)
            log_ddL_dz_inj = atinterp(bk, z_inj, zgrid, log_ddL_dz_grid)

            log_mu, var_u = sel_bias_with_uncertainty(
                bk,
                m1inj, m2inj, dLinj,
                spins_unpack_inj(spins_inj),
                log_p_draw, log_p_incl,
                dL_grid, dc_grid, log_ddL_dz_grid,
                Lambda, Ndraw,
                #zgrid=zgrid,
                rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
                smoothing=smoothing, simplex_repair=simplex_repair, has_m2_break=has_m2_break,
                norm_gauss=norm_gauss,
                param=param,
                interp_vals_mass=interp_vals_mass_jax,
                interp_grids_mass=interp_grids_mass_jax,
                is_observed=is_observed,
                z_grid=zgrid,
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




