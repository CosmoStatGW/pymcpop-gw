import jax.numpy as jnp
from pytensor.link.jax.dispatch.basic import jax_funcify

# IMPORTANT: import the class from the SAME place you use in your model code
from pytensor_tools_new import PopAndSelJAXOp  # <- change to your real module path

@jax_funcify.register(PopAndSelJAXOp)
def jax_funcify_PopAndSelJAXOp(op, **kwargs):
    jax_fwd = op._jax_fwd

    def _impl(m1det, m2det, dLdet, spins_evt,
              m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
              Lambda, Ndraw):
        return jax_fwd(
            jnp.asarray(m1det), jnp.asarray(m2det), jnp.asarray(dLdet), jnp.asarray(spins_evt),
            jnp.asarray(m1inj), jnp.asarray(m2inj), jnp.asarray(dLinj), jnp.asarray(spins_inj),
            jnp.asarray(log_p_draw), jnp.asarray(log_p_incl),
            jnp.asarray(Lambda), jnp.asarray(Ndraw),
        )

    return _impl