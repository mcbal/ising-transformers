from functools import partial
from typing import List

import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import jax.random as jrandom
from einops import rearrange, repeat
from equinox import static_field
from jax import grad, vmap
from jax.numpy import einsum
import jaxopt
from jaxopt import AndersonAcceleration
from jaxopt.tree_util import tree_add, tree_sub

# Ising transformer layer: helper functions.


@eqx.filter_jit
def _phi_covar_inv(t, J):
    V = jnp.diag(t) - J
    # V_inv = jnp.linalg.solve(V, jnp.eye(V.shape[-1]))
    V_inv = jnp.diag(1.0 / t) + jnp.diag(1.0 / t) @ J @ jnp.diag(1.0 / t)
    # jax.experimental.host_callback.id_print((V_inv @ V - jnp.eye(V.shape[-1])).sum())
    return V, V_inv


@eqx.filter_jit
def _phi(t, h, beta, J):
    V, V_inv = _phi_covar_inv(t, J)
    _, logdet = jnp.linalg.slogdet(V)
    return (
        beta * jnp.sum(t, axis=-1) - 0.5 * logdet + beta / 4.0 * einsum("... i f, ... i j, ... j f -> ...", h, V_inv, h)
    )


# ADD TEST TO CHECK IF JAC PHI IS REALLY GRADIENT OF PHI
@eqx.filter_jit
def _jac_phi(t, h, beta, J):
    _, V_inv = _phi_covar_inv(t, J)
    return (
        beta * jnp.ones_like(t)
        - 0.5 * jnp.diagonal(V_inv, axis1=-2, axis2=-1)
        - beta / 4.0 * einsum("... j i, ... j f, ... k f, ... i k -> ... i", V_inv, h, h, V_inv)
    )


@eqx.filter_jit
def _root_fun(t, h, beta, J):
    # assert tree_add(_jac_phi(t, h, beta, J), t) == tree_add(grad(_phi)(t, h, beta, J), t)
    return tree_add(_jac_phi(t, h, beta, J), t)
    # return tree_add(grad(_phi)(t, h, beta, J), t)


@partial(jax.jit, static_argnums=(0,))
def solve_linear_system_fixed_point(matvec, v):
    """Solve linear system matvec(u) = v.
    The solution u* of the system is the fixed point of:
        T(u) = matvec(u) + u - v
    """

    @partial(jax.jit, static_argnums=(1,))
    def fixed_point_fun(u, matvec, v):
        return tree_sub(tree_add(matvec(u), u), v)

    bwd_solver = AndersonAcceleration(
        fixed_point_fun=fixed_point_fun,
        history_size=2,
        ridge=1e-6,
        beta=0.1,  # strong damping to prevent solution from jumping to other, dangerous local minima
        tol=1e-4,
        maxiter=200,
    )
    return jax.jit(bwd_solver.run, static_argnums=(1,))(v, matvec, v)[0]


@eqx.filter_jit
def _t_star(h, beta, J):
    solver = AndersonAcceleration(
        fixed_point_fun=_root_fun,
        history_size=2,
        ridge=1e-6,
        maxiter=20,
        beta=0.1,  # strong damping to prevent solution from jumping to other local minima
        tol=1e-4,
        # jit=True,
        # verbose=True,
        implicit_diff=True,
        implicit_diff_solve=partial(jaxopt.linear_solve.solve_bicgstab, tol=1e-4, maxiter=20)
        # implicit_diff_solve=jax.jit(solve_linear_system_fixed_point, static_argnums=(0)),
    )
    t_init = jnp.ones(*h.shape[:-1], dtype=h.dtype)
    return solver.run(t_init, h, beta, J)[0]


@eqx.filter_jit
def _log_Z(h, beta, J):
    """Compute steepest-descent approximation of log(Z) for large vector dimension."""
    num_spins = h.shape[-2]
    return -0.5 * num_spins * (1.0 + jnp.log(2.0 * beta)) + _phi(_t_star(h, beta, J), h, beta, J)


# Ising transformer layer: module class.


class IsingTransformerLayer(eqx.Module):
    """Differentiable steepest-descent approximation of an Ising vector-spin model.

    Forward pass bathes the spin system in data and returns magnetizations.

    """

    dim: int = static_field()
    dim_head: int = static_field()
    beta: float = static_field()

    norm: eqx.Module
    to_q: eqx.Module
    to_k: eqx.Module

    heads: int = static_field()
    scale: float = static_field()

    def __init__(
        self,
        dim,
        dim_head,
        heads,
        key,
        beta=1.0,
    ):
        super().__init__()

        inner_dim = dim_head * heads
        self.beta = beta
        self.norm = eqx.nn.LayerNorm(dim)

        self.to_q = eqx.nn.Linear(dim, inner_dim, use_bias=False, key=key)
        self.to_k = eqx.nn.Linear(dim, inner_dim, use_bias=False, key=key)

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5

    def __call__(self, x, pos_emb=None, causal_mask=None):
        x = self.norm(x) / jnp.sqrt(self.dim_head)

        def log_Z_heads(x, beta):
            # x = rearrange(x, "...  h n d -> ... n (h d)", h=self.heads)

            q = rearrange(jax.vmap(self.to_q)(x), "... n (h d) -> ... h n d", h=self.heads) * self.dim_head**-0.5
            k = rearrange(jax.vmap(self.to_k)(x), "... n (h d) -> ... h n d", h=self.heads)

            if pos_emb is not None:
                q, k = map(lambda t: apply_rotary_pos_emb(t, pos_emb), (q, k))

            x = rearrange(x, "... n (h d) -> ... h n d", h=self.heads)

            def _log_Z_head(x, beta, q, k):
                attn = einsum("... i d, ... j d -> ... i j", q, k)

                if causal_mask is not None:
                    attn = jnp.where(causal_mask, attn, -1e10)

                # YOU FOKKING SHITHEAD OF COURSE GRADIENT IS ZERO IF J DOES NOT DEPEND ON QKH
                J = jax.nn.softmax(attn, axis=-1) / jnp.sqrt(x.shape[-2] * self.dim_head)
                # J = 0.02 * jnp.ones_like(attn) / jnp.sqrt(x.shape[-2] * self.dim_head)

                # jax.experimental.host_callback.id_print(grad(_log_Z, argnums=)(x, beta, J))

                return _log_Z(x, beta, J)

            return vmap(grad(_log_Z_head), in_axes=(0, None, 0, 0))(x, beta, q, k)

        # x = rearrange(x, "... n (h d) -> ... h n d", h=self.heads)

        return rearrange(
            log_Z_heads(x, self.beta),
            "... h n d -> ... n (h d)",
            h=self.heads,
            n=x.shape[-2],
        )

        # return rearrange(
        #     jnp.diagonal(
        #         rearrange(jax.jacrev(log_Z_heads)(x, self.beta), "... n (h d) -> ... h n d", h=self.heads),
        #         axis1=0,
        #         axis2=1,
        #     ),
        #     "... n d h -> ... n (h d)",
        #     h=self.heads,
        # )


# Transformer model with rotary positional embeddings.


def fixed_pos_embedding(inv_freq, seq):
    sinusoid_inp = einsum("i , j -> i j", jnp.arange(seq), inv_freq)
    sinusoid_inp = repeat(sinusoid_inp, "... d -> ... (d r)", r=2)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


@jax.jit
def rotate_every_two(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x, "... d r -> ... (d r)")


@jax.jit
def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    return (x * cos) + (rotate_every_two(x) * sin)


class IsingTransformer(eqx.Module):
    """Collective of vector-spin models in sequential order.

    Forward pass feeds data into a stack of spin systems and returns final
    magnetizations in response.

    """

    embedding: jnp.ndarray
    inv_freq: jnp.ndarray
    layers: List[List[eqx.Module]]
    final_norm: eqx.Module

    def __init__(self, num_tokens, dim, dim_head, depth, heads, key):
        self.embedding = 0.02 * jrandom.normal(key, (num_tokens, dim))
        self.inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim_head, 2) / dim_head))

        # split key so layers dont start out identical
        self.layers = [IsingTransformerLayer(dim=dim, dim_head=dim_head, heads=heads, key=key) for _ in range(depth)]
        self.final_norm = eqx.nn.LayerNorm(dim)

    def __call__(self, x):
        n = x.shape[-1]
        x = self.embedding[x]

        rotary_emb = fixed_pos_embedding(self.inv_freq, n)
        causal_mask = jnp.tril(jnp.ones((n, n)))

        for layer in self.layers:
            x = layer(x, pos_emb=rotary_emb, causal_mask=causal_mask)

        x = self.final_norm(x)
        out = x @ self.embedding.transpose()
        return out
