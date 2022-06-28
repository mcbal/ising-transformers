from typing import List

import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import jax.random as jrandom
from einops import rearrange, repeat
from equinox import static_field
from jax import grad, vmap
from jax.numpy import einsum
from jaxopt import AndersonAcceleration
from jaxopt.tree_util import tree_add

# Ising transformer layer: helper functions.


@eqx.filter_jit
def _phi_covar_inv(t, J):
    V = jnp.diag(t) - J
    V_inv = jnp.linalg.solve(V, jnp.eye(V.shape[-1]))
    return V, V_inv


@eqx.filter_jit
def _phi(t, h, beta, J):
    V, V_inv = _phi_covar_inv(t, J)
    sign, logdet = jnp.linalg.slogdet(V)
    return (
        beta * jnp.sum(t, axis=-1)
        - 0.5 * sign * logdet
        + beta / 4.0 * einsum("... i f, ... i j, ... j f -> ...", h, V_inv, h)
    )


@eqx.filter_jit
def _jac_phi(t, h, beta, J):
    _, V_inv = _phi_covar_inv(t, J)
    return (
        beta * jnp.ones_like(t)
        - 0.5 * jnp.diagonal(V_inv, axis1=-2, axis2=-1)
        - beta / 4.0 * einsum("... j i, ... j f, ... k f, ... i k -> ... i", V_inv, h, h, V_inv)
    )


@eqx.filter_jit
def root_fun(t, h, beta, J):
    return tree_add(_jac_phi(t, h, beta, J), t)
    # return tree_add(grad(_phi)(t, h, beta, J), t)


@eqx.filter_jit
def t_star(h, beta, J):
    solver = AndersonAcceleration(
        fixed_point_fun=root_fun,
        history_size=2,
        ridge=1e-6,
        maxiter=20,
        beta=0.1,  # strong damping to prevent solution from jumping to other local minima
        tol=1e-4,
        jit=True,
        implicit_diff=True,
    )
    t_init = jnp.ones(*h.shape[:-1], dtype=h.dtype)
    return solver.run(t_init, h, beta, J)[0]


@eqx.filter_jit
def log_Z(h, beta, J):
    """Compute steepest-descent approximation of log(Z) for large vector dimension."""
    num_spins = h.shape[-2]
    return -0.5 * num_spins * (1.0 + jnp.log(2.0 * beta)) + _phi(t_star(h, beta, J), h, beta, J)


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
    neg_mask_value: float = static_field()

    def __init__(
        self,
        *,
        dim,
        dim_head,
        heads,
        key,
        neg_mask_value=-1e10,
        beta=1.0,
        solver_fwd_max_iter=20,
        solver_fwd_tol=1e-3,
        solver_bwd_max_iter=20,
        solver_bwd_tol=1e-4,
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
        self.neg_mask_value = neg_mask_value

    def __call__(self, x, *, pos_emb=None, causal_mask=None):
        x = self.norm(x) / jnp.sqrt(self.dim_head)

        x = rearrange(x, "... n (h d) -> ... h n d", h=self.heads)

        def _magnetizations(x, beta):
            x = rearrange(x, "...  h n d -> ... n (h d)", h=self.heads)

            q = rearrange(jax.vmap(self.to_q)(x), "... n (h d) -> ... h n d", h=self.heads)
            k = rearrange(jax.vmap(self.to_k)(x), "... n (h d) -> ... h n d", h=self.heads)

            if pos_emb is not None:
                q, k = map(lambda t: apply_rotary_pos_emb(t, pos_emb), (q, k))

            x = rearrange(x, "... n (h d) -> ... h n d", h=self.heads)

            def _log_Z_head(
                x,
                q,
                k,
            ):
                attn = einsum("... i d, ... j d -> ... i j", q, k)  # * self.dim_head**-0.5

                if causal_mask is not None:
                    attn = jnp.where(causal_mask, attn, self.neg_mask_value)

                J = jax.nn.softmax(attn, axis=-1) / jnp.sqrt(x.shape[-2] * self.dim_head)

                return log_Z(x, beta, J)

            return vmap(
                grad(_log_Z_head),
                in_axes=(0, 0, 0),
            )(x, q, k)

        return rearrange(
            _magnetizations(x, self.beta),
            "... h n d -> ... n (h d)",
            h=self.heads,
            n=x.shape[-2],
        )


# Transformer model with rotary positional embeddings.


@eqx.filter_jit
def fixed_pos_embedding(inv_freq, seq):
    sinusoid_inp = einsum("i , j -> i j", jnp.arange(seq), inv_freq)
    sinusoid_inp = repeat(sinusoid_inp, "... d -> ... (d r)", r=2)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


@eqx.filter_jit
def rotate_every_two(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x, "... d r -> ... (d r)")


@eqx.filter_jit
def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    return (x * cos) + (rotate_every_two(x) * sin)


class IsingTransformer(eqx.Module):
    """Collective of vector-spin models in sequential order.

    Forward pass feeds data into a stack of spin systems and returns final
    magnetizations in response.

    """

    embedding: jnp.ndarray
    inv_freq: jnp.ndarray = static_field()
    layers: List[List[eqx.Module]]
    final_norm: eqx.Module

    def __init__(self, *, num_tokens, dim, dim_head, depth, heads, key):
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
            print(x)

        x = self.final_norm(x)
        out = x @ self.embedding.transpose()
        return out
