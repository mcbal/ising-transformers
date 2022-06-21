from typing import Callable, List, Tuple, Any
import math

import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax import jit, grad, jacrev, vmap
from jax.numpy import einsum

from equinox import Module, static_field
from einops import rearrange, repeat

import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jaxopt as jopt  # https://github.com/google/jaxopt
from jaxopt import AndersonAcceleration
from jaxopt.base import IterativeSolver
from jaxopt.tree_util import tree_add, tree_sub

import numpy as np

from functools import partial

# Ising transformer layer: helper functions.


@eqx.filter_jit
def root_fun(t, h, beta, J):
    return grad(phi)(t, h, beta, J) + t


@eqx.filter_jit
def phi(t, h, beta, J):
    """Compute scalar `phi` given partition function parameters."""
    # t = t.repeat(1, h.shape[-2]) if t.size(-1) == 1 else t
    V = jnp.diag(t) - J
    V_inv = jnp.linalg.solve(V, jnp.eye(V.shape[-1]))
    print("dong")
    return (
        beta * t.sum(axis=-1)
        - 0.5 * jnp.linalg.slogdet(V)[1]
        + beta / 4.0 * einsum("... i f, ... i j, ... j f -> ...", h, V_inv, h)
    )


@eqx.filter_jit
def approximate_free_energy(t, h, beta, J):
    """Compute steepest-descent approximation of free energy for large vector dimension."""
    num_spins = h.shape[-2]
    return -1.0 / beta * (-0.5 * num_spins * (1.0 + jnp.log(2.0 * beta)) + phi(t, h, beta, J))


@eqx.filter_jit
def solve_linear_system_fixed_point(A, v):
    """Solve linear system A(u) = v.

    The solution u* of the system is the fixed point of:
        T(u) = A(u) + u - v
    """

    def fixed_point_fun(u):
        return tree_sub(tree_add(A(u), u), v)

    bwd_solver = AndersonAcceleration(fixed_point_fun=fixed_point_fun, tol=1e-3, maxiter=20)
    return bwd_solver.run(v)[0]


# Ising transformer layer: module class.
solver = AndersonAcceleration(
    fixed_point_fun=root_fun,
    maxiter=40,
    tol=1e-3,
    implicit_diff=True,
    implicit_diff_solve=solve_linear_system_fixed_point,
)


class IsingTransformerLayer(eqx.Module):
    """Differentiable steepest-descent approximation of an Ising vector-spin model.

    Forward pass bathes the spin system in data and returns magnetizations.

    """

    dim: int
    beta: float = eqx.static_field()

    norm: eqx.Module
    to_q: eqx.Module
    to_k: eqx.Module
    # solver: IterativeSolver = static_field()

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
        self.scale = dim_head**-0.5
        self.neg_mask_value = neg_mask_value

    @eqx.filter_jit
    def __call__(self, x, *, pos_emb, causal_mask, solver):
        n = x.shape[-2]

        h = self.norm(x) / jnp.sqrt(self.dim)

        # fused attention and feedforward projections

        q, k = jax.vmap(self.to_q)(h), jax.vmap(self.to_k)(h)

        # split out heads

        q = rearrange(q, "... n (h d) -> ... h n d", h=self.heads)
        k = rearrange(k, "... n (h d) -> ... h n d", h=self.heads)

        # scale

        q *= self.scale

        # apply rotary embeddings

        q, k = map(lambda t: apply_rotary_pos_emb(t, pos_emb), (q, k))

        # contract

        sim = einsum("... i d, ... j d -> ... i j", q, k) * jnp.sqrt(self.dim)

        # causal mask

        sim = jnp.where(causal_mask, sim, self.neg_mask_value)

        # attention

        attn = jax.nn.softmax(sim, axis=-1)

        #  * np.sqrt(dim)) / np.sqrt(num_spins * dim)
        # J = (torch.einsum('b i f, b j f -> b i j', q, k) * np.sqrt(dim)).softmax(-1) / np.sqrt(num_spins*dim)

        # aggregate values

        # out = einsum("... h i j, ... j d -> ... h i d", attn, v)

        # merge heads (block-diagonal couplings in vector dimension)

        J = rearrange(attn, "... h n d -> ... n (h d)") / jnp.sqrt(self.dim * n)

        # print(jnp.linalg.norm(J))

        # Solve for stationary point of exponential appearing in partition function.
        t0 = 0.5 * jnp.ones(*h.shape[:-1], dtype=h.dtype)

        t_star = solver.run(t0, h, self.beta, J).params
        print("BOING", t_star)
        return grad(approximate_free_energy, argnums=1)(t_star, h, self.beta, J)


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
        self.embedding = jrandom.normal(key, (num_tokens, dim)) * 0.02
        self.inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim_head, 2) / dim_head))

        self.layers = [IsingTransformerLayer(dim=dim, dim_head=dim_head, heads=heads, key=key) for _ in range(depth)]
        self.final_norm = eqx.nn.LayerNorm(dim)

    @eqx.filter_jit()
    def __call__(self, x):
        n = x.shape[-1]
        x = self.embedding[x]

        rotary_emb = fixed_pos_embedding(self.inv_freq, n)
        causal_mask = jnp.tril(jnp.ones((n, n)))

        for layer in self.layers:
            x = layer(x, pos_emb=rotary_emb, causal_mask=causal_mask, solver=solver)

        x = self.final_norm(x)
        out = x @ self.embedding.transpose()
        return out
