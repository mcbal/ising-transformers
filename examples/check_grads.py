import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.test_util import check_grads

from ising_transformers.ising_transformers import IsingTransformerLayer

# jax.config.update("jax_enable_x64", True)

HIDDEN_DIM = 512
BATCH_SIZE = 1
SEQ_LEN = 64
NUM_RUNS = 3

key = jax.random.PRNGKey(0)
model = IsingTransformerLayer(dim=HIDDEN_DIM, heads=8, dim_head=64, key=key)
# print(jax.tree_map(lambda x: x.shape, model))
print(sum(x.size for x in jax.tree_leaves(model)))
model = vmap(model)

seq_keys = random.split(key, NUM_RUNS)
seq = random.normal(seq_keys[0], (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)) * 0.02


def loss(seq):
    return jnp.mean(model(seq))


check_grads(loss, args=(seq,), order=1, modes=["rev"], eps=1e-4)

# Set a step size for finite differences calculations
# eps = 1e-5
# # Check W_grad with finite differences in a random direction
# key, subkey = random.split(key)
# vec = random.normal(subkey, seq.shape)
# unitvec = vec / jnp.sqrt(jnp.vdot(vec, vec))
# W_grad_numerical = (loss(seq + eps / 2.0 * unitvec) - loss(seq - eps / 2.0 * unitvec)) / eps
# print("W_dirderiv_numerical", W_grad_numerical)
# print("W_dirderiv_autodiff", jnp.vdot(jax.grad(loss)(seq), unitvec))
