import time

import jax
from jax import random, vmap
from jax.test_util import check_grads
from jaxopt import AndersonAcceleration

from ising_transformers.ising_transformers import IsingTransformerLayer, root_fun, solve_linear_system_fixed_point


NUM_TOKENS = 10
HIDDEN_DIM = 512
BATCH_SIZE = 3
SEQ_LEN = 64
NUM_RUNS = 3

key = jax.random.PRNGKey(0)
solver = AndersonAcceleration(
    fixed_point_fun=root_fun,
    history_size=2,
    ridge=1e-6,
    maxiter=200,
    beta=0.1,  # strong damping to prevent solution from jumping to other, dangerous local minima
    tol=1e-4,
    implicit_diff=True,
    implicit_diff_solve=solve_linear_system_fixed_point,
)
model = IsingTransformerLayer(dim=HIDDEN_DIM, solver=solver, heads=8, dim_head=64, key=key)
# print(jax.tree_map(lambda x: x.shape, model))
print(sum(x.size for x in jax.tree_leaves(model)))
model = vmap(model)

seq_keys = random.split(key, NUM_RUNS)
seq = random.normal(seq_keys[0], (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)) * 0.02
# print(seq)
t = time.time()


def loss(seq):
    bla = model(seq).sum(axis=-1).sum(axis=-1)
    # print(bla)
    return bla  # (BATCH_SIZE,)


print(f"Including just-in-time compilation {time.time()-t}")


check_grads(loss, args=(seq,), order=1, modes=["rev"], eps=1e-4)
