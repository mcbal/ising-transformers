import time

import equinox
import jax
from jax import jit, random, vmap

from ising_transformers.ising_transformers import IsingTransformer

NUM_TOKENS = 10
BATCH_SIZE = 3
SEQ_LEN = 32
NUM_RUNS = 10

key = jax.random.PRNGKey(0)
model = IsingTransformer(num_tokens=NUM_TOKENS, dim=64, depth=3, heads=8, dim_head=8, key=key)
# print(jax.tree_map(lambda x: x.shape, model))
print(sum(x.size for x in jax.tree_leaves(model)))
# model = vmap(model)

seq_keys = random.split(key, NUM_RUNS)
seq = random.randint(
    seq_keys[0],
    (
        3,
        SEQ_LEN,
    ),
    0,
    NUM_TOKENS,
)
print(seq)
t = time.time()
logits = vmap(model)(seq).block_until_ready()  # (BATCH_SIZE, SEQ_LEN, NUM_TOKENS)
print(logits)
print(f"Including just-in-time compilation {time.time()-t}")

print(type(model))
breakpoint()


@equinox.filter_jit
def batch_loss(model, seq):
    loss_fn = jax.vmap(model)
    return jax.numpy.mean(loss_fn(seq))
    # return jax.numpy.mean((out - jax.numpy.ones_like(out)) ** 2)


t = time.time()
value, grads = equinox.filter_value_and_grad(batch_loss)(model, seq)
print(value)
print(jax.numpy.max(grads.layers[0].to_q.weight))
print(f"Including just-in-time compilation {time.time()-t}")
breakpoint()


for seq_key in seq_keys[1:]:
    seq = random.randint(seq_key, (BATCH_SIZE, SEQ_LEN), 0, NUM_TOKENS)
    print(seq)
    t = time.time()
    value, grads = equinox.filter_value_and_grad(batch_loss)(model, seq)  # (1, 32, 10)
    print(f"Excluding just-in-time compilation {time.time()-t}")
