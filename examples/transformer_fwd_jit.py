import time

import jax
from jax import jit, random, vmap

from ising_transformers.ising_transformers import IsingTransformer

NUM_TOKENS = 10
BATCH_SIZE = 3
SEQ_LEN = 32
NUM_RUNS = 3

key = jax.random.PRNGKey(0)
model = IsingTransformer(num_tokens=NUM_TOKENS, dim=512, depth=3, heads=8, dim_head=64, key=key)
# print(jax.tree_map(lambda x: x.shape, model))
print(sum(x.size for x in jax.tree_leaves(model)))
model = vmap(model)

seq_keys = random.split(key, NUM_RUNS)
seq = random.randint(seq_keys[0], (BATCH_SIZE, SEQ_LEN), 0, NUM_TOKENS)
print(seq)
t = time.time()
logits = model(seq).block_until_ready()  # (BATCH_SIZE, SEQ_LEN, NUM_TOKENS)
print(f"Including just-in-time compilation {time.time()-t}")

for seq_key in seq_keys[1:]:
    seq = random.randint(seq_key, (BATCH_SIZE, SEQ_LEN), 0, NUM_TOKENS)
    print(seq)
    t = time.time()
    logits = model(seq).block_until_ready()  # (1, 32, 10)
    print(f"Excluding just-in-time compilation {time.time()-t}")
