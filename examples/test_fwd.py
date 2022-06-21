import jax
from ising_transformers.ising_transformers import IsingTransformer

key = jax.random.PRNGKey(0)

model = IsingTransformer(num_tokens=10, dim=512, depth=3, heads=1, dim_head=512, key=key)

seq = jax.random.randint(key, (1, 512), 0, 10)
print(seq)

fancymodel = jax.vmap(model)

# print(jax.make_jaxpr(fancymodel)(seq))


logits = fancymodel(seq).block_until_ready()  # (1, 32, 10)

print(logits)

print("ding")
seq = jax.random.randint(key, (1, 512), 0, 10)
print(seq)

logits = fancymodel(seq).block_until_ready()  # (1, 32, 10)
print(logits)
print("dong")
