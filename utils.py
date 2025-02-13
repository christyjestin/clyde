import numpy as np
import jax.numpy as jnp

PATCH_CACHE_DIR = 'patch_cache'
KMEANS_DIR = 'kmeans'

COLOR_MAX = 255

# the grid rearrangement looks like this
# 1 2 3
# 4 5 6 -> 1 4 7 2 5 8 3 6 9
# 7 8 9
def patchify(arr: jnp.array, patch_size: int):
    h, w, _ = arr.shape
    ph, pw = h // patch_size, w // patch_size
    # trim excess and create new patch axis for split rows
    # arr now has the shape ph x p x (pw * p) x 3
    arr = jnp.stack(jnp.split(arr[:ph * patch_size, :pw * patch_size], ph, axis=0))
    # split into cols and concatenate the cols end to end
    arr = jnp.concat(jnp.split(arr, pw, axis=2)) # (ph * pw) x p x p x 3
    # flatten each patch while keeping the color channels
    return arr.reshape((ph * pw, patch_size * patch_size, 3))

# draw `k` all positive vectors from the `size-squared`-dimensional unit sphere; this method is
# from https://en.wikipedia.org/wiki/N-sphere#Uniformly_at_random_on_the_(n_%E2%88%92_1)-sphere
def means_random_init(k, size_squared):
    vecs = jnp.abs(jnp.array(np.random.normal(size=(k, size_squared)), dtype=jnp.float32))
    return vecs / jnp.linalg.norm(vecs, axis=1, keepdims=True)