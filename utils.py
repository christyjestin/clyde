import jax.numpy as jnp

PATCH_CACHE_DIR = 'patch_cache'
KMEANS_DIR = 'kmeans'

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