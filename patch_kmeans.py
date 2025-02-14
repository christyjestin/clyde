import os, glob, argparse
from pathlib import Path

from math import ceil
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp
import jax
from datasets import load_dataset

from utils import *

CLEAR_CACHE_EVERY = 50
SAVE_EVERY = 100

def get_latest_checkpoint(patch_size, k):
    checkpoint_dir = f'{KMEANS_DIR}/{patch_size}/{k}'
    latest = -1
    for file in glob.glob(f'{checkpoint_dir}/*.npy'):
        val = int(Path(file).stem)
        if val > latest:
            latest = val
    return latest


def patch_kmeans(ds, patch_size, k, batch_size, num_iters):
    '''
    We represent a patch (i.e. a square section of the image) by flattening the patch 
    into a 1-D array and storing a min val, a norm, and a mean index `i`. The patch is 
    reconstructed as `patch_hat = min + norm * means[i]` where the means are unit vectors.
    '''
    os.makedirs(f'{KMEANS_DIR}/{patch_size}/{k}', exist_ok=True)
    size_squared = patch_size * patch_size

    latest = get_latest_checkpoint(patch_size, k)
    # load checkpoint if possible; otherwise just randomly init
    if latest != -1:
        means = jnp.load(f'{KMEANS_DIR}/{patch_size}/{k}/{latest}.npy')
    else:
        means = means_random_init(k, size_squared)
    start = latest + 1
    for iter in range(start, start + num_iters):
        ds = ds.shuffle()
        # running tallies (used to compute new means)
        new_means = jnp.zeros_like(means)
        counts = jnp.zeros((k, 1), dtype=jnp.int32)

        num_batches = ceil(ds.num_rows / batch_size)
        for i, batch in tqdm(enumerate(ds.iter(batch_size=batch_size)), total=num_batches, 
                             desc=f"Iteration {iter} of k-means:"):
            # pack batch of patches into single array
            patches = jnp.concat([patchify(jnp.array(img), patch_size) for img in batch['image']])
            # min color over patch
            min_val = jnp.min(patches, axis=1, keepdims=True) # p x 1 x 3
            patches = (patches - min_val).astype(jnp.float32) # p x s x 3
            # treat each color channel as its own vector
            patches = jnp.moveaxis(patches, 2, 1).reshape(-1, size_squared) # 3p x s
            # ignore zero vectors (i.e. flat patches)
            # N.B. these patches will fuck up later calcs if they are left in
            patches = patches[jnp.linalg.norm(patches, axis=1).nonzero()]
            # normalize the patches to be unit vectors
            patches = patches / jnp.linalg.norm(patches, axis=1, keepdims=True)

            # find best fits
            # they're both unit vectors, so cosine similarity is just the dot product
            cos_sims = patches @ means.T
            clusters = jnp.argmax(cos_sims, axis=1)

            # update new means
            # counts_update := number of new patches per cluster
            counts_update = jnp.zeros_like(counts).at[clusters].add(1)
            counts = counts + counts_update
            # N.B. nan_to_num ensures 0 / 0 becomes 0 instead of nan
            alpha = jnp.nan_to_num(counts_update / counts)
            # average over current batch
            # N.B. we do the division step before the summation step to avoid overflow
            means_update = jnp.zeros_like(means).at[clusters].add(patches / counts_update[clusters])
            # take a weighted average of this batch and the running tally
            new_means = alpha * means_update + (1 - alpha) * new_means

            # clear cache to avoid JIT out of memory errors
            if (i + 1) % CLEAR_CACHE_EVERY == 0:
                jax.clear_caches()

            # save progress and reset counters
            if (i + 1) % SAVE_EVERY == 0:
                random_vals = means_random_init(k, size_squared)
                # if the norm is 0, then this cluster was never the closest fit,
                # so we replace it with a random vector
                missed_clusters = jnp.linalg.norm(new_means, axis=1, keepdims=True) == 0
                new_means = jnp.where(missed_clusters, random_vals, new_means)
                # means should be unit vectors
                means = new_means / jnp.linalg.norm(new_means, axis=1, keepdims=True)
                jnp.save(f'{KMEANS_DIR}/{patch_size}/{k}/{iter}.npy', means)
                # reset data structures
                new_means = jnp.zeros_like(means)
                counts = jnp.zeros((k, 1), dtype=jnp.int32)
        # clear cache to avoid build up betweeen iterations
        jax.clear_caches()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--patch_size', type=int, default=20)
    parser.add_argument('--k', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_iters', type=int, default=5)
    args = parser.parse_args()

    ds = load_dataset('nlphuji/flickr30k', split='test').select_columns('image')
    patch_kmeans(ds, args.patch_size, args.k, args.batch_size, args.num_iters)