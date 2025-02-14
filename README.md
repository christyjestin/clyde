# Clyde

Working on some algorithms related to clustering and images using the glorious `jax` library

### Pixel Kmeans

This runs the K-means algorithm over a single image by treating each pixel as a data point and finding `k` groups of similar pixels. Then each pixel is replaced by the mean of its group. This works shocking well with relatively low values of `k` and just a single iteration of the algorithm after initializing the `k` means by selecting random pixels. It obviously still has reconstruction errors, but the images look quite similar.

### Patch Kmeans

This is based on the idea that there are common shapes and patterns that are independent of color i.e. you could create the same shape with black and blue or white and red, and there's some underlying structure that is common to the two cases. To find these patterns, the pipeline breaks an image up into patches, then takes the minimum value and rescales the vector to have a norm of 1. The intuition is that the minimum value of a patch represents a kind off base color, and the norm represents how stark the pattern is, but the final unit vector is what represents the actual underlying shape.

The algorithm then tries to find `k` means to represent these unit vectors after splitting up the color channels and flattening. This algorithm is more lossy, but it's also a much more significant compression.

I'll be honest: I've had a very hard time running this script over the entire dataset for a full five iterations without it crashing my Linux subsystem at some point. There was one jax related code change that did help out with this, and my laptop is likely part of the problem, but **be warned**.

###
