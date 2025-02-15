# Clyde

Working on some algorithms related to clustering and images using the glorious `jax` library. This started out with me trying to create an algorithm that could turn a photo into a realistic painting. I did a ton of experimenting with agglomerative clustering algorithms for that and eventually gave up. Agglomerative algorithms aren't conducive to parallelization, so they took very long to run. Also my primary idea was to just reduce the number of colors used in an image. This isn't a bad algorithm per se, and it doesn't create many artifacts, but there is definitely more to paintings than just a simplified color palette. You can see my work with agglomerative techniques in this [repo](https://github.com/christyjestin/rajaravi).

Here are a couple more parallelizable algorithms I've come up with so far in this (`clyde`) repo:

### Pixel Kmeans

This runs the K-means algorithm over a single image by treating each pixel as a data point and finding `k` groups of similar pixels. Then each pixel is replaced by the mean of its group. This means that there's only `k` total colors used in the entire image. This works shocking well with relatively low values of `k` and just a single iteration of the algorithm after initializing the `k` means with random pixels from the image. It obviously still has reconstruction errors, but you don't notice the flat patches until you look quite closely.

<h3 align="center">  Original Image: </h3>
<img src="https://github.com/user-attachments/assets/485cbd69-0ab7-45b4-8f5b-574546d49e2c"/>
<h3 align="center"> After One Iteration: </h3>
<img src="https://github.com/user-attachments/assets/f49d038c-06ba-4bc8-bf15-aa87a30838a9"/>

### Patch Kmeans

This is based on the idea that there are common shapes and patterns that are independent of color. To find these patterns, the pipeline:
- breaks an image up into patches, flattens each patch, and separates out the color channels
- then for each vector, it subtracts out the minimum value and rescales the vector to have a norm of 1
- repeats the process for many images in a dataset and treats the resulting vectors as data points to run k-means
  
The intuition is that the minimum value of a patch represents a base color, and the norm represents how stark the pattern is, but the final unit vector is what represents the actual underlying shape. Importantly, the best `k` means can be computed over an arbitrary dataset once and then saved to work quickly and out of the box on any new image of any size. In my case, I used the `nlphuji/flickr30k` dataset available on HuggingFace.

This algorithm is more lossy, but it's also a much more significant compression. I'll be honest: I've had a very hard time running this script over the entire dataset for a full five iterations without it crashing my Linux subsystem at some point. There was one jax related code change that did help out with this, and my laptop is likely part of the problem, but **be warned**.

> You don't have to worry about computing the k-means yourself. Instead, I've saved a `.npy` with means for the setting of `patch_size` equal to 16 and `k` equal to 12000. You can run just the visualizer with the saved means as long as you use those settings.

The reconstructed image has gridline artifacts from where the patches are split up, and while you do still see most of the structure, it doesn't look nature, and there is significant loss. Both the artifacts and the accuracy are improved significantly through composite reconstruction: rather than breaking up the image into patches once, shift the patch gridlines to create multiple reconstructions, then average out these shifted reconstructions. This does reduce the level of compression, and it does introduce a blur typical of average methods.

<h3>  Original Image: </h3>
<img src="https://github.com/user-attachments/assets/3cb44b7f-6997-4f99-9d61-3ac318d2cb86"/>
<h3> Base Reconstruction: </h3>
<img src="https://github.com/user-attachments/assets/0cdf4aca-96d8-4f87-bd16-203b27a7a3ed"/>
<h3> Composite Reconstruction with 4 Shifted Components: </h3>
<img src="https://github.com/user-attachments/assets/abdc183a-5261-41df-8f84-b92676fd8ed2"/>
<h3>  Original Image: </h3>
<img src="https://github.com/user-attachments/assets/03dcdfdf-91e6-48ee-93a4-d6ed734532ba"/>
<h3> Base Reconstruction: </h3>
<img src="https://github.com/user-attachments/assets/01605246-c73f-4b33-b7f9-3b8f71f125e0"/>
<h3> Composite Reconstruction with 4 Shifted Components: </h3>
<img src="https://github.com/user-attachments/assets/662b8730-c614-4cb3-b3be-dae89038667a"/>

