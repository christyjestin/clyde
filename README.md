# Clyde

Working on some algorithms related to clustering and images using the glorious `jax` library. This started out with me trying to create an algorithm that could turn a photo into a realistic painting. I did a ton of experimenting with agglomerative clustering algorithms for that and eventually gave up. Agglomerative algorithms aren't conducive to parallelization, so they took very long to run. Also my primary idea was to just reduce the number of colors used in an image. This isn't a bad algorithm per se, and it doesn't create many artifacts, but there is definitely more to paintings than just a simplified color palette. You can see my work with agglomerative techniques in this [repo](https://github.com/christyjestin/rajaravi).

Here are a couple more parallelizable algorithms I've come up with so far in this (`clyde`) repo: Pixel Kmeans, Patch Kmeans, Gradients.

### Pixel Kmeans

This runs the K-means algorithm over a single image by treating each pixel as a data point and finding `k` groups of similar pixels. Then each pixel is replaced by the mean of its group. This means that there's only `k` total colors used in the entire image. This works shocking well with relatively low values of `k` and just a single iteration of the algorithm after initializing the `k` means with random pixels from the image. It obviously still has reconstruction errors, but you don't notice the flat patches until you look quite closely.

<h3 align="center">  Original Image: </h3>
<img src="https://github.com/user-attachments/assets/485cbd69-0ab7-45b4-8f5b-574546d49e2c"/>
<h3 align="center"> After One Iteration: </h3>
<img src="https://github.com/user-attachments/assets/f49d038c-06ba-4bc8-bf15-aa87a30838a9"/>

**Yes I know, it looks incredibly close to the original, but you can see the patches if you look closely at the thighs or shorts.**

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

### Gradients
This is the method that looks closest to paintings, but I haven't found a robust, satisfactory configuration. The idea is very simple: choose some number of anchor points from the original image. Then for all other pixels, simply interpolate between nearby anchor points while giving more weight to anchors that are closer in distance. This creates gradients where color gradually changes. The output reminded me of watercolor or Pointillism, and I'm very confident about the underlying method: the problem is how to choose anchor points. There's a couple heuristics that I've explored, and I haven't really found a solid combination:
- You want to gradually add more detail i.e. you should start with fewer anchor points to capture the overall shapes and then add more points as needed to capture finer details.
- If you're considering adding an anchor to represent a piece of an image, the piece shouldn't have too much variance: if it does, you should wait until you're looking at the image at a more granular level, and then add a larger number of anchor points to represent the greater variation (how much variation you should allow is a hyperparameter - I had it scale with the size of the piece, but there is still a coefficient)
- You only add more anchor points if the reconstruction is sufficiently different from the original image (the error threshold is a hyperparameter)

Other tunable parameters include:
- `k`: how many nearby anchor points do you consider when interpolating; this doesn't seem to matter a ton, but I think it's blurrier with more neighbors i.e. higher values of `k`
- the `patch_size` schedule: as you consider finer and finer pieces of the image, how do the sizes of the pieces decrease over time

Another important detail of the algorithm was how I chose where to put the anchors: I only ever made decisions at the patch level, and the patches were always square; when I decided a patch needed an anchor, I chose a random pixel within that patch. In an ideal case, you can consider patches of any shape and size, and your anchor points are at the centroid of the patch: the reason I didn't just go with the center of each square patch is that it probably would've created an artifact (i.e. you probably would be able to see the grid in some way), and the squares aren't proper patches anyway.

The broader problem I ran into while trying to tune this algorithm is that while you can try a bunch of different things, the real problem you're trying to solve is the tradeoff between looking like a painting and not having obvious errors. You can very easily reduce error by just being less selective with where you put anchors and having more total anchor points. The problem is that the reconstructed image quickly becomes indistinguishable from the original image and quickly stops looking like a painting. It is a very difficult balance, and maybe it can't be distilled into an algorithm — maybe it works with a human in the loop, I'm not sure.

Here's an example of the iterative process (note that the considered patches shrink over time):

<h3> 1st Iteration </h3>
<img src="https://github.com/user-attachments/assets/3bd0eb94-a10a-460b-96dd-93ad30bba63d"/>
<h3> 4th Iteration </h3>
<img src="https://github.com/user-attachments/assets/2ed0a0e2-fd68-4812-a77c-c12673f7f410"/>
<h3> 8th Iteration (Goldilocks) </h3>
<img src="https://github.com/user-attachments/assets/c62e5000-57a9-40cb-b256-ab6b5f707c34"/>
<h3> 12th Iteration (Too Close to Original) </h3>
<img src="https://github.com/user-attachments/assets/cc23826f-2728-4503-b4d7-8c49ea05db97"/>
<br/>
Another thing I discovered while trying to figure out how to choose anchors is that computing the standard deviation of small patches is (imo) shockingly good for edge detection:

### Original Image:
![image](https://github.com/user-attachments/assets/3efeeeae-0f60-4ffe-a20b-72162038e0e7)

### Standard Deviation:
<img width="50%" src="https://github.com/user-attachments/assets/0e0fc2b2-c458-4884-ab59-c12d1490fecd7"/>

<br/>
I found this example to be the most impressive because *The Garden of Earthly Delights* is so packed with objects and has a relatively small image size:

### Original Image:
![image](https://github.com/user-attachments/assets/fb32d03f-930f-4e37-99ad-ac7f82717541)

### Standard Deviation:
![image](https://github.com/user-attachments/assets/ee030b4e-c17a-4451-8635-0000afc4690b)

