
<!-- MathJax -->
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
      inlineMath: [['\\(','\\)'], ['$', '$']]
    }
  });
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<style>
.image-container {
  display: flex; 
  align-items: center; 
  gap: 20px; 
}

.image-container img {
  max-width: 50%;
  height: auto;
}
</style>

# Project 4
## Part 0
Here are the two screenshots of the dataset I captured. I've flipped the axis in the visor tool so the dome is right side up.

<img src="final_images/frustum1.png" alt="" width="300"/>
<img src="final_images/frustum2.png" alt="" width="300"/>

## Part 1
In this part we take a neural network model with width 256, 4 layers total at learning rate 1e-2, and train it to be able to reconstruct an image in 2d. Essentially the model is learning the rgb values of the fox given a positional embedding which is a series of sin and cosine waves that make it easier for the model to encode location. I've discovered that using higher width and layers can cause more instability, and the default settings already have great reproduction qualities.

We use 10 different frequencies along with the base location input (0, 1). This comes to an input dim of 42.

Here are the follow training runs to recreate the fox along with a run for learning on the images I've caputured for my dataset.

<div class="image-container">
  <img src="final_images/fox_0.jpg" width="300"/>
  <img src="final_images/fox_1.jpg" width="300"/>
  <img src="final_images/fox_2.jpg" width="300"/>
  <img src="final_images/fox_3.jpg" width="300"/>
  <img src="final_images/fox_final.jpg" width="300"/>
</div>

<div class="image-container">
  <img src="final_images/fox_mse.jpg" width="300"/>
  <img src="final_images/fox_psnr.jpg" width="300"/>
</div>

Here is the training run for the image in my dataset. This character in the image is Nagi from the anime Blue Lock. I find it hard to reproduce the wood grain and the textures on the face. I think these are high frequency which are hard to reproduce with the given model.

<div class="image-container">
  <img src="final_images/nagi_0.jpg" width="300"/>
  <img src="final_images/nagi_1.jpg" width="300"/>
  <img src="final_images/nagi_2.jpg" width="300"/>
  <img src="final_images/nagi_3.jpg" width="300"/>
  <img src="final_images/nagi_final.jpg" width="300"/>
</div>


<div class="image-container">
  <img src="final_images/nagi_mse.jpg" width="300"/>
  <img src="final_images/nagi_psnr.jpg" width="300"/>
</div>

Here I run a hyperparameter sweep over the fox reproduction training run. I wanted to try some smaller embeddings and smaller width. We can see that model width doesn't have too much difference on reroduction quality, which positional embedding causes the model to platou early likely because the low spacial resolution is making it hard to learn the task. I tested embedding size 10, 20 and widths 128, 256. You can see the performances overlayed on the graph below matching by color.

<div class="image-container">
  <img src="final_images/nagi_sweep_mse.jpg" width="300"/>
  <img src="final_images/nagi_sweep_psnr.jpg" width="300"/>
</div>

# Part 2
To fit the Neural Radiance Field, I first implemented the ray generation pipeline. This involves a `pixel_to_ray` function that calculates a ray's origin from the camera's translation (via the `c2w` matrix) and its direction by un-projecting a pixel coordinate to a 3D point in camera space (using the inverse intrinsic matrix `K`) and then transforming that point to world space. The 'RaysData' class handles dataloading: `sample_rays` randomly selects pixels from all images, adds a 0.5 offset, and calls `pixel_to_ray` to get a batch of rays and their ground-truth colors. The `sample_along_rays` method then discretizes these rays into 3D points, adding random jitter to the sample depths during training.

These 3D points and their ray directions are fed into the `NERFModel`, an MLP that uses positional encoding and a skip connection. The model outputs a volume density (via a ReLU) and a view-dependent color (via a Sigmoid). Finally, the `volrend` function implements the discrete volume rendering equation. It calculates each sample's transmittance and alpha contribution, then computes a weighted sum of the samples' colors along the ray to produce the final estimated pixel color, which is compared against the ground-truth for the MSE loss.

This is the view of the rays that fed into the NERF model for training

<img src="final_images/nagi_cloud.png" width="400"/>
<img src="final_images/lego_vis_2_0.png" width="400"/>
<img src="final_images/lego_vis_2_1.png" width="400"/>

Here is the final result of running my training loop against the provided lego bulldozer dataset.

<img src="final_images/lego_gif.gif" width="300"/>


<div class="image-container">
  <img src="final_images/lego_0.jpg" width="300"/>
  <img src="final_images/lego_1.jpg" width="300"/>
  <img src="final_images/lego_2.jpg" width="300"/>
  <img src="final_images/lego_3.jpg" width="300"/>
  <img src="final_images/lego_final.jpg" width="300"/>
</div>

<div class="image-container">
  <img src="final_images/lego_psnr.jpg" width="300"/>
  <img src="final_images/lego_mse.jpg" width="300"/>
  <img src="final_images/lego_eval_psnr.jpg" width="300"/>
</div>

# Part 2.6
Here I train NERF model on the set of Nagi photos that I've captured. Due to lighting and shadows, there seem to be dark clouds above his head. I think this is due to the face that the model learns high density black pixels at higher z values in the space. At the same time this gif has a pretty low angle, causing rays to encounter these high density black pixels.

### Hyperparameter Changes
I use 64 samples per ray.
To search for good hyperparameters for near and far sampling though we calculate the mean offset distance from origin to camera. I then scale the lafufu dataset training settings of near=0.02 and far=0.5 by ~2, since the mean offset distance ratio between our data and the lafufu dataset is ~2. This produces the point cloud that was provided above and good training convergence. There were no other changes made to the default model architecture and learning rate provided in the directions.

### Code
I use a great class based architecture and work primarily with pytorch. Opting to convert everything immediately into pytorch float tensors. Using classes helps me share code between lego dataset, lafufu dataset, and my dataset. I train with AMP and pytorch compile which help in significantly reducing training time by >10x. I use 1x A5000 ada. I am able to run 4k iterations in 5 minutes under these configs.

<img src="final_images/nagi_nerf_gif.gif" width="300"/>

<div class="image-container">
  <img src="final_images/nagi_nerf_0.jpg" width="300"/>
  <img src="final_images/nagi_nerf_1.jpg" width="300"/>
  <img src="final_images/nagi_nerf_2.jpg" width="300"/>
  <img src="final_images/nagi_nerf_3.jpg" width="300"/>
  <img src="final_images/nagi_nerf_final.jpg" width="300"/>
</div>

<div class="image-container">
  <img src="final_images/nagi_nerf_psnr.jpg" width="300"/>
  <img src="final_images/nagi_nerf_mse.jpg" width="300"/>
</div>
