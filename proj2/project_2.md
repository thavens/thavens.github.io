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
# Fun with Filters
## Part 1.1: Convolutions from Scratch!

``` python
def conv2d_4loop(im: np.ndarray, kernel: np.ndarray, padding: int | str = 0):
    assert im.shape[0] >= kernel.shape[0] and im.shape[1] >= kernel.shape[1], "image should be larger than kernel"
    if isinstance(padding, int):
        padding = ((padding, padding), (padding, padding), (0, 0))
    elif padding == 'same':
        padding = ((kernel.shape[0]//2, kernel.shape[0]//2), (kernel.shape[1]//2, kernel.shape[1]//2), (0, 0))
    elif padding == 'valid':
        padding = ((0, 0), (0, 0), (0, 0))
    else:
        assert False, "padding must be int, 'same', or 'valid'"

    # flip the kernel
    kernel = np.copy(kernel[::-1, ::-1])
    if len(im.shape) == 2:
        im = im[..., None]
    padded_im = np.pad(im, padding) # zero pad
    result_height = padded_im.shape[0] - kernel.shape[0] + 1
    result_width = padded_im.shape[1] - kernel.shape[1] + 1
    channels = im.shape[2]
    result = np.zeros((result_height, result_width, channels))
    for y in range(result_height):
        for x in range(result_width):
            sub_im = padded_im[y : y + kernel.shape[0], x : x + kernel.shape[1]]
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    result[y, x] += sub_im[i, j] * kernel[i, j]

    return result.squeeze()

def conv2d(im: np.ndarray, kernel: np.ndarray, padding: int | str = 0):
    assert im.shape[0] >= kernel.shape[0] and im.shape[1] >= kernel.shape[1], "image should be larger than kernel"
    if isinstance(padding, int):
        padding = ((padding, padding), (padding, padding), (0, 0))
    elif padding == 'same':
        padding = ((kernel.shape[0]//2, kernel.shape[0]//2), (kernel.shape[1]//2, kernel.shape[1]//2), (0, 0))
    elif padding == 'valid':
        padding = ((0, 0), (0, 0), (0, 0))
    else:
        assert False, "padding must be int, 'same', or 'valid'"

    # flip the kernel
    kernel = np.copy(kernel[::-1, ::-1])
    if len(im.shape) == 2:
        im = im[..., None]
    padded_im = np.pad(im, padding) # zero pad
    result_height = padded_im.shape[0] - kernel.shape[0] + 1
    result_width = padded_im.shape[1] - kernel.shape[1] + 1
    channels = im.shape[2]
    result = np.zeros((result_height, result_width, channels))
    for y in range(result_height):
        for x in range(result_width):
            sub_im = padded_im[y : y + kernel.shape[0], x : x + kernel.shape[1]]
            result[y, x] = np.sum(np.multiply(sub_im, kernel[..., None]), axis=(0, 1))

    return result.squeeze()
```
Above is the snippet of code relevant to covolutions. The below code is much faster because we enable numpy to use c code for the tight loops and/or vectorize the multiply and add operations. This code has the tradeoff that we are unable to run kernels that are larger than images, but this is not a limitation for this project usecase.
I get exact answers when comparing to scipy convolve 2d at a variety of image, padding, and kernel sizes.
We handle boundaries by precomputing the number of iterations such that we don't step out of the image bounds. In the 2 loop case boundaries are deferred to numpy in slices. We also handle multiple different cases for padding. Int padding lets you pad all sides by a constant, whereas same lets us pad sides such that the resulting image will have the same size of the image. We make use of this same size padding throughout this project.
In terms of the runtime numbers:
For padding 2, kernel size 3x3 and image size 400x400 we have the following runtime:
```
conv2d 4 loops took 2768.729448 milli seconds
conv2d 2 loops took 591.670513 milli seconds
scipy.signal.convolve2d took 3.002167 milli seconds
```
The runtime between 2 loops and 4 loops will increase much more as kernel size increases.

Me using a box filter:
<img src="me_box.png" width="400">
## Part 1.2: Finite Difference Operator
Here are the partial derivatives in y and x (y on left x on right)
<img src="derivative.png">

Here is a sweep of different thresholds to view the edge quality.
<img src="magnitude.png">
It seems the thresholds of 0.2 to 0.3 are the best. Depending on how much noise from the water you can tolerate / the quality of the edges on the pants.

## Part 1.3: Derivative of Gaussian (DoG) Filter
Here is an example of a gaussian blur kernel in 2d.
<img src="blur_kernel.png">

By applying the blur kernel in convolution with the cameraman photo we get:
<img src="blur.png">

Taking the derivative of the blur kernel wrt y and x yields the following.
<img src="deriv_of_blur.png">

Once we applied the DoG Kernel to the camera man we can get a denoised gradient and therefore smoother edges.
Here we apply the DoG kernel and see the gradient values:
<img src="blur_derivative.png">

Following this we can search for thresholds to achieve higher quality edge detection.
<img src="blur_magnitude.png">
thresh=0.2 is the best in my opinion but thresh=0.1 is able to achieve better edges in the background and pants at the cost of noise due to the water.

What differences do you see?
I see that there is a large improvement in the edge detection noise particularly at lower thresholds.

Verify that you get the same result as before.
**Results are qualitatively and quantitatively very similar.**

# Fun with Frequencies
## 2.1 Image "Sharpening"
Sharpening process is taking the image and applying a gaussian filter. By subtracting the filtered content from the photo we have a high frequency representation. Adding the high frequency features back to the photo will make it sharper.

For context, a blur filter will build a sort of weighted average based on the probability distribution and assign that as the pixel value. This means each pixel is partially the value of all the other pixels around it. This is in contrast with a unsharp mask filter where we subtract with gaussian kernel. This essentially means we are removing the values of the pixels around the central pixel. This attempts to remove gaussian noise in the photo and improves perceived quality by adding high frequency signal.

We can simplify the sharpening process into a single convolution called the unsharp mask filter. $$im + \alpha * (im - im * filter)$$ $$(\alpha + 1) * im * impulse - \alpha * im * filter$$ $$im * ((\alpha + 1) * impulse - \alpha * filter)$$ Where the impulse kernel is a 1 in the center padded to the size of the gaussian filter (unit impulse).

Here we blur the Tajmahal. Next can subtract the blurred Tajmahal from the original to grab the high frequency features.
By adding a set scaled high frequency features we can sharpen the image to remove some gaussian noise.
<div class="image-container">
    <img src="blurred_taj.png">
    <img src="high_freq_taj.png">
    <img src="sharpened_taj.png">
</div>


Finally having the sharpened image of the tajmahal, we demonstrate that effects of various levels of sharpening at alphas [1, 2, 4, 8, 16]:
<img src="taj_alpha_sweep.png">

Here we apply our unsharp masking photo to a second image:
Left: Before, Right: After


<div class="image-container">
    <img src="blurred_coffee.png">
    <img src="high_freq_coffee.png">
    <img src="sharpened_coffee.png">
</div>

These coffee shop photos are also produced at alphas [1, 2, 4, 8, 16]:
<img src="coffee_alpha_sweep.png">

## Part 2.2: Hybrid Images
First, we start with the images of Derek and Nutmeg.
<div class="image-container">
    <img src="hybrid_python/DerekPicture.jpg" width=400>
    <img src="hybrid_python/nutmeg.jpg" width=400>
</div>

Next, we generate the photo alignment using the eyes as the keypoints to match the photo.

<div class="image-container">
<img src="hybrid_python/DerekPicture_aligned.jpg" width=400>
<img src="hybrid_python/nutmeg_aligned.jpg", width=400>
</div>

After taking the aligned images and cropping them to reasonable size we apply a sweep over the frequency cutoff space. We opt to average the image as 20% low frequency and 80% high frequency.
We start with sigma high as 2 and sigma low as 4. The y-axis represents multiples of 2 to sigma low and x-axis represents multiples of 2 to sigma high.
<img src="hybrid_images/cat_hybrid_sweep.png">

Here is a breakdown of the features that we are adding into the final hybrid photo. On the top we see the blur of Derek at sigmas of [4, 8, 16, 32]. On the bottom we have the high frequency features retrieved, similar to the tajmahal, but for Nutmeg, made with sigmas of [2, 4, 8, 16].
<img src="hybrid_images/cat_fft_sweep.png">
<div class="image-container">
    <img src="hybrid_images/cat_hybrid.png" width="400">
    <img src="hybrid_images/cat_hybrid.png" width="200">
</div>

We apply this transformation to 2 other photos:
Before images are on the left and after images are on the right.
First we are cropping the photo to remove the black boxes, then running a sigma sweep and choosing a good looking photo.
<div class="image-container">
    <img src="hybrid_python/lion_aligned.jpg" width="200">
    <img src="hybrid_python/goat_aligned.jpg" width="200">
    <img src="hybrid_images/lion_hybrid.png" width="200">
</div>
<div class="image-container">
    <img src="hybrid_python/cat_aligned.jpg" width="200">
    <img src="hybrid_python/elephant_aligned.jpg" width="200">
    <img src="hybrid_images/elephant_hybrid.png" width=200>
</div>

## Part 2.3: Gaussian and Laplacian Stacks
<img src="hybrid_images/apple_stack.png">
<img src="hybrid_images/orange_stack.png">

## Part 2.4: Multiresolution Blending (a.k.a. the oraple!)
<img src="hybrid_images/mask_stack.png">
<img src="hybrid_images/apple_orange_stack.png">
<img src="hybrid_images/apple_orange.png">


### Captain obama
<img src="hybrid_python/capamerica_aligned.jpg" width=300>
<img src="hybrid_python/obama_aligned.jpg" width=300>
<img src="hybrid_images/obama_mask.png">
<img src="hybrid_images/cap_obama.png">

### Plane evolution
<img src="hybrid_python/f35_aligned.jpg" width=300>
<img src="hybrid_python/f16_aligned.jpg" width=300>

<img src="hybrid_images/f16_f35.png">