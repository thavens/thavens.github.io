from pathlib import Path
import matplotlib.pyplot as plt
from align_image_code import align_images
import argparse
import numpy as np
from skimage.transform import rescale

# get image files for 2 images
parser = argparse.ArgumentParser()
parser.add_argument("--im1", type=Path, default="nutmeg.jpg")
parser.add_argument("--im2", type=Path, default="DerekPicture.jpg")
args = parser.parse_args()

im1 = plt.imread(args.im1) / 255.0
im2 = plt.imread(args.im2) / 255.0

# # rescale 0.1
# im1 = rescale(im1, 0.1, anti_aliasing=True, channel_axis=2)
# im2 = rescale(im2, 0.1, anti_aliasing=True, channel_axis=2)

# # pad both images to the same size.
# h1, w1, _ = im1.shape
# h2, w2, _ = im2.shape
# H = max(h1, h2)
# W = max(w1, w2)
# im1 = np.pad(im1, [(0, H - h1), (0, W - w1), (0, 0)], "constant")
# im2 = np.pad(im2, [(0, H - h2), (0, W - w2), (0, 0)], "constant")
# print(f"Image 1 shape: {im1.shape}")
# print(f"Image 2 shape: {im2.shape}")

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

# save aligned images
output_file1 = args.im1.stem + "_aligned.jpg"
output_file2 = args.im2.stem + "_aligned.jpg"
plt.imsave(output_file1, im1_aligned)
plt.imsave(output_file2, im2_aligned)
