# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import json
import numpy as np
import skimage as sk
import skimage.io as skio
import time
import argparse
import os


args = argparse.ArgumentParser()
args.add_argument("--input_file")
args = args.parse_args()

imname = args.input_file
output_file = f"outputs/{args.input_file}"

base_name, _ = os.path.splitext(output_file)
output_file = base_name + ".jpg"


def align(a: np.ndarray, b: np.ndarray, shift: int = 15):
    # a: shape [h, w]
    # b: shape [h, w]
    h, w = a.shape
    br = b[h // 8 : -h // 8, w // 8 : -w // 8].ravel()
    nbr = np.linalg.norm(br)

    bs = -1
    bsy = 0
    bsx = 0
    img = 0
    for dy in range(-shift, shift):
        for dx in range(-shift, shift):
            new_a = np.roll(a, (dy, dx), axis=(0, 1))[
                h // 8 : -h // 8, w // 8 : -w // 8
            ]
            ar = new_a.ravel()

            score = br @ ar / (np.linalg.norm(ar) * nbr)
            if score > bs:
                bs = score
                bsy = dy
                bsx = dx
                img = np.roll(a, (dy, dx), axis=(0, 1))
    return {"y": bsy, "x": bsx, "image": img}


def multi_scale_align(a: np.ndarray, b: np.ndarray, scales, shift: int = 15):
    h, w = a.shape

    pw = shift * scales[0]
    padded_a = np.pad(a, (pw, pw), constant_values=0.3)

    cy = 0
    cx = 0
    for scale in scales:
        new_a = padded_a[pw : pw + h, pw : pw + w]
        scaled_a = sk.transform.rescale(new_a, 1 / scale, anti_aliasing=True)
        scaled_b = sk.transform.rescale(b, 1 / scale, anti_aliasing=True)

        alignment = align(scaled_a, scaled_b, shift)
        cy, cx = alignment["y"] * scale, alignment["x"] * scale
        padded_a = np.roll(padded_a, (cy, cx), axis=(0, 1))

    new_a = padded_a[pw : pw + h, pw : pw + w]
    return {
        "y": cy,
        "x": cx,
        "image": new_a,
    }


# read in the image
im = skio.imread(imname)
# convert to double (might want to do this later on to save memory)
im = sk.img_as_float(im)

# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(int)
print(im.shape)
# separate color channels
b = im[:height]
g = im[height : 2 * height]
r = im[2 * height : 3 * height]


def crop(a):
    h, w = a.shape
    return a[h // 20 : -h // 20, w // 20 : -w // 20]


b = crop(b)
g = crop(g)
r = crop(r)

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)


print("shifting g")
start = time.time()
r_align = multi_scale_align(r, g, [8, 4, 2, 1], 8)
end = time.time()
print(f"shift time end {end - start}")
print("shifting r")
b_align = multi_scale_align(b, g, [8, 4, 2, 1], 8)
end2 = time.time()
print(f"shift time end {end2 - end}")
shift_data = {
    "red_shift": {
        "y_shift": r_align["y"],
        "x_shift": r_align["x"],
    },
    "green_shift": {
        "y_shift": 0,
        "x_shift": 0,
    },
    "blue_shift": {"y_shift": b_align["y"], "x_shift": b_align["x"]},
}
print(f"Image {os.path.basename(output_file)}: {json.dumps(shift_data)}")

# create a color image
# print(g_align["image"].shape, r_align["image"].shape, b.shape)
im_out = np.dstack([r_align["image"], g, b_align["image"]])
im_out = (im_out * 255).round().astype(np.uint8)
skio.imsave(output_file, im_out)
