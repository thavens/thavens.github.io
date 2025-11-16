# %%
from dataclasses import dataclass
import cv2
import numpy as np
from pathlib import Path
import viser
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
from PIL import Image


# %%
tag_locations = np.array(
    [[90 * (i % 2), 75.67 * (i // 2), 0] for i in range(6)]
)  # shape [6, 3]
corner_locations = np.array(
    [[0, 0, 0], [60, 0, 0], [60, 60, 0], [0, 60, 0]]
)  # Shape [4, 3]
tag_locations = (
    tag_locations[:, None, :] + corner_locations[None, :, :]
)  # shape [6, 4, 3]

# offset such that the middle is the origin
offset = np.max(tag_locations, axis=(0, 1), keepdims=True) / 2
tag_locations = tag_locations - offset

tag_locations /= 1000  # change to meters

print(tag_locations)


# %%
def process_pose(image):
    # Create ArUco dictionary and detector parameters (4x4 tags)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Detect ArUco markers in an image
    # Returns: corners (list of numpy arrays), ids (numpy array)
    # corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
    corners, ids, _ = detector.detectMarkers(image)

    # Check if any markers were detected
    if ids is not None:
        # Process the detected corners
        # corners: list of length N (number of detected tags)
        #   - each element is a numpy array of shape (1, 4, 2) containing the 4 corner coordinates (x, y)
        # ids: numpy array of shape (N, 1) containing the tag IDs for each detected marker
        # Example: if 3 tags detected, corners will be a list of 3 arrays, ids will be shape (3, 1)
        return np.concatenate([corner.squeeze() for corner in corners]), np.concatenate(
            [tag_locations[i] for i in ids.squeeze().tolist()]
        )
    else:
        # No tags detected in this image, skip it
        pass


class Calibrator:
    def load_data(self, path):
        self.fnames = list(Path(path).glob("*.jpg"))
        self.images = []
        for fname in self.fnames:
            image = cv2.imread(fname)[:, :, ::-1]
            self.images.append(image)

    def calibrate(self):
        self.corners_per_img = []
        self.tags_per_img = []
        for image in self.images:
            corners, tags = process_pose(image)
            self.corners_per_img.append(corners.astype(np.float32))
            self.tags_per_img.append(tags.astype(np.float32))

        # tags_per_img = np.array(tags_per_img).astype(np.float32)
        # corners_per_img = np.array(corners_per_img).astype(np.float32)
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(
            self.tags_per_img,
            self.corners_per_img,
            self.images[0].shape[1::-1],
            None,
            None,
        )

    def solve_c2ws(self):
        self.c2ws = []
        for i, (fname, image, corners, tags) in enumerate(
            zip(self.fnames, self.images, self.corners_per_img, self.tags_per_img)
        ):
            H, W, C = image.shape
            success, rvec, tvec = cv2.solvePnP(tags, corners, self.mtx, self.dist)
            rot, _ = cv2.Rodrigues(rvec)
            c2w = np.eye(4)
            c2w[:3, :3] = rot.T
            c2w[:3, 3] = (-rot.T @ tvec).squeeze()
            self.c2ws.append(c2w)

    def visualize(self):
        server = viser.ViserServer(share=True)
        # For each image in your object scan, you'll detect the single ArUco tag and use cv2.solvePnP() to estimate the camera pose.
        for i, (image, c2w) in enumerate(zip(self.images, self.c2ws)):
            H, W, C = image.shape

            server.scene.add_camera_frustum(
                f"/cameras/{i}",  # give it a name
                fov=2 * np.arctan2(H / 2, self.mtx[0, 0]),  # field of view
                aspect=W / H,  # aspect ratio
                scale=0.02,  # scale of the camera frustum change if too small/big
                wxyz=viser.transforms.SO3.from_matrix(
                    c2w[:3, :3]
                ).wxyz,  # orientation in quaternion format
                position=c2w[:3, 3],  # position of the camera
                image=image,  # image to visualize
            )
        return server

    def undistort_images(self):
        h, w = self.images[0].shape[:2]
        # alpha=1 keeps all pixels (more black borders), alpha=0 crops maximally
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), alpha=0, newImgSize=(w, h)
        )
        x, y, w_roi, h_roi = roi
        new_camera_matrix[0, 2] -= x  # cx
        new_camera_matrix[1, 2] -= y

        undistorted_imgs = []
        for image in self.images:
            undistorted_img = cv2.undistort(
                image, self.mtx, self.dist, None, new_camera_matrix
            )

            undistorted_img = undistorted_img[y : y + h_roi, x : x + w_roi]
            undistorted_imgs.append(undistorted_img)
        self.undistorted_imgs = np.array(undistorted_imgs)
        self.new_camera_matrix = new_camera_matrix
        plt.imshow(undistorted_img)

    def save_data(self, file_name):
        images_train, images_val, c2ws_train, c2ws_val = train_test_split(
            self.undistorted_imgs, self.c2ws, test_size=0.01
        )
        np.savez(
            file_name,
            images_train=images_train,  # (N_train, H, W, 3)
            c2ws_train=c2ws_train,  # (N_train, 4, 4)
            images_val=images_val,  # (N_val, H, W, 3)
            c2ws_val=c2ws_val,  # (N_val, 4, 4)
            K=self.new_camera_matrix,
        )


# %%
calibrator = Calibrator()
calibrator.load_data("nagi_images2")
calibrator.calibrate()
calibrator.solve_c2ws()
calibrator.undistort_images()
calibrator.save_data("nagi_dataset.npz")

# %%
calibrator.visualize()
# %%
# fnames = list(Path("calibration-images").glob("*.jpg"))
# images = []
# for fname in fnames:
#     image = cv2.imread(fname)
#     images.append(image)
# corners_per_img = []
# tags_per_img = []
# for fname in fnames:
#     print(fname)
#     image = cv2.imread(fname)
#     images.append(image)
#     corners, tags = process_pose(image)
#     corners_per_img.append(corners)
#     tags_per_img.append(tags)
# tags_per_img = np.array(tags_per_img).astype(np.float32)
# corners_per_img = np.array(corners_per_img).astype(np.float32)
# print(tags_per_img.shape)
# print(corners_per_img.shape)

# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
#     tags_per_img, corners_per_img, images[0].shape[1::-1], None, None
# )

# %%
# server = viser.ViserServer(share=True)
# %%
# For each image in your object scan, you'll detect the single ArUco tag and use cv2.solvePnP() to estimate the camera pose.
# c2ws = []
# for i, (fname, image, corners, tags) in enumerate(
#     zip(fnames, images, corners_per_img, tags_per_img)
# ):
#     H, W, C = image.shape
#     success, rvec, tvec = cv2.solvePnP(tags, corners, mtx, dist)
#     rot, _ = cv2.Rodrigues(rvec)
#     c2w = np.eye(4)
#     c2w[:3, :3] = rot.T
#     c2w[:3, 3] = (-rot.T @ tvec).squeeze()
#     c2ws.append(c2w)

#     server.scene.add_camera_frustum(
#         f"/cameras/{i}",  # give it a name
#         fov=2 * np.arctan2(H / 2, mtx[0, 0]),  # field of view
#         aspect=W / H,  # aspect ratio
#         scale=0.1,  # scale of the camera frustum change if too small/big
#         wxyz=viser.transforms.SO3.from_matrix(
#             c2w[:3, :3]
#         ).wxyz,  # orientation in quaternion format
#         position=c2w[:3, 3],  # position of the camera
#         image=image,  # image to visualize
#     )


# %%
# plt.imshow(images[1])
# plt.figure()
# plt.imshow(cv2.undistort(images[1], mtx, dist))


# %%
def undistort_imgs(images, cmtx, dist, roi):
    new_images = []
    x, y, w_roi, h_roi = roi
    for image in images:
        undistorted_img = cv2.undistort(image, cmtx, dist)
        undistorted_img = undistorted_img[y : y + h_roi, x : x + w_roi]
        new_images.append(undistorted_img)

    return new_images


def undistort_dataset(images, cmtx, dist):
    h, w = images[0].shape[:2]
    # alpha=1 keeps all pixels (more black borders), alpha=0 crops maximally
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        cmtx,
        dist,
        (w, h),
        alpha=0,
    )
    undistored_imgs = undistort_imgs(images, cmtx, dist, roi)
    x, y, w_roi, h_roi = roi
    new_camera_matrix[0, 2] -= x  # cx
    new_camera_matrix[1, 2] -= y  # cy
    return undistored_imgs, new_camera_matrix


# images_train, images_val, c2ws_train, c2ws_val = train_test_split(
#     images, c2ws, test_size=0.1
# )
# np.savez(
#     "my_data.npz",
#     images_train=images_train,  # (N_train, H, W, 3)
#     c2ws_train=c2ws_train,  # (N_train, 4, 4)
#     images_val=images_val,  # (N_val, H, W, 3)
#     c2ws_val=c2ws_val,  # (N_val, 4, 4)
#     c2ws_test=None,  # (N_test, 4, 4)
#     focal=mtx[0][0],  # float
# )


# %%
# training Neural Field
@dataclass
class Embedding:
    L: int = 10

    def __call__(self, x):
        # x shape [bs, 2]
        bs, _ = x.shape
        freqs = 2 ** torch.arange(self.L, device=x.device).view(1, 1, -1)
        c = freqs * torch.pi * x.unsqueeze(-1)  # c shape [bs, 2, L]
        # sin cos order does not matter. We don't need interleave
        sin = torch.sin(c)
        cos = torch.cos(c)
        return torch.concatenate(
            [
                x,
                sin.reshape(bs, -1),
                cos.reshape(bs, -1),
            ],
            dim=-1,
        )


e = Embedding()
temp = e(torch.tensor([[0, 0]]))
temp, temp.shape


# %%
class NeuralFieldNetwork:
    def __init__(self, L=10, width=256):
        self.embedding = Embedding(L)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4 * L + 2, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, 3),
            torch.nn.Sigmoid(),
        )

        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.loss_func = torch.nn.MSELoss()

    @torch.inference_mode()
    def generate_image(self, w, h):
        x = (
            torch.stack(
                torch.meshgrid(
                    torch.linspace(0, 1, w), torch.linspace(0, 1, h), indexing="ij"
                )
            )
            .permute(1, 2, 0)
            .reshape(-1, 2)
        )

        self.model.eval()
        x = self.embedding(x)
        x = self.model(x)
        x = x.reshape(w, h, 3).permute(1, 0, 2)
        return x

    def train(self, dataloader, epochs=10):
        self.model.train()
        self.model = self.model.cuda()
        losses = []
        for epoch in range(epochs):
            for x, y in tqdm.tqdm(dataloader):
                x = x.cuda()
                y = y.cuda()
                self.optim.zero_grad()
                x = self.embedding(x)
                x = self.model(x)
                loss = self.loss_func(x, y)
                loss.backward()
                self.optim.step()
                losses.append(loss.detach())
        self.model = self.model.cpu()
        return [l.cpu().item() for l in losses]


# %%
network = NeuralFieldNetwork()

# dataloader which is just x -> (x, y) / (w, h) coord y -> rgb / 255
fox_image = cv2.imread("fox.jpg")


# %%
class NeuralFieldDataset(Dataset):
    def __init__(self, image):
        self.h, self.w, c = image.shape
        self.image = torch.tensor(image / 255, dtype=torch.float32)
        self.image = self.image.view(-1, 3)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        label = self.image[index]
        x = index % self.w
        y = index // self.w

        return torch.tensor([x / self.w, y / self.h]), label


# %%
neural_field_dataloader = DataLoader(
    NeuralFieldDataset(fox_image), batch_size=10000, shuffle=True
)
losses = network.train(neural_field_dataloader)

# %%
# Model architecture report (number of layers, width, learning rate, and other important details)
# Training progression visualization on both the provided test image and one of your own images
# Final results for 2 choices of max positional encoding frequency and 2 choices of width (2x2 grid)
# PSNR curve for training on one image of your choice


def train_n_save(image: np.ndarray, file_prefix: str, epochs_per: int = 1):
    network = NeuralFieldNetwork()

    neural_field_dataloader = DataLoader(
        NeuralFieldDataset(image), batch_size=10000, shuffle=True
    )

    # save every 2 epoch total 8 epoch
    h, w, _ = image.shape
    losses = []
    for i in range(4):
        image = network.generate_image(w, h)
        plt.imsave(f"{file_prefix}_{i}.jpg", image)
        new_losses = network.train(neural_field_dataloader, epochs_per)
        losses.extend(new_losses)
    image = network.generate_image(w, h)
    plt.imsave(f"{file_prefix}_final.jpg", image)

    # report the log of training
    plt.plot(np.arange(len(losses)), 10 * np.log10(1 / np.array(losses)))
    plt.title("PSNR vs iterations")
    plt.savefig(f"{file_prefix}_psnr.jpg")

    plt.figure()
    plt.plot(np.arange(len(losses)), losses)
    plt.title("MSE loss vs iterations")
    plt.savefig(f"{file_prefix}_mse.jpg")


# %%
fox_image = cv2.imread("fox.jpg")[:, :, ::-1]
train_n_save(fox_image, "final_images/fox")

# %%
nagi_image = cv2.imread("nagi_image.jpg")[:, :, ::-1]
train_n_save(nagi_image, "final_images/nagi", epochs_per=2)


# %%
def hyperparameter_sweep(image: np.ndarray, file_prefix: str, epochs: int = 1):
    neural_field_dataloader = DataLoader(
        NeuralFieldDataset(image), batch_size=10000, shuffle=True
    )

    lines = []
    names = []
    for L in [10, 5]:
        for width in [256, 128]:
            network = NeuralFieldNetwork(L, width)
            losses = network.train(neural_field_dataloader, epochs)
            lines.append(losses)
            names.append(f"L={L},w={width}")

    # report the log of training
    for line, name in zip(lines, names):
        plt.plot(np.arange(len(line)), 10 * np.log10(1 / np.array(line)), label=name)
        plt.title("PSNR vs iterations")
        plt.legend()
    plt.savefig(f"{file_prefix}_psnr.jpg")

    plt.figure()
    for line, name in zip(lines, names):
        plt.plot(np.arange(len(line)), line, label=name)
        plt.title("MSE loss vs iterations")
        plt.legend()
    plt.savefig(f"{file_prefix}_mse.jpg")


# %%
hyperparameter_sweep(fox_image, "final_images/nagi_sweep", epochs=8)

# %%
plt.imshow(nagi_image)
# %%
plt.plot(np.arange(len(losses)), losses)
# %%
plt.plot(np.arange(len(losses)), 10 * np.log10(1 / np.array(losses)))
# %%
losses
# %%
torch.stack(torch.meshgrid(torch.arange(5), torch.arange(3), indexing="ij")).permute(
    1, 2, 0
).view(-1, 2)

# %%
img = network.generate_image(*fox_image.shape[1::-1])
# %%
plt.imshow(img)
# %%
h, w, _ = fox_image.shape
x = torch.tensor([[(i % w) / w, (i // w) / h] for i in range(w * h)])
x[-1]
# %%
network.model.eval()
x = network.embedding(x)
with torch.no_grad():
    x = network.model(x)
x = x.reshape(h, w, 3)
plt.imshow(x)


# %%
def transform(c2w, x_c):
    # transform a point from camera to the world space
    # c2w[:3, :3] = rot.T
    # c2w[:3, 3] = (-rot.T @ tvec).squeeze()
    w2c = torch.linalg.inv(c2w)
    # add 1 to x c
    bs, _ = x_c.shape
    x_c = torch.concatenate([x_c, torch.ones((bs, 1))], dim=-1)
    # x_w = x_c @ w2c
    x_w = torch.einsum("b d, b o d -> b o", x_c, w2c)
    return x_w[:, :3]


def transform2(c2w, x_c):
    # transform a point from camera to the world space
    # c2w[:3, :3] = rot.T
    # c2w[:3, 3] = (-rot.T @ tvec).squeeze()
    # add 1 to x c
    bs, _ = x_c.shape
    x_c = torch.concatenate([x_c, torch.ones((bs, 1))], dim=-1)
    # x_w = x_c @ w2c
    x_w = torch.einsum("b d, b o d -> b o", x_c, c2w)
    return x_w[:, :3]


def test_transform():
    data = np.load("lego_200x200.npz")
    images_train = torch.tensor(data["images_train"] / 255.0, dtype=torch.float)
    # (camera-to-world transformation matrix): [100, 4, 4]
    c2ws_train = torch.tensor(data["c2ws_train"], dtype=torch.float)
    images_val = data["images_val"] / 255.0
    # (camera-to-world transformation matrix): [10, 200, 200, 3]
    c2ws_val = torch.tensor(data["c2ws_val"], dtype=torch.float)
    # (camera-to-world transformation matrix): [60, 4, 4]
    c2ws_test = torch.tensor(data["c2ws_test"], dtype=torch.float)
    focal = data["focal"]  # float
    # assume same camera
    h, w, _ = images_train[0].shape
    K = torch.tensor(
        [
            [focal.item(), 0, w / 2],
            [0, focal.item(), h / 2],
            [0, 0, 1],
        ],
    )
    bs, _, _ = c2ws_val.shape
    x = torch.rand((bs, 3))
    x_hat = transform(torch.linalg.inv(c2ws_val), transform(c2ws_val, x))
    return torch.linalg.norm(x - x_hat)


print("transform error:", test_transform())


def pixel_to_camera(K, uvs, s):
    K_inv = torch.linalg.inv(K)
    bs, _ = uvs.shape
    uv_3d = torch.concat([uvs, torch.ones(bs, 1)], dim=-1)

    return torch.einsum("b i, o i -> b o", uv_3d * s, K_inv)


def pixel_to_ray(K, c2ws, uvs):
    ray_o = c2ws[:, :3, 3]
    X_c = pixel_to_camera(K, uvs, 1)
    X_w = transform2(c2ws, X_c)
    diff = X_w - ray_o
    norm = torch.linalg.norm(diff, dim=-1, keepdims=True)

    dot = torch.einsum("b i, b i -> b", X_w, ray_o)
    dot = dot / torch.linalg.norm(X_w, dim=-1)
    dot = dot / torch.linalg.norm(ray_o, dim=-1)

    ray_d = diff / norm
    return ray_o, ray_d


class RaysData:
    def __init__(self, images, K, c2ws, near=2, far=6):
        self.images = images
        self.num_images, self.h, self.w, _ = images.shape
        self.K = K
        self.c2ws = c2ws
        self.near = near
        self.far = far

        image_uvs = torch.tensor(
            [[(i, j) for j in range(self.h)] for i in range(self.w)]
        ).reshape(-1, 2)
        all_rays_o = []
        all_rays_d = []
        all_pixels = []
        for i in range(self.num_images):
            c2ws = torch.stack([self.c2ws[i]] * len(image_uvs))
            rays_o, rays_d = pixel_to_ray(self.K, c2ws, image_uvs + 0.5)
            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)
            all_pixels.append(self.images[i, image_uvs[:, 1], image_uvs[:, 0]])
        self.rays_o = torch.concat(all_rays_o)
        self.rays_d = torch.concat(all_rays_d)
        self.uvs = torch.concat([image_uvs] * self.num_images)
        self.pixels = torch.concat(all_pixels)

    def __len__(self):
        return len(self.pixels)

    def sample_rays(self, n):
        # pick image
        images_idx = torch.randint(len(self.images), size=[n])
        # pick image loc
        us = torch.randint(self.w, size=[n])
        vs = torch.randint(self.h, size=[n])

        uvs = torch.stack([us, vs], dim=-1)  # shape [bs, 2]
        uvs = uvs + 0.5

        relevant_images = self.images[images_idx]
        relevant_c2ws = self.c2ws[images_idx]

        # image pixes are vs us
        ray_o, ray_d = pixel_to_ray(self.K, relevant_c2ws, uvs)
        return ray_o, ray_d, relevant_images[torch.arange(n), vs, us]

    def sample_along_rays(self, rays_o, rays_d, perturb=False, n_samples=32):
        t_width = (self.far - self.near) / n_samples
        t = torch.linspace(self.near, self.far, n_samples).reshape(
            1, -1, 1
        )  # shape 1, 32, 1
        if perturb:
            t = (
                t + torch.rand((rays_d.shape[0], n_samples)).unsqueeze(-1) * t_width
            )  # shape bs, 32, 1

        samples = rays_o.unsqueeze(1)
        samples = samples + rays_d.unsqueeze(1) * t
        return samples.reshape(-1, 3)


from torch import nn
from torch.nn import functional as F


class NERFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding()
        self.stage1 = nn.ModuleList(
            [
                nn.Linear(63, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            ]
        )
        self.stage2 = nn.ModuleList(
            [
                nn.Linear(63 + 256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
            ]
        )
        self.density = nn.Linear(256, 1)
        self.rgb1 = nn.Linear(256, 256)
        self.rgb2 = nn.Linear(256 + 63, 128)
        self.rgb3 = nn.Linear(128, 3)

    def forward(self, x, r_d):
        emb_input = self.embedding(x)
        r_d = self.embedding(r_d)
        x = emb_input
        for layer in self.stage1:
            x = layer(x)
        x = torch.concat([x, emb_input], dim=-1)
        for layer in self.stage2:
            x = layer(x)
        density = self.density(x)
        density = F.relu(density)
        rgb = self.rgb1(x)
        rgb = torch.concat([rgb, r_d], dim=-1)
        rgb = self.rgb2(rgb)
        rgb = F.relu(rgb)
        rgb = self.rgb3(rgb)
        rgb = F.sigmoid(rgb)
        return density, rgb


def volrend(sigmas, rgbs, step_size):
    # first find T
    prod = sigmas * step_size
    T_component = torch.cumsum(prod, dim=1) - prod
    T = torch.exp(-T_component)  # shape bs, steps, 1
    components = T * (1 - torch.exp(-prod)) * rgbs
    C_hat = torch.sum(components, dim=1)  # shape bs, 3
    return C_hat


import torch

torch.manual_seed(42)
sigmas = torch.rand((10, 64, 1))
rgbs = torch.rand((10, 64, 3))
step_size = (6.0 - 2.0) / 64
rendered_colors = volrend(sigmas, rgbs, step_size)

correct = torch.tensor(
    [
        [0.5006, 0.3728, 0.4728],
        [0.4322, 0.3559, 0.4134],
        [0.4027, 0.4394, 0.4610],
        [0.4514, 0.3829, 0.4196],
        [0.4002, 0.4599, 0.4103],
        [0.4471, 0.4044, 0.4069],
        [0.4285, 0.4072, 0.3777],
        [0.4152, 0.4190, 0.4361],
        [0.4051, 0.3651, 0.3969],
        [0.3253, 0.3587, 0.4215],
    ]
)

assert torch.allclose(rendered_colors, correct, rtol=1e-4, atol=1e-4)


torch.set_float32_matmul_precision("high")


class Trainer:
    def __init__(
        self,
        data_path="lego_200x200.npz",
        batch_size=8192,
        n_samples=32,
        near=2,
        far=6,
    ):
        self.data_path = data_path
        self.model = NERFModel()
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.near = near
        self.far = far

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.load_dataset()

    def load_dataset(self):
        data = np.load(self.data_path, allow_pickle=True)
        print(data.keys())
        images_train = torch.tensor(data["images_train"] / 255.0, dtype=torch.float)
        print("images_train:", images_train.shape)
        # (camera-to-world transformation matrix): [100, 4, 4]
        c2ws_train = torch.tensor(data["c2ws_train"], dtype=torch.float)
        images_val = torch.tensor(data["images_val"] / 255.0, dtype=torch.float)
        # (camera-to-world transformation matrix): [10, 200, 200, 3]
        c2ws_val = torch.tensor(data["c2ws_val"], dtype=torch.float)
        # (camera-to-world transformation matrix): [60, 4, 4]
        if "c2ws_test" in data:
            self.c2ws_test = torch.tensor(data["c2ws_test"], dtype=torch.float)
        if "focal" in data:
            focal = data["focal"]  # float
            # assume same camera
            h, w, _ = images_train[0].shape
            K = torch.tensor(
                [
                    [focal.item(), 0, w / 2],
                    [0, focal.item(), h / 2],
                    [0, 0, 1],
                ],
                dtype=torch.float,
            )
        else:
            K = data["K"]
            K = torch.tensor(K, dtype=torch.float)
        print("c2w matrix:", c2ws_train[0])

        offset_dists = torch.linalg.norm(c2ws_train[:, :3, 3])
        mean_offset = offset_dists.mean(0)
        print("mean offset dist:", mean_offset)

        print("K matrix:", K)
        self.train_dataset = RaysData(images_train, K, c2ws_train, self.near, self.far)
        self.eval_dataset = RaysData(images_val, K, c2ws_val, self.near, self.far)

    def visualize1(self, scale=0.15):
        # --- You Need to Implement These ------
        rays_o, rays_d, pixels = self.train_dataset.sample_rays(
            100
        )  # Should expect (B, 3)
        points = self.train_dataset.sample_along_rays(rays_o, rays_d, perturb=True)
        H, W = self.train_dataset.images.shape[1:3]
        # ---------------------------------------

        server = viser.ViserServer(share=True)
        for i, (image, c2w) in enumerate(
            zip(self.train_dataset.images, self.train_dataset.c2ws)
        ):
            server.add_camera_frustum(
                f"/cameras/{i}",
                fov=2 * np.arctan2(H / 2, np.array(self.train_dataset.K)[0, 0]),
                aspect=W / H,
                scale=scale,
                wxyz=viser.transforms.SO3.from_matrix(np.array(c2w[:3, :3])).wxyz,
                position=np.array(c2w[:3, 3]),
                image=np.array(image),
            )
        for i, (o, d) in enumerate(zip(rays_o, rays_d)):
            server.add_spline_catmull_rom(
                f"/rays/{i}",
                positions=np.stack((o, o + d * self.far)),
            )
        server.add_point_cloud(
            "/samples",
            colors=np.zeros_like(points).reshape(-1, 3),
            points=np.array(points.reshape(-1, 3)),
            point_size=0.02 / 0.15 * scale,
        )

        return server

    def visualize2(self):
        # Visualize Cameras, Rays and Samples

        # --- You Need to Implement These ------
        # This will check that your uvs aren't flipped
        uvs_start = 0
        uvs_end = 40_000
        sample_uvs = self.train_dataset.uvs[
            uvs_start:uvs_end
        ]  # These are integer coordinates of widths / heights (xy not yx) of all the pixels in an image
        # uvs are array of xy coordinates, so we need to index into the 0th image tensor with [0, height, width], so we need to index with uv[:,1] and then uv[:,0]
        assert torch.all(
            self.train_dataset.images[0, sample_uvs[:, 1], sample_uvs[:, 0]]
            == self.train_dataset.pixels[uvs_start:uvs_end]
        )

        # # Uncoment this to display random rays from the first image
        indices = np.random.randint(low=0, high=40_000, size=100)

        # # Uncomment this to display random rays from the top left corner of the image
        # indices_x = np.random.randint(low=100, high=200, size=100)
        # indices_y = np.random.randint(low=0, high=100, size=100)
        # indices = indices_x + (indices_y * 200)

        data = {
            "rays_o": self.train_dataset.rays_o[indices],
            "rays_d": self.train_dataset.rays_d[indices],
        }
        points = self.train_dataset.sample_along_rays(
            data["rays_o"], data["rays_d"], perturb=True
        )
        H, W = self.train_dataset.images.shape[1:3]
        # ---------------------------------------

        server = viser.ViserServer(share=True)
        for i, (image, c2w) in enumerate(
            zip(self.train_dataset.images, self.train_dataset.c2ws)
        ):
            server.add_camera_frustum(
                f"/cameras/{i}",
                fov=2 * np.arctan2(H / 2, np.array(self.train_dataset.K)[0, 0]),
                aspect=W / H,
                scale=0.15,
                wxyz=viser.transforms.SO3.from_matrix(np.array(c2w[:3, :3])).wxyz,
                position=np.array(c2w[:3, 3]),
                image=np.array(image),
            )
        for i, (o, d) in enumerate(zip(data["rays_o"], data["rays_d"])):
            positions = np.stack((o, o + d * 6.0))
            server.add_spline_catmull_rom(
                f"/rays/{i}",
                positions=positions,
            )
        server.add_point_cloud(
            f"/samples",
            colors=np.zeros_like(points).reshape(-1, 3),
            points=np.array(points.reshape(-1, 3)),
            point_size=0.03,
        )

        return server

    @torch.compile()
    def train_1k_step(self, eval: bool = False):
        self.model.train()
        report = []
        evals = []
        for i in tqdm.tqdm(range(1000)):
            self.optim.zero_grad()
            rays_o, rays_d, pixels = self.train_dataset.sample_rays(self.batch_size)
            points = self.train_dataset.sample_along_rays(
                rays_o, rays_d, perturb=True, n_samples=self.n_samples
            )

            rays_d = torch.repeat_interleave(rays_d, self.n_samples, dim=0)

            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                # put through model
                sigmas, rgbs = self.model(
                    points.to(self.device), rays_d.to(self.device)
                )

                sigmas = sigmas.reshape(self.batch_size, self.n_samples, 1)
                rgbs = rgbs.reshape(self.batch_size, self.n_samples, 3)

                step_size = (self.far - self.near) / self.n_samples
                pixels_hat = volrend(sigmas, rgbs, step_size)
                loss = F.mse_loss(pixels_hat, pixels.to(self.device))
            loss.backward()
            self.optim.step()
            report.append(loss.detach())
            if eval and i % 100 == 99:
                evals.append(self.eval())
        if eval:
            return [loss.item() for loss in report], evals
        return [loss.item() for loss in report]

    @torch.inference_mode()
    def eval(self):
        from itertools import batched

        self.model.eval()
        losses = []
        eval_bs = self.batch_size * 4
        for indices in batched(torch.arange(len(self.eval_dataset)), eval_bs):
            real_bs = len(indices)

            rays_o = self.eval_dataset.rays_o[[indices]]
            rays_d = self.eval_dataset.rays_d[[indices]]
            pixels = self.eval_dataset.pixels[[indices]]
            points = self.eval_dataset.sample_along_rays(
                rays_o, rays_d, perturb=False, n_samples=self.n_samples
            )

            rays_d = torch.repeat_interleave(rays_d, self.n_samples, dim=0)

            sigmas, rgbs = self.model(points.to(self.device), rays_d.to(self.device))

            sigmas = sigmas.reshape(real_bs, self.n_samples, 1)
            rgbs = rgbs.reshape(real_bs, self.n_samples, 3)

            step_size = (self.far - self.near) / self.n_samples
            pixels_hat = volrend(sigmas, rgbs, step_size)

            loss = F.mse_loss(pixels_hat, pixels.to(self.device))
            losses.append(loss)

        mse_loss = torch.tensor(losses).mean()
        psnr = 10 * torch.log10(1 / mse_loss)
        return psnr.item()

    @torch.inference_mode()
    def generate_image(self, c2w, h, w):
        # make u and v batch
        # down first
        image_uvs = torch.tensor(
            [[(i, j) for j in range(h)] for i in range(w)]
        ).reshape(-1, 2)
        real_bs = image_uvs.shape[0]

        # get the rays
        rays_o, rays_d = pixel_to_ray(
            self.train_dataset.K, c2w.unsqueeze(0), image_uvs + 0.5
        )

        points = self.train_dataset.sample_along_rays(
            rays_o, rays_d, perturb=False, n_samples=self.n_samples
        )
        rays_d = torch.repeat_interleave(rays_d, self.n_samples, dim=0)

        sigmas, rgbs = self.model(points.to(self.device), rays_d.to(self.device))

        sigmas = sigmas.reshape(real_bs, self.n_samples, 1)
        rgbs = rgbs.reshape(real_bs, self.n_samples, 3)

        step_size = (self.far - self.near) / self.n_samples
        pixels_hat = volrend(sigmas, rgbs, step_size)

        # shape back to image
        pred_image = pixels_hat.reshape(w, h, 3).permute(1, 0, 2)

        return pred_image.cpu().numpy()

    def generate_gif(self, image_index, output_file):
        def look_at_origin(pos):
            # Camera looks towards the origin
            forward = -pos / np.linalg.norm(pos)  # Normalize the direction vector

            # Define up vector (assuming y-up)
            up = np.array([0, 0, 1])

            # Compute right vector using cross product
            right = np.cross(up, forward)
            right = right / np.linalg.norm(right)

            # Recompute up vector to ensure orthogonality
            up = np.cross(forward, right)

            # Create the camera-to-world matrix
            c2w = np.eye(4)
            c2w[:3, 0] = right
            c2w[:3, 1] = up
            c2w[:3, 2] = forward
            c2w[:3, 3] = pos

            return c2w

        def rot_x(phi):
            return np.array(
                [
                    [np.cos(phi), -np.sin(phi), 0, 0],
                    [np.sin(phi), np.cos(phi), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )

        # TODO: Change start position to a good position for your scene such as
        # the translation vector of one of your training camera extrinsics
        START_POS = self.train_dataset.c2ws[image_index][:3, 3].numpy()
        NUM_SAMPLES = 60

        frames: np.ndarray = []
        for phi in tqdm.tqdm(np.linspace(360.0, 0.0, NUM_SAMPLES, endpoint=False)):
            c2w = look_at_origin(START_POS)
            extrinsic = rot_x(phi / 180.0 * np.pi) @ c2w

            # Generate view for this camera pose
            # TODO: Add code for generating a view with your model from the current extrinsic
            frame = self.generate_image(
                torch.tensor(extrinsic, dtype=torch.float),
                self.train_dataset.h,
                self.train_dataset.w,
            )
            frames.append(frame)

        frames = [(frame * 255).astype(np.uint8) for frame in frames]
        imgs = [Image.fromarray(img) for img in frames]
        # duration is the number of milliseconds between frames; this is 40 frames per second
        imgs[0].save(
            output_file, save_all=True, append_images=imgs[1:], duration=50, loop=0
        )

    def generate_gif_from_test(self, output_file):
        # TODO: Change start position to a good position for your scene such as
        # the translation vector of one of your training camera extrinsics
        frames: np.ndarray = []
        for c2w in self.c2ws_test:
            # Generate view for this camera pose
            # TODO: Add code for generating a view with your model from the current extrinsic
            frame = self.generate_image(
                c2w,
                self.train_dataset.h,
                self.train_dataset.w,
            )
            frames.append(frame)

        frames = [(frame * 255).astype(np.uint8) for frame in frames]
        imgs = [Image.fromarray(img) for img in frames]
        # duration is the number of milliseconds between frames; this is 40 frames per second
        imgs[0].save(
            output_file, save_all=True, append_images=imgs[1:], duration=50, loop=0
        )

    def train_save_nerf(self, file_prefix: str, image_idx: int, from_test: bool):
        gen_c2w = self.train_dataset.c2ws[image_idx]
        losses = []
        evals = []
        for i in range(4):
            image = self.generate_image(
                gen_c2w, self.train_dataset.h, self.train_dataset.w
            )
            plt.imsave(f"{file_prefix}_{i}.jpg", image)
            new_losses, new_evals = self.train_1k_step(True)
            losses.extend(new_losses)
            evals.extend(new_evals)
        image = self.generate_image(gen_c2w, self.train_dataset.h, self.train_dataset.w)
        plt.imsave(f"{file_prefix}_final.jpg", image)

        # report the log of training
        plt.plot(np.arange(len(losses)), 10 * np.log10(1 / np.array(losses)))
        plt.title("Training PSNR vs iterations")
        plt.savefig(f"{file_prefix}_psnr.jpg")

        plt.figure()
        plt.plot(np.arange(len(losses)), losses)
        plt.title("Training MSE loss vs iterations")
        plt.savefig(f"{file_prefix}_mse.jpg")

        plt.figure()
        plt.plot(np.arange(len(evals)), evals)
        plt.title("Eval PSNR vs iterations")
        plt.savefig(f"{file_prefix}_eval_psnr.jpg")

        if from_test:
            self.generate_gif_from_test(f"{file_prefix}_gif.gif")
        else:
            self.generate_gif(image_idx, f"{file_prefix}_gif.gif")


# %%
trainer = Trainer(n_samples=64)

# %%
trainer.train_save_nerf("final_images/lego", 2, True)
# %%
# plt.imshow(trainer.train_dataset.images[16])
# %%
losses = trainer.train_1k_step()
# %%
trainer.eval()
# %%
plt.plot(range(len(losses)), losses)
# %%
plt.plot(range(len(losses)), 10 * np.log10(1 / np.array(losses)))
# %%
image = trainer.generate_image(
    trainer.train_dataset.c2ws[1], trainer.train_dataset.h, trainer.train_dataset.w
)
# %%
plt.imshow(image)
# %%
trainer.generate_gif(1, "bulldozer.gif")
# %%
trainer = Trainer(data_path="lafufu_dataset.npz", n_samples=64, near=0.02, far=0.5)
# losses = trainer.train_1k_step()

# %%
image = trainer.generate_image(
    trainer.eval_dataset.c2ws[0], trainer.eval_dataset.h, trainer.eval_dataset.w
)
# %%
plt.imshow(image.cpu().numpy())
# %%
# %%
plt.imshow(trainer.eval_dataset.images[0])
# %%
trainer.generate_gif(0, "lafufu4.gif")

# %%
trainer = Trainer(data_path="nagi_dataset.npz", n_samples=64, near=0.3, far=1)
# %%
trainer.visualize1(0.02)
# %%
losses = trainer.train_1k_step()
# %%
for i in range(2):
    losses += trainer.train_1k_step()

# %%
trainer.visualize2()
# %%
trainer.train_save_nerf("final_images/nagi_nerf", image_idx=16, from_test=False)

# %%
