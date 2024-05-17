
import matplotlib.pyplot as plt
import sys
import numpy as np
import h5py
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from torchvision import models as torchvision_models
from cvtorchvision import cvtransforms
import torch.nn.functional as F

from src.benchmark.pretrain_ssl.datasets.SSL4EO.ssl4eo_dataset_lmdb_old import load_meta_df, LMDBDatasetS2C
from src.benchmark.pretrain_ssl.models.dino import vision_transformer as vits
from src.benchmark.transfer_classification.models.dino.utils import load_pretrained_weights

class FakeArgs:
    arch = 'vit_small'
    avgpool_patchtokens = False
    bands = 'all'
    batch_size_per_gpu = 64
    checkpoint_key = 'teacher'
    data_path = ''
    dist_url = 'env://'
    epochs = 100
    evaluate = False
    gpu = 0
    is_slurm_job = True
    lr = 0.1
    mode = ['s2c']
    n_last_blocks = 4
    num_workers = 18
    patch_size = 16
    resume = True
    seed = 42
    val_freq = 5
    train_frac = 1.0

def load_n_prep_model(model_weights, patch_size=16, n_channels=13, n_classes=0, arch='vit_small', ckpt_key='teacher'):
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=n_classes, in_chans=n_channels)
    model.cuda()
    model.eval()
    load_pretrained_weights(model, model_weights, ckpt_key, arch, patch_size)
    return model

# def get_splitted_image_mask(img_t, model, n_cluster, image_part_size):
#     # Infer patch embeddings
#     pred = [model(i[None, :, :, :].float().to('cuda')) for i in img_t]
#
#     # Drop CLS embedding and reshape
#     pred_reshaped = torch.concat([p[:, 1:, :] for p in pred]).reshape(-1, 384)
#
#     # Set up kmeans
#     kmeans = KMeans(
#         init="random",
#         n_clusters=n_cluster,
#         n_init=10,
#         max_iter=1000,
#         random_state=42
#     )
#
#     # Predict clusters
#     cluster_pred = kmeans.fit_predict(pred_reshaped.to('cpu').detach().numpy())
#
#     # Convert clusters to torch
#     cluster_pred_torch = torch.from_numpy(cluster_pred)
#
#     # Split the tensor
#     n_split = int(cluster_pred_torch.shape[0] / len(img_t))
#     cluster_pred_torch_splitted = cluster_pred_torch.split(n_split)
#
#     # Reshape clusers into a mask
#     new_mask = [i.reshape(14, 14) for i in cluster_pred_torch_splitted]
#
#     # Resize the mask
#     resized_mask = [
#         F.interpolate(i.float()[None, None, :, :], size=(image_part_size, image_part_size), mode='nearest')
#         for i in new_mask
#     ]
#
#     # Return mask
#     return resized_mask

def get_coloured_image(img_rgb_bright, cluster_mask, image_part_size, n_image_split):
    img_rgb_bright_splitted = [
        j for i in np.split(img_rgb_bright, n_image_split, 1) for j in np.split(i, n_image_split, 2)
    ]

    # Set the colour map
    colour_map = {
        0: (15,82,186),
        1: (80,200,120),
        2: (128,0,128),
        3: (224,17,95),
        4: (0,0,128),
        5: (145, 149, 246),
        6: (249, 240, 122),
        7: (251, 136, 180)
    }

    image_list = []

    for i, m in zip(img_rgb_bright_splitted, cluster_mask):

        # Copy the bright images
        img_rgb_bright_copy = i.copy()

        # Separate each channel of the bright image
        img_rgb_first_channel = i[0, :, :].copy()
        img_rgb_second_channel = i[1, :, :].copy()
        img_rgb_third_channel = i[2, :, :].copy()

        ordered_cluster = pd.Series(m.flatten()).value_counts().index

        # Create a coloured mask using the colour map
        for cluster_id in ordered_cluster:
            bool_mask = (m == cluster_id).reshape(image_part_size, image_part_size)
            rgb_values = colour_map[cluster_id]
            img_rgb_first_channel[bool_mask] = rgb_values[0]
            img_rgb_second_channel[bool_mask] = rgb_values[1]
            img_rgb_third_channel[bool_mask] = rgb_values[2]
        coloured_mask = np.concatenate([
            img_rgb_first_channel.reshape(-1, image_part_size, image_part_size),
            img_rgb_second_channel.reshape(-1, image_part_size, image_part_size),
            img_rgb_third_channel.reshape(-1, image_part_size, image_part_size)
        ], 0)

        # Combine the bright image with the coloured mask
        coloured_image = ((img_rgb_bright_copy + coloured_mask) / 2).astype(int)

        image_list.append(coloured_image)

    return image_list

def concatenate_img_list(img_list, n_image_split):
    new_img_list = []
    for i in range(n_image_split):
        new_img_list.append(np.concatenate(img_list[n_image_split*i : n_image_split*i + n_image_split], 2))
    return np.concatenate(new_img_list, 1)


# Set the n image split
n_image_split = 4

# Set the number of cluster
n_cluster = 5

# Set the brightness factor to use
brightness_factor = 1.5

# Set the pre and post odin checkpoint paths
pre_odin_weights = '/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/ssl4eo_ssl/ssl_s2c_new_transforms/checkpoint0095.pth'
post_odin_weights = '/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/odin/run_2024-03-25_18-26/ckp-epoch=03.ckpt'

# Load dataset
train_dataset = LMDBDatasetS2C(
    lmdb_file=None,
    s2c_transform=None,
    normalize=False,
    dtype=None,
    mode=['s2c']
)

# Select images
# img_list = [train_dataset.__getitem__(i) for i in (52, 94, 124, 125)]

img_list = [train_dataset.__getitem__(i) for i in (202, 203, 205, 207, 208, 210, 212, 216)]

# 51 52 94 124, 125

# Select image
img = img_list[5]

# Create RGB image
img_rgb = np.flip(img[0, 1:4, :, :], 0)

# Create bright RGB image
img_rgb_bright = (img_rgb * brightness_factor).clip(0, 255).astype(int)

# Show RGB image
plt.imshow(np.transpose(img_rgb_bright, (1, 2, 0)))
plt.show()

# Select a single image snapshot
img_to_split = img[0]

# Split the image
img_splitted = [j for i in np.split(img_to_split, n_image_split, 1) for j in np.split(i, n_image_split, 2)]

# Get the image part size
image_part_size = img_splitted[0].shape[-1]

# Create transform
val_transform = cvtransforms.Compose([
    cvtransforms.Resize(224),
    cvtransforms.ToTensor(),
])

# Apply transform to image
img_t = [val_transform(np.transpose(i, (1, 2, 0))) for i in img_splitted]

# Load models
pre_odin_model = load_n_prep_model(model_weights=pre_odin_weights)
post_odin_model = load_n_prep_model(model_weights=post_odin_weights)

new_image = img_t[0]

# Create transform
val_transform = cvtransforms.Compose([
    cvtransforms.Resize(448),
    cvtransforms.ToTensor(),
])

new_image_t = val_transform(np.transpose(new_image.numpy(), (1, 2, 0)))

new_torch_image = new_image_t[None, :, :, :].float().to('cuda')


pred = pre_odin_model(new_torch_image)

# Drop CLS embedding and reshape
pred_reshaped = pred[:, 1:, :].reshape(-1, 384)




# Derive the mask
pre_odin_mask = get_splitted_image_mask(img_t, pre_odin_model, n_cluster=n_cluster, image_part_size=image_part_size)
post_odin_mask = get_splitted_image_mask(img_t, post_odin_model, n_cluster=n_cluster, image_part_size=image_part_size)

# Get the coloured images for each mask
pre_odin_images = get_coloured_image(
    img_rgb_bright, cluster_mask=pre_odin_mask, image_part_size=image_part_size, n_image_split=n_image_split
)
post_odin_images = get_coloured_image(
    img_rgb_bright, cluster_mask=post_odin_mask, image_part_size=image_part_size, n_image_split=n_image_split
)

# Concatenate the images together
pre_odin_image = concatenate_img_list(pre_odin_images, n_image_split)
post_odin_image = concatenate_img_list(post_odin_images, n_image_split)

# Show the images
plt.imshow(np.transpose(pre_odin_image, (1, 2, 0)))
plt.show()

plt.imshow(np.transpose(post_odin_image, (1, 2, 0)))
plt.show()





