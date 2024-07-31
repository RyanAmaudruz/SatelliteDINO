
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

from src.benchmark.pretrain_ssl.datasets.SSL4EO.ssl4eo_dataset_dataloader import load_meta_df, LMDBDatasetS2C
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

def get_image_mask(img_t, model, n_cluster):
    # Infer patch embeddings
    pred = model(img_t[None, :, :, :].float().to('cuda'))

    # Drop CLS embedding and reshape
    pred_reshaped = pred[:, 1:, :].reshape(-1, 384)

    # Set up kmeans
    kmeans = KMeans(
        init="random",
        n_clusters=n_cluster,
        n_init=10,
        max_iter=1000,
        random_state=42
    )

    # Predict clusters
    cluster_pred = kmeans.fit_predict(pred_reshaped.to('cpu').detach().numpy())

    # Convert clusters to torch
    cluster_pred_torch = torch.from_numpy(cluster_pred)

    # Reshape clusers into a mask
    new_mask = cluster_pred_torch.reshape(28*2, 28*2)

    # Resize the mask
    resized_mask = F.interpolate(new_mask.float()[None, None, :, :], size=(264, 264), mode='nearest')

    # Return mask
    return resized_mask

def get_coloured_image(img_rgb_bright, cluster_mask):
    # Copy the bright images
    img_rgb_bright_copy = img_rgb_bright.copy()

    # Set the colour map
    colour_map = {
        0: (15,82,186),
        1: (80,200,120),
        2: (128,0,128),
        3: (224,17,95),
        4: (0,0,128),
        5: (145, 149, 246),
        6: (249, 240, 122),
        7: (251, 136, 180),
        8: (244,96,54),
        9: (49,73,94),
        10: (138,145,188),
        11: (137,147,124)
    }
    # Separate each channel of the bright image
    img_rgb_first_channel = img_rgb_bright[0, :, :].copy()
    img_rgb_second_channel = img_rgb_bright[1, :, :].copy()
    img_rgb_third_channel = img_rgb_bright[2, :, :].copy()

    ordered_cluster = pd.Series(cluster_mask.flatten()).value_counts().index

    # Create a coloured mask using the colour map
    for cluster_id in ordered_cluster:
        bool_mask = (cluster_mask == cluster_id).reshape(264, 264)
        rgb_values = colour_map[cluster_id]
        img_rgb_first_channel[bool_mask] = rgb_values[0]
        img_rgb_second_channel[bool_mask] = rgb_values[1]
        img_rgb_third_channel[bool_mask] = rgb_values[2]
    coloured_mask = np.concatenate([
        img_rgb_first_channel.reshape(-1, 264, 264),
        img_rgb_second_channel.reshape(-1, 264, 264),
        img_rgb_third_channel.reshape(-1, 264, 264)
    ], 0)

    # Combine the bright image with the coloured mask
    coloured_image = ((img_rgb_bright_copy + coloured_mask) / 2).astype(int)

    return coloured_image

# Set the number of cluster
n_cluster = 12

# Set the brightness factor to use
brightness_factor = 1.5

# Set the pre and post odin checkpoint paths
pre_odin_weights = '/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/ssl4eo_ssl/ssl_s2c_new_transforms/checkpoint0095.pth'
post_odin_weights1 = '/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/odin/run_2024-03-25_18-26/ckp-epoch=03.ckpt'
post_odin_weights2 = '/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/odin/run_2024-03-26_11-03/ckp-epoch=02.ckpt'


# Load dataset
train_dataset = LMDBDatasetS2C(
    lmdb_file=None,
    s2c_transform=None,
    normalize=False,
    dtype=None,
    mode=['s2c']
)

img_list = [train_dataset.__getitem__(i) for i in (52, 94, 124, 125, 202, 203, 205, 207, 208, 210, 212, 216)]

# 51 52 94 124, 125

# Select image
img = img_list[6]

# Create RGB image
img_rgb = np.flip(img[0, 1:4, :, :], 0)

# Create bright RGB image
img_rgb_bright = (img_rgb * brightness_factor).clip(0, 255).astype(int)

# Show RGB image
plt.imshow(np.transpose(img_rgb_bright, (1, 2, 0)))
plt.show()

# Create transform
val_transform = cvtransforms.Compose([
    cvtransforms.Resize(448*2),
    cvtransforms.ToTensor(),
])

# Apply transform to image
img_t = val_transform(np.transpose(img[0], (1, 2, 0)))

# Load models
pre_odin_model = load_n_prep_model(model_weights=pre_odin_weights)
post_odin_model1 = load_n_prep_model(model_weights=post_odin_weights1)
post_odin_model2 = load_n_prep_model(model_weights=post_odin_weights2)

# Derive the mask
pre_odin_mask = get_image_mask(img_t, pre_odin_model, n_cluster=n_cluster)
post_odin_mask1 = get_image_mask(img_t, post_odin_model1, n_cluster=n_cluster)
post_odin_mask2 = get_image_mask(img_t, post_odin_model2, n_cluster=n_cluster)

# Get the coloured images for each mask
pre_odin_image = get_coloured_image(img_rgb_bright, cluster_mask=pre_odin_mask)
post_odin_image1 = get_coloured_image(img_rgb_bright, cluster_mask=post_odin_mask1)
post_odin_image2 = get_coloured_image(img_rgb_bright, cluster_mask=post_odin_mask2)

# Show the images
plt.imshow(np.transpose(pre_odin_image, (1, 2, 0)))
plt.show()

plt.imshow(np.transpose(post_odin_image1, (1, 2, 0)))
plt.show()

plt.imshow(np.transpose(post_odin_image2, (1, 2, 0)))
plt.show()




