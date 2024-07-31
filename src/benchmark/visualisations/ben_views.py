
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
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.benchmark.pretrain_ssl.datasets.SSL4EO.ssl4eo_dataset_dataloader import load_meta_df, LMDBDatasetS2C
from src.benchmark.pretrain_ssl.models.dino import vision_transformer as vits
from src.benchmark.transfer_classification.datasets.BigEarthNet.bigearthnet_dataset_seco_lmdb_s2_uint8 import \
    LMDBDatasetRA
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

# def get_image_mask(img_t, model, n_cluster):
#     # Infer patch embeddings
#     pred = model(img_t[None, :, :, :].float().to('cuda'))
#
#     # Drop CLS embedding and reshape
#     pred_reshaped = pred[:, 1:, :].reshape(-1, 384)
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
#     # Reshape clusers into a mask
#     new_mask = cluster_pred_torch.reshape(28*2, 28*2)
#
#     # Resize the mask
#     resized_mask = F.interpolate(new_mask.float()[None, None, :, :], size=(264, 264), mode='nearest')
#
#     # Return mask
#     return resized_mask

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

# # Set the number of cluster
# n_cluster = 12
#
# # Set the brightness factor to use
# brightness_factor = 1.5
#
# # Set the pre and post odin checkpoint paths
# pre_odin_weights = '/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/ssl4eo_ssl/ssl_s2c_new_transforms/checkpoint0095.pth'
# post_odin_weights1 = '/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/odin/run_2024-03-25_18-26/ckp-epoch=03.ckpt'
# post_odin_weights2 = '/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/odin/run_2024-03-26_11-03/ckp-epoch=02.ckpt'

dataset_train = LMDBDatasetRA(
    set_type='train',
    transform=None,
    is_slurm_job=None
)

dataset_train.__getitem__(0)

# img_index = np.random.choice(range(len(train_dataset)), size=10, replace=False)
# img_list = [train_dataset.__getitem__(i) for i in img_index]

img_list_temp = [dataset_train.__getitem__(i) for i in range(200)]

img_list = []
store_set = set()

for i, l in img_list_temp:
    string_l = str(l)
    if string_l not in store_set:
        store_set.add(string_l)
        img_list.append((i, l))



img_list = [img_list_0 for i, img_list_0 in enumerate(img_list) if i not in (1, 6, 7, 9, 10, 12)]

label_list = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Agricultural land with nat. vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters'
]

band_indices = {
    'B1': 0,   # Ultra Blue (Coastal and Aerosol) - 443 nm
    'B2': 1,   # Blue - 490 nm
    'B3': 2,   # Green - 560 nm
    'B4': 3,   # Red - 665 nm
    'B5': 4,   # Visible and Near Infrared (VNIR) - 705 nm
    'B6': 5,   # Visible and Near Infrared (VNIR) - 740 nm
    'B7': 6,   # Visible and Near Infrared (VNIR) - 783 nm
    'B8': 7,   # Visible and Near Infrared (VNIR) - 842 nm
    'B8a': 8,  # Visible and Near Infrared (VNIR) - 865 nm
    'B10': 9, # Short Wave Infrared (SWIR) - 1375 nm
    'B11': 10, # Short Wave Infrared (SWIR) - 1610 nm
    'B12': 11  # Short Wave Infrared (SWIR) - 2190 nm
}

# Channel combinations
combinations = [
    ('Natural Color', [band_indices['B4'], band_indices['B3'], band_indices['B2']]),
    ('Color Infrared', [band_indices['B8'], band_indices['B4'], band_indices['B3']]),
    ('Short-Wave Infrared', [band_indices['B12'], band_indices['B8a'], band_indices['B4']]),
    ('Agriculture', [band_indices['B11'], band_indices['B8'], band_indices['B2']]),
    ('Geology', [band_indices['B12'], band_indices['B11'], band_indices['B2']]),
    ('Bathymetric', [band_indices['B4'], band_indices['B3'], band_indices['B1']])
]

# Number of examples
num_examples = len(img_list)
num_combinations = len(combinations)

# Create a figure with subplots arranged in a grid
fig, axs = plt.subplots(num_examples, num_combinations + 1, figsize=(num_combinations * 5, num_examples * 5))

for row, img_lab in enumerate(img_list):
    img, lab = img_lab
    img = np.transpose(img, (2, 0, 1))
    labels = [label_list[i] for i in range(len(lab)) if lab[i] == 1]

    # Add the label text on the left side
    axs[row, 0].text(-0.2, 0.5, '\n'.join(labels), verticalalignment='center', fontsize=18, transform=axs[row, 0].transAxes)
    axs[row, 0].axis('off')

    for col, (title, channels) in enumerate(combinations):

    #     break
    #
    # break
        # Extract the specified channels
        img_comb = img[[channels], :, :]
        img_comb = img_comb.squeeze(axis=0)  # Remove redundant dimension

        # Normalize and scale to 0-255
        min_vals = img_comb.min(axis=(1, 2), keepdims=True)
        max_vals = img_comb.max(axis=(1, 2), keepdims=True)
        img_comb = ((img_comb - min_vals) / (max_vals - min_vals) * 255).clip(0, 255).astype('uint8')

        # Plot the image
        axs[row, col + 1].imshow(np.transpose(img_comb, (1, 2, 0)))
        axs[row, col + 1].axis('off')

        # Add title for the first row only
        if row == 0:
            axs[row, col + 1].set_title(title, fontsize=24)

# Adjust layout and show the plot
# plt.tight_layout()
# plt.show()
# Manually adjust subplot layout
plt.subplots_adjust(left=0.05, right=0.95, top=0.75, bottom=0.02, wspace=0.05, hspace=0.02)


plt.savefig('/gpfs/work5/0/prjs0790/data/new_viz/ben_samples.png')

#
# # Create a figure with subplots arranged in a grid
# fig, axs = plt.subplots(num_examples, num_combinations + 1, figsize=(num_combinations * 5, num_examples * 5))
#
# for row, img_lab in enumerate(img_list):
#     img, lab = img_lab
#     img = np.transpose(img, (2, 0, 1))
#     labels = [label_list[i] for i in range(len(lab)) if lab[i] == 1]
#
#     # Add the label text on the left side
#     axs[row, 0].text(0.5, 0.5, '\n'.join(labels), verticalalignment='center', fontsize=10, transform=axs[row, 0].transAxes)
#     axs[row, 0].axis('off')
#
#     for col, (title, channels) in enumerate(combinations):
#         # Extract the specified channels
#         img_comb = img[channels, :, :]
#         img_comb = img_comb.squeeze(axis=0)  # Remove redundant dimension
#
#         # Normalize and scale to 0-255
#         min_vals = img_comb.min(axis=(1, 2), keepdims=True)
#         max_vals = img_comb.max(axis=(1, 2), keepdims=True)
#         img_comb = ((img_comb - min_vals) / (max_vals - min_vals) * 255).clip(0, 255).astype('uint8')
#
#         # Plot the image
#         axs[row, col + 1].imshow(np.transpose(img_comb, (1, 2, 0)))
#         axs[row, col + 1].axis('off')
#
#         # Add title for the first row only
#         if row == 0:
#             axs[row, col + 1].set_title(title, fontsize=16)
#
# # Adjust layout and show the plot
# plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.2)
# plt.tight_layout()
# plt.savefig('/gpfs/work5/0/prjs0790/data/new_viz/ben_samples.png')