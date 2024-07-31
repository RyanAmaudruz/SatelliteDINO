
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Load dataset
train_dataset = LMDBDatasetS2C(
    lmdb_file=None,
    s2c_transform=None,
    normalize=False,
    dtype=None,
    mode=['s2c']
)

img_list = [train_dataset.__getitem__(i) for i in [52, 94, 124, 125, 202, 208, 210, 212, 216][:-1]]

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
    'B9': 9,   # Short Wave Infrared (SWIR) - 940 nm
    'B10': 10, # Short Wave Infrared (SWIR) - 1375 nm
    'B11': 11, # Short Wave Infrared (SWIR) - 1610 nm
    'B12': 12  # Short Wave Infrared (SWIR) - 2190 nm
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
fig, axs = plt.subplots(num_examples, num_combinations, figsize=(num_combinations * 4, num_examples * 4))

for row, img in enumerate(img_list):
    for col, (title, channels) in enumerate(combinations):
        # Extract the specified channels
        img_comb = img[0, [channels], :, :]
        img_comb = img_comb.squeeze(axis=0)  # Remove redundant dimension

        # Normalize and scale to 0-255
        min_vals = img_comb.min(axis=(1, 2), keepdims=True)
        max_vals = img_comb.max(axis=(1, 2), keepdims=True)
        img_comb = ((img_comb - min_vals) / (max_vals - min_vals) * 255).clip(0, 255).astype('uint8')

        # Plot the image
        axs[row, col].imshow(np.transpose(img_comb, (1, 2, 0)))
        axs[row, col].axis('off')

        # Add title for the first row only
        if row == 0:
            axs[row, col].set_title(title, fontsize=24)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('/gpfs/work5/0/prjs0790/data/new_viz/ssl4eo_samples.png')


