
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

from src.benchmark.pretrain_ssl.datasets.SSL4EO.ssl4eo_dataset_lmdb_new import load_meta_df
from src.benchmark.pretrain_ssl.models.dino import vision_transformer as vits
from src.benchmark.transfer_classification.models.dino.utils import load_pretrained_weights


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torchvision
from torchvision.transforms import CenterCrop, InterpolationMode


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
    # pretrained = '/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/ssl_s2c_2/checkpoint0060.pth'
    # pretrained = '/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/ssl4eo_ssl/ssl_s2c_new_transforms/checkpoint0045.pth'
    # pretrained = '/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/ssl4eo_ssl/ssl_s2c_new_transforms/checkpoint0050.pth'
    # pretrained = '/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/ssl4eo_ssl/ssl_s2c_new_transforms/checkpoint0095.pth'
    # pretrained = '/gpfs/work5/0/prjs0790/data/old_checkpoints/B13_vits16_moco_0099_ckpt.pth'
    # pretrained = '/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/ssl4eo_ssl/moco_s2c_new_transforms/checkpoint_0034.pth.tar'
    pretrained = '/gpfs/home2/ramaudruz/detcon-pytorch/detcon-pytorch/d2z8hvep/checkpoints/epoch=2-step=11768.ckpt'

    # pretrained = '/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/leopart_ssl/leopart_new_transform/leopart-20240301-143933/ckp-epoch=49.ckpt'

    # rank = 0
    resume = True
    seed = 42
    val_freq = 5
    train_frac = 1.0
    # dtype = 'uint8'


args = FakeArgs()


in_channels = 13
# in_channels = 12

# ============ building network ... ============
# if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
if args.arch in vits.__dict__.keys():
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, in_chans=in_channels)
    embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
# otherwise, we check if the architecture is in torchvision models
elif args.arch in torchvision_models.__dict__.keys():
    model = torchvision_models.__dict__[args.arch]()
    embed_dim = model.fc.weight.shape[1]
    model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Identity()
    #model.fc = torch.nn.Linear(2048,19)
# if the network is a XCiT
elif "xcit" in args.arch:
    model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    embed_dim = model.embed_dim
else:
    print(f"Unknow architecture: {args.arch}")
    sys.exit(1)
model.cuda()
model.eval()
# load weights to evaluate
load_pretrained_weights(model, args.pretrained, args.checkpoint_key, args.arch, args.patch_size)
print(f"Model {args.arch} built.")


def load_meta_df():
    meta_df = pd.read_csv('/gpfs/scratch1/shared/ramaudruz/s2c_un/ssl4eo_s2_l1c_full_extract_metadata.csv')
    temp_var = meta_df['patch_id'].astype(str)
    meta_df['patch_id'] = temp_var.map(lambda x: (7 - len(x)) * '0' + x)
    meta_df['file_name'] = meta_df['patch_id'] + '/' + meta_df['timestamp']
    return meta_df


# Load image
# image_path = "/gpfs/work5/0/prjs0790/data/random_pics/Screenshot 2024-03-20 at 18.30.52.png"
# image_path = '/gpfs/work5/0/prjs0790/data/random_pics/Screenshot 2024-03-22 at 13.44.36.png'
# image_path = '/gpfs/work5/0/prjs0790/data/random_pics/Screenshot 2024-03-22 at 13.44.44.png'
# image_path = '/gpfs/work5/0/prjs0790/data/random_pics/Screenshot 2024-03-22 at 13.44.20.png'

image_path = '/gpfs/work5/0/prjs0790/data/random_pics/000000000474.jpg'

image = mpimg.imread(image_path)

# Show raw image
plt.imshow(image)
plt.show()

# Convert image to numpy array
img1 = torch.from_numpy(image)

# Remove alpha channel and reshuffle axis
img2 = img1[:, :, :3].permute(2, 0, 1)

# Show the image without the alpha channel
plt.imshow(img1[:, :, :3])
plt.show()

img3 = img2[:, :498 , :332]


img_parts = {}
img_parts[0] = img3[:, :166, :166]
img_parts[1] = img3[:, :166, 166:]
img_parts[2] = img3[:, 166:332, :166]
img_parts[3] = img3[:, 166:332, 166:]
img_parts[4] = img3[:, 332:, :166]
img_parts[5] = img3[:, 332:, 166:]

# Resize image parts to 224
resize_t = torchvision.transforms.RandomResizedCrop(
    224,
    scale=(1, 1),
    ratio=(1, 1),
    interpolation=InterpolationMode.BILINEAR
)

img2_cropped = {k: resize_t(v) for k, v in img_parts.items()}

# Show the cropped image
plt.imshow(img2_cropped[0].permute(1, 2, 0))
plt.show()


new_band_image = {}
for k, v in img2_cropped.items():
    new_band_image_tmp = torch.zeros(1, 13, 224, 224)
    new_band_image_tmp[:, 1:4, :, :] = v.flip(0)
    new_band_image[k] = new_band_image_tmp

# Predict the dense rep and reshape
pred_dic = {k: model(v.float().to('cuda')) for k, v in new_band_image.items()}
pred_res_dic = {k: v[:, 1:, :].reshape(14, 14, 384) for k, v in pred_dic.items()}

# Reshape for kmeans
mask_sh = torch.concat([v.reshape(-1, 384) for v in pred_res_dic.values()])

kmeans = KMeans(
    init="random",
    n_clusters=10,
    n_init=10,
    max_iter=1000,
    random_state=42
)

cluster_pred = kmeans.fit_predict(mask_sh.to('cpu').detach().numpy())

cluster_pred_torch = torch.from_numpy(cluster_pred)

new_mask_dic = {i: s for i, s in enumerate(cluster_pred_torch.split(196))}

new_mask_res_dic = {k: v.reshape(14, 14) for k, v in new_mask_dic.items()}

resized_masks = {
    k : F.interpolate(v.float()[None, None, :, :], size=(224, 224), mode='nearest')
    for k, v in new_mask_res_dic.items()
}

final_image_patch = {}
for k, v in resized_masks.items():

    df_t = pd.Series(v.flatten()).value_counts()

    img1rgb2_new = img2_cropped[k].clone()

    colour_map = {
        0: (15,82,186),
        1: (80,200,120),
        2: (128,0,128),
        3: (224,17,95),
        4: (0,0,128)
    }
    i = 0

    img1rgb2_first_channel = img2_cropped[k][0, :, :].clone()
    img1rgb2_second_channel = img2_cropped[k][1, :, :].clone()
    img1rgb2_third_channel = img2_cropped[k][2, :, :].clone()


    for k1, v1 in dict(zip(df_t.index, df_t.values)).items():

        first_cat = (v == k1).reshape(224, 224)

        if i == 5:
            break
        else:
            rgb_values = colour_map[i]
            i += 1

        img1rgb2_first_channel[first_cat] = rgb_values[0]
        img1rgb2_second_channel[first_cat] = rgb_values[1]
        img1rgb2_third_channel[first_cat] = rgb_values[2]

    img1rgb2_new = ((img1rgb2_new + np.concatenate([
        img1rgb2_first_channel.reshape(-1, 224, 224),
        img1rgb2_second_channel.reshape(-1, 224, 224),
        img1rgb2_third_channel.reshape(-1, 224, 224)
    ], 0)) /2).numpy().astype(int)

    final_image_patch[k] = img1rgb2_new


final_image = np.concatenate([
    np.concatenate([final_image_patch[0], final_image_patch[2],final_image_patch[4]], 1),
    np.concatenate([final_image_patch[1], final_image_patch[3],final_image_patch[5]], 1)
], 2)


plt.imshow(np.transpose(final_image, (1, 2, 0)))
plt.savefig(f'/gpfs/home2/ramaudruz/SSL4EO-S12/patch_vis/real_image_dense.png')


plt.imshow(np.transpose(img2_cropped[k], (1, 2, 0)))
plt.show()





