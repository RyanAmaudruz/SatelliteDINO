
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

from src.benchmark.pretrain_ssl.datasets.SSL4EO.ssl4eo_dataset_lmdb_old import load_meta_df
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

class LMDBDatasetS2C(Dataset):

    def __init__(self, lmdb_file, is_slurm_job=False, s1_transform=None, s2a_transform=None, s2c_transform=None, subset=None, normalize=False, mode=['s1','s2a','s2c'], dtype='raw'):
        self.lmdb_file = lmdb_file
        self.s1_transform = s1_transform
        self.s2a_transform = s2a_transform
        self.s2c_transform = s2c_transform
        # self.is_slurm_job = is_slurm_job
        self.subset = subset
        self.normalize = normalize
        self.mode = mode
        self.dtype = dtype
        self.meta_df = load_meta_df()
        self.patch_id_list = self.meta_df['patch_id'].unique().tolist()
        self.length = len(self.patch_id_list)

    def __getitem__(self, index):

        patch_id = self.patch_id_list[index]

        with h5py.File('/gpfs/scratch1/shared/ramaudruz/s2c_un/s2c_264_light_new.h5', 'r') as f:
            data = np.array(f.get(patch_id))

        if self.s2c_transform is not None:
            data = self.s2c_transform(data)

        return data

    def __len__(self):
        return self.length


train_dataset = LMDBDatasetS2C(
    lmdb_file=None,
    s2c_transform=None,
    # is_slurm_job=args.is_slurm_job,
    normalize=False,
    dtype=None,
    mode=args.mode
)

img_list = [train_dataset.__getitem__(i) for i in range(50)]

img_list3 = [i.max() for i in img_list]

img_list2 = [np.flip(i[0, 1:4, :, :], 1) for i in img_list]

img1 = img_list[6]

img1rgb = np.flip(img1[0, 1:4, :, :], 1)

img1rgb2 = (img1rgb * 2).clip(0, 256).astype(int)

img1rgb = img1[0, 1:4, :, :]

#
# plt.imshow(np.transpose(img1rgb2))
# plt.show()

from cvtorchvision import cvtransforms




val_transform = cvtransforms.Compose([
    # cvtransforms.Resize(256),
    # cvtransforms.CenterCrop(224),
    cvtransforms.Resize(224),
    cvtransforms.ToTensor(),
    #cvtransforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

first_split = np.split(img1[0], 2, 1)
sec_split = [np.split(i, 2, 2) for i in first_split]

pred_list1 = []
pred_list10 = []
for s_list in sec_split:
    pred_list2 = []
    pred_list20 = []
    for arr in s_list:
        arr_t = val_transform(np.transpose(arr, (1, 2, 0)))
        pred_list2.append(model(arr_t[None, :, :, :].float().to('cuda')))
        pred_list20.append(model(arr_t[None, :, :, :].float().to('cuda')))
    pred_list1.append(pred_list2)
    pred_list10.append(pred_list20)



pred_list1[0][0][:, 1:, :].reshape(14, 14, 384)


data_list = []
for p1 in pred_list1:
    data_list.append(torch.concat([p[:, 1:, :].reshape(14, 14, 384) for p in p1], 0))

mask = torch.concat(data_list, 1).to('cpu')

mask_sh = mask.reshape(-1, 384)



kmeans = KMeans(
    init="random",
    n_clusters=21,
    n_init=10,
    max_iter=1000,
    random_state=42
)

cluster_pred = kmeans.fit_predict(mask_sh.detach().numpy())

cluster_pred_torch = torch.from_numpy(cluster_pred)

# new_mask = cluster_pred_torch.reshape(56, 56)
new_mask = cluster_pred_torch.reshape(28, 28)



import torch.nn.functional as F

resized_masks = F.interpolate(new_mask.float()[None, None, :, :], size=(264, 264), mode='nearest')

df_t = pd.Series(resized_masks.flatten()).value_counts()

for k, v in dict(zip(df_t.index, df_t.values)).items():

    first_cat = (resized_masks == k).reshape(264, 264)

    img1rgb2_new = img1rgb2.copy()
    img1rgb2_copy = img1rgb2[0, :, :].copy()

    img1rgb2_copy[first_cat] += 50
    img1rgb2_copy = np.clip(img1rgb2_copy, 0, 255)

    img1rgb2_new[0, :, :] = img1rgb2_copy



    plt.imshow(np.transpose(img1rgb2_new))
    plt.savefig(f'round_1_cluster_vis-cat_{k}-count_{v}.png')



img1rgb2_new = img1rgb2.copy()

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
i = 0

img1rgb2_first_channel = img1rgb2[0, :, :].copy()
img1rgb2_second_channel = img1rgb2[1, :, :].copy()
img1rgb2_third_channel = img1rgb2[2, :, :].copy()


for k, v in dict(zip(df_t.index, df_t.values)).items():

    first_cat = (resized_masks == k).reshape(264, 264)

    if i == 8:
        break
    else:
        rgb_values = colour_map[i]
        i += 1

    img1rgb2_first_channel[first_cat] = rgb_values[0]
    img1rgb2_second_channel[first_cat] = rgb_values[1]
    img1rgb2_third_channel[first_cat] = rgb_values[2]

img1rgb2_new = ((img1rgb2_new + np.concatenate([
    img1rgb2_first_channel.reshape(-1, 264, 264),
    img1rgb2_second_channel.reshape(-1, 264, 264),
    img1rgb2_third_channel.reshape(-1, 264, 264)
], 0)) /2).astype(int)

plt.imshow(np.transpose(img1rgb2_new))
plt.savefig(f'/gpfs/home2/ramaudruz/SSL4EO-S12/patch_vis/round_1_cluster_vis-cat_NEW.png')













#
# # plt.imshow(img1rgb)
# #
# #
# #
# # plt.imshow(np.transpose(img1rgb, (1, 2, 0)))
#
#
#
# 264 / 28
# img1rgb2
#
#
#
#
# np.transpose(img1rgb2)
# plt.show()
#
#
#
#



