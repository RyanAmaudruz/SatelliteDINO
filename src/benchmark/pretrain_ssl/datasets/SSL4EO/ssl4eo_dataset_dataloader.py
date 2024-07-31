import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd


def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def load_meta_df():
    meta_df = pd.read_csv('/gpfs/scratch1/shared/ramaudruz/s2c_un/ssl4eo_s2_l1c_full_extract_metadata.csv')
    temp_var = meta_df['patch_id'].astype(str)
    meta_df['patch_id'] = temp_var.map(lambda x: (7 - len(x)) * '0' + x)
    meta_df['file_name'] = meta_df['patch_id'] + '/' + meta_df['timestamp']
    return meta_df


class LMDBDatasetS2C(Dataset):

    def __init__(self, lmdb_file, s1_transform=None, s2a_transform=None, s2c_transform=None, subset=None, normalize=False, mode=['s1','s2a','s2c'], dtype='raw'):
        self.lmdb_file = lmdb_file
        self.s1_transform = s1_transform
        self.s2a_transform = s2a_transform
        self.s2c_transform = s2c_transform
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

