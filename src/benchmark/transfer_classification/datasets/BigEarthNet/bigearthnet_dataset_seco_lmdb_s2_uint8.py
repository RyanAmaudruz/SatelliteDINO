import pickle

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import lmdb
from tqdm import tqdm

### SSL4EO stats 
S1_MEAN = [-12.54847273, -20.19237134]
S1_STD = [5.25697717, 5.91150917]

S2A_MEAN = [752.40087073, 884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 2422.21248945, 2517.76053101, 2581.64687018, 2645.51888987, 2368.51236873, 1805.06846033]

S2A_STD = [1108.02887453, 1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 1416.51487101, 1474.78900051, 1439.3086061, 1582.28010962, 1455.52084939, 1343.48379601]

NEW_LABELS = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
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

GROUP_LABELS = {
    'Continuous urban fabric': 'Urban fabric',
    'Discontinuous urban fabric': 'Urban fabric',
    'Non-irrigated arable land': 'Arable land',
    'Permanently irrigated land': 'Arable land',
    'Rice fields': 'Arable land',
    'Vineyards': 'Permanent crops',
    'Fruit trees and berry plantations': 'Permanent crops',
    'Olive groves': 'Permanent crops',
    'Annual crops associated with permanent crops': 'Permanent crops',
    'Natural grassland': 'Natural grassland and sparsely vegetated areas',
    'Sparsely vegetated areas': 'Natural grassland and sparsely vegetated areas',
    'Moors and heathland': 'Moors, heathland and sclerophyllous vegetation',
    'Sclerophyllous vegetation': 'Moors, heathland and sclerophyllous vegetation',
    'Inland marshes': 'Inland wetlands',
    'Peatbogs': 'Inland wetlands',
    'Salt marshes': 'Coastal wetlands',
    'Salines': 'Coastal wetlands',
    'Water bodies': 'Inland waters',
    'Water courses': 'Inland waters',
    'Coastal lagoons': 'Marine waters',
    'Estuaries': 'Marine waters',
    'Sea and ocean': 'Marine waters'
}


def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

class Subset(Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def random_subset(dataset, frac, seed=None):
    rng = np.random.default_rng(seed)
    indices = rng.choice(range(len(dataset)), int(frac * len(dataset)))
    return Subset(dataset, indices)


class _RepeatSampler(object):
    """
    Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(DataLoader):
    """
    Dataloader that reuses workers.
    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


def make_lmdb(dataset, lmdb_file, num_workers=6):
    loader = InfiniteDataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x[0])
    env = lmdb.open(lmdb_file, map_size=1099511627776)

    txn = env.begin(write=True)
    for index, (sample, target) in tqdm(enumerate(loader), total=len(dataset), desc='Creating LMDB'):
        sample = np.array(sample)
        obj = (sample.tobytes(), sample.shape, target.tobytes())
        txn.put(str(index).encode(), pickle.dumps(obj))
        if index % 10000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()

    env.sync()
    env.close()


class LMDBDataset(Dataset):

    def __init__(self, lmdb_file, is_slurm_job=False, transform=None, normalize=False):
        self.lmdb_file = lmdb_file
        self.transform = transform
        self.is_slurm_job = is_slurm_job
        self.normalize = normalize

        if not self.is_slurm_job:
            self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
            with self.env.begin(write=False) as txn:
                self.length = txn.stat()['entries']            
        else:
            # Workaround to have length from the start for ImageNet since we don't have LMDB at initialization time
            self.env = None
            if 'train' in self.lmdb_file:
                self.length = 300000
            elif 'val' in self.lmdb_file:
                self.length = 100000
            elif 'test' in self.lmdb_file:
                self.length = 100000
            else:
                raise NotImplementedError

    def _init_db(self):
        
        self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

    def __getitem__(self, index):
        if self.is_slurm_job:
            # Delay loading LMDB data until after initialization
            if self.env is None:
                self._init_db()
        
        with self.env.begin(write=False) as txn:
            data = txn.get(str(index).encode())

        #sample_s2_bytes, sample_s2_shape, sample_s1_bytes, sample_s1_shape, target_bytes = pickle.loads(data)
        sample_s2_bytes, sample_s2_shape, target_bytes = pickle.loads(data)
        sample = np.frombuffer(sample_s2_bytes, dtype=np.uint8).reshape(sample_s2_shape)
        #sample_s1 = np.frombuffer(sample_s1_bytes, dtype=np.float32).reshape(sample_s1_shape)

        target = np.frombuffer(target_bytes, dtype=np.float32)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return self.length


class LMDBDatasetRA(Dataset):

    def __init__(self, set_type, is_slurm_job=False, transform=None, normalize=False):
        self.set_type = set_type
        self.transform = transform
        self.is_slurm_job = is_slurm_job
        self.normalize = normalize
        self.patches_to_exclude = set(self.get_patches_to_exclude())
        self.ref_df = pd.read_csv(f'/var/node433/local/ryan_a/data/new_ben/bigearthnet/s2a_128_all_{set_type}_content.csv')

        self.ref_list = [
            p for p in self.ref_df['t'].map(lambda x: x.split("',")[0][2:]).tolist()
            if p not in self.patches_to_exclude
        ]
        self.length = len(self.ref_list)
        metadata_df = pd.read_csv('/var/node433/local/ryan_a/data/new_ben/bigearthnet/metadata_df.csv')
        metadata_df['labels'] = metadata_df['labels'].map(lambda x: eval(x))
        label_map = dict(zip(metadata_df['dir_name'], metadata_df['labels']))
        self.target_map = {}
        for p in self.ref_list:
            temp_labels = label_map[p]
            target = np.zeros((len(NEW_LABELS),), dtype=np.float32)
            for label in temp_labels:
                if label in GROUP_LABELS:
                    target[NEW_LABELS.index(GROUP_LABELS[label])] = 1
                elif label not in set(NEW_LABELS):
                    continue
                else:
                    target[NEW_LABELS.index(label)] = 1
            self.target_map[p] = target


        self.target_ref = pd.read_csv(f'/var/node433/local/ryan_a/data/new_ben/bigearthnet/bigearthnet-{set_type}.txt')

        # if not self.is_slurm_job:
        #     self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        #     with self.env.begin(write=False) as txn:
        #         self.length = txn.stat()['entries']
        # else:
        #     # Workaround to have length from the start for ImageNet since we don't have LMDB at initialization time
        #     self.env = None
        #     if 'train' in self.lmdb_file:
        #         self.length = 300000
        #     elif 'val' in self.lmdb_file:
        #         self.length = 100000
        #     elif 'test' in self.lmdb_file:
        #         self.length = 100000
        #     else:
        #         raise NotImplementedError

    def get_patches_to_exclude(self):
        patches_to_exclude = []
        with open('/var/node433/local/ryan_a/data/new_ben/bigearthnet/patches_with_seasonal_snow.csv', 'r') as f:
            patches_to_exclude += [p.strip() for p in f.readlines()]
        with open('/var/node433/local/ryan_a/data/new_ben/bigearthnet/patches_with_cloud_and_shadow.csv', 'r') as f:
            patches_to_exclude += [p.strip() for p in f.readlines()]
        return patches_to_exclude

    def _init_db(self):

        self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

    def __getitem__(self, index):
        # if self.is_slurm_job:
        #     # Delay loading LMDB data until after initialization
        #     if self.env is None:
        #         self._init_db()
        #
        # with self.env.begin(write=False) as txn:
        #     data = txn.get(str(index).encode())


        with h5py.File(f'/var/node433/local/ryan_a/data/new_ben/bigearthnet/s2a_128_all_{self.set_type}.h5', 'r') as f:
            sample = np.array(f.get(self.ref_list[index])).astype('int16')

        target = self.target_map[self.ref_list[index]]

        # #sample_s2_bytes, sample_s2_shape, sample_s1_bytes, sample_s1_shape, target_bytes = pickle.loads(data)
        # sample_s2_bytes, sample_s2_shape, target_bytes = pickle.loads(data)
        # sample = np.frombuffer(sample_s2_bytes, dtype=np.uint8).reshape(sample_s2_shape)
        # #sample_s1 = np.frombuffer(sample_s1_bytes, dtype=np.float32).reshape(sample_s1_shape)
        #
        # target = np.frombuffer(target_bytes, dtype=np.float32)

        mean = [340.76769064, 429.9430203, 614.21682446, 590.23569706, 950.68368468, 1792.46290469, 2075.46795189, 2218.94553375, 2266.46036911, 2246.0605464, 1594.42694882, 1009.32729131]
        std = [554.81258967, 572.41639287, 582.87945694, 675.88746967, 729.89827633, 1096.01480586, 1273.45393088, 1365.45589904, 1356.13789355, 1302.3292881, 1079.19066363, 818.86747235]

        # 1st attempt
        # sample = (sample - np.array(mean).reshape(-1, 1, 1)) / np.array(std).reshape(-1, 1, 1)

        # 2nd attempt
        mean_shaped = np.array(mean).reshape(-1, 1, 1)
        std_shaped = np.array(std).reshape(-1, 1, 1)
        min_value = mean_shaped - 2 * std_shaped
        max_value = mean_shaped + 2 * std_shaped
        sample = (sample - min_value) / (max_value - min_value) * 255.0
        sample = np.clip(sample, 0, 255).astype(np.uint8)

        sample = sample.astype('float32').transpose(1, 2, 0)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return self.length


class LMDBDatasetRALeo(LMDBDatasetRA):
    def __getitem__(self, index):
        with h5py.File(f'/gpfs/work5/0/prjs0790/data/bigearthnet/s2a_128_all_{self.set_type}.h5', 'r') as f:
            sample = np.array(f.get(self.ref_list[index])).astype('int16')

        target = self.target_map[self.ref_list[index]]

        mean = [340.76769064, 429.9430203, 614.21682446, 590.23569706, 950.68368468, 1792.46290469, 2075.46795189, 2218.94553375, 2266.46036911, 2246.0605464, 1594.42694882, 1009.32729131]
        std = [554.81258967, 572.41639287, 582.87945694, 675.88746967, 729.89827633, 1096.01480586, 1273.45393088, 1365.45589904, 1356.13789355, 1302.3292881, 1079.19066363, 818.86747235]

        # 1st attempt
        # sample = (sample - np.array(mean).reshape(-1, 1, 1)) / np.array(std).reshape(-1, 1, 1)

        # 2nd attempt
        # mean_shaped = np.array(mean).reshape(-1, 1, 1)
        # std_shaped = np.array(std).reshape(-1, 1, 1)
        # min_value = mean_shaped - 2 * std_shaped
        # max_value = mean_shaped + 2 * std_shaped
        # sample = (sample - min_value) / (max_value - min_value) * 255.0
        # sample = np.clip(sample, 0, 255).astype(np.uint8)

        # sample = sample.astype('float32').transpose(1, 2, 0) / 20

        sample = sample.astype('float32').transpose(1, 2, 0) / 20

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


if __name__ == '__main__':
    import os
    import argparse
    import time
    import torch
    from torchvision import transforms
    from cvtorchvision import cvtransforms
    import cv2
    import random
    import pdb


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/p/scratch/hai_dm4eo/wang_yi/data/BigEarthNet_LMDB_raw/')
    parser.add_argument('--train_frac', type=float, default=1.0)
    args = parser.parse_args()

    test_loading_time = False
    seed = 42
    

    
    augmentation = [
        cvtransforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        cvtransforms.RandomHorizontalFlip(),
        cvtransforms.ToTensor(),
    ]
    train_transforms = cvtransforms.Compose(augmentation)
    
    train_dataset = LMDBDataset(
        lmdb_file=os.path.join(args.data_dir, 'train_B12_B2.lmdb'),
        transform=train_transforms
    )

    print(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=0)    
    for idx, (img,target) in enumerate(train_loader):
        if idx>1:
            break
        print(img.shape, img.dtype)