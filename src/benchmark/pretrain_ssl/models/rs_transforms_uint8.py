import numpy as np
import torch
import random
import cv2
import torchvision
from PIL import Image
from torchvision.transforms.functional import adjust_hue



# class RandomBrightness(object):
#     """ Random Brightness """
#
#     def __init__(self, brightness=0.4):
#         self.brightness = brightness
#
#     def __call__(self, sample):
#         s = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
#         img = sample * s
#
#         return img.clip(0, 255)
#
# class RandomContrast(object):
#     """ Random Contrast """
#
#     def __init__(self, contrast=0.4):
#         self.contrast = contrast
#
#     def __call__(self, sample):
#         s = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
#         mean = np.mean(sample, axis=(0, 1))
#
#         return ((sample - mean) * s + mean).clip(0, 255)
#
#
# class RandomSaturation(object):
#     """ Random Contrast """
#
#     def __init__(self, saturation=0.4):
#         self.saturation = saturation
#
#     def __call__(self, sample):
#         s = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
#         mean = np.expand_dims(sample.mean(-1), -1)
#         return ((sample - mean) * s + mean).clip(0, 255)
#
#
# class RandomHue(object):
#     """ Random Contrast """
#     def __init__(self, hue=0.1):
#         self.hue = hue
#
#     def __call__(self, sample):
#         sample = sample.astype(np.uint8)
#         sample[:, :, 1:4] = np.flip(np.array(
#             adjust_hue(Image.fromarray(np.flip(sample[:, :, 1:4], 2)), hue_factor=self.hue)
#         ), 2)
#         return sample
#
# class ToGray(object):
#     def __init__(self, out_channels):
#         self.out_channels = out_channels
#     def __call__(self,sample):
#         gray_img = np.mean(sample, axis=-1)
#         gray_img = np.tile(gray_img, (self.out_channels, 1, 1))
#         gray_img = np.transpose(gray_img, [1, 2, 0])
#         return gray_img.astype(np.uint8)
#
class RandomChannelDrop(object):
    """ Random Channel Drop """

    def __init__(self, min_n_drop=1, max_n_drop=8):
        self.min_n_drop = min_n_drop
        self.max_n_drop = max_n_drop

    def __call__(self, sample):
        n_channels = random.randint(self.min_n_drop, self.max_n_drop)
        channels = np.random.choice(range(sample.shape[0]), size=n_channels, replace=False)

        for c in channels:
            sample[c, :, :] = 0
        return sample

#
# class GaussianBlur(object):
#     """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
#
#     def __init__(self, sigma=[.1, 2.]):
#         self.sigma = sigma
#
#     def __call__(self, x):
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         #x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         #return x
#         return cv2.GaussianBlur(x,(0,0),sigma)
#
#
# class Solarize(object):
#
#     def __init__(self, threshold=0.5):
#         self.threshold = threshold
#
#     def __call__(self, x):
#         x1 = x.copy()
#         one = np.ones(x.shape) * 255
#         x1[x<self.threshold] = one[x<self.threshold] - x[x<self.threshold]
#
#         return x1.astype(np.uint8)

# class RandomBrightness(object):
#     """ Random Brightness """
#
#     def __init__(self, brightness=0.4):
#         self.brightness = brightness
#
#     def __call__(self, sample):
#         s = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
#         img = sample * s
#
#         return img.clip(0, 255)
#
# class RandomContrast(object):
#     """ Random Contrast """
#
#     def __init__(self, contrast=0.4):
#         self.contrast = contrast
#
#     def __call__(self, sample):
#         s = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
#         mean = np.mean(sample, axis=(0, 1))
#
#         return ((sample - mean) * s + mean).clip(0, 255)
#
#
# class RandomSaturation(object):
#     """ Random Contrast """
#
#     def __init__(self, saturation=0.4):
#         self.saturation = saturation
#
#     def __call__(self, sample):
#         s = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
#         mean = np.expand_dims(sample.mean(-1), -1)
#         return ((sample - mean) * s + mean).clip(0, 255)
#
#
# class RandomHue(object):
#     """ Random Contrast """
#     def __init__(self, hue=0.1):
#         self.hue = hue
#
#     def __call__(self, sample):
#         sample = sample.astype(np.uint8)
#         sample[:, :, 1:4] = np.flip(np.array(
#             adjust_hue(Image.fromarray(np.flip(sample[:, :, 1:4], 2)), hue_factor=self.hue)
#         ), 2)
#         return sample
#
# class ToGray(object):
#     def __init__(self, out_channels):
#         self.out_channels = out_channels
#     def __call__(self,sample):
#         gray_img = np.mean(sample, axis=-1)
#         gray_img = np.tile(gray_img, (self.out_channels, 1, 1))
#         gray_img = np.transpose(gray_img, [1, 2, 0])
#         return gray_img.astype(np.uint8)
#
# class RandomChannelDrop(object):
#     """ Random Channel Drop """
#
#     def __init__(self, min_n_drop=1, max_n_drop=8):
#         self.min_n_drop = min_n_drop
#         self.max_n_drop = max_n_drop
#
#     def __call__(self, sample):
#         n_channels = random.randint(self.min_n_drop, self.max_n_drop)
#         channels = np.random.choice(range(sample.shape[0]), size=n_channels, replace=False)
#
#         for c in channels:
#             sample[c, :, :] = 0
#         return sample
#
#
# class GaussianBlur(object):
#     """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
#
#     def __init__(self, sigma=[.1, 2.]):
#         self.sigma = sigma
#
#     def __call__(self, x):
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         #x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         #return x
#         return cv2.GaussianBlur(x,(0,0),sigma)
#
#
# class Solarize(object):
#
#     def __init__(self, threshold=0.5):
#         self.threshold = threshold
#
#     def __call__(self, x):
#         x1 = x.copy()
#         one = np.ones(x.shape) * 255
#         x1[x<self.threshold] = one[x<self.threshold] - x[x<self.threshold]
#
#         return x1.astype(np.uint8)
#
class RandomSensorDrop_S1S2(object):
    """ Random Channel Drop """

    def __init__(self):
        pass

    def __call__(self, sample):
        sensor = np.random.choice([1,2], replace=False)

        if sensor==2:
            sample[:13, :, :] = 0
        elif sensor==1:
            sample[13:,:,:] = 0

        return sample
#
# class SensorDrop_S1S2(object):
#     def __init__(self, sensor):
#         self.sensor = sensor
#     def __call__(self,sample):
#         if self.sensor == 'S1':
#             sample[13:,:,:] = 0
#         elif self.sensor == 'S2':
#             sample[:13,:,:] = 0
#         return sample
#
#
# class RandomSensorDrop_RGBD(object):
#     """ Random Channel Drop """
#
#     def __init__(self):
#         pass
#
#     def __call__(self, sample):
#         sensor = np.random.choice([1,2], replace=False, p=[0.8,0.2])
#
#         if sensor==2:
#             sample[:3, :, :] = 0
#         elif sensor==1:
#             sample[3:,:,:] = 0
#
#         return sample
#
# class SensorDrop_RGBD(object):
#     def __init__(self, sensor):
#         self.sensor = sensor
#     def __call__(self,sample):
#         if self.sensor == 'D':
#             sample[3:,:,:] = 0
#         elif self.sensor == 'RGB':
#             sample[:3,:,:] = 0
#         return sample


class RandomBrightness(object):
    """ Random Brightness """

    def __init__(self, brightness=0.4):
        self.brightness = brightness

    def __call__(self, sample):
        s = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        img = sample * s

        return img.clip(0, 1)

class RandomContrast(object):
    """ Random Contrast """

    def __init__(self, contrast=0.4):
        self.contrast = contrast

    def __call__(self, sample):
        s = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        mean = sample.mean(axis=(1, 2))[:, None, None]
        return ((sample - mean) * s + mean).clip(0, 1)


class RandomSaturation(object):
    """ Random Contrast """

    def __init__(self, saturation=0.4):
        self.saturation = saturation

    def __call__(self, sample):

        s = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        mean = sample.mean(axis=0)[None, :, :]
        return ((sample - mean) * s + mean).clip(0, 1)


class RandomHue(object):
    """ Random Contrast """
    def __init__(self, hue=0.1):
        self.hue = hue

    def __call__(self, sample):
        rgb_channels = sample[1:4, :, :].flip(0)
        h = np.random.uniform(0 - self.hue, self.hue)
        rgb_channels_hue_mod = adjust_hue(rgb_channels, hue_factor=h)
        sample[1:4, :, :] = rgb_channels_hue_mod.flip(0)
        return sample

class ToGray(object):
    def __init__(self, out_channels):
        self.out_channels = out_channels
    def __call__(self,sample):
        nc = sample.shape[0]
        return sample.mean(axis=0)[None, :, :].expand(nc, -1, -1)

class RandomChannelDrop(object):
    """ Random Channel Drop """

    def __init__(self, min_n_drop=1, max_n_drop=8):
        self.min_n_drop = min_n_drop
        self.max_n_drop = max_n_drop

    def __call__(self, sample):
        n_channels = random.randint(self.min_n_drop, self.max_n_drop)
        channels = np.random.choice(range(sample.shape[0]), size=n_channels, replace=False)

        for c in channels:
            sample[c, :, :] = 0
        return sample


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
        self.transform = None

    def __call__(self, x):
        if self.transform is None:
            img_size = x.shape[-1]
            kernel_size = int(img_size * 0.1)
            # Make kernel size odd
            if kernel_size % 2 == 0:
                kernel_size = kernel_size + 1
            self.transform = torchvision.transforms.GaussianBlur(kernel_size, self.sigma)
        return self.transform(x)



class Solarize(object):

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, x):
        x1 = x.clone()
        one = torch.ones(x.shape)
        bool_check = x > self.threshold
        x1[bool_check] = one[bool_check] - x[bool_check]
        return x1
