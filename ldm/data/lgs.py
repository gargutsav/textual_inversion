import os
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random

class LGS(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 center_crop=False,
                 per_image_tokens=False
                 ):

        self.data_root = data_root
        self.metadata = pd.read_csv(f'{self.data_root}/{set}_subset.csv')

        self.image_paths = [os.path.join(self.data_root, 'imgs_256_04_27', f_path) for f_path in self.metadata['images'].values]
        self.captions = [caption for caption in self.metadata['caption'].values]

        print("KJSDFHJKFHKSFDJKHKFHSKDJHKFDHJFHS")
        print(len(self.image_paths), len(self.captions))
        print(self.image_paths[:5])
        print(self.captions[:5])

        self.num_images = len(self.image_paths)
        self._length = self.num_images 

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.center_crop = center_crop
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        example["caption"] = self.captions[i % self.num_images]
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")
            
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example