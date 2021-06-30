import os, glob
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

label_dict = {'positive': 1, 'negative': 0}


class CovidDataSet(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = None
        if annotations_file is not None:
            self.img_labels = pd.read_csv(annotations_file, delimiter=" ",
                                          names=['patient id', 'filename', 'class', 'data source'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        if self.img_labels is not None:
            self.len = len(self.img_labels)
        else:
            self.len = sum([len(glob.glob(img_dir + s)) for s in ['*.jpg', '*.png', '*.jpeg']])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        # image = read_image(img_path)
        image = Image.open(img_path).convert("RGB")
        label = label_dict[self.img_labels.iloc[idx, 2]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class ImageSortingDataSet(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = None
        if annotations_file is not None:
            self.img_labels = pd.read_csv(annotations_file, delimiter=" ",
                                          names=['patient id', 'filename', 'class', 'data source'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        if self.img_labels is not None:
            self.len = len(self.img_labels)
        else:
            self.len = sum([len(glob.glob(img_dir + s)) for s in ['*.jpg', '*.png', '*.jpeg']])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        # image = read_image(img_path)
        image = Image.open(img_path).convert("RGB")
        label = label_dict[self.img_labels.iloc[idx, 2]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class MyTopCropTransform:
    """
    crop the top <ratio> of the image
    """

    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, x):
        c, h, w = x.shape
        top = int(h * self.ratio)
        return transforms.functional.crop(x, top, 0, h - top, w)
