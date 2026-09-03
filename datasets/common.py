# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import sys
import time
# sys.path.append('/home/zhaohuazhong/DGReID/')
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageOps
from reidutils.file_io import PathManager
from collections import defaultdict
import random
import torch

class SyntheticOcclusionAugment:
    def __init__(self, p=0.5, grid_size=(16, 8)):
        self.p = p
        self.grid_size = grid_size
        
    def __call__(self, img_tensor):
        # img_tensor is [3, H, W]
        _, H, W = img_tensor.shape
        num_parts = 3
        # Mask is [3, 1] representing visibility of Head, Torso, Legs (1=visible, 0=occluded)
        mask = torch.ones(num_parts, 1, dtype=torch.float32)
        
        if random.random() > self.p:
            return img_tensor, mask
            
        # Random occlusion block
        # Size between 0.2 and 0.5 of image width
        s_w = int(random.uniform(0.2, 0.6) * W)
        s_h = int(random.uniform(0.2, 0.6) * H)
        
        if s_w <= 0 or s_h <= 0:
            return img_tensor, mask
            
        x0 = random.randint(0, W - s_w)
        y0 = random.randint(0, H - s_h)
        x1 = x0 + s_w
        y1 = y0 + s_h
        
        # Apply occlusion (mean color or random noise, let's use mean color 0 since it's normalized)
        # Using 0 which is roughly mean for InstanceNorm or ImageNet mean
        img_tensor[:, y0:y1, x0:x1] = 0.0
        
        # Calculate which parts are occluded.
        # We split H into 3 parts (Head, Torso, Legs)
        part_h = H / num_parts
        for i in range(num_parts):
            p_y0 = i * part_h
            p_y1 = (i + 1) * part_h
            
            # Intersection of (y0, y1) and (p_y0, p_y1)
            inter_y0 = max(y0, p_y0)
            inter_y1 = min(y1, p_y1)
            
            if inter_y1 > inter_y0:
                # Occlusion overlaps with this part
                # If overlap area is significant, mark as occluded (0)
                overlap_ratio = (inter_y1 - inter_y0) / part_h
                if overlap_ratio > 0.3:
                    mask[i, 0] = 0.0 # Occluded
                    
        return img_tensor, mask

def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.
    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"
    Returns:
        image (np.ndarray): an HWC image
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)

        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)

        # handle formats not supported by PIL
        elif format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]

        # handle grayscale mixed in RGB images
        elif len(image.shape) == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=-1)

        image = Image.fromarray(image)

        return image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True,last_id=0, is_train=False):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        self.last_id = last_id
        self.is_train = is_train
        self.occ_aug = SyntheticOcclusionAugment(p=0.5) if is_train else None

        pid_set = set()
        pids = []
        cam_set = set()
        domains = set()
        domain = []
        dcmain = defaultdict(set)
        # self.p2d = defaultdict(set)
        for i in img_items:
            pid_set.add(i[1])
            pids.append(i[1])
            cam_set.add(i[2])
            domains.add(i[3])
            domain.append(i[3])
            dcmain[i[3]].add(i[2])
        p2d = dict(zip(pids,domain))
        domains = list(domains)
        domains.sort()

        self.demains = {}

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        self.domains = {list(domains)[i]: i for i in range(len(list(domains)))}

        self.p2d = {}

        print(self.domains)
        if relabel:
            self.pid_dict = dict([(p, i+self.last_id) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])
            for p, d in p2d.items():
                self.p2d[self.pid_dict[p]] = self.domains[d]
        else:
            for p, d in p2d.items():
                self.p2d[p] = self.domains[d]
        for elm in dcmain.keys():
            self.demains[elm] = dict([(p, i) for i, p in enumerate(dcmain[elm])])





    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        domain = img_item[3]
        img = read_image(img_path)

        # pdb.set_trace()
        if self.transform is not None: img = self.transform(img)
        
        if self.is_train and self.occ_aug is not None:
            img, occ_mask = self.occ_aug(img)
        else:
            occ_mask = torch.ones(3, 1, dtype=torch.float32)

        if self.relabel:
            pid_agg = self.pid_dict[pid]

            rescamid = self.cam_dict[camid]
            # pid_expert = int(pid.split('_')[-1])
            return img, pid_agg, rescamid, 1, img_path, self.domains[domain],self.demains[domain][camid], occ_mask
        else:
            return img, pid, camid, 1, img_path, self.domains[domain],self.demains[domain][camid], occ_mask

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        trackid = 1
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, img_path.split('/')[-1]


if __name__ == "__main__":
    read_image('/data/zhz_dataset/ReID/cuhk03/images_labeled/1_001_2_08.png')
