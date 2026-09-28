import os
import glob
import re
import numpy as np
import scipy.io as sio
from .bases import BaseImageDataset
from . import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class cuhk_sysu(BaseImageDataset):
    """
    CUHK-SYSU Person Re-ID Dataset
    Reference:
    Xiao et al. Joint Detection and Identification Feature Learning for Person Search. CVPR 2017.
    """
    dataset_dir = 'cuhk_sysu'

    def __init__(self, root='', verbose=True, pid_begin=0, combineall=False, **kwargs):
        super(cuhk_sysu, self).__init__()

        # Locate root directory of cuhk_sysu
        candidates = [
            os.path.join(root, 'cuhk_sysu'),
            os.path.join(root, 'cuhk-sysu'),
            'cuhk_sysu',
            './cuhk_sysu',
            '/content/btp_repo/data/cuhk_sysu',
            '/content/btp_repo/cuhk_sysu',
            '/content/drive/MyDrive/datasets/cuhk_sysu',
        ]
        self.dataset_dir = os.path.join(root, 'cuhk_sysu')
        for cand in candidates:
            if os.path.exists(os.path.join(cand, 'annotation')):
                self.dataset_dir = cand
                break

        # Locate Train.mat (supports test/train_test/Train.mat or annotation/Train.mat)
        self.mat_path = None
        for root_d, _, files in os.walk(self.dataset_dir):
            for f in files:
                if 'train' in f.lower() and f.endswith('.mat'):
                    self.mat_path = os.path.join(root_d, f)
                    break
            if self.mat_path:
                break

        # Locate Image directory (supports Image/SSM or Image or cropped_images)
        self.img_dir = os.path.join(self.dataset_dir, 'Image', 'SSM')
        if not os.path.exists(self.img_dir):
            for cand_img in [os.path.join(self.dataset_dir, 'Image'),
                             os.path.join(self.dataset_dir, 'cropped_images'),
                             os.path.join(self.dataset_dir, 'images')]:
                if os.path.isdir(cand_img):
                    self.img_dir = cand_img
                    break

        train = self._process_train_mat(self.mat_path, self.img_dir)
        query = []
        gallery = []

        if verbose:
            print(f"=> CUHK-SYSU loaded  [dir: {self.dataset_dir}]")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _process_train_mat(self, mat_path, img_dir):
        if not mat_path or not os.path.exists(mat_path):
            print(f"⚠️ CUHK-SYSU: Train.mat not found in {self.dataset_dir}")
            return []

        mat = sio.loadmat(mat_path)
        key = 'Train' if 'Train' in mat else 'train' if 'train' in mat else [k for k in mat.keys() if not k.startswith('__')][0]
        train_records = mat[key].flat

        dataset = []
        for item in train_records:
            # Extract idname -> pid (e.g. 'p10573' -> 10573)
            id_val = item['idname']
            while isinstance(id_val, np.ndarray) and id_val.size == 1:
                id_val = id_val[0]
            raw_str = str(id_val)
            digits = re.sub(r'\D', '', raw_str)
            pid = int(digits) if digits else 0

            # Extract scene array
            scenes = item['scene']
            while isinstance(scenes, np.ndarray) and scenes.size == 1 and scenes.dtype == object:
                scenes = scenes[0]

            for sc in scenes.flat:
                im_val = sc['imname']
                while isinstance(im_val, np.ndarray) and im_val.size == 1:
                    im_val = im_val[0]
                im_str = str(im_val)
                img_path = os.path.join(img_dir, im_str)
                dataset.append((img_path, pid, 0, 1))

        return dataset
