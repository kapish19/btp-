import os
import glob
import re
import scipy.io
import torch
from .bases import BaseImageDataset, auto_locate_dataset_dir
from . import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class cuhk03(BaseImageDataset):
    """
    CUHK03
    Reference:
    Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.
    Zhong et al. Re-ranking Person Re-identification with k-reciprocal Encoding. CVPR 2017.
    """
    dataset_dir = 'cuhk03'

    def __init__(self, root='', verbose=True, pid_begin=0, combineall=False, **kwargs):
        super(cuhk03, self).__init__()
        self.dataset_dir = auto_locate_dataset_dir(
            'cuhk03', root=root,
            markers=['cuhk-03.mat', 'cuhk03_new_protocol_config_detected.mat', 'cuhk03_release', 'images_detected', 'images_labeled']
        )
        
        # Auto-detect if unzipped inside an archive subfolder
        if os.path.exists(os.path.join(self.dataset_dir, 'archive')):
            self.dataset_dir = os.path.join(self.dataset_dir, 'archive')

        self.data_dir = os.path.join(self.dataset_dir, 'cuhk03_release')
        self.raw_mat_path = os.path.join(self.data_dir, 'cuhk-03.mat')
        
        # Check protocol mat files
        self.detected_mat_path = os.path.join(self.dataset_dir, 'cuhk03_new_protocol_config_detected.mat')
        self.labeled_mat_path = os.path.join(self.dataset_dir, 'cuhk03_new_protocol_config_labeled.mat')

        required_files = [self.dataset_dir]
        self.check_before_run(required_files)

        train, query, gallery = self._process_data()

        if verbose:
            print("=> CUHK03 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _process_data(self):
        train, query, gallery = [], [], []
        # Check detected vs labeled images
        det_dir = os.path.join(self.dataset_dir, 'images_detected')
        lbl_dir = os.path.join(self.dataset_dir, 'images_labeled')
        
        target_dir = det_dir if os.path.exists(det_dir) else lbl_dir
        if not os.path.exists(target_dir):
            target_dir = self.data_dir

        # Process images directly if folders exist
        if os.path.exists(target_dir):
            for root, dirs, files in os.walk(target_dir):
                for f in files:
                    if f.endswith('.png') or f.endswith('.jpg'):
                        img_path = os.path.join(root, f)
                        # extract pid and camid
                        parts = f.split('_')
                        if len(parts) >= 3:
                            try:
                                camid = int(parts[0])
                                pid = int(parts[1])
                                train.append((img_path, pid, camid, 2))
                            except Exception:
                                pass

        # Split into train/query/gallery logically
        if len(train) > 0:
            total = len(train)
            split_tr = int(total * 0.7)
            split_q = int(total * 0.85)
            query = train[split_tr:split_q]
            gallery = train[split_q:]
            train = train[:split_tr]

        return train, query, gallery
