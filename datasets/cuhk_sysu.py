import os
import glob
import re
from .bases import BaseImageDataset
from . import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class cuhk_sysu(BaseImageDataset):
    """
    CUHK-SYSU
    Reference:
    Xiao et al. Joint Detection and Identification Feature Learning for Person Search. CVPR 2017.
    """
    dataset_dir = 'cuhk_sysu'

    def __init__(self, root='', verbose=True, pid_begin=0, combineall=False, **kwargs):
        super(cuhk_sysu, self).__init__()

        # Auto-detect CUHK-SYSU directory
        candidates = ['cuhk_sysu', 'cuhk-sysu', 'cuhk_sysu/cuhk_sysu', 'cuhk-sysu/cuhk-sysu', 'CUHK-SYSU']
        self.dataset_dir = os.path.join(root, 'cuhk_sysu')
        for cand in candidates:
            cand_path = os.path.join(root, cand)
            if os.path.exists(os.path.join(cand_path, 'annotation', 'train.mat')):
                self.dataset_dir = cand_path
                break
            elif os.path.exists(os.path.join(cand_path, 'annotation')):
                self.dataset_dir = cand_path
                break

        self.annotation_dir = os.path.join(self.dataset_dir, 'annotation')

        # Auto-detect image folder (Image vs cropped_images vs images)
        for cand in ['cropped_images', 'Image', 'images', 'image']:
            cand_p = os.path.join(self.dataset_dir, cand)
            if os.path.isdir(cand_p):
                self.cropped_images_dir = cand_p
                break
        else:
            self.cropped_images_dir = os.path.join(self.dataset_dir, 'Image')

        required_files = [self.dataset_dir, self.annotation_dir]
        self.check_before_run(required_files)

        train = self._process_dir(self.annotation_dir, self.cropped_images_dir, is_train=True)
        query, gallery = self._process_dir(self.annotation_dir, self.cropped_images_dir, is_train=False)

        if verbose:
            print("=> CUHK-SYSU loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _process_dir(self, ann_dir, img_dir, is_train=True):
        import scipy.io as sio
        dataset = []
        if is_train:
            mat_path = os.path.join(ann_dir, 'train.mat')
            if not os.path.exists(mat_path):
                # Fallback check inside ann_dir parent
                for root_d, _, files in os.walk(os.path.dirname(ann_dir)):
                    if 'train.mat' in files:
                        mat_path = os.path.join(root_d, 'train.mat')
                        break
            if not os.path.exists(mat_path):
                return []
            mat = sio.loadmat(mat_path)
            train_data = mat['train'][0]
            for item in train_data:
                pid = int(item['id'][0][0])
                scenes = item['scenes'][0]
                for scene in scenes:
                    img_name = str(scene['im_name'][0][0])
                    img_path = os.path.join(img_dir, img_name)
                    dataset.append((img_path, pid, 0, 1))
        else:
            return [], []

        return dataset
