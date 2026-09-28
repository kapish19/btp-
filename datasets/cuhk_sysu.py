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
        candidates = [
            os.path.join(root, 'cuhk_sysu'),
            os.path.join(root, 'cuhk-sysu'),
            os.path.join(root, 'cuhk_sysu', 'cuhk_sysu'),
            'cuhk_sysu',
            './cuhk_sysu',
            '../cuhk_sysu',
            '/content/btp_repo/cuhk_sysu',
            '/content/btp_repo/data/cuhk_sysu',
            '/content/drive/MyDrive/datasets/cuhk_sysu',
            '/content/drive/MyDrive/cuhk_sysu',
        ]
        self.dataset_dir = os.path.join(root, 'cuhk_sysu')
        for cand_path in candidates:
            if os.path.exists(os.path.join(cand_path, 'annotation')):
                self.dataset_dir = cand_path
                break

        self.annotation_dir = os.path.join(self.dataset_dir, 'annotation')

        # Auto-detect image folder (Image/SSM vs Image vs cropped_images)
        for cand in ['cropped_images', 'Image/SSM', 'Image', 'images', 'image']:
            cand_p = os.path.join(self.dataset_dir, cand)
            if os.path.isdir(cand_p):
                self.cropped_images_dir = cand_p
                break
        else:
            self.cropped_images_dir = os.path.join(self.dataset_dir, 'Image')

        required_files = [self.dataset_dir]
        self.check_before_run(required_files)

        train = self._process_dir(self.annotation_dir, self.cropped_images_dir, is_train=True)
        query, gallery = self._process_dir(self.annotation_dir, self.cropped_images_dir, is_train=False)

        if verbose:
            print("=> CUHK-SYSU loaded  [dir: {}]".format(self.dataset_dir))
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
            mat_path = None
            # Search case-insensitively for *train*.mat anywhere inside dataset_dir
            for root_d, _, files in os.walk(self.dataset_dir):
                for f in files:
                    if 'train' in f.lower() and f.endswith('.mat'):
                        mat_path = os.path.join(root_d, f)
                        break
                if mat_path:
                    break

            if not mat_path or not os.path.exists(mat_path):
                print(f"⚠️ CUHK-SYSU: No *train*.mat found in {self.dataset_dir}")
                return []

            print(f"  ↳ CUHK-SYSU: loading annotations from {mat_path}")
            mat = sio.loadmat(mat_path)
            
            # Handle key variations ('train' vs 'Train')
            key = 'train' if 'train' in mat else 'Train' if 'Train' in mat else list(mat.keys())[-1]
            train_data = mat[key][0]

            # Verify and resolve which image folder contains the scene images
            if len(train_data) > 0 and len(train_data[0]['scenes'][0]) > 0:
                sample_name = str(train_data[0]['scenes'][0][0]['im_name'][0][0])
                if not os.path.exists(os.path.join(img_dir, sample_name)):
                    for cand in [os.path.join(self.dataset_dir, 'Image', 'SSM'),
                                 os.path.join(self.dataset_dir, 'Image'),
                                 os.path.join(self.dataset_dir, 'cropped_images')]:
                        if os.path.exists(os.path.join(cand, sample_name)):
                            img_dir = cand
                            self.cropped_images_dir = cand
                            break

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
