import os
import glob
import re
from .bases import BaseImageDataset
from . import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class MSMT17(BaseImageDataset):
    """
    MSMT17
    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.
    URL: http://www.pkuvmc.com/publications/msmt17.html
    """
    dataset_dir = 'MSMT17_V2'

    def __init__(self, root='', verbose=True, pid_begin=0, combineall=False, **kwargs):
        super(MSMT17, self).__init__()

        self.dataset_dir = self._find_msmt17_root(root)

        self.list_train_path   = os.path.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path     = os.path.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path   = os.path.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = os.path.join(self.dataset_dir, 'list_gallery.txt')

        # Detect train/test subfolders
        train_cands = ['mask_train_v2', 'train_v2', 'train', 'mask_train', 'images_train']
        test_cands  = ['mask_test_v2',  'test_v2',  'test',  'mask_test',  'images_test']

        self.mask_train_dir = self._pick_existing(self.dataset_dir, train_cands, 'mask_train_v2')
        self.mask_test_dir  = self._pick_existing(self.dataset_dir, test_cands,  'mask_test_v2')

        train   = self._process_dir(self.list_train_path,   self.mask_train_dir, relabel=True)
        val     = self._process_dir(self.list_val_path,     self.mask_train_dir, relabel=False)
        query   = self._process_dir(self.list_query_path,   self.mask_test_dir,  relabel=False)
        gallery = self._process_dir(self.list_gallery_path, self.mask_test_dir,  relabel=False)

        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train   = train
        self.query   = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _find_msmt17_root(self, root):
        dir_candidates = ['MSMT17_V2', 'MSMT17_V1', 'MSMT17', 'msmt17', 'MSMT17_V3']

        # 1. Flat check
        for cand in dir_candidates:
            p = os.path.join(root, cand)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, 'list_train.txt')):
                return p

        # 2. Check 1-level nested (e.g. data/MSMT17_V2/MSMT17_V1)
        for cand in dir_candidates:
            outer = os.path.join(root, cand)
            if os.path.isdir(outer):
                for sub in os.listdir(outer):
                    inner = os.path.join(outer, sub)
                    if os.path.isdir(inner) and os.path.exists(os.path.join(inner, 'list_train.txt')):
                        return inner

        # 3. Check root itself
        if os.path.exists(os.path.join(root, 'list_train.txt')):
            return root

        return os.path.join(root, 'MSMT17_V2')

    def _pick_existing(self, base, candidates, default):
        for c in candidates:
            p = os.path.join(base, c)
            if os.path.isdir(p):
                try:
                    if len(os.listdir(p)) > 0:
                        return p
                except Exception:
                    return p
        return os.path.join(base, default)

    def _process_dir(self, list_path, img_dir, relabel=False):
        if not os.path.exists(list_path):
            return []

        with open(list_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            return []

        # Determine path structure ONCE across first 50 lines (zero disk lag on Drive/Kaggle)
        use_basename = False
        for test_line in lines[:50]:
            parts = test_line.strip().split()
            if not parts: continue
            test_rel = parts[0]
            test_base = os.path.basename(test_rel)
            if os.path.exists(os.path.join(img_dir, test_base)):
                use_basename = True
                break
            elif os.path.exists(os.path.join(img_dir, test_rel)):
                use_basename = False
                break

        pid_container = set()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                pid_container.add(int(parts[1]))

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            img_rel_path = parts[0]
            pid = int(parts[1])
            if relabel:
                pid = pid2label[pid]

            fname = os.path.basename(img_rel_path) if use_basename else img_rel_path
            camid = 0
            try:
                camid = int(os.path.basename(img_rel_path).split('_')[2])
            except Exception:
                pass

            full_path = os.path.join(img_dir, fname)
            dataset.append((full_path, pid, camid, 0))

        return dataset
