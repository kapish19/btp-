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

        # ----------------------------------------------------------------
        # Step 1: Find the folder that actually contains list_train.txt
        # Works regardless of nesting depth or folder name variant
        # Searches: data/MSMT17_V2, data/MSMT17_V1, data/MSMT17,
        #           data/MSMT17_V2/MSMT17_V1, data/msmt17/MSMT17_V1, etc.
        # ----------------------------------------------------------------
        self.dataset_dir = self._find_msmt17_root(root)

        self.list_train_path  = os.path.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path    = os.path.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path  = os.path.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path= os.path.join(self.dataset_dir, 'list_gallery.txt')

        # ----------------------------------------------------------------
        # Step 2: Locate train / test image subfolders
        # ----------------------------------------------------------------
        train_cands = ['mask_train_v2', 'train_v2', 'train', 'mask_train', 'images_train']
        test_cands  = ['mask_test_v2',  'test_v2',  'test',  'mask_test',  'images_test']

        self.mask_train_dir = self._pick_existing(self.dataset_dir, train_cands, 'mask_train_v2')
        self.mask_test_dir  = self._pick_existing(self.dataset_dir, test_cands,  'mask_test_v2')

        train   = self._process_dir(self.list_train_path,   self.mask_train_dir, relabel=True)
        val     = self._process_dir(self.list_val_path,     self.mask_train_dir, relabel=False)
        query   = self._process_dir(self.list_query_path,   self.mask_test_dir,  relabel=False)
        gallery = self._process_dir(self.list_gallery_path, self.mask_test_dir,  relabel=False)

        if verbose:
            print("=> MSMT17 loaded  [dir: {}]".format(self.dataset_dir))
            self.print_dataset_statistics(train, query, gallery)

        self.train   = train
        self.query   = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _find_msmt17_root(self, root):
        """
        Searches up to 3 levels deep for a directory that contains list_train.txt.
        Falls back to data/MSMT17_V2 if nothing is found (graceful degradation).
        """
        # Direct candidates under root
        dir_candidates = ['MSMT17_V2', 'MSMT17_V1', 'MSMT17', 'msmt17', 'MSMT17_V3']

        # 1. Check flat: root/<candidate>
        for cand in dir_candidates:
            path = os.path.join(root, cand)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, 'list_train.txt')):
                return path

        # 2. Check one level deeper: root/<candidate>/<sub>
        for cand in dir_candidates:
            outer = os.path.join(root, cand)
            if not os.path.isdir(outer):
                continue
            for sub in os.listdir(outer):
                inner = os.path.join(outer, sub)
                if os.path.isdir(inner) and os.path.exists(os.path.join(inner, 'list_train.txt')):
                    return inner

        # 3. Glob search within root (handles any naming)
        matches = glob.glob(os.path.join(root, '**/list_train.txt'), recursive=True)
        if matches:
            # Take the shallowest match
            matches.sort(key=lambda x: x.count(os.sep))
            return os.path.dirname(matches[0])

        # 4. Fallback — return default path even if it doesn't exist
        # (will produce empty dataset with clear path info)
        fallback = os.path.join(root, 'MSMT17_V2')
        print(f"⚠️  MSMT17: list_train.txt not found under '{root}'. "
              f"Searched: {dir_candidates}. Defaulting to {fallback}")
        return fallback

    def _pick_existing(self, base, candidates, default):
        for c in candidates:
            p = os.path.join(base, c)
            if os.path.isdir(p):
                return p
        return os.path.join(base, default)

    def _process_dir(self, list_path, img_dir, relabel=False):
        if not os.path.exists(list_path):
            return []

        with open(list_path, 'r') as f:
            lines = f.readlines()

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

            fname = os.path.basename(img_rel_path)
            camid = 0
            try:
                camid = int(fname.split('_')[2])
            except Exception:
                pass

            # Try relative path first, then just filename
            full_path = os.path.join(img_dir, img_rel_path)
            if not os.path.exists(full_path):
                full_path = os.path.join(img_dir, fname)

            if os.path.exists(full_path):
                dataset.append((full_path, pid, camid, 0))

        return dataset
