import os
import glob
import re
from .bases import BaseImageDataset

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
        
        # Auto-detect MSMT17 folder case-insensitively
        possible_dirs = ['MSMT17_V2', 'MSMT17_V1', 'MSMT17', 'msmt17']
        main_dir = None
        
        for p_dir in possible_dirs:
            if os.path.exists(os.path.join(root, p_dir)):
                main_dir = p_dir
                break
                
        if main_dir is None and os.path.exists(root):
            for d in os.listdir(root):
                if 'msmt17' in d.lower() and os.path.isdir(os.path.join(root, d)):
                    main_dir = d
                    break

        if main_dir is None:
            main_dir = 'MSMT17_V2'

        self.dataset_dir = os.path.join(root, main_dir)
        self.list_train_path = os.path.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = os.path.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path = os.path.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = os.path.join(self.dataset_dir, 'list_gallery.txt')

        # Auto-detect train/test image folder
        mask_train_cand = ['mask_train_v2', 'train_v2', 'train', 'mask_train', 'images_train']
        for cand in mask_train_cand:
            if os.path.exists(os.path.join(self.dataset_dir, cand)):
                self.mask_train_dir = os.path.join(self.dataset_dir, cand)
                break
        else:
            self.mask_train_dir = os.path.join(self.dataset_dir, 'mask_train_v2')

        mask_test_cand = ['mask_test_v2', 'test_v2', 'test', 'mask_test', 'images_test']
        for cand in mask_test_cand:
            if os.path.exists(os.path.join(self.dataset_dir, cand)):
                self.mask_test_dir = os.path.join(self.dataset_dir, cand)
                break
        else:
            self.mask_test_dir = os.path.join(self.dataset_dir, 'mask_test_v2')

        train = self._process_dir(self.list_train_path, self.mask_train_dir, relabel=True)
        val = self._process_dir(self.list_val_path, self.mask_train_dir, relabel=False)
        query = self._process_dir(self.list_query_path, self.mask_test_dir, relabel=False)
        gallery = self._process_dir(self.list_gallery_path, self.mask_test_dir, relabel=False)

        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _process_dir(self, list_path, img_dir, relabel=False):
        if not os.path.exists(list_path):
            return []

        with open(list_path, 'r') as f:
            lines = f.readlines()

        dataset = []
        pid_container = set()
        for line in lines:
            line_str = line.strip()
            if not line_str: continue
            parts = line_str.split()
            img_rel_path = parts[0]
            pid = int(parts[1])
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        for line in lines:
            line_str = line.strip()
            if not line_str: continue
            parts = line_str.split()
            img_rel_path = parts[0]
            pid = int(parts[1])
            if relabel:
                pid = pid2label[pid]
            
            # Extract camera ID from filename
            fname = os.path.basename(img_rel_path)
            camid = 0
            try:
                camid = int(fname.split('_')[2])
            except Exception:
                pass
                
            full_img_path = os.path.join(img_dir, img_rel_path)
            if not os.path.exists(full_img_path):
                # Fallback check directly in img_dir/fname
                full_img_path = os.path.join(img_dir, fname)
                
            if os.path.exists(full_img_path):
                dataset.append((full_img_path, pid, camid, 0))

        return dataset
