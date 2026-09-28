# encoding: utf-8
"""
Occluded-DukeMTMC dataset loader.
Zero-shot test only — no training split.
Same filename format as DukeMTMC-reID: NNNN_cN_fNNNNNNN.jpg
Structure: bounding_box_train/, bounding_box_test/ (gallery), query/
"""

import glob
import os.path as osp
import re
from .bases import ImageDataset, auto_locate_dataset_dir
from . import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class OccludedDuke(ImageDataset):
    """Occluded-DukeMTMC dataset (test-only for zero-shot evaluation).

    Reference:
        Miao et al. Pose-Guided Feature Alignment for Occluded Person
        Re-Identification. ICCV 2019.

    Dataset statistics:
        - identities: 1110 (702 train, 519 test)
        - All query images are occluded.
        - ~10% gallery images are occluded.
    """
    dataset_dir = 'Occluded_Duke'
    dataset_name = "occluded_duke"

    def __init__(self, root='', **kwargs):
        self.root = root
        self.dataset_dir = auto_locate_dataset_dir(
            'Occluded_Duke', root=root,
            markers=['Occluded_Duke', 'DukeMTMC-reID']
        )

        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        required_files = [
            self.dataset_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        # Build pid/cam dicts from gallery (test set)
        self.pid_dict_test, self.cam_dict_test = self._build_dict(self.gallery_dir)

        # This dataset is test-only, but we still parse train for completeness
        train = self._process_dir(self.train_dir, is_train=True)
        query = self._process_dir(self.query_dir, is_train=False)
        gallery = self._process_dir(self.gallery_dir, is_train=False)

        super(OccludedDuke, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path, is_train=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        # First pass: collect pids and cams
        pid_set = set()
        cam_set = set()
        for img_path in img_paths:
            match = pattern.search(osp.basename(img_path))
            if match is None:
                continue
            pid, camid = int(match.group(1)), int(match.group(2))
            if pid == -1:
                continue
            pid_set.add(pid)
            cam_set.add(camid)

        pids = sorted(list(pid_set))
        cams = sorted(list(cam_set))

        if is_train:
            pid_dict = dict([(p, i) for i, p in enumerate(pids)])
            cam_dict = dict([(p, i) for i, p in enumerate(cams)])
        else:
            pid_dict = self.pid_dict_test
            cam_dict = self.cam_dict_test

        data = []
        for img_path in img_paths:
            match = pattern.search(osp.basename(img_path))
            if match is None:
                continue
            pid, camid = int(match.group(1)), int(match.group(2))
            if pid == -1:
                continue
            if is_train:
                pid_mapped = self.dataset_name + "_" + str(pid_dict[pid])
                camid_mapped = self.dataset_name + "_" + str(cam_dict[camid])
                data.append((img_path, pid_mapped, camid_mapped, 'OccludedDuke'))
            else:
                pid_mapped = pid_dict[pid]
                camid_mapped = cam_dict[camid]
                data.append((img_path, int(pid_mapped), int(camid_mapped), 'OccludedDuke'))

        return data

    def _build_dict(self, dir_path):
        """Build pid->index and cam->index dicts from a directory."""
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_set = set()
        cam_set = set()
        for img_path in img_paths:
            match = pattern.search(osp.basename(img_path))
            if match is None:
                continue
            pid, camid = int(match.group(1)), int(match.group(2))
            if pid == -1:
                continue
            pid_set.add(pid)
            cam_set.add(camid)

        pids = sorted(list(pid_set))
        cams = sorted(list(cam_set))

        pid_dict = dict([(p, i) for i, p in enumerate(pids)])
        cam_dict = dict([(p, i) for i, p in enumerate(cams)])

        return pid_dict, cam_dict
