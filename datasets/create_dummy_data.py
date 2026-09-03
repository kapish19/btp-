"""
Generate synthetic dummy datasets that mimic the exact directory structure and
filename conventions of each real dataset, for local smoke testing without
needing the actual data downloads.

Creates tiny subsets (a few PIDs, a few images each) in the same format
that the existing dataset loaders expect.
"""

import os
import numpy as np
from PIL import Image


def _create_image(path, size=(128, 256)):
    """Create a random RGB image and save as JPEG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(img).save(path)


def create_dummy_market1501(root, num_train_pids=8, num_test_pids=4, imgs_per_pid=4):
    """
    Market-1501 format:
      Market-1501-v15.09.15/bounding_box_train/PPPP_cC_sS_FFFFFF.jpg
      Market-1501-v15.09.15/query/PPPP_cC_sS_FFFFFF.jpg
      Market-1501-v15.09.15/bounding_box_test/PPPP_cC_sS_FFFFFF.jpg
    Filename pattern: {pid:04d}_c{camid}s{seqid}_{frame:06d}.jpg
    """
    base = os.path.join(root, 'Market-1501-v15.09.15')
    train_dir = os.path.join(base, 'bounding_box_train')
    query_dir = os.path.join(base, 'query')
    gallery_dir = os.path.join(base, 'bounding_box_test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(gallery_dir, exist_ok=True)

    # Train PIDs: 1 to num_train_pids (0 is background/junk)
    for pid in range(1, num_train_pids + 1):
        for img_idx in range(imgs_per_pid):
            camid = (img_idx % 6) + 1  # cameras 1-6
            fname = f'{pid:04d}_c{camid}s1_{img_idx:06d}.jpg'
            _create_image(os.path.join(train_dir, fname))

    # Test PIDs: start after train PIDs
    for pid in range(num_train_pids + 1, num_train_pids + num_test_pids + 1):
        for img_idx in range(imgs_per_pid):
            camid = (img_idx % 6) + 1
            fname = f'{pid:04d}_c{camid}s1_{img_idx:06d}.jpg'
            # Put first image as query, rest as gallery
            if img_idx == 0:
                _create_image(os.path.join(query_dir, fname))
            else:
                _create_image(os.path.join(gallery_dir, fname))
        # Also put at least one gallery image per query pid with same cam (or just ensure cam 1 is in gallery)
        camid = 1
        fname = f'{pid:04d}_c{camid}s1_{imgs_per_pid:06d}.jpg'
        _create_image(os.path.join(gallery_dir, fname))


def create_dummy_msmt17(root, num_train_pids=8, num_test_pids=4, imgs_per_pid=4):
    """
    MSMT17_V2 format:
      MSMT17_V2/mask_train_v2/PPPP_CCC_FFFFFFFF.jpg
      MSMT17_V2/mask_test_v2/PPPP_CCC_FFFFFFFF.jpg
      MSMT17_V2/list_train.txt  (imgpath pid)
      MSMT17_V2/list_val.txt
      MSMT17_V2/list_query.txt
      MSMT17_V2/list_gallery.txt
    """
    base = os.path.join(root, 'MSMT17_V2')
    train_dir = os.path.join(base, 'mask_train_v2')
    test_dir = os.path.join(base, 'mask_test_v2')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_lines = []
    val_lines = []
    query_lines = []
    gallery_lines = []

    # Train
    for pid in range(num_train_pids):
        for img_idx in range(imgs_per_pid):
            camid = (img_idx % 15) + 1
            fname = f'{pid:04d}_{img_idx:03d}_{camid:02d}_{img_idx:08d}.jpg'
            _create_image(os.path.join(train_dir, fname))
            train_lines.append(f'{fname} {pid}\n')

    # Val (small subset)
    for pid in range(num_train_pids, num_train_pids + 2):
        for img_idx in range(2):
            camid = (img_idx % 15) + 1
            fname = f'{pid:04d}_{img_idx:03d}_{camid:02d}_{img_idx:08d}.jpg'
            _create_image(os.path.join(train_dir, fname))
            val_lines.append(f'{fname} {pid}\n')

    # Test (query + gallery)
    for pid in range(num_test_pids):
        for img_idx in range(imgs_per_pid):
            camid = (img_idx % 15) + 1
            fname = f'{pid:04d}_{img_idx:03d}_{camid:02d}_{img_idx:08d}.jpg'
            _create_image(os.path.join(test_dir, fname))
            if img_idx == 0:
                query_lines.append(f'{fname} {pid}\n')
            else:
                gallery_lines.append(f'{fname} {pid}\n')
        # Extra gallery
        camid = 1
        fname = f'{pid:04d}_{imgs_per_pid:03d}_{camid:02d}_{imgs_per_pid:08d}.jpg'
        _create_image(os.path.join(test_dir, fname))
        gallery_lines.append(f'{fname} {pid}\n')

    with open(os.path.join(base, 'list_train.txt'), 'w') as f:
        f.writelines(train_lines)
    with open(os.path.join(base, 'list_val.txt'), 'w') as f:
        f.writelines(val_lines)
    with open(os.path.join(base, 'list_query.txt'), 'w') as f:
        f.writelines(query_lines)
    with open(os.path.join(base, 'list_gallery.txt'), 'w') as f:
        f.writelines(gallery_lines)


def create_dummy_cuhk_sysu(root, num_pids=10, imgs_per_pid=3):
    """
    CUHK-SYSU format:
      cuhk_sysu/cropped_images/p{pid}_s{scene}.jpg
    """
    data_dir = os.path.join(root, 'cuhk_sysu', 'cropped_images')
    os.makedirs(data_dir, exist_ok=True)

    for pid in range(1, num_pids + 1):
        for scene in range(1, imgs_per_pid + 1):
            fname = f'p{pid:04d}_s{scene}.jpg'
            _create_image(os.path.join(data_dir, fname))


def create_dummy_cuhk03(root, num_train_pids=8, num_test_pids=4, imgs_per_pid=4):
    """
    CUHK03 with new protocol (CUHK03-NP).
    For smoke testing, we create pre-processed JSON splits directly,
    bypassing the .mat file preprocessing.

    The cuhk03.py loader does: img_path.split('datasets/')[-1]
    which strips everything up to and including 'datasets/'.
    For train: the result is used as-is (relative path).
    For test: root is prepended.

    So we store paths like: '<root>/datasets/cuhk03/images_labeled/X.png'
    After splitting: 'cuhk03/images_labeled/X.png'
    For test: root + 'cuhk03/images_labeled/X.png' -> absolute path

    Structure:
      cuhk03/images_labeled/CAMPID_PID_VIEWID_IMGID.png
      cuhk03/splits_new_labeled.json
    """
    import json

    base = os.path.join(root, 'cuhk03')
    img_dir = os.path.join(base, 'images_labeled')
    det_dir = os.path.join(base, 'images_detected')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(det_dir, exist_ok=True)

    # Also create the cuhk03_release dir and dummy mat (to pass check_before_run)
    release_dir = os.path.join(base, 'cuhk03_release')
    os.makedirs(release_dir, exist_ok=True)

    # Create a 'datasets' symlink/directory structure so paths resolve
    # The loader uses path.split('datasets/')[-1] which gives 'cuhk03/images_labeled/...'
    # For train items, this relative path needs to resolve from cwd
    # For test items, root is prepended: root + 'cuhk03/images_labeled/...'

    train_data = []
    query_data = []
    gallery_data = []

    # We'll use a path format: datasets/cuhk03/images_labeled/X.png
    # so split('datasets/')[-1] = 'cuhk03/images_labeled/X.png'
    # and root + result = abs path (for test), or we put images there (for train)

    # Train identities
    for pid in range(num_train_pids):
        for img_idx in range(imgs_per_pid):
            campid = (img_idx // 2) + 1
            viewid = (img_idx % 2) + 1
            img_name = f'{campid:01d}_{pid + 1:03d}_{viewid:01d}_{img_idx + 1:02d}.png'
            img_path = os.path.join(img_dir, img_name)
            _create_image(img_path)
            # also create for detected just in case
            _create_image(os.path.join(det_dir, img_name))
            camid = viewid - 1  # 0-based
            # Use absolute path with 'datasets/' marker for the JSON
            stored_path = os.path.join(root, 'datasets', 'cuhk03', 'images_labeled', img_name)
            train_data.append([stored_path, pid, camid, 'cuhk03'])

    # Test identities
    for pid in range(num_train_pids, num_train_pids + num_test_pids):
        for img_idx in range(imgs_per_pid):
            campid = (img_idx // 2) + 1
            viewid = (img_idx % 2) + 1
            img_name = f'{campid:01d}_{pid + 1:03d}_{viewid:01d}_{img_idx + 1:02d}.png'
            img_path = os.path.join(img_dir, img_name)
            _create_image(img_path)
            _create_image(os.path.join(det_dir, img_name))
            camid = viewid - 1
            stored_path = os.path.join(root, 'datasets', 'cuhk03', 'images_labeled', img_name)
            if img_idx == 0:
                query_data.append([stored_path, pid, camid, 'cuhk03'])
            else:
                gallery_data.append([stored_path, pid, camid, 'cuhk03'])

    split = [{
        'train': train_data,
        'query': query_data,
        'gallery': gallery_data,
        'num_train_pids': num_train_pids,
        'num_train_imgs': len(train_data),
        'num_query_pids': num_test_pids,
        'num_query_imgs': len(query_data),
        'num_gallery_pids': num_test_pids,
        'num_gallery_imgs': len(gallery_data),
    }]

    # Write the splits JSON files
    for split_name in ['splits_new_labeled.json', 'splits_new_detected.json',
                       'splits_classic_labeled.json', 'splits_classic_detected.json']:
        with open(os.path.join(base, split_name), 'w') as f:
            json.dump(split, f)

    # Create dummy .mat files (just empty files to pass existence check)
    for mat_name in ['cuhk03_new_protocol_config_detected.mat',
                     'cuhk03_new_protocol_config_labeled.mat']:
        mat_path = os.path.join(base, mat_name)
        if not os.path.exists(mat_path):
            with open(mat_path, 'w') as f:
                f.write('')

    # Create cuhk-03.mat in release dir
    mat_path = os.path.join(release_dir, 'cuhk-03.mat')
    if not os.path.exists(mat_path):
        with open(mat_path, 'w') as f:
            f.write('')

    # Create a symlink so stripped paths resolve:
    # 'cuhk03/images_labeled/X.png' needs to resolve from cwd AND
    # root + 'cuhk03/images_labeled/X.png' needs to work
    # Since root already has cuhk03/images_labeled, the root + path works.
    # For train (no root prepend), we need images at 'cuhk03/images_labeled/' relative to cwd.
    # The simplest fix: our smoke test will set cwd to data_path, or we'll
    # patch the reader. Actually, looking at the loader more carefully:
    # Train paths get stored as-is after stripping 'datasets/' prefix, e.g.
    # 'cuhk03/images_labeled/X.png' — this is relative. The CommDataset __getitem__
    # calls read_image(img_path) which will try to open this relative path.
    # For the smoke test, we'll create a 'datasets/cuhk03/images_labeled' dir structure
    # under the root, and also create a top-level symlink.
    datasets_img_dir = os.path.join(root, 'datasets', 'cuhk03', 'images_labeled')
    os.makedirs(datasets_img_dir, exist_ok=True)
    # Copy images to the datasets path too
    import shutil
    for fname in os.listdir(img_dir):
        src = os.path.join(img_dir, fname)
        dst = os.path.join(datasets_img_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)



def create_dummy_occluded_duke(root, num_train_pids=8, num_test_pids=4, imgs_per_pid=4):
    """
    Occluded-DukeMTMC format (same as DukeMTMC-reID):
      Occluded_Duke/bounding_box_train/PPPP_cN_fNNNNNNN.jpg
      Occluded_Duke/query/PPPP_cN_fNNNNNNN.jpg
      Occluded_Duke/bounding_box_test/PPPP_cN_fNNNNNNN.jpg
    """
    base = os.path.join(root, 'Occluded_Duke')
    train_dir = os.path.join(base, 'bounding_box_train')
    query_dir = os.path.join(base, 'query')
    gallery_dir = os.path.join(base, 'bounding_box_test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(gallery_dir, exist_ok=True)

    # Train
    for pid in range(1, num_train_pids + 1):
        for img_idx in range(imgs_per_pid):
            camid = (img_idx % 8) + 1
            frame = img_idx * 100 + 1
            fname = f'{pid:04d}_c{camid}_f{frame:07d}.jpg'
            _create_image(os.path.join(train_dir, fname))

    # Test
    for pid in range(num_train_pids + 1, num_train_pids + num_test_pids + 1):
        for img_idx in range(imgs_per_pid):
            camid = (img_idx % 8) + 1
            frame = img_idx * 100 + 1
            fname = f'{pid:04d}_c{camid}_f{frame:07d}.jpg'
            if img_idx == 0:
                _create_image(os.path.join(query_dir, fname))
            else:
                _create_image(os.path.join(gallery_dir, fname))
        # Extra gallery
        camid = 1
        frame = imgs_per_pid * 100 + 1
        fname = f'{pid:04d}_c{camid}_f{frame:07d}.jpg'
        _create_image(os.path.join(gallery_dir, fname))


def create_all_dummy_datasets(root, num_pids=8, imgs_per_pid=4):
    """Create all dummy datasets under root for smoke testing."""
    print(f"Creating dummy datasets under {root}...")
    create_dummy_market1501(root, num_train_pids=num_pids, num_test_pids=max(4, num_pids // 2),
                            imgs_per_pid=imgs_per_pid)
    print("  ✓ Market-1501")

    create_dummy_msmt17(root, num_train_pids=num_pids, num_test_pids=max(4, num_pids // 2),
                        imgs_per_pid=imgs_per_pid)
    print("  ✓ MSMT17")

    create_dummy_cuhk_sysu(root, num_pids=num_pids, imgs_per_pid=imgs_per_pid)
    print("  ✓ CUHK-SYSU")

    create_dummy_cuhk03(root, num_train_pids=num_pids, num_test_pids=max(4, num_pids // 2),
                        imgs_per_pid=imgs_per_pid)
    print("  ✓ CUHK03")

    create_dummy_occluded_duke(root, num_train_pids=num_pids, num_test_pids=max(4, num_pids // 2),
                               imgs_per_pid=imgs_per_pid)
    print("  ✓ Occluded-DukeMTMC")
    print("Done.")


if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else 'data'
    create_all_dummy_datasets(root)
