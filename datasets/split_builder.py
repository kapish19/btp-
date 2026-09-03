"""
Split builder for CLIP-FGDI Protocol 2 and Occluded-Duke benchmarks.

Protocol 2 (leave-one-out):
    Given held_out_domain in {Market, MSMT17, cuhk_sysu, cuhk03},
    train on the other 3, test zero-shot on the held-out one.
    All 4 rotations via config change alone.

Occluded-Duke (all-4-source):
    Train on Market + MSMT17 + CUHK-SYSU + CUHK03,
    test zero-shot on Occluded-DukeMTMC.
"""

from collections import defaultdict
from . import DATASET_REGISTRY
from cfgs.cfgs import ALL_SOURCE_DATASETS, DATASET_CLASSES


def subsample_by_pid(items, max_pids):
    """Keep only the first `max_pids` unique PIDs and all their images.

    Args:
        items: list of (img_path, pid, camid, domain) tuples
        max_pids: max number of unique PIDs to keep

    Returns:
        Filtered list of items.
    """
    if max_pids <= 0:
        return items

    pid_set = set()
    pid_order = []
    for item in items:
        pid = item[1]
        if pid not in pid_set:
            pid_set.add(pid)
            pid_order.append(pid)

    selected_pids = set(pid_order[:max_pids])
    return [item for item in items if item[1] in selected_pids]


def build_protocol2_split(held_out_domain, data_path, debug_subset_ids=0, combine_all=False):
    """
    Build Protocol 2 leave-one-out split.

    Args:
        held_out_domain: str, one of {Market, MSMT17, cuhk_sysu, cuhk03}
        data_path: str, root path where all datasets live
        debug_subset_ids: int, if >0, subsample this many PIDs per source dataset
        combine_all: bool, whether to combine train+query+gallery for source datasets

    Returns:
        train_items: list of (img_path, pid, camid, domain) for all source datasets combined
        test_datasets: dict of {name: dataset_obj} for the held-out domain
        source_classes: list of int, class counts for each source dataset
        source_datasets: list of str, source dataset names
    """
    assert held_out_domain in ALL_SOURCE_DATASETS, \
        f"held_out_domain must be one of {ALL_SOURCE_DATASETS}, got {held_out_domain}"

    source_datasets = [d for d in ALL_SOURCE_DATASETS if d != held_out_domain]
    source_classes = [DATASET_CLASSES[d] for d in source_datasets]

    # Build training data from source datasets
    train_items = []
    for d in source_datasets:
        dataset = DATASET_REGISTRY.get(d)(root=data_path, combineall=combine_all)
        items = dataset.train
        if debug_subset_ids > 0:
            items = subsample_by_pid(items, debug_subset_ids)
        train_items.extend(items)

    # Build test data from held-out dataset
    test_dataset = DATASET_REGISTRY.get(held_out_domain)(root=data_path)
    test_datasets = {held_out_domain: test_dataset}

    return train_items, test_datasets, source_classes, source_datasets


def build_occluded_duke_split(data_path, debug_subset_ids=0, combine_all=False):
    """
    Build all-4-source -> Occluded-DukeMTMC split.

    Args:
        data_path: str, root path where all datasets live
        debug_subset_ids: int, if >0, subsample this many PIDs per source dataset
        combine_all: bool, whether to combine train+query+gallery for source datasets

    Returns:
        train_items: list of (img_path, pid, camid, domain) for all 4 source datasets
        test_datasets: dict of {name: dataset_obj} for Occluded-DukeMTMC
        source_classes: list of int, class counts for each source dataset
        source_datasets: list of str, source dataset names
    """
    source_datasets = list(ALL_SOURCE_DATASETS)
    source_classes = [DATASET_CLASSES[d] for d in source_datasets]

    # Build training data from all 4 source datasets
    train_items = []
    for d in source_datasets:
        dataset = DATASET_REGISTRY.get(d)(root=data_path, combineall=combine_all)
        items = dataset.train
        if debug_subset_ids > 0:
            items = subsample_by_pid(items, debug_subset_ids)
        train_items.extend(items)

    # Build test data from Occluded-DukeMTMC
    test_dataset = DATASET_REGISTRY.get("OccludedDuke")(root=data_path)
    test_datasets = {"OccludedDuke": test_dataset}

    return train_items, test_datasets, source_classes, source_datasets


def get_actual_class_counts(train_items, source_datasets):
    """
    After subsampling, compute actual class counts per source dataset.
    This is needed when debug_subset_ids > 0, since the class counts
    will be smaller than the full dataset counts.

    Args:
        train_items: list of (img_path, pid, camid, domain) tuples
        source_datasets: list of dataset name strings

    Returns:
        list of int, actual class counts per source dataset (in order)
    """
    # Group PIDs by domain
    domain_pids = defaultdict(set)
    for item in train_items:
        domain = item[3]
        pid = item[1]
        domain_pids[domain].add(pid)

    # Map dataset names to their domain strings used in data items
    # The domain string in items is set by each dataset loader
    domain_name_map = {
        "Market": "Market",
        "MSMT17": "MSMT17",
        "cuhk_sysu": "cuhk_sysu",
        "cuhk03": "cuhk03",
    }

    counts = []
    for d in source_datasets:
        domain_key = domain_name_map.get(d, d)
        counts.append(len(domain_pids.get(domain_key, set())))

    return counts
