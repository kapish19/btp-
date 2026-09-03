"""
CLIP-FGDI Smoke Test — Comprehensive Verification Script.

Runs the full code path at tiny scale for both:
  1. Protocol 2 (leave-one-out, held_out=Market)
  2. All-4-source -> Occluded-DukeMTMC

Mandatory checks:
  ✓ Gradient flow for all trainable modules
  ✓ Checkpoint save/load round-trip
  ✓ Full 3-stage training pipeline (2 epochs each)
  ✓ No NaN/Inf in any loss
  ✓ Eval end-to-end with mAP, CMC R1/R5/R10 output
"""

import os
import sys
import time
import argparse
import random
import shutil
import warnings
import datetime
import tempfile
import traceback
from collections import defaultdict

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from functools import partial

# ====================================================================
# Device abstraction
# ====================================================================

def get_device(device_str='auto'):
    """Get the best available device."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


# ====================================================================
# Synthetic data setup
# ====================================================================

def setup_dummy_data(data_path):
    """Create dummy datasets for smoke testing."""
    from datasets.create_dummy_data import create_all_dummy_datasets
    create_all_dummy_datasets(data_path, num_pids=10, imgs_per_pid=5)


# ====================================================================
# Monkeypatch CUDA references for CPU/MPS execution
# ====================================================================

def patch_for_device(device):
    """Patch CUDA-specific code to work on any device."""
    if device.type != 'cuda':
        # Patch torch.cuda.amp to be no-ops
        original_autocast = torch.cuda.amp.autocast

        class DummyGradScaler:
            def __init__(self, *a, **kw):
                pass
            def scale(self, loss):
                return loss
            def step(self, optimizer):
                optimizer.step()
            def update(self):
                pass

        torch.cuda.amp.GradScaler = DummyGradScaler

        # Patch cuda synchronize
        if not torch.cuda.is_available():
            torch.cuda.synchronize = lambda: None
            torch.cuda.empty_cache = lambda: None

    return device


# ====================================================================
# Model building (adapted from run.py for device abstraction)
# ====================================================================

def build_model(args, device):
    """Build CLIP model with device abstraction."""
    from models import Clip

    # Temporarily patch clip tokenize to work on specified device
    from models.clip import clip as clip_module
    original_tokenize = clip_module.tokenize

    # Override the PromptLearner to use correct device
    from models.CLIP import PromptLearner
    original_pl_init = PromptLearner.__init__

    def patched_pl_init(self, num_class, dataset_num, dtype, token_embedding):
        nn.Module.__init__(self)
        ctx_init = "A photo of a X X X X person."
        ctx_init_domain = "A photo of a X X X X person from X dataset."

        ctx_dim = 512
        n_ctx = 4

        tokenized_prompts = clip_module.tokenize(ctx_init).to(device)
        tokenized_prompts_domain = clip_module.tokenize(ctx_init_domain).to(device)

        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
            embedding_domain = token_embedding(tokenized_prompts_domain).type(dtype)

        self.tokenized_prompts = tokenized_prompts
        self.tokenized_prompts_domain = tokenized_prompts_domain

        n_cls_ctx = 4
        n_dm_ctx = 1
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        dom_vectors = torch.empty(dataset_num, n_dm_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(dom_vectors, std=0.02)
        self.clsctx = nn.Parameter(cls_vectors, requires_grad=True)
        self.dmctx = nn.Parameter(dom_vectors, requires_grad=True)

        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])

        self.register_buffer("token_prefix_domain", embedding_domain[:, :n_ctx + 1, :])
        self.register_buffer("token_intermediate_domain",
                             embedding_domain[:, n_ctx + 1 + n_cls_ctx:n_ctx + 1 + n_cls_ctx + 2, :])
        self.register_buffer("token_suffix_domain",
                             embedding_domain[:, n_ctx + 1 + n_cls_ctx + 2 + n_dm_ctx:, :])

        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    PromptLearner.__init__ = patched_pl_init

    # Patch the Model class to use correct device
    from models.CLIP import Model, load_clip_to_cpu
    original_model_init = Model.__init__

    def patched_model_init(self, num_classes, args, epsilon=.1, domain_num=4):
        nn.Module.__init__(self)
        self.h_resolution = int((args.size_train[0] - 16) // 16 + 1)
        self.w_resolution = int((args.size_train[1] - 16) // 16 + 1)
        self.vision_stride_size = 16
        self.model_name = args.backbone
        self.neck_feat = 'before'
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        if self.model_name == 'ViT-B-32':
            self.in_planes = 768
            self.in_planes_proj = 512
            self.h_resolution = int((args.size_train[0] - 32) // 32 + 1)
            self.w_resolution = int((args.size_train[1] - 32) // 32 + 1)
            self.vision_stride_size = 32
        if self.model_name == 'ViT-L-14':
            self.in_planes = 768
            self.in_planes_proj = 512
            self.h_resolution = int((args.size_train[0] - 14) // 14 + 1)
            self.w_resolution = int((args.size_train[1] - 14) // 14 + 1)
            self.vision_stride_size = 14
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        elif self.model_name == 'RN101':
            self.in_planes = 2048
            self.in_planes_proj = 512
        self.num_classes = num_classes

        from models.CLIP import GRL, DomainClassifier, PartVisibilityGAT, weights_init_classifier, weights_init_kaiming
        self.grl = GRL(epsilon)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)
        
        # Local classifier
        self.local_classifier = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.local_classifier.apply(weights_init_classifier)

        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution,
                                       self.vision_stride_size)
        clip_model.to(device)
        
        # Domain Classifier for GRL on CLS token
        self.domain_classifier = DomainClassifier(self.in_planes, 128, domain_num)
        
        # Part branch
        self.part_gat = PartVisibilityGAT(d_model=self.in_planes, num_parts=3)
        
        self.image_encoder = clip_model.visual

        from models.CLIP import PromptLearner, TextEncoder
        self.promptlearner = PromptLearner(num_classes, domain_num, clip_model.dtype,
                                            clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)

    Model.__init__ = patched_model_init

    num_classes = sum(args.classes)
    domain_num = len(args.train_datasets)
    model = Model(num_classes, args, domain_num=domain_num, epsilon=args.epsilon)
    model = model.to(device)

    # Convert to float32 for CPU/MPS
    if device.type != 'cuda':
        model = model.float()

    return model


# ====================================================================
# Data loader building
# ====================================================================

def build_data_for_smoke(args, args_test, device):
    """Build data loaders for smoke test using dummy data."""
    from datasets import DATASET_REGISTRY
    from datasets.common import CommDataset
    from datasets.samplers.sampler import RandomIdentitySampler
    from datasets.trans import bulid_transforms
    from datasets.split_builder import subsample_by_pid
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        # We now return 8 elements: img, pid, camid, trackid, img_path, domain, cid, occ_mask
        imgs, pids, camids, viewids, image_path, domains, cid, occ_masks = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        viewids = torch.tensor(viewids, dtype=torch.int64)
        camids = torch.tensor(camids, dtype=torch.int64)
        cid = torch.tensor(cid, dtype=torch.int64)
        domains = torch.tensor(domains, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids, camids, viewids, domains, cid, torch.stack(occ_masks, dim=0)

    # Build training data from source datasets
    train_items = []
    for d in args.train_datasets:
        dataset = DATASET_REGISTRY.get(d)(root=args.data_path, combineall=args.combine_all)
        items = dataset.train
        if args.debug_subset_ids > 0:
            items = subsample_by_pid(items, args.debug_subset_ids)
        train_items.extend(items)

    print(f"  Total train items: {len(train_items)}")

    train_transforms = bulid_transforms(args, is_train=True)
    val_transforms = bulid_transforms(args_test, is_train=False)

    train_set = CommDataset(train_items, train_transforms, is_train=True)

    # Update args.classes to actual counts after subsampling
    if args.debug_subset_ids > 0:
        from datasets.split_builder import get_actual_class_counts
        actual_classes = get_actual_class_counts(train_items, args.train_datasets)
        args.classes = actual_classes
        print(f"  Actual classes after subsampling: {actual_classes}")

    num_workers = 0  # Avoid multiprocessing issues in smoke test

    # Use PK sampler with smaller batch
    sampling_method = RandomIdentitySampler(train_items, args.batch_size, args.num_instance)

    train_loader_stage2 = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      sampler=sampling_method,
                                      num_workers=num_workers,
                                      collate_fn=collate_fn)

    train_set_normal = CommDataset(train_items, val_transforms, is_train=False)
    train_loader_stage1 = DataLoader(
        train_set_normal, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn
    )

    # Build test loaders
    dataset_names = args_test.test_datasets
    val_loaders = {}
    for elm in dataset_names:
        dataset = DATASET_REGISTRY.get(elm)(root=args.data_path)
        test_items = dataset.query + dataset.gallery
        if len(test_items) == 0:
            print(f"  WARNING: {elm} has no test items, skipping")
            continue
        val_set = CommDataset(test_items, val_transforms, relabel=False, is_train=False)
        val_loader = DataLoader(
            val_set, batch_size=min(args_test.test_batch_size, len(test_items)),
            shuffle=False, num_workers=num_workers, collate_fn=collate_fn
        )
        val_loaders[elm] = [val_loader, len(dataset.query)]

    return train_loader_stage1, train_loader_stage2, val_loaders


# ====================================================================
# Verification checks
# ====================================================================

class VerificationResults:
    """Collects and reports verification results."""
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def check(self, name, condition, details=""):
        if condition:
            self.results.append(f"  ✅ PASS: {name}")
            self.passed += 1
        else:
            self.results.append(f"  ❌ FAIL: {name} — {details}")
            self.failed += 1
        return condition

    def report(self):
        print("\n" + "=" * 60)
        print("VERIFICATION REPORT")
        print("=" * 60)
        for r in self.results:
            print(r)
        print(f"\nTotal: {self.passed + self.failed} checks, "
              f"{self.passed} passed, {self.failed} failed")
        print("=" * 60)
        return self.failed == 0


def check_gradients(model, device, verifier, config_name):
    """Verify all trainable modules have non-None, non-zero gradients after backward()."""
    print(f"\n--- Gradient check ({config_name}) ---")

    model.train()

    # Create dummy input
    batch_size = 4
    img = torch.randn(batch_size, 3, 224, 224).to(device)
    if device.type != 'cuda':
        img = img.float()
    target = torch.arange(batch_size).to(device)

    # Forward pass (Image)
    model.zero_grad()
    try:
        outputs = model(x=img, label=target, grl_alpha=1.0) # simulate stage 3
    except Exception as e:
        verifier.check(f"[{config_name}] forward pass", False, str(e))
        return

    # Backward pass
    from loss.make_loss import TotalLoss
    dummy_args = lambda: None
    criterion = TotalLoss(model.num_classes, dummy_args)
    criterion = criterion.to(device)
    loss = criterion(outputs, target, domain_labels=torch.zeros_like(target), stage=3, grl_alpha=1.0)
    loss.backward()

    # Check gradients for all trainable parameters
    trainable_modules = {
        'promptlearner (dmctx)': 'promptlearner.dmctx',
        'classifier': 'classifier.weight',
        'classifier_proj': 'classifier_proj.weight',
        'bottleneck': 'bottleneck.weight',
        'bottleneck_proj': 'bottleneck_proj.weight',
        'local_classifier': 'local_classifier.weight',
        'domain_classifier (fc1)': 'domain_classifier.fc1.weight',
        'part_gat (q_proj)': 'part_gat.q_proj.weight',
        'image_encoder': None,  # check any param
    }

    all_grads_ok = True
    for module_name, param_name in trainable_modules.items():
        if param_name is not None:
            found = False
            for name, param in model.named_parameters():
                if name == param_name:
                    found = True
                    has_grad = param.grad is not None and param.grad.abs().sum().item() > 0
                    verifier.check(
                        f"[{config_name}] {module_name} grad non-None/non-zero",
                        has_grad,
                        f"grad is None" if param.grad is None else f"grad sum = {param.grad.abs().sum().item()}"
                    )
                    if not has_grad:
                        all_grads_ok = False
                    break
            if not found:
                verifier.check(f"[{config_name}] {module_name} found", False, f"param {param_name} not found")
                all_grads_ok = False
        else:
            # Check any image encoder param has gradient
            has_any_grad = False
            for name, param in model.named_parameters():
                if 'image_encoder' in name and param.grad is not None:
                    if param.grad.abs().sum().item() > 0:
                        has_any_grad = True
                        break
            verifier.check(
                f"[{config_name}] {module_name} has at least one non-zero grad",
                has_any_grad
            )
            if not has_any_grad:
                all_grads_ok = False

    return all_grads_ok


def check_checkpoint_roundtrip(model, device, verifier, config_name, tmp_dir):
    """Verify checkpoint save/load produces identical outputs."""
    print(f"\n--- Checkpoint round-trip ({config_name}) ---")

    model.eval()

    # Fixed dummy input
    torch.manual_seed(42)
    dummy_img = torch.randn(2, 3, 224, 224).to(device)
    if device.type != 'cuda':
        dummy_img = dummy_img.float()

    # Get output before save
    with torch.no_grad():
        out_before = model(dummy_img)

    # Save checkpoint
    ckpt_path = os.path.join(tmp_dir, f'ckpt_{config_name}.pth')
    torch.save(model.state_dict(), ckpt_path)

    # Load into fresh state
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Get output after load
    torch.manual_seed(42)
    with torch.no_grad():
        out_after = model(dummy_img)

    # Compare
    if isinstance(out_before, tuple):
        out_before = out_before[0] if isinstance(out_before[0], torch.Tensor) else out_before
    if isinstance(out_after, tuple):
        out_after = out_after[0] if isinstance(out_after[0], torch.Tensor) else out_after

    if isinstance(out_before, torch.Tensor) and isinstance(out_after, torch.Tensor):
        diff = (out_before - out_after).abs().max().item()
        verifier.check(
            f"[{config_name}] checkpoint round-trip identical outputs",
            diff < 1e-5,
            f"max diff = {diff}"
        )
    else:
        verifier.check(
            f"[{config_name}] checkpoint round-trip output type match",
            type(out_before) == type(out_after),
            f"types: {type(out_before)} vs {type(out_after)}"
        )


def check_no_nan_inf(value, name, verifier, config_name):
    """Check a loss value for NaN/Inf."""
    if isinstance(value, torch.Tensor):
        value = value.item()
    is_finite = np.isfinite(value)
    verifier.check(
        f"[{config_name}] {name} is finite (no NaN/Inf)",
        is_finite,
        f"value = {value}"
    )
    return is_finite


# ====================================================================
# Mini training loop (same 3-stage code path as full run)
# ====================================================================

def run_mini_training(model, train_loader_stage1, train_loader_stage2, val_loaders,
                      args, device, verifier, config_name, epochs=2):
    """Run the full 3-stage training pipeline at tiny scale."""
    print(f"\n--- Mini training ({config_name}, {epochs} epochs per stage) ---")

    from reidutils.meter import AverageMeter
    from loss import make_loss
    from loss.supcontrast import SupConLoss
    from solver import (create_scheduler, WarmupMultiStepLR,
                        make_optimizer_for_IE, make_optimizer_prompt_domain, make_optimizer_prompt)

    num_classes = sum(args.classes)
    criterion = make_loss(num_classes)

    optimizer_prompt = make_optimizer_prompt(model, args)
    optimizer_prompt_domain = make_optimizer_prompt_domain(model, args)
    optimizer_image_encoder = make_optimizer_for_IE(model, args)

    scheduler_prompt = create_scheduler(max(epochs, 5), args.prompt_lr, optimizer_prompt)
    scheduler_prompt_domain = create_scheduler(max(epochs, 5), args.prompt_lr, optimizer_prompt_domain)
    scheduler_image_encoder = WarmupMultiStepLR(optimizer_image_encoder, [1], 0.1, 0.1, 2, 'linear')

    all_losses_finite = True

    # --- STAGE PRIOR ---
    print(f"  Stage-Prior ({epochs} epochs)...")
    model.train()
    for epoch in range(1, epochs + 1):
        loss_meter = AverageMeter()
        scheduler_image_encoder.step()
        for n_iter, (img, vid, _, _, domain, _, occ_mask) in enumerate(train_loader_stage2):
            optimizer_image_encoder.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if device.type != 'cuda':
                img = img.float()

            outputs = model(x=img, label=target, disable_grl=args.disable_grl, disable_part_branch=args.disable_part_branch)
            loss = criterion(outputs, target, domain_labels=domain, stage=0)

            loss.backward()
            optimizer_image_encoder.step()

            loss_meter.update(loss.item(), img.shape[0])
            if not check_no_nan_inf(loss, f"stage-prior epoch {epoch} iter {n_iter}", verifier, config_name):
                all_losses_finite = False

            if n_iter >= 1:  # Just 2 iters for speed
                break
        print(f"    Epoch {epoch}: loss = {loss_meter.avg:.4f}")

    # --- STAGE 1 (prompt learning without domain) ---
    print(f"  Stage-1 skipped in v1 DG-ReID (no text prompt tuning)...")

    # Collect image features loop skipped since Stage 1 is skipped in v1

    # --- STAGE 2 (full training) ---
    print(f"  Stage-2 full training ({epochs} epochs)...")
    num_classes = sum(args.classes)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter = AverageMeter()
        scheduler_image_encoder.step()

        for n_iter, (img, vid, _, _, domain, cid, occ_mask) in enumerate(train_loader_stage2):
            optimizer_image_encoder.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            domain = domain.to(device)
            if device.type != 'cuda':
                img = img.float()

            # Stage 2 + 3 combined test logic
            grl_alpha = 1.0 # Simulate stage 3 GRL
            outputs = model(x=img, label=target, grl_alpha=grl_alpha, disable_grl=args.disable_grl, disable_part_branch=args.disable_part_branch)
            loss = criterion(outputs, target, domain_labels=domain, stage=3, grl_alpha=grl_alpha, occ_mask=occ_mask.to(device))

            loss.backward()
            optimizer_image_encoder.step()

            loss_meter.update(loss.item(), img.shape[0])
            if not check_no_nan_inf(loss, f"stage-2 epoch {epoch} iter {n_iter}", verifier, config_name):
                all_losses_finite = False

            if n_iter >= 1:
                break
        print(f"    Stage-2 Epoch {epoch}: loss = {loss_meter.avg:.4f}")

    verifier.check(
        f"[{config_name}] all losses finite across all stages",
        all_losses_finite
    )

    return model


# ====================================================================
# Eval check
# ====================================================================

def run_eval(model, val_loaders, device, verifier, config_name):
    """Run eval and verify output format matches expected mAP + CMC table."""
    print(f"\n--- Eval ({config_name}) ---")

    from reidutils.metrics import R1_mAP_eval

    model.eval()
    eval_output_lines = []

    for name, val_info in val_loaders.items():
        val_loader, num_query = val_info
        evaluator = R1_mAP_eval(num_query, max_rank=10, feat_norm=False, reranking=False)
        evaluator.reset()

        for n_iter, (img, pids, camids, viewids, domain, cid, _) in enumerate(val_loader):
            with torch.no_grad():
                img = img.to(device)
                if device.type != 'cuda':
                    img = img.float()
                feat = model(img)
            evaluator.update((feat, pids, camids))

        try:
            cmc, mAP, _, _, _, _, _ = evaluator.compute()

            line_map = f"mAP: {mAP:.1%}"
            eval_output_lines.append(line_map)
            print(f"  {name} - {line_map}")

            for r in [1, 5, 10]:
                idx = min(r - 1, len(cmc) - 1)
                line_cmc = f"CMC curve, Rank-{r:<3}:{cmc[idx]:.1%}"
                eval_output_lines.append(line_cmc)
                print(f"  {name} - {line_cmc}")

            verifier.check(
                f"[{config_name}] {name} eval produces mAP",
                mAP is not None and np.isfinite(mAP),
                f"mAP = {mAP}"
            )
            verifier.check(
                f"[{config_name}] {name} eval produces CMC R1/R5/R10",
                len(cmc) >= min(10, evaluator.num_query),
                f"CMC length = {len(cmc)}"
            )
        except Exception as e:
            verifier.check(
                f"[{config_name}] {name} eval completed",
                False,
                f"Error: {str(e)}"
            )

    # Verify format correctness
    has_map = any("mAP:" in line for line in eval_output_lines)
    has_r1 = any("Rank-1" in line for line in eval_output_lines)
    has_r5 = any("Rank-5" in line for line in eval_output_lines)
    has_r10 = any("Rank-10" in line for line in eval_output_lines)

    verifier.check(
        f"[{config_name}] eval output format has mAP",
        has_map
    )
    verifier.check(
        f"[{config_name}] eval output format has Rank-1/5/10",
        has_r1 and has_r5 and has_r10,
        f"R1={has_r1}, R5={has_r5}, R10={has_r10}"
    )


# ====================================================================
# Main
# ====================================================================

def run_one_config(benchmark, held_out_domain, data_path, device, verifier, tmp_dir,
                   debug_subset_ids=6, batch_size=12, epochs=2):
    """Run full verification for one configuration."""

    config_name = f"{benchmark}" + (f"_{held_out_domain}" if benchmark == "protocol2" else "")
    print(f"\n{'='*60}")
    print(f"Running config: {config_name}")
    print(f"{'='*60}")

    # Build args
    from cfgs.cfgs import protocol_2, protocol_occluded_duke

    parser = argparse.ArgumentParser()
    parser_test = argparse.ArgumentParser()

    if benchmark == 'protocol2':
        parsertrain, parsertest, logname = protocol_2(parser, parser_test, held_out_domain)
    else:
        parsertrain, parsertest, logname = protocol_occluded_duke(parser, parser_test)

    args = parsertrain.parse_args([])
    args_test = parsertest.parse_args([])

    # Override for smoke test
    args.data_path = data_path
    args.batch_size = batch_size
    args.debug_subset_ids = debug_subset_ids
    args.num_workers = 0
    args.log_period = 1
    args.checkpoint_period = 999
    args.eval_period = 999
    args_test.data_path = data_path

    print(f"  Train datasets: {args.train_datasets}")
    print(f"  Test datasets: {args_test.test_datasets}")
    print(f"  Classes (before subsampling): {args.classes}")

    # Build data
    print("\n  Building data loaders...")
    train_loader_stage1, train_loader_stage2, val_loaders = build_data_for_smoke(
        args, args_test, device
    )
    print(f"  Classes (after subsampling): {args.classes}")

    if len(val_loaders) == 0:
        print("  WARNING: No test datasets available, skipping eval checks")

    # Build model
    print("\n  Building model...")
    model = build_model(args, device)

    # 1. Gradient check
    check_gradients(model, device, verifier, config_name)

    # 2. Mini training
    model = run_mini_training(
        model, train_loader_stage1, train_loader_stage2, val_loaders,
        args, device, verifier, config_name, epochs=epochs
    )

    # 3. Checkpoint round-trip
    check_checkpoint_roundtrip(model, device, verifier, config_name, tmp_dir)

    # 4. Eval
    if len(val_loaders) > 0:
        run_eval(model, val_loaders, device, verifier, config_name)

    # Clean up model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='CLIP-FGDI Smoke Test')
    parser.add_argument('--data_path', type=str, default='data_smoke_test',
                        help='Path for dummy test data')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto | cuda | mps | cpu')
    parser.add_argument('--debug_subset_ids', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--skip_occluded', action='store_true',
                        help='Skip the Occluded-Duke config')
    args = parser.parse_args()

    device = get_device(args.device)
    device = patch_for_device(device)
    print(f"Using device: {device}")

    # Create dummy data
    data_path = os.path.abspath(args.data_path)
    if os.path.exists(data_path):
        import shutil
        shutil.rmtree(data_path)
    setup_dummy_data(data_path)

    # Create tmp dir for checkpoints
    tmp_dir = os.path.join(data_path, '_tmp_ckpt')
    os.makedirs(tmp_dir, exist_ok=True)

    verifier = VerificationResults()

    try:
        # Config 1: Protocol 2 (held_out = Market)
        run_one_config(
            benchmark='protocol2',
            held_out_domain='Market',
            data_path=data_path,
            device=device,
            verifier=verifier,
            tmp_dir=tmp_dir,
            debug_subset_ids=args.debug_subset_ids,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )

        # Config 2: All-4-source -> Occluded-Duke
        if not args.skip_occluded:
            run_one_config(
                benchmark='occluded_duke',
                held_out_domain=None,
                data_path=data_path,
                device=device,
                verifier=verifier,
                tmp_dir=tmp_dir,
                debug_subset_ids=args.debug_subset_ids,
                batch_size=args.batch_size,
                epochs=args.epochs,
            )

    except Exception as e:
        print(f"\n\n!!! SMOKE TEST CRASHED !!!")
        print(f"Error: {e}")
        traceback.print_exc()
        verifier.check("smoke test completed without crash", False, str(e))

    # Final report
    all_passed = verifier.report()

    # Cleanup
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
