import torch


def make_optimizer_prompt(model,cfg):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "clsctx" in key:
            lr = cfg.prompt_lr
            weight_decay = 1e-4
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

        if 'dcgrl' in key:
            lr = cfg.lamda
            weight_decay = 1e-4
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

    if cfg.optimizer == 'SGD':
        optimizer = getattr(torch.optim, cfg.optimizer)(params, momentum=0.95)
    elif cfg.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.prompt_lr, weight_decay=1e-4)
    else:
        optimizer = getattr(torch.optim, 'Adam')(params)
    return optimizer

def make_optimizer_prompt_domain(model,cfg):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "dmctx" in key:
            lr = cfg.prompt_lr
            weight_decay = 1e-4
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

        if "dc" in key and 'dcgrl' not in key:
            print(key)
            lr = cfg.lamda
            weight_decay = 1e-4
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

    if cfg.optimizer == 'SGD':
        optimizer = getattr(torch.optim, cfg.optimizer)(params, momentum=0.95)
    elif cfg.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.prompt_lr, weight_decay=1e-4)
    else:
        optimizer = getattr(torch.optim, 'Adam')(params)
    return optimizer


def make_optimizer_for_IE(model, cfg):
    params = []
    keys = []
    head_lr = getattr(cfg, 'prompt_lr', 0.00035)
    base_lr = getattr(cfg, 'image_encoder_lr', 5e-6)
    weight_decay = getattr(cfg, 'weight_decay', 1e-4)

    for key, value in model.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue

        # Classifiers, PartVisibilityGAT, domain_classifier, and BN bottleneck heads
        # are newly initialized and must be trained with head_lr (0.00035).
        # Pretrained CLIP visual backbone uses base_lr (5e-6).
        is_head = any(h in key for h in ['classifier', 'part_gat', 'domain_classifier', 'bottleneck', 'promptlearner'])
        lr = head_lr if is_head else base_lr

        if "bias" in key:
            lr = lr * 2.0

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]

    if cfg.optimizer == 'SGD':
        optimizer = getattr(torch.optim, cfg.optimizer)(params, momentum=0.95)
    elif cfg.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, weight_decay=weight_decay)
    else:
        optimizer = getattr(torch.optim, 'Adam')(params)

    return optimizer
