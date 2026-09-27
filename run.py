import os
import time
import argparse
import warnings
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")
from reidutils.meter import AverageMeter
from reidutils.metrics import R1_mAP_eval
from cfgs import *
from reidutils import setup_logger
import datetime
from models import *
from functools import partial
from torch.cuda import amp
from torch import nn
from datasets.build import build_data_loader
from loss import make_loss
from solver import create_scheduler, WarmupMultiStepLR, make_optimizer_for_IE, \
    make_optimizer_prompt_domain, make_optimizer_prompt
import sys
import numpy as np

sys.path.append('/')


def get_model(args):
    if args.model == 'ViT':
        model = ViT(img_size=args.size_train,
                    stride_size=16,
                    drop_path_rate=0.1,
                    drop_rate=0.,
                    attn_drop_rate=0.,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    qkv_bias=True)
        model.load_param(args.pretrain_vit_path)
    elif args.model == 'gfnet':
        model = GFNet(
            img_size=(384, 128),
            patch_size=16, embed_dim=384, depth=19, mlp_ratio=4, drop_path_rate=0.15,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        model.load_param(args.pretrain_gfnet_path)
    elif args.model == 'clip':
        model = Clip(sum(args.classes), args,domain_num=len(args.train_datasets),epsilon=args.epsilon)
    else:
        model = ViT(img_size=args.size_train,
                    stride_size=16,
                    drop_path_rate=0.1,
                    drop_rate=0.,
                    attn_drop_rate=0.,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    qkv_bias=True)
        model.load_param(args.pretrain_path)
    return model


def train(train_loader, model, criterion, optimizer, scheduler, testloaders, args_train, logger_train, logger_test, log_path, epochs=60):
    logger_train.info('start training')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    
    for epoch in range(1, epochs + 1):
        model.train()
        start_time = time.time()
        loss_meter.reset()
        scheduler.step()
        
        # Calculate simulated GRL alpha based on epoch
        # Warmup GRL lambda: 2 / (1 + exp(-10 * epoch / total_epochs)) - 1
        grl_alpha = 2.0 / (1.0 + np.exp(-10.0 * epoch / epochs)) - 1.0
        
        for n_iter, (img, vid, _, _, domain, cid, occ_mask) in enumerate(train_loader):
            optimizer.zero_grad()

            img = img.to(device)
            target = vid.to(device)
            domain = domain.to(device)
            
            with amp.autocast(enabled=True):
                outputs = model(x=img, label=target, grl_alpha=grl_alpha, 
                                disable_grl=args_train.disable_grl, 
                                disable_part_branch=args_train.disable_part_branch)
                loss = criterion(outputs, target, domain_labels=domain, stage=3, 
                                 grl_alpha=grl_alpha, occ_mask=occ_mask.to(device))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            # torch.cuda.synchronize()
            if (n_iter + 1) % args_train.log_period == 0:
                logger_train.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, (n_iter + 1), len(train_loader),
                            loss_meter.avg, scheduler.get_lr()[0]))
        if epoch % args_train.checkpoint_period == 0:
            torch.save(model.state_dict(),
                       os.path.join(log_path,
                                    args_train.model + str(datetime.datetime.now()) + '_epoch_{}.pth'.format(epoch)))

        if epoch % args_train.eval_period == 0:
            test(testloaders, model, logger_test)

def test(testloaders, model, logger_test):
    model.eval()
    maps, r1s, r5s, r10s = [], [], [], []
    for name, val_loader in testloaders.items():
        evaluator = R1_mAP_eval(val_loader[1], max_rank=10, feat_norm=False, reranking=False)
        evaluator.reset()
        logger_test.info("Validation Results of {}: ".format(name))
        for n_iter, (img, pids, camids, viewids, domain, cid, _) in enumerate(val_loader[0]):
            with torch.no_grad():
                img = img.to(device)
                feat = model(img)
                evaluator.update((feat, pids, camids))
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger_test.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger_test.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        logger_test.info("-" * 30)
        torch.cuda.empty_cache()
        maps.append(mAP)
        r1s.append(cmc[0])
        r5s.append(cmc[4])
        r10s.append(cmc[9])
    logger_test.info("Average Results :")
    logger_test.info("Average, mAP:{:.1%}".format(sum(maps) / len(maps)))
    logger_test.info("Average, Rank-1:{:.1%}".format(sum(r1s) / len(r1s)))
    logger_test.info("Average, Rank-5:{:.1%}".format(sum(r5s) / len(r5s)))
    logger_test.info("Average, Rank-10:{:.1%}".format(sum(r10s) / len(r10s)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser_test = argparse.ArgumentParser(description='test')
    
    # Parse just the benchmark/domain first
    temp_args, _ = parser.parse_known_args()
    benchmark = getattr(temp_args, 'benchmark', 'protocol2')
    held_out_domain = getattr(temp_args, 'held_out_domain', 'Market')
    
    if benchmark == 'protocol2':
        parsertrain, parsertest, logname = protocol_2(parser, parser_test, held_out_domain)
    else:
        parsertrain, parsertest, logname = protocol_occluded_duke(parser, parser_test)

    args_train = parsertrain.parse_known_args()[0]
    args_test = parsertest.parse_known_args()[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    time_now = str(datetime.datetime.now())[:-7]
    log_path = os.path.join(args_train.log_path, logname + '_' + args_train.backbone + '_' + time_now)
    logger_train = setup_logger(args_train.model + '_' + args_train.backbone + '_train', log_path, if_train=True)
    logger_test = setup_logger(args_train.model + '_' + args_train.backbone + '_test', log_path, if_train=False)
    logger_train.info("Log saved in- {}".format(log_path))
    logger_train.info("Training cfgs- {}".format(str(args_train)))
    logger_train.info("Running protocol- {}->{}".format(args_train.train_datasets, args_test.test_datasets))

    model = get_model(args_train).to(device)
    train_loader_stage1, train_loader_stage2, val_loaders = build_data_loader(args_train, args_test)
    criterion = make_loss(sum(args_train.classes))

    optimizer_image_encoder = make_optimizer_for_IE(model, args_train)
    scheduler_image_encoder = WarmupMultiStepLR(optimizer_image_encoder, [30, 50], 0.1, 0.1, 10, 'linear')

    train(train_loader=train_loader_stage2,
          model=model,
          criterion=criterion,
          optimizer=optimizer_image_encoder,
          scheduler=scheduler_image_encoder,
          testloaders=val_loaders,
          args_train=args_train,
          logger_train=logger_train,
          logger_test=logger_test,
          log_path=log_path,
          epochs=args_train.image_encoder_epoch)

    test(val_loaders, model, logger_test)
