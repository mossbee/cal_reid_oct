import logging
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from PIL import Image
import numpy as np

from data.transforms import build_transforms


def create_supervised_trainer(model, optimizer, loss_fn,using_cal,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        model.to(device)
        
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target = batch

        img = img.cuda()
        target = target.cuda()
        if using_cal:
            score,score_hat, feat = model(img)
            loss = loss_fn(score, score_hat, feat, target)
        else:
            score,feat = model(img)
            loss = loss_fn(score, feat, target)

        loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()

        return loss.item(), acc.item()

    return Engine(_update)


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)


def _compute_verification_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    # scores: similarity, higher is more similar
    # labels: 1 for same, 0 for different
    order = np.argsort(scores)[::-1]
    scores = scores[order]
    labels = labels[order]

    # Threshold sweep over all unique scores
    thresholds = np.r_[[-np.inf], (scores[:-1] + scores[1:]) / 2.0, [np.inf]]

    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    num_pos = tp[-1] if tp.size > 0 else 0
    num_neg = fp[-1] if fp.size > 0 else 0

    # Add starting point (0,0)
    tpr = np.r_[0.0, tp / max(1, num_pos)]
    fpr = np.r_[0.0, fp / max(1, num_neg)]

    # ACC at best threshold
    # For each threshold between scores, positives above threshold
    # Compute accuracy via TPR and TNR
    tnr = 1.0 - fpr
    acc = (tpr * (num_pos / max(1, num_pos + num_neg)) + tnr * (num_neg / max(1, num_pos + num_neg)))
    best_acc = float(np.max(acc)) if acc.size > 0 else 0.0

    # AUC via trapezoidal rule
    auc = float(np.trapz(tpr, fpr))

    # EER where FPR == FNR
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)

    # TAR at FAR targets
    def tar_at_far(target_far: float) -> float:
        if fpr.size == 0:
            return 0.0
        j = np.searchsorted(fpr, target_far, side='right') - 1
        j = np.clip(j, 0, len(tpr) - 1)
        return float(tpr[j])

    tar_far_1e_2 = tar_at_far(1e-2)
    tar_far_1e_3 = tar_at_far(1e-3)

    return {
        'ACC': best_acc,
        'AUC': auc,
        'EER': eer,
        'TAR@FAR=1e-2': tar_far_1e_2,
        'TAR@FAR=1e-3': tar_far_1e_3,
    }


def _load_pairs(pairs_file: str, image_root: str) -> List[Tuple[str, str, int]]:
    pairs: List[Tuple[str, str, int]] = []
    with open(pairs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel1, rel2, label_str = line.split()
            p1 = os.path.join(image_root, rel1)
            p2 = os.path.join(image_root, rel2)
            label = int(label_str)
            pairs.append((p1, p2, label))
    return pairs


def _extract_embeddings(model, paths: List[str], batch_size: int, device: str, cfg) -> Dict[str, np.ndarray]:
    model.eval()
    transforms = build_transforms(cfg, is_train=False)
    embeddings: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            images = []
            for p in batch_paths:
                img = Image.open(p).convert('RGB')
                images.append(transforms(img))
            data = torch.stack(images, dim=0).to(device)
            feat = model(data)
            feat = _l2_normalize(feat).cpu().numpy()
            for p, e in zip(batch_paths, feat):
                embeddings[p] = e
    return embeddings


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    using_cal = cfg.MODEL.CAL


    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")

    trainer = create_supervised_trainer(model, optimizer, loss_fn, using_cal,device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=5, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model, 'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] \nLoss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss'], 
                                engine.state.metrics['avg_acc'], scheduler.get_lr()[0]))

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    best_acc = {'value': 0.0}

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_verification_eval(engine):
        epoch = engine.state.epoch
        pairs_file = getattr(cfg.TEST, 'PAIRS_FILE', '')
        if pairs_file and ((epoch % eval_period == 0) or (epoch == epochs)):
            if not os.path.isfile(pairs_file):
                logger.warning("Pairs file not found: {}".format(pairs_file))
                return
            image_root = cfg.DATASETS.IMAGE_ROOT
            pairs = _load_pairs(pairs_file, image_root)
            if len(pairs) == 0:
                logger.warning("No pairs loaded from {}".format(pairs_file))
                return
            unique_paths = sorted(set([p for p, _, _ in pairs] + [q for _, q, _ in pairs]))
            batch_size = cfg.TEST.IMS_PER_BATCH
            embeddings = _extract_embeddings(model, unique_paths, batch_size, device, cfg)

            # Build score and label arrays
            scores = []
            labels = []
            for p1, p2, lab in pairs:
                v1 = embeddings[p1]
                v2 = embeddings[p2]
                s = float(np.dot(v1, v2))
                scores.append(s)
                labels.append(lab)
            scores = np.asarray(scores, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.int32)

            metrics = _compute_verification_metrics(scores, labels)
            logger.info("Verification Results - Epoch: {}".format(epoch))
            logger.info("ACC: {:.4f} AUC: {:.4f} EER: {:.4f} TAR@1e-2: {:.4f} TAR@1e-3: {:.4f}".format(
                metrics['ACC'], metrics['AUC'], metrics['EER'], metrics['TAR@FAR=1e-2'], metrics['TAR@FAR=1e-3']
            ))

            # Save best by ACC
            if metrics['ACC'] > best_acc['value']:
                best_acc['value'] = metrics['ACC']
                best_path = os.path.join(output_dir, '{}_best_acc.pth'.format(cfg.MODEL.NAME))
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'acc': best_acc['value']}, best_path)



    trainer.run(train_loader, max_epochs=epochs)
