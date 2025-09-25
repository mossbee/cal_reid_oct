from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler
from .transforms import build_transforms
import numpy as np
import torch
import matplotlib.pyplot as plt

def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    # Initialize dataset, allowing custom list-based dataset
    names = cfg.DATASETS.NAMES
    if isinstance(names, (list, tuple)) and len(names) == 1:
        name = names[0]
    else:
        name = names

    if name == 'list_reid':
        dataset = init_dataset('list_reid', root=cfg.DATASETS.IMAGE_ROOT, list_path=cfg.DATASETS.TRAIN_LIST)
    else:
        dataset = init_dataset(cfg.DATASETS.NAMES)
    
    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, 'train', train_transforms)

    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    # Build val_loader only if not using pairs-based evaluation
    if hasattr(cfg.TEST, 'PAIRS_FILE') and cfg.TEST.PAIRS_FILE:
        val_loader = None
        num_query = 0
    else:
        val_set = ImageDataset(dataset.query + dataset.gallery, 'test', val_transforms)
        val_loader = DataLoader(
            val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )
        num_query = len(dataset.query)

    return train_loader, val_loader, num_query, num_classes
