import torchvision.transforms as T


from .transforms import RandomErasing
from .autoaugment import ImageNetPolicy


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        # For our vehicle dataset: images are 256x256 already; do not resize or flip.
        transform = T.Compose([
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        # For verification eval we also avoid resizing and flipping; rely on original 256x256
        transform = T.Compose([
            T.ToTensor(),
            normalize_transform
        ])

    return transform
