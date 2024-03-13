#!/usr/bin/env python3

"""Data_process loader."""
import logging
import torch

from torch.utils.data.sampler import RandomSampler, SequentialSampler

from RLRRDatasets.fgvc_json_dataset import CUB200Dataset, CarsDataset, DogsDataset, FlowersDataset, NabirdsDataset

logger = logging.getLogger(__name__)

_DATASET_CATALOG = {
    "CUB_200_2011": CUB200Dataset,
    'OxfordFlowers': FlowersDataset,
    'StanfordCars': CarsDataset,
    'StanfordDogs': DogsDataset,
    "NABirds": NabirdsDataset,
}


def _construct_loader(cfg, split, batch_size, shuffle, drop_last, data_path):
    """Constructs the data loader for the given dataset."""
    dataset_name = cfg.dataset_name

    assert (
        dataset_name in _DATASET_CATALOG.keys()
    ), "Dataset '{}' not supported".format(dataset_name)
    dataset = _DATASET_CATALOG[dataset_name](cfg, split, data_path)

    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader(cfg):
    """Train loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="train",
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        data_path=cfg.dataset_dir
    )

def construct_trainval_loader(cfg, drop_last=False):
    """Train loader wrapper."""
    
    return _construct_loader(
        cfg=cfg,
        split="trainval",
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=drop_last,
        data_path=cfg.dataset_dir
    )

def construct_test_loader(cfg):
    """Test loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="test",
        batch_size=256,
        shuffle=False,
        drop_last=False,
        data_path=cfg.dataset_dir
    )


def construct_val_loader(cfg):
    """Validation loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="val",
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        data_path=cfg.dataset_dir
    )
