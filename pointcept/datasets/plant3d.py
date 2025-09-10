# -*- coding:utf-8 -*-
"""
coding   : utf-8
@Project ：Pointcept-1.5.1 
@File    ：plant3d.py
@IDE     ：PyCharm 
@Author  ：yangxin6
@Date    ：2025/4/1 上午9:39 
"""
import glob
import os
from typing import Sequence

import numpy as np
from .defaults import DefaultDataset
from .builder import DATASETS
from torch.utils.data import Dataset
from .transform import Compose
from ..utils.logger import get_root_logger


@DATASETS.register_module()
class PlantClsDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/modelnet40_normal_resampled",
        class_names=None,
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache_data=False,
        loop=1,
    ):
        super(PlantClsDataset, self).__init__()
        self.data_root = data_root
        self.class_names = dict(zip(class_names, range(len(class_names))))
        self.split = split
        self.transform = Compose(transform)
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.cache_data = cache_data
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.cache = {}
        if test_mode:
            # TODO: Optimize
            pass

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self):
        assert isinstance(self.split, str)
        split_path = os.path.join(
            self.data_root, "{}.txt".format(self.split)
        )
        data_list = np.loadtxt(split_path, dtype="str")
        return data_list

    def get_data(self, idx):
        data_idx = idx % len(self.data_list)
        if self.cache_data:
            coord, normal, category = self.cache[data_idx]
        else:
            # data_shape = "_".join(self.data_list[data_idx].split("_")[0:-1])
            data_id_split = self.data_list[data_idx].split("_")
            data_shape = data_id_split[0]
            data_name = "_".join(data_id_split[1:])
            data_path = os.path.join(
                self.data_root, data_shape, f"{data_name}.txt"
            )
            data = np.loadtxt(data_path).astype(np.float32)
            coord, normal = data[:, 0:3], data[:, 6:9]
            category = np.array([self.class_names[data_shape]])
            if self.cache_data:
                self.cache[data_idx] = (coord, normal, category, self.data_list[data_idx])
        data_dict = dict(coord=coord, normal=normal, category=category, name=self.data_list[data_idx])
        return data_dict

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        assert idx < len(self.data_list)
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def get_data_name(self, idx):
        data_idx = idx % len(self.data_list)
        return self.data_list[data_idx]

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

@DATASETS.register_module()
class Plant3dSemTxTDataset(DefaultDataset):
    # class2id = np.array([0, 1, 2, 3])

    def __init__(
            self,
            split="train",
            data_root="data/plant3d",
            transform=None,
            test_mode=False,
            test_cfg=None,
            loop=1,
            ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data_list(self):
        if self.split == "predict":
            data_list = [self.data_root]
        elif isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.txt"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.txt"))
        else:
            raise NotImplementedError
        return data_list
    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data = np.loadtxt(data_path)
        coord = data[:, :3]
        color = data[:, 3:6]
        segment = data[:, -1].astype(int)
        scene_id = os.path.basename(data_path)[:-4]

        data_dict = dict(
            coord=coord,
            color=color,
            segment=segment,
            name=scene_id,
        )

        return data_dict

@DATASETS.register_module()
class Plant3dSemEdgeTxTDataset(DefaultDataset):
    # class2id = np.array([0, 1, 2, 3])

    def __init__(
            self,
            split="train",
            data_root="data/plant3d",
            transform=None,
            test_mode=False,
            test_cfg=None,
            loop=1,
            ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data_list(self):
        if self.split == "predict":
            data_list = [self.data_root]
        elif isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.txt"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.txt"))
        else:
            raise NotImplementedError
        return data_list
    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data = np.loadtxt(data_path)
        coord = data[:, :3]
        color = data[:, 3:6]
        edge = data[:, -2].astype(int)
        segment = data[:, -1].astype(int)
        scene_id = os.path.basename(data_path)[:-4]

        data_dict = dict(
            coord=coord,
            color=color,
            segment=segment,
            edge=edge,
            name=scene_id,
        )

        return data_dict


@DATASETS.register_module()
class Plant3dNormalsSemTxTDataset(DefaultDataset):
    # class2id = np.array([0, 1, 2, 3])

    def __init__(
            self,
            split="train",
            data_root="data/plant3d",
            transform=None,
            test_mode=False,
            test_cfg=None,
            loop=1,
            ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data_list(self):
        if self.split == "predict":
            data_list = [self.data_root]
        elif isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.txt"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.txt"))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data = np.loadtxt(data_path)
        coord = data[:, :3]
        color = data[:, 3:6]
        normals = data[:, 6:9]
        segment = data[:, -1].astype(int)
        scene_id = os.path.basename(data_path)[:-4]

        data_dict = dict(
            coord=coord,
            color=color,
            segment=segment,
            normals=normals,
            name=scene_id,
        )

        return data_dict


@DATASETS.register_module()
class Plant3dInstEdgeTxTDataset(DefaultDataset):
    # class2id = np.array([0, 1, 2, 3])

    def __init__(
            self,
            split="train",
            data_root="data/plant3d",
            transform=None,
            test_mode=False,
            test_cfg=None,
            loop=1,
            ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data_list(self):
        if self.split == "predict":
            data_list = [self.data_root]
        elif isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.txt"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.txt"))
        else:
            raise NotImplementedError
        return data_list
    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data = np.loadtxt(data_path)
        coord = data[:, :3]
        color = data[:, 3:6]
        edge = data[:, -3].astype(int)
        segment = data[:, -2].astype(int)
        inst = data[:, -1].astype(int)
        scene_id = os.path.basename(data_path)[:-4]

        data_dict = dict(
            coord=coord,
            color=color,
            segment=segment,
            instance=inst,
            edge=edge,
            name=scene_id,
        )

        return data_dict


@DATASETS.register_module()
class Plant3dNormalsTxTDataset(DefaultDataset):
    # class2id = np.array([0, 1, 2, 3])

    def __init__(
            self,
            split="train",
            data_root="data/plant3d",
            transform=None,
            test_mode=False,
            test_cfg=None,
            loop=1,
            ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data_list(self):
        if self.split == "predict":
            data_list = [self.data_root]
        elif isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.txt"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.txt"))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data = np.loadtxt(data_path)
        coord = data[:, :3]
        normals = data[:, 3:6]
        segment = data[:, -2].astype(int)
        instance = data[:, -1].astype(int)
        scene_id = os.path.basename(data_path)[:-4]

        data_dict = dict(
            coord=coord,
            segment=segment,
            instance=instance,
            normal=normals,
            name=scene_id,
        )

        return data_dict


@DATASETS.register_module()
class Plant3dColorTxTDataset(DefaultDataset):
    # class2id = np.array([0, 1, 2, 3])

    def __init__(
            self,
            split="train",
            data_root="data/plant3d",
            transform=None,
            test_mode=False,
            test_cfg=None,
            loop=1,
            ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data_list(self):
        if self.split == "predict":
            data_list = [self.data_root]
        elif isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.txt"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.txt"))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data = np.loadtxt(data_path)
        coord = data[:, :3]
        color = data[:, 3:6]
        segment = data[:, -2].astype(int)
        instance = data[:, -1].astype(int)
        scene_id = os.path.basename(data_path)[:-4]

        data_dict = dict(
            coord=coord,
            segment=segment,
            instance=instance,
            color=color,
            name=scene_id,
        )

        return data_dict