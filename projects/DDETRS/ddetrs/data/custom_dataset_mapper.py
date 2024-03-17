# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import re

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
import random
from transformers import AutoTokenizer
from collections import defaultdict
from transformers import RobertaTokenizerFast
from itertools import compress

__all__ = ["DetrDatasetMapper", "ObjDescription"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class DetrDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DETR.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, dataset_name=None, is_train=True, test_categories=None):
        # test_categories: categories to detect during testing
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.mask_on = cfg.MODEL.MASK_ON
        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        if not self.is_train:
            self.test_prompt = 'detect'
        
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.is_train and 'describe' in dataset_dict["task"]: #refcoco不进行图像翻转
            tfm_gens = self.tfm_gens[1:]
        else:
            tfm_gens = self.tfm_gens

        if self.crop_gen is None:
            image, transforms = T.apply_transform_gens(tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    tfm_gens[:-1] + self.crop_gen + tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            if 'describe' in dataset_dict["task"]:
                new_annotations = []
                for anno in dataset_dict["annotations"]:
                    new_anno = {}
                    xx, yy, ww, hh = anno['bbox']
                    x1, y1, x2, y2 = xx, yy, xx+ww, yy+hh
                    new_anno['bbox'] = [x1, y1, x2, y2]
                    new_anno['object_description'] = anno['object_description']
                    new_annotations.append(new_anno)
                dataset_dict.pop("annotations", None)
                dataset_dict["targets"] = new_annotations
                dataset_dict['test_prompt'] = dataset_dict["task"]
                return dataset_dict
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict['test_prompt'] = self.test_prompt
            return dataset_dict

        if dataset_dict['anno_type'] == 'image':
            dataset_dict['annotations'] = []

        if "annotations" in dataset_dict:

            if len(dataset_dict["annotations"]) > 0:
                object_descriptions = [an['object_description'] for an in dataset_dict["annotations"]]
            else:
                object_descriptions = []

            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")

            instances.gt_object_descriptions = ObjDescription(object_descriptions)
  
            if hasattr(instances, "gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = utils.filter_empty_instances(instances)
            if dataset_dict['anno_type'] != 'image':
                if len(instances) == 0:
                    return None 
            dataset_dict["instances"] = instances
        # import pdb;pdb.set_trace()

        return dataset_dict

class ObjDescription:
    def __init__(self, object_descriptions):
        self.data = object_descriptions

    def __getitem__(self, item):
        assert type(item) == torch.Tensor
        assert item.dim() == 1
        if len(item) > 0:
            assert item.dtype == torch.int64 or item.dtype == torch.bool
            if item.dtype == torch.int64:
                return ObjDescription([self.data[x.item()] for x in item])
            elif item.dtype == torch.bool:
                return ObjDescription(list(compress(self.data, item)))

        return ObjDescription(list(compress(self.data, item)))

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "ObjDescription({})".format(self.data)
