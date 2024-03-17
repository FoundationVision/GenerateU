#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
DDETRS Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import sys
import itertools
import time
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results, DatasetEvaluators, LVISEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.projects.ddetrs import add_ddetrsvluni_config
import logging
from collections import OrderedDict
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass
import logging
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer, TrainerBase
from torch.nn.parallel import DistributedDataParallel
from detectron2.utils.env import TORCH_VERSION
# Unification
from detectron2.projects.ddetrs.data.custom_dataset_dataloader import build_custom_train_loader
from detectron2.projects.ddetrs.data.custom_dataset_mapper import DetrDatasetMapper #CustomDatasetMapper
from detectron2.projects.ddetrs.data.custom_build_augmentation import build_custom_augmentation

from detectron2.projects.ddetrs import build_detection_test_loader

# layer-wise learning rate decay for ConvNext
def get_num_layer_layer_wise(var_name_full, num_max_layer=12):
    assert var_name_full.startswith("detr.detr.backbone.0.")
    var_name = var_name_full.replace("detr.detr.backbone.0.", "")
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed"):
        return 0
    elif var_name.startswith("backbone.downsample_layers"):
        stage_id = int(var_name.split('.')[2])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    elif var_name.startswith("backbone.stages"):
        stage_id = int(var_name.split('.')[2])
        block_id = int(var_name.split('.')[3])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    else:
        return num_max_layer + 1

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        assert cfg.UNI == True
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "lvis":
            evaluator_list.append(LVISEvaluator(dataset_name, cfg, True, output_folder, max_dets_per_image=cfg.TEST.NUM_TEST_QUERIES))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DetrDatasetMapper(cfg, is_train=True)
        data_loader = build_custom_train_loader(cfg, mapper=mapper)
        return data_loader


    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DetrDatasetMapper(cfg, dataset_name, is_train=False)
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        return data_loader 

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
                # no weight decay for grn of convnext-v2 
                if cfg.MODEL.BACKBONE.NAME == "D2ConvNeXtV2" and "grn" in key:
                    weight_decay = 0.0
            elif "sampling_offsets" in key or "reference_points" in key:
                lr = lr * cfg.SOLVER.LINEAR_PROJ_MULTIPLIER
            elif "text_encoder" in key or "lang_layers" in key:
                lr = cfg.SOLVER.LANG_LR
            elif "class_generate" in key:
                lr = cfg.SOLVER.VL_LR
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # Only support Sparse R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = DDETRSVLUniWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ddetrsvluni_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
            # fix ap
            os.system(f"python3 tools/evaluate_ap_fixed.py datasets/lvis/lvis_v1_minival.json  {cfg.OUTPUT_DIR}/inference/lvis_instances_results.json   output/")
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()
    if comm.is_main_process():
        os.system(f"python3 tools/evaluate_ap_fixed.py datasets/lvis/lvis_v1_minival.json  {cfg.OUTPUT_DIR}/inference/lvis_instances_results.json   output/")
    return 


def new_argument_parser():
    parser = default_argument_parser()
    parser.add_argument("--uni", type=int, default=1, help="whether to use a unified model for multiple tasks")
    return parser


if __name__ == "__main__":
    args = new_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
