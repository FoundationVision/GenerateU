# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from typing import Dict, List
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from fvcore.nn import giou_loss, smooth_l1_loss

from .backbone.masked_backbone import MaskedBackbone
from .models.deformable_detr.backbone import Joiner
from .models.deformable_detr.deformable_detr import DeformableDETR, SetCriterion
from .models.deformable_detr.matcher import HungarianMatcher, HungarianMatcherVL
from .models.deformable_detr.position_encoding import PositionEmbeddingSine
from .models.deformable_detr.deformable_transformer import DeformableTransformer
from .models.segmentation_condInst_new_encodfpn import DETRsegmVLUni, segmentation_postprocess
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import NestedTensor
import torchvision.ops as ops

from collections import OrderedDict
from einops import repeat
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)
from detectron2.structures import BoxMode
import cv2
from skimage import color
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from .data.custom_dataset_mapper import ObjDescription
import clip
import numpy as np
import random 
__all__ = ["DDETRSVL"]

"""
DDETRS supports traditional (not language-guided) object detection & visual grounding.
DDETRSVL is designed for language-guided object detection & visual grounding.
DDETRSVLUni is designed for Unified language-guided object detection & visual grounding
"""
@META_ARCH_REGISTRY.register()
class DDETRSVLUni(nn.Module):
    """
    Implement DDETRSVLUni
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.demo_only = False
        self.use_amp = cfg.SOLVER.AMP.ENABLED

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.mask_stride = cfg.MODEL.DDETRS.MASK_STRIDE
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_thres = cfg.MODEL.DDETRS.MASK_THRES
        self.new_mask_head = cfg.MODEL.DDETRS.NEW_MASK_HEAD
        self.use_raft = cfg.MODEL.DDETRS.USE_RAFT
        self.freeze_detr = cfg.FREEZE_DETR
        self.use_rel_coord = cfg.MODEL.DDETRS.USE_REL_COORD
        hidden_dim = cfg.MODEL.DDETRS.HIDDEN_DIM
        self.num_queries = cfg.MODEL.DDETRS.NUM_OBJECT_QUERIES
        self.num_classes = cfg.MODEL.DDETRS.NUM_CLASSES
        self.topk_for_mapping = cfg.TEST.TOPK_FOR_MAPPING

        # Transformer parameters:
        nheads = cfg.MODEL.DDETRS.NHEADS
        dim_feedforward = cfg.MODEL.DDETRS.DIM_FEEDFORWARD
        dec_layers = cfg.MODEL.DDETRS.DEC_LAYERS

        num_feature_levels = cfg.MODEL.DDETRS.NUM_FEATURE_LEVELS
        two_stage = cfg.MODEL.DDETRS.TWO_STAGE
        two_stage_num_proposals = cfg.MODEL.DDETRS.TWO_STAGE_NUM_PROPOSALS

        
        # Loss parameters:
        mask_weight = cfg.MODEL.DDETRS.MASK_WEIGHT
        dice_weight = cfg.MODEL.DDETRS.DICE_WEIGHT
        giou_weight = cfg.MODEL.DDETRS.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DDETRS.L1_WEIGHT
        class_weight = cfg.MODEL.DDETRS.CLASS_WEIGHT
        deep_supervision = cfg.MODEL.DDETRS.DEEP_SUPERVISION

        focal_alpha = cfg.MODEL.DDETRS.FOCAL_ALPHA

        set_cost_class = cfg.MODEL.DDETRS.SET_COST_CLASS
        set_cost_bbox = cfg.MODEL.DDETRS.SET_COST_BOX
        set_cost_giou = cfg.MODEL.DDETRS.SET_COST_GIOU


        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels  # only take [c3 c4 c5] from resnet and gengrate c6 later
        backbone.strides = d2_backbone.feature_strides

        transformer_class = DeformableTransformer
        transformer = transformer_class(
        d_model= hidden_dim,
        nhead=nheads,
        num_encoder_layers=cfg.MODEL.DDETRS.ENC_LAYERS,
        num_decoder_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        dropout=cfg.MODEL.DDETRS.DROPOUT,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=num_feature_levels,
        dec_n_points=cfg.MODEL.DDETRS.DEC_N_POINTS,
        enc_n_points=cfg.MODEL.DDETRS.ENC_N_POINTS,
        two_stage=two_stage,
        two_stage_num_proposals=two_stage_num_proposals
        )
        
        detr_class = DeformableDETR
        model = detr_class(
        backbone,
        transformer,
        num_classes_list=self.num_classes,
        num_queries=self.num_queries,
        num_feature_levels=num_feature_levels,
        aux_loss=deep_supervision,
        with_box_refine=True,
        two_stage=two_stage,
        mixed_selection=cfg.MODEL.DDETRS.MIXED_SELECTION,
        cfg=cfg)

        model_class = DETRsegmVLUni
        self.detr = model_class(model, freeze_detr=self.freeze_detr, rel_coord=self.use_rel_coord,
        new_mask_head=self.new_mask_head, use_raft=self.use_raft, mask_out_stride=self.mask_stride, 
        lang_as_tgt=cfg.MODEL.LANG_AS_TGT, decouple_tgt=cfg.MODEL.DECOUPLE_TGT, cls_pool_type=cfg.MODEL.CLS_POOL_TYPE,
        use_iou_branch=cfg.MODEL.USE_IOU_BRANCH, cfg=cfg)

        self.detr.to(self.device)

        # building criterion
        matcher = HungarianMatcher(
            cost_class=set_cost_class,
            cost_bbox=set_cost_bbox,
            cost_giou=set_cost_giou)

        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight, \
            "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)


        losses = ['labels', 'boxes', 'masks']

        criterion_class = SetCriterion
        self.criterion = criterion_class(self.num_classes, matcher, weight_dict, losses, focal_alpha=focal_alpha,
        still_cls_for_encoder=cfg.MODEL.STILL_CLS_FOR_ENCODER, cfg=cfg)
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.clip_model, _ = clip.load("ViT-L/14@336px")
        self.clip_model.to(self.device)

        for p in self.clip_model.parameters():
                p.requires_grad_(False)

        zs_weight_path = cfg.MODEL.TEXT.ZERO_SHOT_WEIGHT

        self.lvis_embed = torch.tensor(
                np.load(zs_weight_path), 
                dtype=torch.half).permute(1, 0).contiguous() # D x C
        self.lvis_embed = self.lvis_embed / self.lvis_embed.norm(dim=0, keepdim=True)
        self.lvis_embed.to(self.device)
        self.beam_size = cfg.MODEL.TEXT.BEAM_SIZE

        self.num_test_queries = cfg.TEST.NUM_TEST_QUERIES

        self.to(self.device)
        

    def forward(self, batched_inputs, do_postprocess=True):
        images = self.preprocess_image(batched_inputs)
        if self.training:
            dataset_source = batched_inputs[0]['dataset_source']
            for anno_per_image in batched_inputs:
                assert dataset_source == anno_per_image['dataset_source']

        if self.training:
            ann_type = batched_inputs[0]['anno_type']
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, batched_inputs)
            clip_object_descriptions_features = self.language_feature_forward(targets)
            output, loss_dict = self.detr.forward(images, targets, self.criterion, train=True, clip_object_descriptions_features=clip_object_descriptions_features, dataset_source=dataset_source, ann_type=ann_type)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            output, loss_dict = self.detr.inference(images, None, self.criterion, train=False)
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            pred_object_descriptions = output['pred_object_descriptions']
            logprobs = output['logprobs']
            mask_pred = output["pred_masks"] if self.mask_on else None
            if self.detr.use_iou_branch:
                iou_pred = output["pred_boxious"]
            else:
                iou_pred = [None]
            results = self.inference(box_cls,logprobs, box_pred, pred_object_descriptions, mask_pred, images.image_sizes, iou_pred=iou_pred)
            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = segmentation_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results

    def language_feature_forward(self, targets):
        max_len = max([len(target['gt_object_descriptions']) for target in targets])
        clip_object_descriptions_features = torch.zeros(len(targets), max_len, 768).to(self.device)
        for ii, target in enumerate(targets):
            text = [clip.tokenize(['a '+ pred_object]) for pred_object in target['gt_object_descriptions']]
            text = torch.stack(text, dim=0).squeeze(1).to(self.device)
            text_features = self.clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            clip_object_descriptions_features[ii:ii+1, :text_features.size(0), :] = text_features
        return clip_object_descriptions_features


    def prepare_targets(self, targets, batched_inputs):
        new_targets = []
        for targets_per_image, batched_input in zip(targets, batched_inputs):
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)

            gt_object_descriptions = []
            for gt_object_description in targets_per_image.gt_object_descriptions.data:
                if ', ' in gt_object_description:
                    gt_object_descriptions += random.sample(gt_object_description.split(', '), 1)
                else:
                    gt_object_descriptions += [gt_object_description]
            
            if self.mask_on and hasattr(targets_per_image, 'gt_masks'):
                gt_masks = targets_per_image.gt_masks.tensor
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks, "image_size": image_size_xyxy, "gt_object_descriptions": gt_object_descriptions})
            else:
                if batched_input['anno_type'] == 'image':
                    new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "image_size": image_size_xyxy, "gt_object_descriptions": gt_object_descriptions, "image_description":batched_input['image_description']})
                else:
                    new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "image_size": image_size_xyxy, "gt_object_descriptions": gt_object_descriptions})
        return new_targets

    def inference(self, box_cls, logprobs, box_pred, pred_object_descriptions, mask_pred, image_sizes, score_thres=0.0, iou_pred=None):
        max_num_inst = self.num_test_queries
        beam_size = self.beam_size
        assert len(box_cls) == len(image_sizes)
        results = []

        for i, (logits_per_image, box_pred_per_image, output_text, image_size, iou_per_image) in enumerate(zip(
            box_cls, box_pred, pred_object_descriptions,image_sizes, iou_pred
        )):
            pred_object_des = []
            for ii in range(0,len(output_text),beam_size):
                pred_object_des.append(', '.join(output_text[ii:ii+beam_size]))

            if False: #self.ota:
                # NMS
                logits_per_image = convert_grounding_to_od_logits(logits_per_image.unsqueeze(0), num_classes, positive_map_label_to_token) #[1, 900, 256]-->[1, 900, 80]
            else:
                logits_per_image = logits_per_image[..., 0:1]
                prob = logits_per_image.sigmoid()
                # cls x iou
                if iou_per_image is not None:
                    compose_prob = prob * iou_per_image.sigmoid() # (num_query, C)
                # filter low-score predictions
                if score_thres > 0.0:
                    valid_mask = (prob > score_thres)
                    num_valid = torch.sum(valid_mask).item()
                    num_inst = min(num_valid, max_num_inst)
                    prob[~valid_mask] = -1.0 # set to invalid status
                else:
                    num_inst = max_num_inst
                
                if max_num_inst > 300:
                    nms_scores,idxs = torch.max(prob,1)
                    boxes_before_nms = box_cxcywh_to_xyxy(box_pred_per_image)
                    keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.9)  
                    compose_prob = compose_prob[keep_indices]
                    box_pred_per_image = box_pred_per_image[keep_indices]

                    keep_pred_object_des = []
                    for keep_indice in keep_indices:
                        keep_pred_object_des.append(pred_object_des[keep_indice])
                    pred_object_des = keep_pred_object_des

                text = []
                for pred_object in pred_object_des:
                    if len(pred_object)>75:
                        pred_object = pred_object[:75]
                    text.append(clip.tokenize(['a '+ pred_object]))
                text = torch.stack(text, dim=0).squeeze(1).to(self.device)
                text_features = self.clip_model.encode_text(text)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                similarity = (100.0 * text_features @ self.lvis_embed.to(self.device)).softmax(dim=-1)
                sim_values, sim_indices = similarity.topk(self.topk_for_mapping)
                compose_prob = compose_prob * sim_values

                topk_values, topk_indexes = torch.topk(compose_prob.view(-1), compose_prob.view(-1).size(0), dim=0)
                topk_boxes = torch.div(topk_indexes, compose_prob.size(1), rounding_mode='floor')
                scores_per_image = topk_values
                box_pred_per_image = box_pred_per_image[topk_boxes]
                o_pred_object_des = [pred_object_des[tki] for tki in topk_boxes] #topk_indexes] #
                if mask_pred is not None:
                    mask_pred_i = mask_pred[i][topk_boxes]
            
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.pred_object_descriptions = ObjDescription(o_pred_object_des)
            labels_per_image = sim_indices.view(-1)[topk_indexes] #.squeeze(1)

            if self.mask_on:
                N, C, H, W = mask_pred_i.shape
                mask = F.interpolate(mask_pred_i, size=(H*self.mask_stride, W*self.mask_stride), mode='bilinear', align_corners=False)
                mask = mask.sigmoid() > self.mask_thres
                mask = mask[:,:,:image_size[0],:image_size[1]]
                result.pred_masks = mask
                
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results



    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        #######TODO: why ImageList
        images = ImageList.from_tensors(images)
        return images

