import itertools
import json
import os
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
import numpy as np
import pycocotools.mask as mask_util
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco

from detectron2.evaluation.evaluator import DatasetEvaluator
import torch
import torchvision.ops.boxes as box_ops

from nlgeval.pycocoevalcap.meteor.meteor import Meteor
import detectron2.utils.comm as comm

class GRiTCOCOEvaluator(COCOEvaluator):
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])

            if len(prediction) > 1:
                self._predictions.append(prediction)

    def _eval_predictions(self, predictions, img_ids=None):
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )

        coco_results = self.convert_classname_to_id(coco_results)

        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    use_fast_impl=self._use_fast_impl,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

    def convert_classname_to_id(self, results):
        outputs = []
        class_name_to_id = {}
        categories = sorted(self._coco_api.dataset['categories'], key=lambda x: x['id'])

        for cat in categories:
            if '/' in cat['name']:
                text = ''
                for xx in cat['name'].split('/'):
                    text += xx
                    text += ' '
                text = text[:-1]
            else:
                text = cat['name']
            class_name_to_id[text] = cat['id']

        for pred in results:
            if pred['object_descriptions'] in class_name_to_id:
                pred['category_id'] = class_name_to_id[pred['object_descriptions']]
                del pred['object_descriptions']
                outputs.append(pred)

        return outputs


class GRiTVGEvaluator(COCOEvaluator):
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            assert input["image_id"] == int(input['file_name'].split('/')[-1].split('.')[0])
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"], output_logits=True)
                h = input['height']
                w = input['width']
                scale = 720.0 / max(h, w)
                scaled_inst = []
                for inst in prediction["instances"]:
                    inst['bbox'][0] = inst['bbox'][0] * scale
                    inst['bbox'][1] = inst['bbox'][1] * scale
                    inst['bbox'][2] = inst['bbox'][2] * scale
                    inst['bbox'][3] = inst['bbox'][3] * scale
                    scaled_inst.append(inst)
                if len(scaled_inst) > 0:
                    prediction["instances"] = scaled_inst
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def _eval_predictions(self, predictions, img_ids=None):
        '''
        This is only for saving the results to json file
        '''
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "vg_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

def merge_boxes(boxes, thr):
    """
    :param boxes: (N, 4)
    :param thr: float > 0
    :return:
    """

    assert thr > 0

    ix = []
    iou_matrix = box_ops.box_iou(boxes, boxes)  # (N, N)

    while True:

        good = torch.ge(iou_matrix, thr)
        good_sum = torch.sum(good, 0).view(-1)  # (N,)

        topnum, topix = good_sum.max(dim=0)
        if topnum.item() == 0:
            break

        mergeix = torch.nonzero(good[topix.item()]).view(-1)

        ix.append(mergeix)
        iou_matrix.index_fill_(0, mergeix, 0)
        iou_matrix.index_fill_(1, mergeix, 0)

    return ix

def pluck_boxes(ix, boxes, text=None):
    """multiple ground truth annotations can be on top of each other, and instead group many overlapping boxes into one,
       with multiple caption references.
    :param ix: list (length N) of LongTensors giving indices to boxes/text
    :param boxes: (M, 4)
    :param text: list of strings
    :return: boxes Nx4, and text[] of length N
    """

    total_n = len(ix)
    new_boxes = torch.zeros(total_n, 4).type_as(boxes).to(boxes.device)
    new_text = [] if text is not None else None

    for i in range(total_n):
        ixi = ix[i]  # all index that on top of each other
        n = ixi.nelement()
        bsub = boxes.index_select(0, ixi)
        newbox = torch.mean(bsub, 0)

        new_boxes[i] = newbox

        if text is not None:
            texts = []
            for j in range(n):
                texts.append(text[ixi[j]])

            new_text.append(texts)

    return new_boxes, new_text

class NEWGRiTVGEvaluator(DatasetEvaluator):
    def __init__(self, distributed): #,special_token_list

        self.all_scores = []
        self.records = []
        self.npos = 0
        self._cpu_device = torch.device("cpu")
        self._distributed = distributed
        self.special_token_list = ['<bos>',] #special_token_list  # tokens in it are not used when score_captions eg. '<bos>'

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            # assert input["image_id"] == int(input['file_name'].split('/')[-1].split('.')[0])
            prediction = {"image_id": input["image_id"]}
            img_info = input['file_name']
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_vg_json(instances, input["image_id"], output_logits=False)
                h = input['height']
                w = input['width']
                scale = 1 #720.0 / max(h, w)
                scaled_inst = []
                for inst in prediction["instances"]:
                    inst['bbox'][0] = inst['bbox'][0] * scale
                    inst['bbox'][1] = inst['bbox'][1] * scale
                    inst['bbox'][2] = inst['bbox'][2] * scale
                    inst['bbox'][3] = inst['bbox'][3] * scale
                    scaled_inst.append(inst)
                if len(scaled_inst) > 0:
                    prediction["instances"] = scaled_inst
                
                scores = instances.scores
                boxes = torch.tensor([inst['bbox'] for inst in scaled_inst])
                text = [inst['object_descriptions'] for inst in scaled_inst]
                target_boxes = torch.tensor([target['bbox'] for target in input['targets']])
                target_text = [target['object_description'] for target in input['targets']]

            assert scores.nelement() > 0, '{} {} {}'.format(scores, boxes, text)
            assert scores.shape[0] == boxes.shape[0]
            assert scores.shape[0] == len(text)
            assert target_boxes.shape[0] == len(target_text)
            assert boxes.ndim == 2

            # make sure we're on CPU
            boxes = boxes.cpu().double()
            scores = scores.view(-1).cpu()
            target_boxes = target_boxes.cpu().double()

            # merge ground truth boxes that overlap by >= 0.7
            merged_ix = merge_boxes(target_boxes, 0.7)
            merged_boxes, merged_text = pluck_boxes(merged_ix, target_boxes, target_text)

            # === Sort detections by decreasing confidence ====
            sorted_scores, sorted_idx = torch.sort(scores, 0, True)  # true makes order descending

            nd = scores.shape[0]  # number of detections
            nt = merged_boxes.shape[0]  # number of gt boxes
            used = torch.zeros(nt)

            iou_matrix = box_ops.box_iou(boxes, merged_boxes)  # (nd, nt)

            for d in range(nd):  # for each detection in descending order of confidence

                cand_idx = sorted_idx[d]

                # assign the box to its best match in true boxes
                largest_iou, gt_idx = iou_matrix[cand_idx].max(0)

                ok = 1
                if largest_iou.item() > 0 and used[gt_idx.item()] == 0:
                    used[gt_idx.item()] = 1
                else:
                    ok = 0

                # record the best box, the overlap, and the fact that we need to score the language match
                record = {
                    'ok': ok,
                    'iou': largest_iou.item(),
                    'candidate': text[cand_idx],
                    'references': merged_text[gt_idx] if largest_iou.item() > 0 else [],
                    'img_info': img_info
                }

                self.records.append(record)
            
            self.npos += nt
            self.all_scores.append(sorted_scores)

    def score_captions(self):

        references = {}
        candidates = {}
        for i, record in enumerate(self.records):
            references[i] = [' '.join(token for token in ref.split() if token not in self.special_token_list)
                                for ref in record['references']]
            candidates[i] = [' '.join(token for token in record['candidate'].split()
                                if token not in self.special_token_list)]

        if len(references) == 0 or len(candidates) == 0:
            return 0., [0. for _ in range(len(self.records))]

        meteor_scorer = Meteor()
        meteor, meteor_scores = meteor_scorer.compute_score(references, candidates)
        meteor_scorer.close()

        return meteor, meteor_scores
    def evaluate(self, img_ids=None):
        
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self.all_scores, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self.all_scores


        min_overlaps = [0.3, 0.4, 0.5, 0.6, 0.7]
        min_meteors = [-1, 0, 0.05, 0.1, 0.15, 0.2, 0.25]

        # concatenate everything across all images
        scores = torch.cat(self.all_scores, dim=0)
        # evaluate all records and get their METEOR scores
        _, meteors = self.score_captions()

        if True: #verbose:
            for k, record in enumerate(self.records):
                if record['iou'] > 0 and record['ok'] == 1 and k % 1000 == 0:
                    assert isinstance(record['references'], list)

                    info_txt = 'IOU: {:.3f} OK: {} SCORE: {:.3F} METEOR: {:.3f}'.format(record['iou'], record['ok'],
                                                                                        scores[k].item(), meteors[k])
                    if record['img_info'] is not None:
                        info_txt = 'IMG_INFO: {} '.format(record['img_info']) + info_txt
                    else:
                        info_txt = 'IDX: {} '.format(k) + info_txt

                    print(info_txt)

                    print('PRED:')
                    print(record['candidate'])

                    print('GT:')
                    for gt_sent in record['references']:
                        print(gt_sent)

                    print('-'*20)

        # lets now do the evaluation
        sorted_scores, sorted_ix = torch.sort(scores, 0, True)

        ap_results = {}
        det_results = {}

        for min_overlap in min_overlaps:
            for min_meteor in min_meteors:

                # go down the list and build tp,fp arrays
                n = sorted_scores.nelement()
                tp = torch.zeros(n)
                fp = torch.zeros(n)

                for i in range(n):
                    # pull up the relevant record
                    ii = sorted_ix[i].item()
                    r = self.records[ii]

                    if len(r['references']) == 0:
                        fp[i] = 1  # nothing aligned to this predicted box in the ground truth
                    else:
                        # ok something aligned. Lets check if it aligned enough, and correctly enough
                        if r['iou'] >= min_overlap and r['ok'] == 1 and meteors[ii] > min_meteor:
                            tp[i] = 1
                        else:
                            fp[i] = 1

                fp = torch.cumsum(fp, dim=0)
                tp = torch.cumsum(tp, dim=0)
                rec = tp / self.npos  # recall
                prec = tp / (fp + tp)  # precision

                # compute max-interpolated average precision
                ap = 0
                apn = 0
                for t in torch.arange(0, 1, 0.01).tolist():
                    mask = torch.ge(rec, t)
                    prec_masked = prec * mask
                    p = torch.max(prec_masked)

                    ap = ap + p.item()
                    apn = apn + 1
                ap = ap / apn

                if min_meteor == -1:
                    det_results['iou_{}'.format(min_overlap)] = ap
                else:
                    ap_results['iou_{}_meteor_{}'.format(min_overlap, min_meteor)] = ap

        map = sum(ap_results.values()) / len(ap_results)
        detmap = sum(det_results.values()) / len(det_results)

        results = {
            'map': map,
            'ap_breakdown': ap_results,
            'detmap': detmap,
            'det_breakdown': det_results,
        }

        return results
    def _eval_predictions(self, predictions, img_ids=None):
        '''
        This is only for saving the results to json file
        '''
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "vg_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()



def instances_to_coco_json(instances, img_id, output_logits=False):
    """
        Add object_descriptions and logit (if applicable) to
        detectron2's instances_to_coco_json
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    object_descriptions = instances.pred_object_descriptions.data
    if output_logits:
        logits = instances.logits.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
            'object_descriptions': object_descriptions[k],
        }
        if output_logits:
            result["logit"] = logits[k]

        results.append(result)
    return results


def instances_to_vg_json(instances, img_id, output_logits=False):
    """
        Add object_descriptions and logit (if applicable) to
        detectron2's instances_to_coco_json
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    # boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    object_descriptions = instances.pred_object_descriptions.data
    if output_logits:
        logits = instances.logits.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
            'object_descriptions': object_descriptions[k],
        }
        if output_logits:
            result["logit"] = logits[k]

        results.append(result)
    return results
