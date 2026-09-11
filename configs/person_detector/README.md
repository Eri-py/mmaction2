# Person detector configs (not mmaction2 models)

**Not mmaction2 models.** Copies of two configs from
[`mmdetection`](https://github.com/open-mmlab/mmdetection), kept here so the skeleton-based recognition pipeline
(`notebooks/demo_skeleton.ipynb`) works without requiring that full repo to be cloned separately. mmaction2 did
not train either of these — the benchmark numbers below are copied from their original source project, not
reproduced here.

**Pipeline role:** these are the first stage of top-down pose estimation — find *where* each person is (a
bounding box), before [`configs/skeleton_coord/`](../skeleton_coord/) estimates *where their joints are* within
that box. Output is boxes only; no keypoints, no pose.

> [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
> Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian. *IEEE TPAMI*, 2017.

## `faster-rcnn_r50_fpn_2x_coco_infer.py`

Source: [`mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py`](https://github.com/open-mmlab/mmdetection/blob/main/configs/faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py)

| Backbone | Style | Schedule | box AP (COCO val, 80 classes) | Checkpoint |
|---|---|---|---|---|
| R-50-FPN | pytorch | 2x | 38.4 | [weights](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth) |

Used by the skeleton pipeline, filtered to keep only the COCO "person" class (`det_cat_id=0`) — the other 79
classes go unused.

## `faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py`

Same base architecture, fine-tuned on a person-only subset of COCO for higher person-detection accuracy (at the
cost of no longer detecting anything else).

| Backbone | Style | Fine-tuned on | box AP (person class) | Checkpoint |
|---|---|---|---|---|
| R-50-FPN | caffe | person | 55.8 | [weights](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth) |

Not currently used by any notebook in this repo — kept as an alternative if the general-purpose detector above
proves too slow or noisy for a person-only use case.
