# Skeleton-coordinate (pose) estimator config (not an mmaction2 model)

**Not an mmaction2 model.** A copy of one config from [`mmpose`](https://github.com/open-mmlab/mmpose), kept here
so the skeleton-based recognition pipeline (`notebooks/demo_skeleton.ipynb`) works without requiring that full
repo to be cloned separately. mmaction2 did not train this — the benchmark numbers below are copied from its
original source project, not reproduced here.

**Pipeline role:** this estimates *coordinates* — 17 (x, y) keypoint positions per detected person box — it does
not draw or classify anything. It runs after [`configs/person_detector/`](../person_detector/) (which finds *where
the person is*) and before [`configs/skeleton/posec3d/`](../skeleton/posec3d/) (which takes the coordinate
sequence over time and classifies *what action* it represents). Drawing the visible skeleton onto video frames is
a separate step again, done by `mmpose`'s own visualizer in the notebook — not by this model.

## `td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py`

> [Deep High-Resolution Representation Learning for Human Pose Estimation](http://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html)
> Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong. *CVPR*, 2019.

Source: [`mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py`](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py)

| Arch | Input size | AP | AP@0.5 | AP@0.75 | AR | Checkpoint |
|---|---|---|---|---|---|---|
| HRNet-w32 | 256x192 | 0.746 | 0.904 | 0.819 | 0.799 | [weights](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth) |

"Top-down" means it runs once per detected person box, not once per whole image — see this repo's own notes on
top-down vs. bottom-up pose estimation for why that tradeoff was chosen here.
