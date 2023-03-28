## Abstract

Object detection is a fundamental task in computer vision with various applications, including autonomous driving, surveillance, and robotics. In this study, we evaluate several object detection algorithms within the MMDetection framework, including Sparse R-CNN, Cascade R-CNN, Faster R-CNN, Mask R-CNN, and YOLOv3, on the COCO 2017 train validation sets. We use several average precision and recall scores to evaluate the performance of each model.

## Introduction

Object detection algorithms are designed to locate and classify objects within an image. There are various object detection algorithms available in the literature, and the MMDetection framework provides a flexible platform for developing and evaluating these algorithms. In this study, we focus on evaluating five popular object detection algorithms within the MMDetection framework: Sparse R-CNN, Cascade R-CNN, Faster R-CNN, Mask R-CNN, and YOLOv3.



## Results

Table 1 shows the average precision and recall scores for each of the five object detection algorithms on the COCO 2017 validation set.

| Model | AP@0.50:0.95 | AP@0.50 | AP@0.75 | AR@0.50:0.95 |
|---|---|---|---|---|
| Sparse R-CNN | 0.373 | 0.580 | 0.403 | 0.516 |
| Cascade R-CNN | 0.380 | 0.593 | 0.412 | 0.533 |
| Faster R-CNN | 0.370 | 0.578 | 0.401 | 0.514 |
| Mask R-CNN | 0.389 | 0.605 | 0.427 | 0.539 |
| YOLOv3 |  |  |  |  |

