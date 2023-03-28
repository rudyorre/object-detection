# Comparing Object Detection Algorithms in MMDetection: A Study on COCO 2017 Dataset

## Abstract
Object detection is a fundamental task in computer vision with various applications, including autonomous driving, surveillance, and robotics. In this study, we evaluate several object detection algorithms within the MMDetection framework, including Sparse R-CNN, Cascade R-CNN, Faster R-CNN, Mask R-CNN, and YOLOv3, on the COCO 2017 train validation sets. We use several average precision and recall scores to evaluate the performance of each model.

## Introduction
Object detection algorithms are designed to locate and classify objects within an image. There are various object detection algorithms available in the literature, and the MMDetection framework provides a flexible platform for developing and evaluating these algorithms. In this study, we focus on evaluating five popular object detection algorithms within the MMDetection framework: Sparse R-CNN, Cascade R-CNN, Faster R-CNN, Mask R-CNN, and YOLOv3. To get to the details of this study, some fundamentals of object detection must first be covered.

### Intersection over Union (IoU)
Unlike standard computer vision classification problems, the goal of object detection is to not only detect objects in an image, but to also find where they are in that image. This is done typically through a tight bounding box enclosing the object, which encodes both location and size. To compare a ground truth bounding box to a model-predicted bounding box, we can use Intersection over Union (IoU). IoU is a commonly used metric for evaluating the performance of object detection algorithms. IoU measures the overlap between the predicted bounding box and the ground truth bounding box for a given object.

To calculate IoU, the area of intersection between the predicted and ground truth bounding boxes is divided by the area of union between the two bounding boxes. This bounds IoU between [0,1], where 1 indicates perfect overlap and 0 indicates no overlap.

$$IoU(A,B) = \frac{|A \cap B|}{|A \cup B|}$$

The paper that first introduced IoU as a metric for object detection is "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" by Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. The paper was published in 2015 and proposed a new deep learning architecture for object detection called Faster R-CNN. The authors used IoU as a key evaluation metric to measure the accuracy of their model on the PASCAL VOC dataset. Since then, IoU has become a standard metric for evaluating object detection models.

### Average Precision (AP)
Average precision (AP) is a common evaluation metric used in object detection to measure the accuracy of predicted bounding boxes. It is a single scalar value that summarizes the precision-recall curve for a given model and dataset at different levels of IoU (typically ranging between 0.5 and 0.96). AP measures the quality of the predicted bounding boxes by computing the area under the precision-recall curve, which plots the precision (fraction of correct detections among all predicted detections) versus the recall (fraction of true objects detected among all true objects).

The average precision (AP) is defined as the average of the maximum precisions at different recall levels. This means that the AP metric penalizes models that have low precision at high recall levels, which is important in object detection because detecting all objects of interest is usually more important than detecting some objects with high precision.

### COCO Dataset
The COCO (Common Objects in Context) dataset is a large object detection dataset that was introduced in 2014. Of the 330,000 total images, each has been annotated with the classes and bounding boxes for the target objects. The dataset includes 80 different object categories, such as people, animals, vehicles, and household objects. This study will be evaluating models using the 2017 COCO train validation set in particular, which has exactly 5000 labeled images.

### YOLOv3
YOLOv3 (You Only Look Once) is described as a real-time object detection algorithm that identifies specific objects in videos, live feeds, or images. This strong balance of having fast speeds and accurate predictions have made the YOLO family of algorithms to be particularly important in object detection. Their architecture is also relatively simple, making them more accessible for researchers. YOLOv3 at its core is a convolutional neural network that performs object detection by dividing an image into a grid and predicting bounding boxes and class probabilities for each grid cell. As suggested by its name, YOLOv3 is a single-stage detector, meaning it predicts the classes and locations in a single forward pass of the network. This in particular gives it its speed, however at the cost of lower accuracy.

### Faster R-CNN
Faster R-CNN is a popular object detection algorithm that uses a two-stage detection process (unlike YOLOv3's one-stage) to generate region proposals and refine them using a classification and regression network. The Region Proposal Network (RPN) generates a set of candidate regions of interest in the image, while the Classifier Network labels each region as being part of a class or just the background.

![AltText]({{ '/assets/images/team04/fastrcnn.png' | relative_url }}) {: style="width: 600px; max-width: 100%;"} Figure 2. Increase in Number of Object Detection Publications From 1998 to 2001 [1].

The shared feature map for the RPN and classifier networks allow Faster R-CNN to perform object detection in real-time applications while still demonstrating excellent performance.

### Mask R-CNN
Mask R-CNN is an extension of Faster R-CNN that adds a segmentation branch to the network to generate object masks in addition to bounding boxes. This allows for both bounding box detections as well as instance segmentation. Although this is a dramatic improvement over Faster R-CNN in terms of use cases, we are including Mask R-CNN to see if there is a performance tradeoff (or improvement) over its predecessor.

### Sparse R-CNN
Sparse R-CNN is a recently proposed object detection algorithm that uses sparse convolutional layers to reduce the computational cost of the backbone network. Sparse R-CNN achieves this by selecting features that aren't necessarily the most informational, but the ones that are most correlated with the ground truth. It also made non-trivial improvements on top of Faster R-CNN in the training scheme to improve performance.

### Cascade R-CNN
Cascade R-CNN is an extension of Faster R-CNN that improves the accuracy of bounding box proposals through a multi-stage detection process. This cascade of detectors helps against degrading performance when increasing the IoU threshold. We aim to see if Cascade R-CNN improves not only its baseline AP scores above Faster R-CNN, but especially at higher IoU values such as 0.75 and above.

## Method
We use the MMDetection framework to implement and evaluate the five object detection algorithms on the COCO 2017 train validation sets. We train each model on the train set and evaluate it on the validation set using several average precision and recall scores, including AP@0.50:0.95, AP@0.50, AP@0.75, and AR@0.50:0.95.

## Results





# TODO: insert stuff below to rest of the article

### Innovation
We hope to compare the average precision and recall performance between these two models to identify strengths and weakpoints between them. We plan on exploring Pascal VOC, COCO, CityScapes, LVIS, etc. standard datasets and perhaps using more niche datasets such as KITTI to focus on more complex areas of object detection. Another possibility is perhaps obfuscating existing image datasets and see how it affects performance.

### Code
<ol>
  <li>MMDetection Colab - https://colab.research.google.com/drive/1ervLVmxrpWRMnTGFSmjHa9TIRtAr2ag6?usp=sharing</li>
  <li>YOLO Github - https://github.com/rudyorre/object-detection</li>
</ol> 

### Three Relevant Papers
- 'MMDetection: Open MMLab Detection Toolbox and Benchmark' [Code](https://github.com/open-mmlab/mmdetection) [Paper](https://arxiv.org/abs/1906.07155) [1]
- 'SSD: Single Shot MultiBox Detector' [Code](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) [Paper](https://arxiv.org/abs/1512.02325) [2]
- 'Real Time Object/Face Detection Using YOLO-v3' [Code](https://github.com/shayantaherian/Object-detection) [Paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [3]

## Reference
[1] Chen, K., Wang, J., Pang, J., Cao, Y., Xiong, Y., Li, X., Sun, S., Feng, W., Liu, Z., Xu, J., Zhang, Z., Cheng, D., Zhu, C., Cheng, T., Zhao, Q., Li, B., Lu, X., Zhu, R., Wu, Y., … Lin, D. (2019, June 17). MMDetection: Open mmlab detection toolbox and benchmark. arXiv.org. Retrieved January 29, 2023, from https://arxiv.org/abs/1906.07155.
[2] Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.-Y., &amp; Berg, A. C. (2016, December 29). SSD: Single shot multibox detector. arXiv.org. Retrieved January 29, 2023, from https://arxiv.org/abs/1512.02325.
[3] Redmon, J., &amp; Farhadi, A. (n.d.). Yolov3: An Incremental Improvement - pjreddie.com. YOLOv3: An Incremental Improvement. Retrieved January 30, 2023, from https://pjreddie.com/media/files/papers/YOLOv3.pdf.