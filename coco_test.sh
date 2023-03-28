mkdir coco_test
cd coco_test

# COCO 2017 val images
mkdir images
cd images
curl http://images.cocodataset.org/zips/val2017.zip -o val2017.zip
unzip val2017.zip
rm val2017.zip

# COCO 2017 val annotations
cd ../
curl http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o trainval2017.zip
unzip trainval2017.zip
rm trainval2017.zip

# YOLO v3 weights
cd yolo-v3
curl https://pjreddie.com/media/files/yolov3.weights -o yolov3.weights
cd ../