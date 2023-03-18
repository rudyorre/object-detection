mkdir coco_test
cd coco_test

mkdir images
cd images
curl http://images.cocodataset.org/zips/test2017.zip -o test2017.zip
unzip test2017.zip
rm test2017.zip

cd ../
curl http://images.cocodataset.org/annotations/image_info_test2017.zip -o image_info_test2017.zip
unzip image_info_test2017.zip
rm image_info_test2017.zip
