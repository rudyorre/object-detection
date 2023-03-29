import sys
# sys.path.append('../mmdetection/mmdet/datasets/')
# import builders
sys.path.append('../mmdetection/mmdet/datasets/')
# from builder import DATASETS
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
# import mmdet

@DATASETS.register_module()
class XMLCustomDataset(CustomDataset):
    """XML dataset for detection.
    Args:
        min_size (int | float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_size``, it would be add to ignored field.
        img_subdir (str): Subdir where images are stored. Default: JPEGImages.
        ann_subdir (str): Subdir where annotations are. Default: Annotations.
    """
    CLASSES = ('mask', 'no-mask')

    def __init__(self, min_size=None, img_subdir='JPEGImages', ann_subdir='Annotations', **kwargs):
        assert self.CLASSES or kwargs.get('classes', None), 'CLASSES in `XMLDataset` can not be None.'
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        super(XMLCustomDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size
        print(self.CLASSES)