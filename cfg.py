from mmcv import Config
from mmdet.apis import set_random_seed
from dataset import XMLCustomDataset
import os

cfg = Config.fromfile(os.path.join('mmdetection', 'configs', 'yolo', 'yolov3_d53_320_273e_coco.py'))
print(f"Default Config:\n{cfg.pretty_text}")

# Modify dataset type and path.
cfg.dataset_type = 'XMLCustomDataset'
cfg.data_root = 'input/data_root/'
cfg.data.test.type = 'XMLCustomDataset'
cfg.data.test.data_root = 'input/data_root/'
cfg.data.test.ann_file = 'dataset/ImageSets/Main/val.txt'
cfg.data.test.img_prefix = 'dataset/'
cfg.data.train.type = 'XMLCustomDataset'
cfg.data.train.data_root = 'input/data_root/'
cfg.data.train.ann_file = 'dataset/ImageSets/Main/train.txt'
cfg.data.train.img_prefix = 'dataset/'
cfg.data.val.type = 'XMLCustomDataset'
cfg.data.val.data_root = 'input/data_root/'
cfg.data.val.ann_file = 'dataset/ImageSets/Main/val.txt'
cfg.data.val.img_prefix = 'dataset/' 

# Batch size (samples per GPU).
cfg.data.samples_per_gpu = 16

# Modify number of classes as per the model head.
cfg.model.bbox_head.num_classes = 7

# Comment/Uncomment this to training from scratch/fine-tune according to the 
# model checkpoint path. 
cfg.load_from = 'checkpoints/yolov3_d53_320_273e_coco-421362b6.pth'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.008 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 5

# The output directory for training. As per the model name.
cfg.work_dir = 'outputs/yolov3_d53_320_273e_coco'

# Evaluation Metric.
cfg.evaluation.metric = 'mAP'

# Evaluation times.
cfg.evaluation.interval = 1

# Checkpoint storage interval.
cfg.checkpoint_config.interval = 15

# Set random seed for reproducible results.
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'
cfg.runner.max_epochs = 100

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')
]

# We can initialize the logger for training and have a look
# at the final config used for training
print('#'*50)
print(f'Config:\n{cfg.pretty_text}')