import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(config):
    aug_cfg = config.get('augmentation', {})
    
    return A.Compose([
        A.RandomResizedCrop(
            height=config['training']['image_size'],
            width=config['training']['image_size'],
            scale=tuple(aug_cfg.get('crop_scale', [0.8, 1.0]))
        ),
        A.HorizontalFlip(p=aug_cfg.get('hflip', 0.5)),
        A.VerticalFlip(p=aug_cfg.get('vflip', 0.5)),
        A.ShiftScaleRotate(
            shift_limit=0.0625, 
            scale_limit=0.1, 
            rotate_limit=aug_cfg.get('rotation', 30), 
            p=0.5
        ),
        A.RandomBrightnessContrast(
            p=aug_cfg.get('brightness_contrast', 0.2)
        ),
        A.GaussNoise(
            var_limit=(10.0, 50.0), 
            p=0.2
        ),
        A.MotionBlur(
            blur_limit=aug_cfg.get('motion_blur_kernel', [5, 15]), 
            p=0.2
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def get_val_transforms(config):
    return A.Compose([
        A.Resize(height=config['training']['image_size'], width=config['training']['image_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
