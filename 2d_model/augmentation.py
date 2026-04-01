from albumentations import *


# augmentation function
def albumentation_aug(p=1.0, crop_size_row = 448, crop_size_col = 448 ):
    return Compose([
        RandomCrop(crop_size_row, crop_size_col, always_apply=True, p=1),
        CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
        RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, brightness_by_max=True, p=0.4),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.1),
        HorizontalFlip(always_apply=False, p=0.5),
        VerticalFlip(always_apply=False, p=0.5),
        RandomRotate90(always_apply=False, p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, interpolation=1, 
                         border_mode=4, always_apply=False, p=0.1),

    ], p=p)
