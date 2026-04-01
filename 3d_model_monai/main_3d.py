# script to perfrom 3D segmentation on ISPY1 dataset using Monai
# Amirreza Mahbod

########################################################################################################################
# import libs
import os
from glob import glob
import random
import time
from sklearn.model_selection import KFold, StratifiedKFold
from random import shuffle
import monai
import matplotlib
import pandas as pd
from metric import *
from preprocess import *

from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    EnsureTyped,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ResizeWithPadOrCrop,
    ResizeWithPadOrCropd
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil


print_config()


from params_3d import opts
########################################################################################################################
## disabeling warning msg
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
import warnings

warnings.simplefilter('ignore')
import sys

sys.stdout.flush()  # resolving tqdm problem
########################################################################################################################
# ISPY1 dataset
train_volum = glob('{}*{}'.format(opts['base_path'] + '/', '.nii.gz'))
train_volum.sort()

mask_volum = glob('{}*{}'.format(opts['base_masks'] + '/', '.nii.gz'))
mask_volum.sort()

print('total nuber of samples:', len(train_volum))
rand_num = random.randint(0, len(train_volum))
print('random train volume:', train_volum[rand_num])
print('random mask volume:', mask_volum[rand_num])

########################################################################################################################
# main cross-validation loop
start_time_train = time.time()
kf = KFold(n_splits=opts['k_fold'], random_state=opts['random_seed_num'], shuffle=True)
kf.get_n_splits(train_volum)
current_fold = 1
dice_3d_fold = []
sample_dice = []
sample_name = []

# create folders

for idx, [train_v_index, test_v_index] in enumerate(kf.split(train_volum)):
    shuffle(train_v_index)
    shuffle(test_v_index)
    train_v_name_cv = [train_volum[name] for name in train_v_index]
    train_mask_v_name_cv = [mask_volum[name] for name in train_v_index]

    test_v_name_cv = [train_volum[name] for name in test_v_index]
    test_mask_v_name_cv = [mask_volum[name] for name in test_v_index]


    train_files = [{"image": image_name, "label": label_name} for image_name, label_name in
                  zip(train_v_name_cv, train_mask_v_name_cv)]
    val_files = [{"image": image_name, "label": label_name} for image_name, label_name in
                  zip(test_v_name_cv, test_mask_v_name_cv)]

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(opts['crop_size'], opts['crop_size'], opts['crop_size']),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
                allow_smaller= True,
            ),
            ResizeWithPadOrCropd(keys=["image", "label"],
                                 spatial_size=(opts['crop_size'], opts['crop_size'], opts['crop_size']),
                                 mode='constant')
            ,
            #user can also add other random transforms
            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=1.0, spatial_size=(opts['crop_size'], opts['crop_size'], opts['crop_size']),
            #     rotate_range=(0, 0, np.pi/15),
            #     scale_range=(0.1, 0.1, 0.1)),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ]
    )

    # check_ds = Dataset(data=val_files, transform= val_transforms)
    # check_loader = DataLoader(check_ds, batch_size=1)
    # check_data = first(check_loader)
    # image, label = (check_data["image"][0][0], check_data["label"][0][0])
    # print(f"image shape: {image.shape}, label shape: {label.shape}")
    # # plot the slice [:, :, 80]
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title("image")
    # plt.imshow(image[:, :, 80], cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.title("label")
    # plt.imshow(label[:, :, 80])
    # plt.show()

    #train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    train_ds = Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

    #val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    max_epochs = opts['epoch_num']
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    root_dir = opts['model_save_path']
    print('===========================================================')
    print('training model for fold {}:'.format(current_fold))
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (opts['crop_size'], opts['crop_size'], opts['crop_size'])
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(root_dir, "unet_3d_{}.pth".format(current_fold)))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

    # model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
    # model.eval()
    # with torch.no_grad():
    #     for i, val_data in enumerate(val_loader):
    #         roi_size = (160, 160, 160)
    #         sw_batch_size = 4
    #         val_outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model)
    #         # plot the slice [:, :, 80]
    #         plt.figure("check", (18, 6))
    #         plt.subplot(1, 3, 1)
    #         plt.title(f"image {i}")
    #         plt.imshow(val_data["image"][0, 0, :, :, 80], cmap="gray")
    #         plt.subplot(1, 3, 2)
    #         plt.title(f"label {i}")
    #         plt.imshow(val_data["label"][0, 0, :, :, 80])
    #         plt.subplot(1, 3, 3)
    #         plt.title(f"output {i}")
    #         plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 80])
    #         plt.show()
    #         if i == 2:
    #             break

    # Inference on Test Set
    #test_images = sorted(glob(os.path.join(data_dir, "imagesTs", "*.nii.gz")))
    test_images = test_v_name_cv
    test_data = [{"image": image} for image in test_images]

    test_org_transforms = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys=["image"], axcodes="RAS"),
            CropForegroundd(keys=["image"], source_key="image"),
        ]
    )

    test_org_ds = Dataset(data=test_data, transform=test_org_transforms)

    test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)

    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=test_org_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True, to_onehot=2),
            SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=opts['results_save_path_3d'], output_postfix="seg",
                       resample=False),
        ]
    )

    model.load_state_dict(torch.load(os.path.join(root_dir, "unet_3d_{}.pth".format(current_fold))))
    model.eval()

    test_mask_v_name_cv = [mask_volum[name] for name in test_v_index]

    dice_3d = []
    with torch.no_grad():
        for test_count, test_data in enumerate(test_org_loader):
            GT = readImageVolume(test_mask_v_name_cv[test_count], False)
            test_inputs = test_data["image"].to(device)
            roi_size = (opts['crop_size'], opts['crop_size'], opts['crop_size'])
            sw_batch_size = 4
            test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
            an = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
            test_data = [post_transforms(i) for i in decollate_batch(test_data)]
            #print(GT.shape)
            #print(test_data[0]['pred'][1].shape)
            dice_3d.append(get_dice_1(GT, test_data[0]['pred'][1]))
            sample_dice.append(get_dice_1(GT, test_data[0]['pred'][1]))
            name_mask = get_id_from_file_path(test_mask_v_name_cv[test_count], '.nii.gz')
            sample_name.append(name_mask)


    import gc
    del model
    # del history
    gc.collect()

    # print('average 2ddice for fold{}: {:.2f}'.format(current_fold, np.mean(dice_2d[current_fold-1, :])))
    print('average 3d_dice for fold{}: {:.2f}'.format(current_fold, np.mean(dice_3d)))
    dice_3d_fold.append(np.mean(dice_3d))
    print(dice_3d)

    current_fold = current_fold + 1

stop_time_train = time.time()
print('===========================================================')
print('Training time for {} - fold cross validation is:'.format(opts['k_fold']),
      (stop_time_train - start_time_train) / 60, 'min')
print(dice_3d_fold)
############################################################################################
## save all used parameters
opts['time'] = (stop_time_train - start_time_train) / 60
opts['dice'] = dice_3d_fold
opts['dice_mean'] = np.mean(dice_3d_fold)
opts['pytorch version used'] = torch.__version__
df_params = pd.DataFrame({'values': opts})
df_params.to_csv(opts['results_save_path_3d'] + 'params.csv', index=True, index_label='params')
df_dice_details = pd.DataFrame({'sample name': sample_name, 'sample Dice': sample_dice})
df_dice_details.to_csv(opts['results_save_path_3d'] + 'dice_details.csv', index=True, index_label='params')

