# script to perfrom 2D and 2.5D segmentation on volumetric medical images
# Amirreza Mahbod
# Pre-processing and data gerebarion inspired by: https://github.com/madsendennis/notebooks/tree/master/volume_segmentation_with_unet
########################################################################################################################
# import libs
import os
from glob import glob
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import time
import cv2
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
from albumentations import *
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import skimage.morphology
from skimage.io import imsave
from skimage.morphology import remove_small_objects
import tqdm
from random import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
import nibabel as nib
import random
from tqdm import tqdm
########################################################################################################################
# import functions from other scripts
from params import opts
from losses import *
from models import Attention_UNet, Attention_UNet_shallow, Attention_ResUNet_shallow, binary_unet, step_decay_schedule
from metric import *
from data_gen import *
from preprocess import *
from postprocess import *
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
# Check GPU availabelity and important package version
print("Tensorflow version:", tf.__version__)
print("Tensorflow version:", tf.keras.__version__)
## gpu configuration (comment the next two lines if use CPU)
from gpu_setting import gpu_setting

gpu_setting(opts)
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

# stats (from matlab check)
# total number of slices = 87600
# total number of Z slices = 22360 (68.7% are empty masks)
# total number of Y slices = 32620 (76.3% are empty masks)
# total number of X slices = 32620 (75.2% are empty masks)

########################################################################################################################
# main cross-validation loop
start_time_train = time.time()
kf = KFold(n_splits=opts['k_fold'], random_state=opts['random_seed_num'], shuffle=True)
kf.get_n_splits(train_volum)
current_fold = 1
dice_2d = np.zeros([opts['k_fold'], 500])  # just for sanity check
dice_3d_fold = []

for idx, [train_v_index, test_v_index] in enumerate(kf.split(train_volum)):
    shuffle(train_v_index)
    shuffle(test_v_index)
    train_v_name_cv = [train_volum[name] for name in train_v_index]
    train_mask_v_name_cv = [mask_volum[name] for name in train_v_index]

    test_v_name_cv = [train_volum[name] for name in test_v_index]
    test_mask_v_name_cv = [mask_volum[name] for name in test_v_index]

    imageSliceOutput_train = opts['cv_data'] + str(idx + 1) + '/train/img/'
    maskSliceOutput_train = opts['cv_data'] + str(idx + 1) + '/train/mask/'

    imageSliceOutput_test = opts['cv_data'] + str(idx + 1) + '/test/img/'
    maskSliceOutput_test = opts['cv_data'] + str(idx + 1) + '/test/mask/'
    ####################################################################################################################
    print('===========================================================')
    print('creating 2D (non-zero) train and test image for fold {}'.format(idx + 1))
    for i in tqdm(range(len(train_v_name_cv))):
        img = readImageVolume(train_v_name_cv[i], True)
        mask = readImageVolume(train_mask_v_name_cv[i], False)
        name_img = get_id_from_file_path(train_v_name_cv[i], '_DCE_0001_N3_zscored.nii.gz')
        name_mask = get_id_from_file_path(train_mask_v_name_cv[i], '.nii.gz')
        # print(train_img[i], img.shape, np.min(img), np.max(img))
        # print(train_mask[i], mask.shape, np.min(mask), np.max(mask))
        numOfSlices_img = sliceAndSaveVolumeImage(img, mask, name_img, imageSliceOutput_train, maskSliceOutput_train)

    for i in tqdm(range(len(test_v_name_cv))):
        img = readImageVolume(test_v_name_cv[i], True)
        mask = readImageVolume(test_mask_v_name_cv[i], False)
        name_img = get_id_from_file_path(test_v_name_cv[i], '_DCE_0001_N3_zscored.nii.gz')
        name_mask = get_id_from_file_path(test_mask_v_name_cv[i], '.nii.gz')
        #print(train_img[i], img.shape, np.min(img), np.max(img))
        #print(train_mask[i], mask.shape, np.min(mask), np.max(mask))
        numOfSlices_img = sliceAndSaveVolumeImage(img, mask, name_img, imageSliceOutput_test, maskSliceOutput_test)
    ####################################################################################################################
    train_img = glob('{}*{}'.format(opts['cv_data'] + str(idx + 1) + '/train/img/', '.png'))
    train_mask = glob('{}*{}'.format(opts['cv_data'] + str(idx + 1) + '/train/mask/', '.png'))

    test_img = glob('{}*{}'.format(opts['cv_data'] + str(idx + 1) + '/test/img/', '.png'))
    test_mask = glob('{}*{}'.format(opts['cv_data'] + str(idx + 1) + '/test/mask/', '.png'))

    # creating validation set
    validation_set_img = []
    validation_set_label = []
    for counter in range(200):
        dim = (224, 224)
        val_img = cv2.imread(test_img[counter])
        val_img = val_img / 255
        val_img = cv2.resize(val_img, dim)

        val_label = cv2.imread(test_mask[counter], -1)
        val_label = (val_label / 255).astype(np.uint8)
        val_label = cv2.resize(val_label, dim, cv2.IMREAD_UNCHANGED).astype(np.uint8)

        validation_set_img.append(val_img)
        validation_set_label.append(val_label)

    validation_set_img = np.array(validation_set_img)
    validation_set_label = np.array(validation_set_label)

    # sanity check
    # plt.figure(figsize=(15,30))
    # idx = np.random.randint(1, 200)
    # plt.subplot(3,2,1)
    # plt.imshow(validation_set_img[idx])
    # plt.subplot(3,2,2)
    # plt.imshow(validation_set_label[idx])
    # plt.show()

    model_path = opts['model_save_path'] + 'Attention_ResUNet_shallow_{}.h5'.format(current_fold)
    logger = CSVLogger(opts['model_save_path'] + 'Attention_ResUNet_shallow_{}.log'.format(current_fold))
    LR_drop = step_decay_schedule(initial_lr=opts['init_LR'],
                                  decay_factor=opts['LR_decay_factor'],
                                  epochs_drop=opts['LR_drop_after_nth_epoch'])

    model_raw = Attention_ResUNet_shallow((224, 224, 3), dropout_rate=0.1, batch_norm=True)
    # model_raw = binary_unet(opts['number_of_channel'], opts['init_LR'])
    checkpoint = ModelCheckpoint(model_path, monitor='val_dice_coef', verbose=1,
                                 save_best_only=True, mode='max', save_weights_only=True)
    print('===========================================================')
    print('training model for fold {}:'.format(current_fold))
    history = model_raw.fit_generator(data_gen(train_img,
                                               train_mask,
                                               opts['batch_size'],
                                               1,
                                               opts['crop_size'], opts['crop_size'],
                                               distance_unet_flag=0,
                                               augment=True,
                                               BACKBONE_model= '',
                                               use_pretrain_flag= False),
                                      validation_data=data_gen(test_img,
                                                               test_mask,
                                                               opts['batch_size'],
                                                               1,
                                                               opts['crop_size'], opts['crop_size'],
                                                               distance_unet_flag=0,
                                                               augment= False,
                                                               BACKBONE_model= '',
                                                               use_pretrain_flag= False),
                                      validation_steps=1,
                                      epochs=opts['epoch_num'], verbose=1,
                                      callbacks=[checkpoint, logger, LR_drop],
                                      steps_per_epoch=(len(train_img) // opts['batch_size']) // opts['quick_run'])

    model_raw.load_weights(opts['model_save_path'] + 'Attention_ResUNet_shallow_{}.h5'.format(current_fold))

    ## predication on validation set
    preds_val = model_raw.predict(validation_set_img, verbose=1, batch_size=1)
    preds_val_t = (preds_val > opts['treshold']).astype(np.uint8)
    preds_val_orgSize = []
    for i in range(len(preds_val)):
        preds_val_orgSize.append(cv2.resize(np.squeeze(preds_val[i]),  # note in original preds_test
                                            (validation_set_img.shape[1], validation_set_img.shape[
                                                1])))  # Perform a sanity check on some random training samples
    preds_val_orgSize_t = (np.array(preds_val_orgSize) > opts['treshold']).astype(np.uint8)

    ## saving results of the validation set (optional)
    res = np.array(preds_val_orgSize)
    count_val_up_ep = []
    size_val_orgSize_ep = []
    print(len(res))
    for i in range(len(res)):
        lab_mask = skimage.morphology.label(res[i] > .5)
        if np.sum(lab_mask) == 0:
            size_val_orgSize_ep.append(0)
            count_val_up_ep.append(0)
        else:
            mask_labels, mask_sizes = np.unique(lab_mask, return_counts=True)
            size_val_orgSize_ep.append(np.mean(mask_sizes[1:]))
            count_val_up_ep.append(max(mask_labels))
        # imsave(opts['results_save_path'] + '/{}_mask.png'.format(get_id_from_file_path(test_img[i], '.png')),np.round(res[i]))
        # imsave(opts['results_save_path'] + '/{}_img.png'.format(get_id_from_file_path(test_img[i], '.png')), validation_set_img[i])
        dice_2d[current_fold - 1, i] = get_dice_1(validation_set_label[i], np.round(res[i]))
    #########################################################################
    # 3d preiction
    dice_3d = []
    for test_v_counter in tqdm(range(len(test_v_name_cv))):
        imgTargetNii = nib.load(test_v_name_cv[test_v_counter])
        imgMaskNii = nib.load(test_mask_v_name_cv[test_v_counter])

        imgTarget = normalizeImageIntensityRange(imgTargetNii.get_fdata())
        imgMask = imgMaskNii.get_fdata()
        # print(np.max(imgMask), np.min(imgMask), np.shape(imgMask))

        v_prediction = predictVolume(imgTarget, model_raw, toBin=True)
        dice_3d.append(get_dice_1(imgMask, v_prediction))
        # print(dice_3d)
        temp = nib.Nifti1Image(v_prediction, imgTargetNii.affine)
        nib.save(temp, os.path.join(opts['results_save_path_3d'],
                                    get_id_from_file_path(test_mask_v_name_cv[test_v_counter], '')))

    import gc

    del model_raw
    del history
    gc.collect()

    print('average 2ddice for fold{}: {:.2f}'.format(current_fold, np.mean(dice_2d[current_fold - 1, :])))
    print('average 3ddice for fold{}: {:.2f}'.format(current_fold, np.mean(dice_3d)))
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
opts['tf version used'] = tf.__version__
opts['keras version used'] = tf.keras.__version__
df_params = pd.DataFrame({'values': opts})
df_params.to_csv(opts['results_save_path_3d'] + 'params.csv', index=True, index_label='params')
