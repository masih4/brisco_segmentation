import numpy as np
import nibabel as nib
import os
import cv2
from params_3d import opts
#from patchify import patchify, unpatchify
#from tensorflow.keras.callbacks import LearningRateScheduler

SLICE_DECIMATE_IDENTIFIER = 3
########################################################################################################################
# Normalize image
def normalizeImageIntensityRange(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))
########################################################################################################################
# Read image or mask volume
def readImageVolume(imgPath, normalize=False):
    img = nib.load(imgPath).get_fdata()
    if normalize:
        return normalizeImageIntensityRange(img)
    else:
        return img
########################################################################################################################
# Save 3d patches
def save_3dpatches(img, mask, filename, path1, path2):
    img_patches = patchify(img, (64, 64, 64), step=64)
    mask_patches = patchify(mask, (64, 64, 64), step=64)

########################################################################################################################
# Save volume slice to file
def saveSlice(img, mask, fname, path1, path2):
    img = np.uint8(img * 255)
    mask = np.uint8(mask * 255)
    fout1 = os.path.join(path1, f'{fname}.png')
    fout2 = os.path.join(path2, f'{fname}.png')
    if len(np.unique(mask)) > 1:
        cv2.imwrite(fout1, img)
        cv2.imwrite(fout2, mask)
    #print(f'[+] Slice saved: {fout}', end='\r')
########################################################################################################################
# Slice image in all directions and save
def sliceAndSaveVolumeImage(vol1, vol2,  fname, path1, path2):
    (dimx, dimy, dimz) = vol1.shape
    #print(dimx, dimy, dimz)
    cnt = 0
    if opts['SLICE_X']:
        cnt += dimx
        #print('Slicing X: ')
        for i in range(dimx):
            #saveSlice(vol[i,:,:], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_x', path)
            saveSlice(vol1[i,:,:], vol2[i,:,:], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_X', path1, path2)


    if opts['SLICE_Y']:
        cnt += dimy
        #print('Slicing Y: ')
        for i in range(dimy):
            #saveSlice(vol[:,i,:], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_y', path)
            saveSlice(vol1[:,i,:], vol2[:,i,:], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_Y', path1, path2)

    if opts['SLICE_Z']:
        cnt += dimz
        #print('Slicing Z: ')
        for i in range(dimz):
            saveSlice(vol1[:,:,i], vol2[:,:,i], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_z', path1, path2)
    return cnt
########################################################################################################################
def get_id_from_file_path(file_path, indicator):
    return file_path.split(os.path.sep)[-1].replace(indicator, '')

########################################################################################################################
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, epochs_drop=1000):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/epochs_drop))

    return LearningRateScheduler(schedule, verbose = 1)