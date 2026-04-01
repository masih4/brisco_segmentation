import numpy as np
from params import opts
import cv2
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects


from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from losses import *
from tensorflow.keras.callbacks import LearningRateScheduler

def scaleImg(img, height, width):
    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

def predictVolume(inImg, model, toBin=True):
    (xMax, yMax, zMax) = inImg.shape

    outImgX = np.zeros((xMax, yMax, zMax))
    outImgY = np.zeros((xMax, yMax, zMax))
    outImgZ = np.zeros((xMax, yMax, zMax))

    cnt = 0.0
    if opts['SLICE_X']:
        cnt += 1.0
        for i in range(xMax):
            img = scaleImg(inImg[i,:,:], opts['crop_size'], opts['crop_size'])
            img_3d = np.zeros((img.shape[0],img.shape[1],3))
            img_3d[:,:,0] =img
            img_3d[:,:,1] =img
            img_3d[:,:,2] =img
            img_4d = img_3d[np.newaxis,:,:]
            tmp = model.predict(img_4d, verbose=False)[0,:,:,0]
            # tmp2 = scaleImg(tmp, yMax, zMax)
            # print(np.max(tmp2), np.min(tmp2), np.shape(tmp2))
            # rr = remove_small_objects(tmp2, min_size=30, connectivity=2)
            # tmp2[rr == 0] = 0
            # outImgX[i,:,:] = tmp2
            outImgX[i,:,:] = scaleImg(tmp, yMax, zMax)

    if opts['SLICE_Y']:
        cnt += 1.0
        for i in range(yMax):
            img = scaleImg(inImg[:,i,:], opts['crop_size'], opts['crop_size'])
            img_3d = np.zeros((img.shape[0],img.shape[1],3))
            img_3d[:,:,0] =img
            img_3d[:,:,1] =img
            img_3d[:,:,2] =img
            img_4d = img_3d[np.newaxis,:,:]
            tmp = model.predict(img_4d, verbose=False)[0,:,:,0]
            # tmp2 = scaleImg(tmp, xMax, zMax)
            # rr = remove_small_objects(tmp2, min_size=30, connectivity=2)
            # tmp2[rr == 0] = 0
            # outImgY[:,i,:] = tmp2
            outImgY[:,i,:] = scaleImg(tmp, xMax, zMax)
    if opts['SLICE_Z']:
        cnt += 1.0
        for i in range(zMax):
            img = scaleImg(inImg[:,:,i], opts['crop_size'], opts['crop_size'])
            img_3d = np.zeros((img.shape[0],img.shape[1],3))
            img_3d[:,:,0] =img
            img_3d[:,:,1] =img
            img_3d[:,:,2] =img
            img_4d = img_3d[np.newaxis,:,:]
            tmp = model.predict(img_4d, verbose=False)[0,:,:,0]
            # tmp2 = scaleImg(tmp, xMax, yMax)
            # rr = remove_small_objects(tmp2, min_size=30, connectivity=2)
            # tmp2[rr == 0] = 0
            # outImgZ[:,:,i] = tmp2
            outImgZ[:,:,i] = scaleImg(tmp, xMax, yMax)
    if opts['SLICE_Z'] and opts['SLICE_Y'] and opts['SLICE_X']:
        outImg = (outImgX + outImgY + outImgZ)/cnt
    elif opts['SLICE_Z'] and not opts['SLICE_Y']:
        outImg =  outImgZ
    elif opts['SLICE_Y'] and not opts['SLICE_Z']:
        outImg =  outImgY
    elif opts['SLICE_X']:
        outImg =  outImgX
    elif opts['SLICE_Z'] and opts['SLICE_Y']:
        outImg =  (outImgZ + outImgY)/2
    if(toBin):
        outImg[outImg>0.5] = 1.0
        outImg[outImg<=0.5] = 0.0
        #print(np.max(outImg), np.min(outImg), np.shape(outImg))
        #outImg_t = outImg > 0
        #rr = remove_small_objects(outImg_t, min_size=500, connectivity=2)
        #outImg[rr == 0] = 0
        #outImg = binary_fill_holes(outImg)
        #outImg = np.array(outImg).astype('float')
        #print(outImg.dtype)
    return outImg