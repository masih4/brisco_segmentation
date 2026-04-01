# data generator related functions
from augmentation import *
import os
import cv2
import numpy as np

def get_id_from_file_path(file_path, indicator):
    return file_path.split(os.path.sep)[-1].replace(indicator, '')
#############################################################################################################
def chunker(seq, seq2, size):
    return ([seq[pos:pos + size], seq2[pos:pos + size]] for pos in range(0, len(seq), size))
#############################################################################################################
def data_gen(list_files, list_files2, batch_size, p , size_row, size_col, distance_unet_flag = 0,
             augment= False, BACKBONE_model = None, use_pretrain_flag = 1):
    crop_size_row = size_row
    crop_size_col = size_col
    aug = albumentation_aug(p, crop_size_row, crop_size_col)

    while True:
        for batch in chunker(list_files,list_files2, batch_size):
            X = []
            Y = []

            for count in range(len(batch[0])):
                x = cv2.resize(cv2.imread(batch[0][count]), (224,224))
                #x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                x_mask = cv2.resize(cv2.imread(batch[1][count], cv2.IMREAD_GRAYSCALE), (224,224))
                x_mask[x_mask<128] =0
                x_mask[x_mask>128] =255
                #print(np.unique(x_mask))
                
                x_mask_temp = np.zeros((x_mask.shape[0], x_mask.shape[1]))
                x_mask_temp[x_mask == 255] = 1

                #print(x.dtype, x_mask_temp.dtype)
                

                if distance_unet_flag == False:
                    if augment:
                        augmented = aug(image= x, mask= x_mask_temp)
                        x = augmented['image']
                        if use_pretrain_flag == 1:
                            x = preprocess_input(x)
                        x_mask_temp = augmented['mask']
                        x = x/255
                    else:
                        x = x/255    
                    X.append(x)
                    Y.append(x_mask_temp)
                else:
                    if augment:
                        augmented = aug(image=x, mask=x_mask)
                        x = augmented['image']
                        if use_pretrain_flag == 1:
                            x = preprocess_input(x)
                        x_mask = augmented['mask']
                        x = x/255
                    else:
                        x = x/255  
                        
                    X.append(x)
                    x_mask = (x_mask - np.min(x_mask))/ (np.max(x_mask) - np.min(x_mask) + 0.0000001)
                    Y.append(x_mask)

                del x_mask
                del x_mask_temp
                del x
            Y = np.expand_dims(np.array(Y), axis=3)
            Y = np.array(Y)
            yield np.array(X), np.array(Y)
