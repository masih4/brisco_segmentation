# set all hyper parameters
opts = {}
opts['number_of_channel'] = 3
opts['treshold'] = 0.5
opts['epoch_num'] = 60
opts['quick_run'] = 1
opts['batch_size'] = 16
opts['random_seed_num'] = 19
opts['k_fold'] = 5
opts['save_val_results'] = 1
opts['init_LR'] = 0.001
opts['LR_decay_factor'] = 0.5
opts['LR_drop_after_nth_epoch'] = 12
opts['crop_size'] = 224
opts['base_path'] = '/media/masih/wd/projects/ISPY1/data/modified_dataset/images_bias_corrected_resampled_zscored_nifti_first_post/'
opts['base_masks'] = '/media/masih/wd/projects/ISPY1/data/modified_dataset/masks_stv_manual/'
opts['cv_data'] = '/media/masih/wd/projects/ISPY1/data/kfold_cv_all_noNorm/fold'

opts['tf_version']= 2

opts['SLICE_X'] = True
opts['SLICE_Y'] = True
opts['SLICE_Z'] = True


## output directories
opts['model_save_path'] ='/media/masih/wd/projects/ISPY1/results/attention_resnet_trainall_testall_noNorm/'
opts['results_save_path'] = '/media/masih/wd/projects/ISPY1/results/attention_resnet_trainall_testall_noNorm/validation/'
opts['results_save_path_3d'] = '/media/masih/wd/projects/ISPY1/results/attention_resnet_trainall_testall_noNorm/3d_pred/'
