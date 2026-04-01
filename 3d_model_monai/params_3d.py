# set all hyper parameters
opts = {}
opts['number_of_channel'] = 3
opts['treshold'] = 0.5
opts['epoch_num'] = 500
opts['quick_run'] = 1
opts['batch_size'] = 4
opts['random_seed_num'] = 19
opts['k_fold'] = 5
opts['save_val_results'] = 1
opts['init_LR'] = 0.0001
opts['LR_decay_factor'] = 0.5
opts['LR_drop_after_nth_epoch'] = 12
opts['crop_size'] = 64
opts[
    'base_path'] = '../../data/modified_dataset/images_bias_corrected_resampled_zscored_nifti_first_post/'
opts['base_masks'] = '../../data/modified_dataset/masks_stv_manual/'
opts['tf_version'] = 2

opts['normalise'] = False

# output directories
opts['model_save_path'] = '../../results/3D/unet3d_monai/'
opts['results_save_path_3d'] = '../../results/3D/unet3d_monai/test/'
