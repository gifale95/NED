"""Create and save the metadata for NED's synthetic fMRI responses.

Parameters
----------
sub : int
	Number of sed subject.
roi : str
	Used ROI.
ned_dir : str
	Neural encoding dataset directory.

"""

import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default=1) # [1, 2, 3, 4, 5, 6, 7, 8]
parser.add_argument('--roi', type=str, default='V1') # ['V1', 'V2', 'V3', 'hV4', 'OFA', 'FFA-1', 'FFA-2', 'OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'PPA', 'RSC', 'OPA', 'EBA', 'FBA-2', 'early', 'midventral', 'midlateral', 'midparietal', 'parietal', 'lateral', 'ventral']
parser.add_argument('--ned_dir', default='../neural_encoding_dataset', type=str)
args = parser.parse_args()

print('>>> Create metadata <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Prepare the metadata
# =============================================================================
metadata = {}

# fMRI metadata
fmri = {}
betas_dir = os.path.join(args.ned_dir, 'model_training_datasets',
	'training_dataset-nsd', 'model-fwrf', 'neural_data', 'nsd_betas_sub-'+
	format(args.sub,'02')+'_roi-'+args.roi+'.npy')
betas_dict = np.load(betas_dir, allow_pickle=True).item()
fmri['ncsnr'] = betas_dict['ncsnr']
fmri['roi_mask_volume'] = betas_dict['roi_mask_volume']
fmri['fmri_affine'] = betas_dict['fmri_affine']
metadata['fmri'] = fmri

# Encoding models metadata
encoding_models = {}
res_dir = os.path.join(args.ned_dir, 'results', 'encoding_accuracy',
	'modality-fmri', 'training_dataset-nsd', 'model-fwrf',
	'encoding_accuracy.npy')
accuracy = np.load(res_dir, allow_pickle=True).item()
encoding_accuracy = {}
encoding_accuracy['r2'] = accuracy['r2']['s'+str(args.sub)+'_'+args.roi]
encoding_accuracy['noise_ceiling'] = \
	accuracy['noise_ceiling']['s'+str(args.sub)+'_'+args.roi]
encoding_accuracy['noise_normalized_encoding'] = \
	accuracy['noise_normalized_encoding']['s'+str(args.sub)+'_'+args.roi]
encoding_models['encoding_accuracy'] = encoding_accuracy
train_val_test_nsd_image_splits = {}
train_val_test_nsd_image_splits['train_img_num'] = betas_dict['train_img_num']
train_val_test_nsd_image_splits['val_img_num'] = betas_dict['val_img_num']
train_val_test_nsd_image_splits['test_img_num'] = betas_dict['test_img_num']
encoding_models['train_val_test_nsd_image_splits'] = \
	train_val_test_nsd_image_splits
encoding_models['train_val_test_nsd_image_splits'] = \
	train_val_test_nsd_image_splits
metadata['encoding_models'] = encoding_models


# =============================================================================
# Save the metadata
# =============================================================================
save_dir = os.path.join(args.ned_dir, 'encoding_models', 'modality-fmri',
	'training_dataset-nsd', 'model-fwrf', 'metadata')

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'metadata_sub-' + format(args.sub, '02') + '_roi-' + args.roi

np.save(os.path.join(save_dir, file_name), metadata)

