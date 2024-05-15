"""Prepare the NSD betas for later fwRF GNet training. Load the 1.8mm fMRI NSD
betas, z-score them at each scanning session, and select the voxels of the ROI
of interest. This script additionally saves the used voxels masks in 3D volume
space, and the ncsnr of the used voxels.

Parameters
----------
sub : int
	Numbers of the used NSD subjects.
roi : str
	Used ROI.
nsd_dir : str
	Directory of the NSD.
ned_dir : str
	Neural encoding dataset directory.

"""

import argparse
import os
import numpy as np
from scipy.io import loadmat
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default=1) # [1, 2, 3, 4, 5, 6, 7, 8]
parser.add_argument('--roi', type=str, default='V1') # ['V1', 'V2', 'V3', 'hV4', 'OFA', 'FFA-1', 'FFA-2', 'OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'PPA', 'RSC', 'OPA', 'EBA', 'FBA-2', 'early', 'midventral', 'midlateral', 'midparietal', 'parietal', 'lateral', 'ventral']
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
parser.add_argument('--ned_dir', default='../neural_encoding_dataset', type=str)
args = parser.parse_args()

print('>>> Prepare NSD betas <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Get the ROI mask indices
# =============================================================================
# Select the ROI "family" from the chosen ROI
if args.roi in ['V1', 'V2', 'V3', 'hV4']:
	roi_family = 'prf-visualrois'
elif args.roi in ['OFA', 'FFA-1', 'FFA-2', 'mTL-face', 'aTL-faces']:
	roi_family = 'floc-faces'
elif args.roi in ['OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'mTL-words']:
	roi_family = 'floc-words'
elif args.roi in ['OPA', 'PPA', 'RSC']:
	roi_family = 'floc-places'
elif args.roi in ['EBA', 'FBA-1', 'FBA-2', 'mTL-bodies']:
	roi_family = 'floc-bodies'
elif args.roi in ['early', 'midventral', 'midlateral', 'midparietal', \
	'ventral', 'lateral', 'parietal']:
	roi_family = 'streams'

# Mask indices of the prf-visualrois
roi_masks_dir = os.path.join(args.nsd_dir, 'nsddata', 'ppdata', 'subj'+
	format(args.sub, '02'), 'func1pt8mm', 'roi', roi_family+'.nii.gz')
all_roi_mask_volume = nib.load(roi_masks_dir).get_fdata()
volume_shape = all_roi_mask_volume.shape
# Mapping dictionaries of the prf-visualrois
roi_maps_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer', 'subj'+
	format(args.sub, '02'), 'label', roi_family+'.mgz.ctab')
roi_map = pd.read_csv(roi_maps_dir, delimiter=' ', header=None,
	index_col=0).to_dict()[1]
# Get the voxels of the ROI of interest
if args.roi in ['V1', 'V2', 'V3']:
	rois = [args.roi+'v', args.roi+'d']
	roi_mask_volume = np.zeros(volume_shape, dtype=bool)
	roi_value = [k for k, v in roi_map.items() if v in rois]
	for r in roi_value:
		roi_mask_volume[all_roi_mask_volume==r] = True
else:
	roi_value = [k for k, v in roi_map.items() if v == args.roi][0]
	roi_mask_volume = all_roi_mask_volume == roi_value


# =============================================================================
# From volumes to vectors and vice versa
# =============================================================================
# From volume of all voxels to vector of used voxels
roi_vector = roi_mask_volume[roi_mask_volume]
# From vector of used voxels to volume of all voxels
all_rois_volume = np.zeros(volume_shape)
all_rois_volume[roi_mask_volume] = roi_vector


# =============================================================================
# Get order and ID of the presented images
# =============================================================================
# Load the experimental design info
nsd_expdesign = loadmat(os.path.join(args.nsd_dir, 'nsddata', 'experiments',
	'nsd', 'nsd_expdesign.mat'))
# Subtract 1 since the indices start with 1 (and not 0)
masterordering = nsd_expdesign['masterordering'] - 1
subjectim = nsd_expdesign['subjectim'] - 1

# Completed sessions per subject
if args.sub in (1, 2, 5, 7):
	sessions = 40
elif args.sub in (3, 6):
	sessions = 32
elif args.sub in (4, 8):
	sessions = 30

# Image presentation matrix of the selected subject
image_per_session = 750
tot_images = sessions * image_per_session
img_presentation_order = subjectim[args.sub-1,masterordering[0]][:tot_images]

# Get the train (subject-unique) image condition numbers
train_img_num = subjectim[args.sub-1,1000:]
train_img_idx = np.where(np.isin(subjectim[args.sub-1], train_img_num))[0]

# Get the val/test (shared) image condition numbers. The test images are the
# 515 shared images with 3 repeats for all subjects, and the validation images
# the remaining 485 NSD shared images
min_sess = 30
min_images = min_sess * 750
min_img_presentation = img_presentation_order[:min_images]
test_part = subjectim[args.sub-1,:1000]
val_img_num = []
test_img_num = []
for i in range(len(test_part)):
	if len(np.where(min_img_presentation == test_part[i])[0]) == 3:
		test_img_num.append(test_part[i])
	else:
		val_img_num.append(test_part[i])
val_img_num = np.asarray(val_img_num)
test_img_num = np.asarray(test_img_num)
val_img_idx = np.where(np.isin(subjectim[args.sub-1], val_img_num))[0]
test_img_idx = np.where(np.isin(subjectim[args.sub-1], test_img_num))[0]


# =============================================================================
# Prepare the fMRI betas
# =============================================================================
betas_dir = os.path.join(args.nsd_dir, 'nsddata_betas', 'ppdata', 'subj'+
	format(args.sub, '02'), 'func1pt8mm', 'betas_fithrf_GLMdenoise_RR')

for s in tqdm(range(sessions)):
	# Load the fMRI betas
	file_name = 'betas_session' + format(s+1, '02') + '.nii.gz'
	betas_sess = nib.load(os.path.join(betas_dir, file_name))
	# Get affine
	if s == 0:
		fmri_affine = betas_sess.affine
	betas_sess = betas_sess.get_fdata()
	# Mask the ROI voxels
	betas_sess = np.transpose(betas_sess[roi_mask_volume])
	# Convert back to decimal format and divide by 300
	betas_sess = betas_sess.astype(np.float32) / 300
	# Z-score the betas of each voxel within each scan session
	scaler = StandardScaler()
	betas_sess = scaler.fit_transform(betas_sess)
	betas_sess = np.nan_to_num(betas_sess)
	# Store the betas
	if s == 0:
		betas = betas_sess
	else:
		betas = np.append(betas, betas_sess, 0)


# =============================================================================
# Prepare the noise ceiling SNR
# =============================================================================
# Load the noise ceiling SNR
ncsnr = nib.load(os.path.join(betas_dir, 'ncsnr.nii.gz')).get_fdata()

# Select the voxels falling within the ROI mask
ncsnr = ncsnr[roi_mask_volume]


# =============================================================================
# Save the fMRI data
# =============================================================================
prepared_betas = {
	'volume_shape': volume_shape,
	'roi_mask_volume': roi_mask_volume,
	'fmri_affine': fmri_affine,
	'betas': betas,
	'ncsnr': ncsnr,
	'masterordering': masterordering,
	'subjectim': subjectim,
	'img_presentation_order': img_presentation_order,
	'train_img_num': train_img_num,
	'train_img_idx': train_img_idx,
	'val_img_num': val_img_num,
	'val_img_idx': val_img_idx,
	'test_img_num': test_img_num,
	'test_img_idx': test_img_idx
	}

save_dir = os.path.join(args.ned_dir, 'model_training_datasets',
	'training_dataset-nsd', 'model-fwrf', 'neural_data')

if not os.path.isdir(save_dir):
	os.makedirs(save_dir)

output_file = os.path.join(save_dir, 'nsd_betas_sub-'+format(args.sub,'02')+
	'_roi-'+args.roi+'.npy')

np.save(output_file, prepared_betas)

