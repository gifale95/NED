"""Use the trained fwRF GNet ecoding models to predict the brain responses
for the NSD or ImageNet val split images.
The 'lateral' and 'ventral' ROIs were too big to be predicted by a single fwrf
model (not enough GPU RAM), and therefore were split into two partitions.

Parameters
----------
sub : int
	Number of sed subject.
roi : str
	Used ROI.
imageset : str
	Name of imageset for which the neural responses are predicted.
ned_dir : str
	Neural encoding dataset directory.
nsd_dir : str
	Directory of the NSD.
imagenet_dir : str
	Directory of the ImageNet dataset.
things_dir : str
	Directory of the THINGS database.

"""

import argparse
import os
import random
import numpy as np
import torch
import torchvision
from torchvision import transforms as trn
from tqdm import tqdm
from PIL import Image
import nibabel as nib
import h5py
import pandas as pd
from scipy.io import loadmat

from src_new.load_nsd import image_feature_fn
from src_new.torch_joint_training_unpacked_sequences import *
from src_new.torch_gnet import Encoder
from src_new.torch_mpf import Torch_LayerwiseFWRF

parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default=1)
parser.add_argument('--roi', type=str, default='V1')
parser.add_argument('--imageset', type=str, default='nsd')
#parser.add_argument('--ned_dir', default='/home/ale/aaa_stuff/PhD/projects/neural_encoding_dataset', type=str)
#parser.add_argument('--nsd_dir', default='/media/ale/Elements/PhD/datasets/natural-scenes-dataset', type=str)
#parser.add_argument('--imagenet_dir', default='/media/ale/Elements/PhD/datasets/ILSVRC2012/images', type=str)
#parser.add_argument('--things_dir', default='/media/ale/Elements/PhD/datasets/things_database', type=str)
#parser.add_argument('--ned_dir', default='/home/ale/scratch/projects/neural_encoding_dataset', type=str)
#parser.add_argument('--nsd_dir', default='/home/ale/scratch/datasets/natural-scenes-dataset', type=str)
#parser.add_argument('--imagenet_dir', default='/home/ale/scratch/datasets/image_sets/ILSVRC2012', type=str)
#parser.add_argument('--things_dir', default='/home/ale/scratch/datasets/image_sets/things_database', type=str)
parser.add_argument('--ned_dir', default='/scratch/giffordale95/projects/neural_encoding_dataset', type=str)
parser.add_argument('--nsd_dir', default='/scratch/giffordale95/datasets/natural-scenes-dataset', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/giffordale95/datasets/image_sets/ILSVRC2012', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
args = parser.parse_args()

print('>>> Synthesize neural responses <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Set random seeds to make results reproducible
# =============================================================================
# Random seeds
seed = (args.sub * 100) + (np.sum([ord(c) for c in args.roi]))
seed = int(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Generator object for DataLoader random batching
g_cpu = torch.Generator()
g_cpu.manual_seed(seed)


# =============================================================================
# Computing resources
# =============================================================================
# Checking for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
	torch.backends.cudnn.enabled=True
	print ('#device:', torch.cuda.device_count())
	print ('device#:', torch.cuda.current_device())
	print ('device name:', torch.cuda.get_device_name(
		torch.cuda.current_device()))

batch_size = 100


# =============================================================================
# Import the image datasets
# =============================================================================
if args.imageset == 'nsd':
	img_dir = os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli',
		'nsd', 'nsd_stimuli.hdf5')
	img_dataset = h5py.File(img_dir, 'r').get('imgBrick')

elif args.imageset == 'imagenet_val':
	img_dataset = torchvision.datasets.ImageNet(root=args.imagenet_dir,
		split='val')

elif args.imageset == 'things':
	image_concept_index = np.squeeze(pd.read_csv(os.path.join(args.things_dir,
		'THINGS', 'Metadata', 'Concept-specific', 'image_concept_index.csv'),
		header=None).values) - 1
	image_paths_df = pd.read_csv(os.path.join(args.things_dir, 'THINGS',
		'Metadata', 'Image-specific', 'image_paths.csv'), header=None)
	unique_id_df = pd.read_csv(os.path.join(args.things_dir, 'THINGS',
		'Metadata', 'Concept-specific', 'unique_id.csv'), header=None)
	image_paths = {}
	for i in range(len(image_concept_index)):
		image_paths[i] = image_paths_df[0][i]
	unique_id = {}
	for i in range(len(unique_id_df)):
		unique_id[i] = unique_id_df[0][i]
	img_dataset = image_paths


# =============================================================================
# Load the trained encoding model weights
# =============================================================================
# Subjects and ROI directory
nsd_subjects = [args.sub]
if args.roi in ['lateral', 'ventral']:
	subj_roi_dir_1 = 'sub-' + format(args.sub, '02') + '_roi-' + args.roi + '_split-1'
	subj_roi_dir_2 = 'sub-' + format(args.sub, '02') + '_roi-' + args.roi + '_split-2'
else:
	subj_roi_dir = 'sub-' + format(args.sub, '02') + '_roi-' + args.roi

# Total model directory
if args.roi in ['lateral', 'ventral']:
	model_dir_1 = os.path.join(args.ned_dir, 'dataset', 'modality-fmri',
		'training_dataset-nsd', 'model-fwrf', 'trained_models_weights',
		'weights_'+subj_roi_dir_1+'.pt')
	model_dir_2 = os.path.join(args.ned_dir, 'dataset', 'modality-fmri',
		'training_dataset-nsd', 'model-fwrf', 'trained_models_weights',
		'weights_'+subj_roi_dir_2+'.pt')
else:
	model_dir = os.path.join(args.ned_dir, 'dataset', 'modality-fmri',
		'training_dataset-nsd', 'model-fwrf', 'trained_models_weights',
		'weights_'+subj_roi_dir+'.pt')

# Load the model
if args.roi in ['lateral', 'ventral']:
	trained_model_1 = torch.load(model_dir_1, map_location=torch.device('cpu'))
	trained_model_2 = torch.load(model_dir_2, map_location=torch.device('cpu'))
	stim_mean = trained_model_1['stim_mean']
else:
	trained_model = torch.load(model_dir, map_location=torch.device('cpu'))
	stim_mean = trained_model['stim_mean']


# =============================================================================
# Model instantiation
# =============================================================================
# Voxel number
if args.roi in ['lateral', 'ventral']:
	nnv_1 = {}
	nnv_2 = {}
	nnv_1[args.sub] = len(trained_model_1['best_params']['fwrfs'][args.sub]['b'])
	nnv_2[args.sub] = len(trained_model_2['best_params']['fwrfs'][args.sub]['b'])
else:
	nnv = {}
	nnv[args.sub] = len(trained_model['best_params']['fwrfs'][args.sub]['b'])

# Load 20 images:
# (Images × Image channels × Resized image height × Resized image width)
img_chan = 3
resize_px = 227
stim_data = {}
stim_data[nsd_subjects[0]] = np.zeros((20, img_chan, resize_px, resize_px),
	dtype=np.float32)
for i in range(20):
	if args.imageset == 'nsd':
		img = img_dataset[i]
		img = Image.fromarray(np.uint8(img))
	elif args.imageset == 'imagenet_val':
		img, _ = img_dataset.__getitem__(i)
	elif args.imageset == 'things':
		img = Image.open(os.path.join(args.things_dir, 'THINGS',
			img_dataset[i])).convert('RGB')
	min_size = min(img.size)
	transform = trn.Compose([
		trn.CenterCrop(min_size),
		trn.Resize((resize_px,resize_px))
	])
	img = transform(img)
	img = np.asarray(img)
	img = img.transpose(2,0,1)
	img = image_feature_fn(img)
	stim_data[i] = img

# Model functions
_log_act_fn = lambda _x: torch.log(1 + torch.abs(_x))*torch.tanh(_x)

def _model_fn(_ext, _con, _x):
	'''model consists of an extractor (_ext) and a connection model (_con)'''
	_y, _fm, _h = _ext(_x)
	return _con(_fm)

def _pred_fn(_ext, _con, xb):
	return _model_fn(_ext, _con, torch.from_numpy(xb).to(device))

# Shared encoder model
if args.roi in ['lateral', 'ventral']:
	shared_model_1 = Encoder(mu=stim_mean, trunk_width=64,
		use_prefilter=1).to(device)
	shared_model_2 = Encoder(mu=stim_mean, trunk_width=64,
		use_prefilter=1).to(device)
	rec, fmaps, h = shared_model_1(torch.from_numpy(
		stim_data[nsd_subjects[0]]).to(device))
else:
	shared_model = Encoder(mu=stim_mean, trunk_width=64,
		use_prefilter=1).to(device)
	rec, fmaps, h = shared_model(torch.from_numpy(
		stim_data[nsd_subjects[0]]).to(device))

# Subject specific FWRF models
if args.roi in ['lateral', 'ventral']:
	subject_fwrfs_1 = {s: Torch_LayerwiseFWRF(fmaps, nv=nnv_1[s], pre_nl=_log_act_fn, \
		post_nl=_log_act_fn, dtype=np.float32).to(device) for s in nsd_subjects}
	subject_fwrfs_2 = {s: Torch_LayerwiseFWRF(fmaps, nv=nnv_2[s], pre_nl=_log_act_fn, \
		post_nl=_log_act_fn, dtype=np.float32).to(device) for s in nsd_subjects}
else:
	subject_fwrfs = {s: Torch_LayerwiseFWRF(fmaps, nv=nnv[s], pre_nl=_log_act_fn, \
		post_nl=_log_act_fn, dtype=np.float32).to(device) for s in nsd_subjects}


# =============================================================================
# Load the pretrained weights into the model
# =============================================================================
if args.roi in ['lateral', 'ventral']:
	shared_model_1.load_state_dict(trained_model_1['best_params']['enc'])
	shared_model_2.load_state_dict(trained_model_2['best_params']['enc'])
	for s,sd in subject_fwrfs_1.items():
		sd.load_state_dict(trained_model_1['best_params']['fwrfs'][s])
	for s,sd in subject_fwrfs_2.items():
		sd.load_state_dict(trained_model_2['best_params']['fwrfs'][s])
else:
	shared_model.load_state_dict(trained_model['best_params']['enc'])
	for s,sd in subject_fwrfs.items():
		sd.load_state_dict(trained_model['best_params']['fwrfs'][s])

if args.roi in ['lateral', 'ventral']:
	shared_model_1.eval()
	shared_model_2.eval()
	for s,sd in subject_fwrfs_1.items():
		sd.eval()
	for s,sd in subject_fwrfs_2.items():
		sd.eval()
else:
	shared_model.eval()
	for s,sd in subject_fwrfs.items():
		sd.eval()


# =============================================================================
# Predict the responses to all images
# =============================================================================
if args.roi in ['lateral', 'ventral']:
	predicted_fmri_1 = np.zeros((len(img_dataset), nnv_1[args.sub]),
		dtype=np.float32)
	predicted_fmri_2 = np.zeros((len(img_dataset), nnv_2[args.sub]),
		dtype=np.float32)
else:
	predicted_fmri = np.zeros((len(img_dataset), nnv[args.sub]),
		dtype=np.float32)

if args.roi in ['lateral', 'ventral']:
	for i in tqdm(range(len(img_dataset))):
		if args.imageset == 'nsd':
			img = img_dataset[i]
			img = Image.fromarray(np.uint8(img))
		elif args.imageset == 'imagenet_val':
			img, _ = img_dataset.__getitem__(i)
		elif args.imageset == 'things':
			img = Image.open(os.path.join(args.things_dir, 'THINGS',
				img_dataset[i])).convert('RGB')
		min_size = min(img.size)
		transform = trn.Compose([
			trn.CenterCrop(min_size),
			trn.Resize((resize_px,resize_px))
		])
		img = transform(img)
		img = np.asarray(img)
		img = img.transpose(2,0,1)
		img = image_feature_fn(img)
		img = np.expand_dims(img, 0)
		sd_1 = subject_fwrfs_1[args.sub]
		sd_2 = subject_fwrfs_2[args.sub]
		with torch.no_grad():
			predicted_fmri_1[i] = subject_pred_pass(_pred_fn,
				shared_model_1, sd_1, img, batch_size)
			predicted_fmri_2[i] = subject_pred_pass(_pred_fn,
				shared_model_2, sd_2, img, batch_size)
else:
	with torch.no_grad():
		for s, sd in subject_fwrfs.items():
			for i in tqdm(range(len(img_dataset))):
				if args.imageset == 'nsd':
					img = img_dataset[i]
					img = Image.fromarray(np.uint8(img))
				elif args.imageset == 'imagenet_val':
					img, _ = img_dataset.__getitem__(i)
				elif args.imageset == 'things':
					img = Image.open(os.path.join(args.things_dir, 'THINGS',
						img_dataset[i])).convert('RGB')
				min_size = min(img.size)
				transform = trn.Compose([
					trn.CenterCrop(min_size),
					trn.Resize((resize_px,resize_px))
				])
				img = transform(img)
				img = np.asarray(img)
				img = img.transpose(2,0,1)
				img = image_feature_fn(img)
				img = np.expand_dims(img, 0)
				predicted_fmri[i] = subject_pred_pass(_pred_fn, shared_model, sd,
					img, batch_size)

if args.roi in ['lateral', 'ventral']:
	predicted_fmri = np.append(predicted_fmri_1, predicted_fmri_2, 1)


# =============================================================================
# Save the synthetic fMRI responses
# =============================================================================
save_dir = os.path.join(args.ned_dir, 'dataset', 'modality-fmri',
	'training_dataset-nsd', 'model-fwrf', 'synthetic_neural_responses',
	'imageset-'+args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'synthetic_neural_responses_training_dataset-nsd_model-fwrf_' + \
	'imageset-' + args.imageset + '_sub-' + format(args.sub, '02') + \
	'_roi-' + args.roi + '.h5'

# Save the h5py file
with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
	f.create_dataset('synthetic_neural_responses', data=predicted_fmri,
		dtype=np.float32)

# Read the h5py file
# data = h5py.File('predicted_data.h5', 'r')
# synthetic_neural_responses = data.get('synthetic_neural_responses')


# =============================================================================
# Prepare the metadata
# =============================================================================
metadata = {}
if args.roi in ['lateral', 'ventral']:
	trained_model = trained_model_1

# fMRI metadata
fmri = {}
fmri['ncsnr'] = trained_model['betas_info'][args.sub]['ncsnr']
fmri['roi_mask_volume'] = \
	trained_model['betas_info'][args.sub]['roi_mask_volume']
nii_volume = nib.load(os.path.join(args.nsd_dir, 'nsddata_betas', 'ppdata',
	'subj0'+str(args.sub), 'func1pt8mm', 'betas_fithrf_GLMdenoise_RR',
	'ncsnr.nii.gz'))
fmri['fmri_affine'] = nii_volume.affine
metadata['fmri'] = fmri

# Encoding models metadata
encoding_models = {}
res_dir = os.path.join(args.ned_dir, 'results',
	'encoding_models_prediction_accuracy', 'modality-fmri',
	'training_dataset-nsd', 'model-fwrf', 'prediction_accuracy.npy')
accuracy = np.load(res_dir, allow_pickle=True).item()
encoding_accuracy = {}
encoding_accuracy['r2'] = accuracy['r2']['s'+str(args.sub)+'_'+args.roi]
encoding_accuracy['noise_ceiling'] = \
	accuracy['noise_ceiling']['s'+str(args.sub)+'_'+args.roi]
encoding_accuracy['noise_normalized_encoding'] = \
	accuracy['noise_normalized_encoding']['s'+str(args.sub)+'_'+args.roi]
encoding_models['encoding_accuracy'] = encoding_accuracy
train_val_test_nsd_image_splits = {}
train_val_test_nsd_image_splits['train_img_num'] = \
	trained_model['betas_info'][args.sub]['train_img_num']
train_val_test_nsd_image_splits['val_img_num'] = \
	trained_model['betas_info'][args.sub]['val_img_num']
train_val_test_nsd_image_splits['test_img_num'] = \
	trained_model['betas_info'][args.sub]['test_img_num']
encoding_models['train_val_test_nsd_image_splits'] = \
	train_val_test_nsd_image_splits
encoding_models['train_val_test_nsd_image_splits'] = \
	train_val_test_nsd_image_splits
metadata['encoding_models'] = encoding_models

# NSD metadata
if args.imageset == 'nsd':
	nsd_labels = {}
	nsd_expdesign = loadmat(os.path.join(args.nsd_dir, 'nsddata', 'experiments',
		'nsd', 'nsd_expdesign.mat'))
	nsd_labels['subjectim'] = nsd_expdesign['subjectim'] - 1
	nsd_labels['masterordering'] = nsd_expdesign['masterordering'] - 1
	metadata['nsd_labels'] = nsd_labels

# ImageNet-val metadata
elif args.imageset == 'imagenet_val':
	imagenet_val_labels = {}
	imagenet_val_labels['label_number'] = np.load(os.path.join(
		args.imagenet_dir, 'labels_val.npy'))
	imagenet_val_labels['label_names'] = np.load(os.path.join(
		args.imagenet_dir, 'imagenet_label_names.npy'), allow_pickle=True).item()
	metadata['imagenet_val_labels'] = imagenet_val_labels

# THINGS database metadata
if args.imageset == 'things':
	things_labels = {}
	things_labels['image_concept_index'] = image_concept_index
	things_labels['image_paths'] = image_paths
	things_labels['unique_id'] = unique_id
	metadata['things_labels'] = things_labels


# =============================================================================
# Save the metadata
# =============================================================================
file_name = 'synthetic_neural_responses_metadata_training_dataset-' + \
	'nsd_model-fwrf_imageset-' + args.imageset + '_sub-' + \
	format(args.sub, '02') + '_roi-' + args.roi

np.save(os.path.join(save_dir, file_name), metadata)

