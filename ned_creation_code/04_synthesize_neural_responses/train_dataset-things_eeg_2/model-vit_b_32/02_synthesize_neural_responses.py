"""Use the feature maps and the trained ecoding models to synthesize EEG
responses to images.

Parameters
----------
sub : int
	Number of sed subject.
n_synthetic_repeats : int
	Number synthetic data trials.
imageset : str
	Name of imageset for which the neural responses are predicted.
tot_img_partitions : int
	Total amount of image partitions in which the feature maps were divided.
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
import numpy as np
import pandas as pd
from joblib import load
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default=1) # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
parser.add_argument('--n_synthetic_repeats', default=4, type=int)
parser.add_argument('--imageset', type=str, default='nsd') # ['nsd', 'things', 'imagenet-val']
parser.add_argument('--tot_img_partitions', type=int, default=20)
parser.add_argument('--ned_dir', default='../neural_encoding_dataset', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
parser.add_argument('--imagenet_dir', default='../ILSVRC2012', type=str)
parser.add_argument('--things_dir', default='../things_database', type=str)
args = parser.parse_args()

print('>>> Synthesize neural responses <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the EEG channels and time information
# =============================================================================
data_dir = os.path.join(args.ned_dir, 'model_training_datasets',
	'train_dataset-things_eeg_2', 'model-vit_b_32', 'neural_data')

data = np.load(os.path.join(data_dir, 'eeg_sub-'+format(args.sub,'02')+
	'_split-test.npy'), allow_pickle=True).item()

ch_names = data['ch_names']
times = data['times']


# =============================================================================
# Load the trained regression weights
# =============================================================================
weights_dir = os.path.join(args.ned_dir, 'dataset', 'modality-eeg',
	'train_dataset-things_eeg_2', 'model-vit_b_32', 'trained_models_weights',
	'LinearRegression_param_sub-'+format(args.sub, '02')+'.joblib')

regression_weights = load(weights_dir)


# =============================================================================
# Load the extracted model features
# =============================================================================
for p in range(args.tot_img_partitions):
	fmap_dir = os.path.join(args.ned_dir, 'results',
		'synthesize_neural_responses', 'modality-eeg',
		'train_dataset-things_eeg_2', 'model-vit_b_32', 'imageset-'+
		args.imageset, 'feature_maps_partition-'+format(p+1, '02')+'.npy')
	if p == 0:
		X = np.load(fmap_dir)
	else:
		X = np.append(X, np.load(fmap_dir), 0)

# The encoding models are trained using only the first 250 PCs
X = X[:,:250]


# =============================================================================
# Synthesize the EEG responses
# =============================================================================
predicted_eeg = []

for r in range(args.n_synthetic_repeats):

	# Synthesize the EEG responses
	pred_eeg = regression_weights[r].predict(X)
	pred_eeg = pred_eeg.astype(np.float32)

	# Reshape the synthetic EEG data to (Images x Channels x Time)
	pred_eeg = np.reshape(pred_eeg, (len(pred_eeg), len(ch_names),
		len(times)))
	predicted_eeg.append(pred_eeg)
	del pred_eeg

# Reshape to: (Images x Repeats x Channels x Time)
predicted_eeg = np.swapaxes(np.asarray(predicted_eeg), 0, 1)
predicted_eeg = predicted_eeg.astype(np.float32)


# =============================================================================
# Save the synthetic EEG responses
# =============================================================================
save_dir = os.path.join(args.ned_dir, 'synthetic_neural_responses',
	'modality-eeg', 'train_dataset-things_eeg_2', 'model-vit_b_32',
	'imageset-'+args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'synthetic_neural_responses_training_dataset-things_eeg_2_' + \
	'model-vit_b_32_imageset-' + args.imageset + '_sub-' + \
	format(args.sub, '02') + '.h5'

# Save the h5py file
with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
	f.create_dataset('synthetic_neural_responses', data=predicted_eeg,
		dtype=np.float32)

# Read the h5py file
# data = h5py.File('predicted_data.h5', 'r')
# synthetic_neural_responses = data.get('synthetic_neural_responses')


# =============================================================================
# Prepare the metadata
# =============================================================================
metadata = {}

# EEG metadata
eeg = {}
eeg['ch_names'] = ch_names
eeg['times'] = times
metadata['eeg'] = eeg

# Encoding models metadata
encoding_models = {}
res_dir = os.path.join(args.ned_dir, 'results', 'encoding_accuracy',
	'modality-eeg', 'train_dataset-things_eeg_2', 'model-vit_b_32',
	'encoding_accuracy.npy')
accuracy = np.load(res_dir, allow_pickle=True).item()
encoding_accuracy = {}
encoding_accuracy['correlation_all_repetitions'] = np.reshape(
	accuracy['correlation_all_repetitions']['s'+str(args.sub)],
	(len(ch_names), len(times)))
encoding_accuracy['correlation_single_repetitions'] = np.reshape(
	accuracy['correlation_single_repetitions']['s'+str(args.sub)],
	(args.n_synthetic_repeats, len(ch_names), len(times)))
encoding_accuracy['noise_ceiling_upper'] = np.reshape(
	accuracy['noise_ceiling_upper']['s'+str(args.sub)],
	(len(ch_names), len(times)))
encoding_accuracy['noise_ceiling_lower'] = np.reshape(
	accuracy['noise_ceiling_lower']['s'+str(args.sub)],
	(len(ch_names), len(times)))
encoding_models['encoding_accuracy'] = encoding_accuracy
encoding_models['train_test_things_image_splits'] = np.load(
	os.path.join(data_dir, 'image_metadata.npy'), allow_pickle=True).item()
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
if args.imageset == 'imagenet_val':
	imagenet_val_labels = {}
	imagenet_val_labels['label_number'] = np.load(os.path.join(
		args.imagenet_dir, 'labels_val.npy'))
	imagenet_val_labels['label_names'] = np.load(os.path.join(
		args.imagenet_dir, 'imagenet_label_names.npy'), allow_pickle=True).item()
	metadata['imagenet_val_labels'] = imagenet_val_labels

# THINGS database metadata
if args.imageset == 'things':
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
	things_labels = {}
	things_labels['image_concept_index'] = image_concept_index
	things_labels['image_paths'] = image_paths
	things_labels['unique_id'] = unique_id
	metadata['things_labels'] = things_labels


# =============================================================================
# Save the metadata
# =============================================================================
file_name = 'synthetic_neural_responses_metadata_training_dataset-' + \
	'things_eeg_2_model-vit_b_32_imageset-' + args.imageset + '_sub-' + \
	format(args.sub, '02')

np.save(os.path.join(save_dir, file_name), metadata)

