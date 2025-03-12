"""Create and save the metadata for NED's synthetic EEG responses.

Parameters
----------
sub : int
	Number of sed subject.
n_synthetic_repeats : int
	Number synthetic data trials.
ned_dir : str
	Neural encoding dataset directory.
	https://github.com/gifale95/NED

"""

import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default=1) # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
parser.add_argument('--n_synthetic_repeats', default=4, type=int)
parser.add_argument('--ned_dir', default='../neural_encoding_dataset', type=str)
args = parser.parse_args()

print('>>> Create metadata <<<')
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


# =============================================================================
# Save the metadata
# =============================================================================
save_dir = os.path.join(args.ned_dir, 'encoding_models', 'modality-eeg',
	'train_dataset-things_eeg_2', 'model-vit_b_32',
	'metadata')

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'metadata_sub-' + format(args.sub, '02')

np.save(os.path.join(save_dir, file_name), metadata)

