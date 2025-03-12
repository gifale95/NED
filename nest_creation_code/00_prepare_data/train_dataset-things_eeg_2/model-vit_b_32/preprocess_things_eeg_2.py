"""Preprocess the raw EEG data from the THINGS EEG 2 dataset (Gifford et al.,
2022):
	- channel selection
	- filtering,
	- epoching,
	- current source density transform,
	- frequency downsampling,
	- baseline correction,
	- zscoring of each recording sessions.

After preprocessing, the EEG data is reshaped to:
(Image conditions x EEG repetitions x EEG channels x EEG time points).

The data of the test and train EEG partitions is saved independently.

Parameters
----------
sub : int
	Used subject.
n_ses : int
	Number of EEG sessions.
lowpass : float
	Lowpass filter frequency.
highpass : float
	Highpass filter frequency.
tmin : float
	Start time of the epochs in seconds, relative to stimulus onset.
tmax : float
	End time of the epochs in seconds, relative to stimulus onset.
baseline_correction : int
	Whether to baseline correct [1] or not [0] the data.
baseline_mode : str
	Whether to apply 'mean' or 'zscore' baseline correction mode.
csd : int
	Whether to transform the data into current source density [1] or not [0].
sfreq : int
	Downsampling frequency.
things_eeg_2_dir : str
	Directory of the THINGS EEG2 dataset.
	https://osf.io/3jk45/
ned_dir : str
	Neural encoding dataset directory.
	https://github.com/gifale95/NED

"""

import argparse
from preprocessing_utils import epoching
from preprocessing_utils import zscore
from preprocessing_utils import compute_ncsnr
from preprocessing_utils import save_prepr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int) # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
parser.add_argument('--n_ses', default=4, type=int)
parser.add_argument('--lowpass', default=100, type=float)
parser.add_argument('--highpass', default=0.03, type=float)
parser.add_argument('--tmin', default=-.1, type=float)
parser.add_argument('--tmax', default=.6, type=float)
parser.add_argument('--baseline_correction', default=1, type=int)
parser.add_argument('--baseline_mode', default='zscore', type=str)
parser.add_argument('--csd', default=1, type=int)
parser.add_argument('--sfreq', default=200, type=int)
parser.add_argument('--things_eeg_2_dir', default='../things_eeg_2', type=str)
parser.add_argument('--ned_dir', default='../neural_encoding_dataset', type=str)
args = parser.parse_args()

print('>>> EEG data preprocessing <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220


# =============================================================================
# Epoch and sort the data
# =============================================================================
# After preprocessing, the EEG data is reshaped to:
# (Image conditions x EEG repetitions x EEG channels x EEG time points)
# This step is applied independently to the data of each partition and session.
epoched_test, _, ch_names, times = epoching(args, 'test', seed)
epoched_train, img_conditions_train, _, _ = epoching(args, 'training', seed)


# =============================================================================
# z-scorings
# =============================================================================
# z-scoring is applied independently to the data of each session.
zscored_test, zscored_train = zscore(args, epoched_test, epoched_train)
del epoched_test, epoched_train


# =============================================================================
# Compute the noise ceiling SNR using the test data split
# =============================================================================
ncsnr = compute_ncsnr(zscored_test)


# =============================================================================
# Merge and save the preprocessed data
# =============================================================================
# In this step the data of all sessions is merged into the shape:
# (Image conditions x EEG repetitions x EEG channels x EEG time points)
# Then, the preprocessed data of the test and training data partitions is saved.
save_prepr(args, zscored_test, zscored_train, img_conditions_train, ch_names,
	times, seed, ncsnr)

