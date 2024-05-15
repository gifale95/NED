"""Fit a linear regression to predict EEG data using the DNN feature maps as
predictors. The linear regression is trained using the training images EEG data
(Y) and feature maps (X). A separate model is trained for each EEG channel and
time point, and also for each of the four EEG train data repeats: in this way,
for each input image we can have four different instances of synthetic EEG
response.

Parameters
----------
sub : int
	Used subject.
ned_dir : str
	Neural encoding dataset directory.

"""

import argparse
from utils import load_dnn_data
from utils import load_eeg_data
from utils import train_regression


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int) # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
parser.add_argument('--ned_dir', default='../neural_encoding_dataset', type=str)
args = parser.parse_args()

print('>>> Train encoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the DNN feature maps
# =============================================================================
X_train = load_dnn_data(args, 'train')


# =============================================================================
# Load the EEG data
# =============================================================================
y_train = load_eeg_data(args)


# =============================================================================
# Train the linear regression and save the regression weights
# =============================================================================
train_regression(args, X_train, y_train)

