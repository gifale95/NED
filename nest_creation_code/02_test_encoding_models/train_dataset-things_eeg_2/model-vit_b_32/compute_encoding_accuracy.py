"""Use the trained encoding models to synthesize the EEG responses for the
THINGS EEG 2 test images, and compute the encoding accuracy. This code also
computes the split-half noise ceiling.

Parameters
----------
all_subs : list of int
	List with all subject numbers.
n_synthetic_repeats : int
	Number synthetic data trials.
n_iter : int
	Number of analysis iterations.
n_boot_iter : int
	Number of bootstrap iterations for the confidence intervals.
ned_dir : str
	Neural encoding dataset directory.
	https://github.com/gifale95/NED

"""

import argparse
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr
from sklearn.utils import resample
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--all_subs', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument('--n_synthetic_repeats', default=4, type=int)
parser.add_argument('--n_iter', default=10, type=int)
parser.add_argument('--n_boot_iter', default=10000, type=int)
parser.add_argument('--ned_dir', default='../neural_encoding_dataset', type=str)
args = parser.parse_args()

print('>>> Encoding models accuracy <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# # Random seed for reproducible results
# =============================================================================
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Import the THINGS EEG2 test image feature maps
# =============================================================================
fmaps_dir = os.path.join(args.ned_dir, 'encoding_models', 'modality-eeg',
	'train_dataset-things_eeg_2', 'model-vit_b_32',
	'encoding_models_weights', 'pca_feature_maps_test.npy')
X_test = np.load(fmaps_dir)

# The encoding models are trained using the first 250 PCs
X_test = X_test[:,:250]


# =============================================================================
# Output variables
# =============================================================================
correlation_all_repetitions = {}
correlation_single_repetitions = {}
noise_ceiling_upper = {}
noise_ceiling_lower = {}


# =============================================================================
# Load the trained regression weights
# =============================================================================
for sub in tqdm(args.all_subs, leave=False):

	eeg_pred = []

	weights_dir = os.path.join(args.ned_dir, 'encoding_models', 'modality-eeg',
		'train_dataset-things_eeg_2', 'model-vit_b_32',
		'encoding_models_weights', 'LinearRegression_param_sub-'+
		format(sub, '02')+'.joblib')
	regression_weights = load(weights_dir)


# =============================================================================
# Predict the responses for the test images
# =============================================================================
	for r in range(len(regression_weights)):
		reg = LinearRegression()
		reg.coef_ = regression_weights['rep-'+str(r+1)]['coef_']
		reg.intercept_ = regression_weights['rep-'+str(r+1)]['intercept_']
		reg.n_features_in_ = \
			regression_weights['rep-'+str(r+1)]['n_features_in_']
		eeg_pred.append(reg.predict(X_test))
		del reg
	del regression_weights

	# Rehape to: (Images x Repeats x Features)
	eeg_pred = np.swapaxes(np.asarray(eeg_pred), 0, 1)


# =============================================================================
# Load the biological test data
# =============================================================================
	data_dir = os.path.join(args.ned_dir, 'model_training_datasets',
		'train_dataset-things_eeg_2', 'model-vit_b_32', 'neural_data',
		'eeg_sub-'+format(sub,'02')+'_split-test.npy')
	data_dict = np.load(data_dir, allow_pickle=True).item()
	eeg_bio = data_dict['preprocessed_eeg_data']
	eeg_bio = np.reshape(eeg_bio, (len(eeg_bio), eeg_bio.shape[1], -1))
	ch_names = data_dict['ch_names']
	times = data_dict['times']
	del data_dict


# =============================================================================
# Correlate the biological and predicted data
# =============================================================================
	corr_sub_all_rep = np.zeros((args.n_iter, eeg_pred.shape[2]))
	corr_sub_single_rep = np.zeros((args.n_iter, args.n_synthetic_repeats,
		eeg_pred.shape[2]))
	nc_up_sub = np.zeros((args.n_iter, eeg_pred.shape[2]))
	nc_low_sub = np.zeros((args.n_iter, eeg_pred.shape[2]))

	# Average across all the biological data repetitions for the noise ceiling
	# upper bound calculation
	eeg_bio_avg_all = np.mean(eeg_bio, 1)

	# Loop over iterations
	for i in range(args.n_iter):
		# Random data repetitions index
		shuffle_idx = resample(np.arange(0, eeg_bio.shape[1]), replace=False,
			n_samples=int(eeg_bio.shape[1]/2))
		# Average across one half of the biological data repetitions
		eeg_bio_avg_half_1 = np.mean(np.delete(eeg_bio, shuffle_idx, 1), 1)
		# Average across the other half of the biological data repetitions for
		# the noise ceiling lower bound calculation
		eeg_bio_avg_half_2 = np.mean(eeg_bio[:,shuffle_idx], 1)

		# Compute the correlation and noise ceiling
		for f in range(eeg_bio.shape[2]):
			# Noise ceiling lower boud
			nc_low_sub[i,f] = corr(eeg_bio_avg_half_2[:,f],
				eeg_bio_avg_half_1[:,f])[0]
			# Noise ceiling upper bound
			nc_up_sub[i,f] = corr(eeg_bio_avg_all[:,f],
				eeg_bio_avg_half_1[:,f])[0]
			# Encoding accuracy (all repeats)
			corr_sub_all_rep[i,f] = corr(np.mean(eeg_pred[:,:,f], 1),
				eeg_bio_avg_half_1[:,f])[0]
			# Encoding accuracy (single repeats)
			for r in range(args.n_synthetic_repeats):
				corr_sub_single_rep[i,r,f] = corr(eeg_pred[:,r,f],
					eeg_bio_avg_half_1[:,f])[0]

	# Average the results across iterations
	correlation_all_repetitions['s'+str(sub)] = np.mean(corr_sub_all_rep, 0)
	correlation_single_repetitions['s'+str(sub)] = np.mean(
		corr_sub_single_rep, 0)
	noise_ceiling_upper['s'+str(sub)] = np.mean(nc_up_sub, 0)
	noise_ceiling_lower['s'+str(sub)] = np.mean(nc_low_sub, 0)

	del eeg_bio, eeg_pred, corr_sub_all_rep, corr_sub_single_rep, nc_up_sub, \
		nc_low_sub


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
# Extract the data
corr_all_rep_list = []
corr_single_rep_list = []
for s in args.all_subs:
	corr_all_rep_list.append(correlation_all_repetitions['s'+str(s)])
	corr_single_rep_list.append(correlation_single_repetitions['s'+str(s)])
corr_all_rep_list = np.asarray(corr_all_rep_list)
corr_single_rep_list = np.asarray(corr_single_rep_list)

# Empty CI matrices
ci_lower_all_repetitions = np.zeros((corr_all_rep_list.shape[1]),
	dtype=np.float32)
ci_upper_all_repetitions = np.zeros((corr_all_rep_list.shape[1]),
	dtype=np.float32)
ci_lower_single_repetitions = np.zeros((args.n_synthetic_repeats,
	corr_single_rep_list.shape[2]), dtype=np.float32)
ci_upper_single_repetitions = np.zeros((args.n_synthetic_repeats,
	corr_single_rep_list.shape[2]), dtype=np.float32)

# Compute the CIs
for f in range(corr_all_rep_list.shape[1]):
	sample_dist_all_rep = np.zeros(args.n_boot_iter)
	sample_dist_single_rep = np.zeros((args.n_boot_iter,
		args.n_synthetic_repeats))
	for i in tqdm(range(args.n_boot_iter)):
		idx = resample(np.arange(len(args.all_subs)))
		sample_dist_all_rep[i] = np.mean(corr_all_rep_list[idx,f])
		for r in range(args.n_synthetic_repeats):
			sample_dist_single_rep[i,r] = np.mean(corr_single_rep_list[idx,r,f])
	ci_lower_all_repetitions[f] = np.percentile(sample_dist_all_rep, 2.5)
	ci_upper_all_repetitions[f] = np.percentile(sample_dist_all_rep, 97.5)
	ci_lower_single_repetitions[:,f] = np.percentile(sample_dist_single_rep,
		2.5, axis=0)
	ci_upper_single_repetitions[:,f] = np.percentile(sample_dist_single_rep,
		97.5, axis=0)


# =============================================================================
# t-tests & multiple comparisons correction
# =============================================================================
# p_values = {}
# significance = {}
# p_values_mat = np.zeros((len(args.all_rois)))
# for r, roi in enumerate(args.all_rois):
# 	mean_corr = []
# 	for s in args.all_subs:
# 		mean_corr.append(np.mean(
# 			noise_normalized_encoding['s'+str(s)+'_'+roi]))
# 	mean_corr = np.asarray(mean_corr)
# 	_, p_values_mat[r] = ttest_1samp(mean_corr, 0,
# 		alternative='greater')
# sig, p_val, _, _ = multipletests(p_values_mat, 0.05, 'bonferroni')
# for r, roi in enumerate(args.all_rois):
# 	significance[roi] = sig[r]
# 	p_values[roi] = p_val[r]


# =============================================================================
# Save the results
# =============================================================================
correlation_results = {
	'correlation_all_repetitions': correlation_all_repetitions,
	'correlation_single_repetitions': correlation_single_repetitions,
	'noise_ceiling_upper': noise_ceiling_upper,
	'noise_ceiling_lower': noise_ceiling_lower,
	'ci_lower_all_repetitions': ci_lower_all_repetitions,
	'ci_upper_all_repetitions': ci_upper_all_repetitions,
	'ci_lower_single_repetitions': ci_lower_single_repetitions,
	'ci_upper_single_repetitions': ci_upper_single_repetitions,
	'ch_names': ch_names,
	'times': times
# 	'p_values': p_values,
# 	'significance': significance
}

save_dir = os.path.join(args.ned_dir, 'results', 'encoding_accuracy',
	'modality-eeg', 'train_dataset-things_eeg_2', 'model-vit_b_32')

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'encoding_accuracy.npy'

np.save(os.path.join(save_dir, file_name), correlation_results)

