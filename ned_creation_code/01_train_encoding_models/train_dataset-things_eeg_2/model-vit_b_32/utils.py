def load_dnn_data(args, things_split):
	"""Load the DNN feature maps of the training images.

	Parameters
	----------
	args : Namespace
		Input arguments.
	things_split : str
		THINGS EEG 2 images split: 'train' or 'test'.

	Returns
	-------
	X : float
		Images feature maps.

	"""

	import numpy as np
	import os

	### Load the DNN feature maps ###
	data_dir = os.path.join(args.ned_dir, 'encoding_models', 'modality-eeg',
		'train_dataset-things_eeg_2', 'model-vit_b_32',
		'encoding_models_weights', 'pca_feature_maps_'+things_split+'.npy')
	# Load the feature maps
	X = np.load(data_dir)

	### Only use the first 250 PCs to encode EEG responses ###
	X = X[:,:250]

	### Output ###
	return X


def load_eeg_data(args):
	"""Load the EEG training data of the subject of interest.

	Parameters
	----------
	args : Namespace
		Input arguments.

	Returns
	-------
	y_train : list of float
		Training EEG data.

	"""

	import os
	import numpy as np

	### Load the EEG training data ###
	data_dir = os.path.join(args.ned_dir, 'model_training_datasets',
		'train_dataset-things_eeg_2', 'model-vit_b_32', 'neural_data',
		'eeg_sub-'+format(args.sub,'02')+'_split-train.npy')
	data = np.load(os.path.join(data_dir), allow_pickle=True).item()

	### Append the four EEG repetitions to a list ###
	y_train = []
	for r in range(data['preprocessed_eeg_data'].shape[1]):
		# Reshape to (Samples x Features)
		y_train.append(np.reshape(data['preprocessed_eeg_data'][:,r],
			(len(data['preprocessed_eeg_data']), -1)))

	### Output ###
	return y_train


def train_regression(args, X_train, y_train):
	"""Train a linear regression on the training images DNN feature maps (X)
	and training EEG data (Y), and save the trained weights.

	Parameters
	----------
	args : Namespace
		Input arguments.
	X_train : float
		Training images feature maps.
	y_train : list of float
		Training EEG data.

	"""

	import os
	import numpy as np
	from sklearn.linear_model import LinearRegression
	import copy

	### Fit the regression at each EEG repeat, time-point and channel ###
	regression_weights = {}
	for i, y in enumerate(y_train):
		reg = LinearRegression()
		reg.fit(X_train, y)
		reg_dict = {
			'coef_': reg.coef_,
			'intercept_': reg.intercept_,
			'n_features_in_': reg.n_features_in_
			}
		reg_param['rep-'+str(r+1)] = copy.deepcopy(reg_dict)
		del reg_dict

	### Save the trained regression weights ###
	save_dir = os.path.join(args.ned_dir, 'encoding_models', 'modality-eeg',
		'train_dataset-things_eeg_2', 'model-vit_b_32',
		'encoding_models_weights')
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)
	file_name = 'LinearRegression_param_sub-' + format(args.sub, '02') + '.npy'
	np.save(os.path.join(save_dir, file_name), reg_param)

