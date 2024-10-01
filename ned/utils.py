import os
import numpy as np
from tqdm import tqdm
import torch
from copy import deepcopy
import torchvision
from torchvision import transforms as trn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from ned.models.fwrf.load_nsd import image_feature_fn
from ned.models.fwrf.torch_joint_training_unpacked_sequences import *
from ned.models.fwrf.torch_gnet import Encoder
from ned.models.fwrf.torch_mpf import Torch_LayerwiseFWRF


def get_model_fmri_nsd_fwrf(ned_dir, subject, roi, device):
	"""
	Load the feature-weighted receptive field (fwrf) encoding model
	(St-Yves & Naselaris, 2018).

	This code is an adapted version of the code from the paper:
	Allen, E.J., St-Yves, G., Wu, Y., Breedlove, J.L., Prince, J.S., Dowdle,
		L.T., Nau, M., Caron, B., Pestilli, F., Charest, I. and Hutchinson,
		J.B., 2022. A massive 7T fMRI dataset to bridge cognitive neuroscience
		and artificial intelligence. Nature neuroscience, 25(1), pp.116-126.

	The original code, written by Ghislain St-Yves, can be found here:
	https://github.com/styvesg/nsd_gnet8x

	The 'lateral' and 'ventral' ROIs were too big to be modeled by a
	single fwrf model (not enough GPU RAM), and therefore were split
	into two partitions.

	Parameters
	----------
	ned_dir : str
		Path to the "neural_encoding_dataset" folder.
	subject : int
		Subject number for which the encoding model was trained.
	roi : str
		Only required if modality=='fmri'. Name of the Region of Interest
		(ROI) for which the fMRI encoding model was trained.
	device : str
		Whether the encoding model is stored on the 'cpu' or 'cuda'. If 'auto',
		the code will use GPU if available, and otherwise CPU.

	Returns
	-------
	encoding_model : dict
		Neural encoding model.
	"""

	### Load the trained encoding model weights ###
	# Trained model directory
	if roi in ['lateral', 'ventral']:
		model_dir_1 = os.path.join(ned_dir, 'encoding_models', 'modality-fmri',
			'train_dataset-nsd', 'model-fwrf', 'encoding_models_weights',
			'weights_'+'sub-'+format(subject,'02')+'_roi-'+roi+'_split-1'+'.pt')
		model_dir_2 = os.path.join(ned_dir, 'encoding_models', 'modality-fmri',
			'train_dataset-nsd', 'model-fwrf', 'encoding_models_weights',
			'weights_'+'sub-'+format(subject,'02')+'_roi-'+roi+'_split-2'+'.pt')
	else:
		model_dir = os.path.join(ned_dir, 'encoding_models', 'modality-fmri',
			'train_dataset-nsd', 'model-fwrf', 'encoding_models_weights',
			'weights_'+'sub-'+format(subject,'02')+'_roi-'+roi+'.pt')

	# Load the model
	if roi in ['lateral', 'ventral']:
		trained_model_1 = torch.load(model_dir_1,
			map_location=torch.device('cpu'))
		trained_model_2 = torch.load(model_dir_2,
			map_location=torch.device('cpu'))
		stim_mean = trained_model_1['stim_mean']
	else:
		trained_model = torch.load(model_dir,
			map_location=torch.device('cpu'))
		stim_mean = trained_model['stim_mean']

	### Model instantiation ###
	# Voxel number
	if roi in ['lateral', 'ventral']:
		nnv_1 = {}
		nnv_2 = {}
		nnv_1[subject] = len(
			trained_model_1['best_params']['fwrfs'][subject]['b'])
		nnv_2[subject] = len(
			trained_model_2['best_params']['fwrfs'][subject]['b'])
	else:
		nnv = {}
		nnv[subject] = len(trained_model['best_params']['fwrfs'][subject]['b'])

	# Create a 4-D dummy images array for model instantiation of shape:
	# (Images × Image channels × Resized image height × Resized image width)
	img_chan = 3
	resize_px = 227
	dummy_images = np.random.randint(0, 255,
		(20, img_chan, resize_px, resize_px))
	dummy_images = image_feature_fn(dummy_images)

	# Shared encoder model
	if roi in ['lateral', 'ventral']:
		shared_model_1 = Encoder(mu=stim_mean, trunk_width=64,
			use_prefilter=1).to(device)
		shared_model_2 = Encoder(mu=stim_mean, trunk_width=64,
			use_prefilter=1).to(device)
		rec, fmaps, h = shared_model_1(torch.from_numpy(
			dummy_images).to(device))
	else:
		shared_model = Encoder(mu=stim_mean, trunk_width=64,
			use_prefilter=1).to(device)
		rec, fmaps, h = shared_model(torch.from_numpy(dummy_images).to(device))

	# Subject specific fwrf models
	_log_act_fn = lambda _x: torch.log(1 + torch.abs(_x))*torch.tanh(_x)
	if roi in ['lateral', 'ventral']:
		subject_fwrfs_1 = {s: Torch_LayerwiseFWRF(
			fmaps, nv=nnv_1[s], pre_nl=_log_act_fn, post_nl=_log_act_fn,
			dtype=np.float32).to(device) for s in [subject]}
		subject_fwrfs_2 = {s: Torch_LayerwiseFWRF(
			fmaps, nv=nnv_2[s], pre_nl=_log_act_fn, post_nl=_log_act_fn,
			dtype=np.float32).to(device) for s in [subject]}
	else:
		subject_fwrfs = {s: Torch_LayerwiseFWRF(
			fmaps, nv=nnv[s], pre_nl=_log_act_fn, post_nl=_log_act_fn,
			dtype=np.float32).to(device) for s in [subject]}

	### Load the pretrained weights into the model ###
	if roi in ['lateral', 'ventral']:
		shared_model_1.load_state_dict(
			trained_model_1['best_params']['enc'])
		shared_model_2.load_state_dict(
			trained_model_2['best_params']['enc'])
		for s,sd in subject_fwrfs_1.items():
			sd.load_state_dict(trained_model_1['best_params']['fwrfs'][s])
		for s,sd in subject_fwrfs_2.items():
			sd.load_state_dict(trained_model_2['best_params']['fwrfs'][s])
	else:
		shared_model.load_state_dict(trained_model['best_params']['enc'])
		for s,sd in subject_fwrfs.items():
			sd.load_state_dict(trained_model['best_params']['fwrfs'][s])
	if roi in ['lateral', 'ventral']:
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

	### Store the encoding model into a dictionary ###
	encoding_model = {}
	if roi in ['lateral', 'ventral']:
		encoding_model['shared_model_1'] = shared_model_1
		encoding_model['shared_model_2'] = shared_model_2
		encoding_model['subject_fwrfs_1'] = subject_fwrfs_1
		encoding_model['subject_fwrfs_2'] = subject_fwrfs_2
		encoding_model['nnv_1'] = nnv_1
		encoding_model['nnv_2'] = nnv_2
	else:
		encoding_model['shared_model'] = shared_model
		encoding_model['subject_fwrfs'] = subject_fwrfs
		encoding_model['nnv'] = nnv
	encoding_model['resize_px'] = resize_px

	### Output ###
	return encoding_model


def encode_fmri_nsd_fwrf(encoding_model, images, device):
	"""
	Synthesize fMRI responses for the input images using the feature-weighted
	receptive field (fwrf) encoding model (St-Yves & Naselaris, 2018).

	This code is an adapted version of the code from the paper:
	Allen, E.J., St-Yves, G., Wu, Y., Breedlove, J.L., Prince, J.S., Dowdle,
		L.T., Nau, M., Caron, B., Pestilli, F., Charest, I. and Hutchinson,
		J.B., 2022. A massive 7T fMRI dataset to bridge cognitive neuroscience
		and artificial intelligence. Nature neuroscience, 25(1), pp.116-126.

	The original code, written by Ghislain St-Yves, can be found here:
	https://github.com/styvesg/nsd_gnet8x

	The 'lateral' and 'ventral' ROIs were too big to be modeled by a
	single fwrf model (not enough GPU RAM), and therefore were split
	into two partitions.

	Parameters
	----------
	encoding_model : dict
		Neural encoding model.
	images : int
		Images for which the neural responses are synthesized. Must be a 4-D
		numpy array of shape (Batch size x 3 RGB Channels x Width x Height)
		consisting of integer values in the range [0, 255]. Furthermore, the
		images must be of square size (i.e., equal width and height).
	device : str
		Whether to work on the 'cpu' or 'cuda'.

	Returns
	-------
	synthetic_fmri_responses : float
		Synthetic fMRI responses for the input stimulus images, of shape:
		(Images x N ROI Voxels).
	"""

	### Extract model parameters ###
	subject = encoding_model['args']['subject']
	roi = encoding_model['args']['roi']
	resize_px = encoding_model['resize_px']

	### Model functions ###
	def _model_fn(_ext, _con, _x):
		"""model consists of an extractor (_ext) and a connection model (_con)"""
		_y, _fm, _h = _ext(_x)
		return _con(_fm)
	def _pred_fn(_ext, _con, xb):
		return _model_fn(_ext, _con, torch.from_numpy(xb).to(device))

	### Synthesize the fMRI responses to images ###
	# Empty synthetic fMRI responses variables of shape: (Images x Voxels)
	if roi in ['lateral', 'ventral']:
		synthetic_fmri_responses_1 = np.zeros((len(images),
			encoding_model['nnv_1'][subject]), dtype=np.float32)
		synthetic_fmri_responses_2 = np.zeros((len(images),
			encoding_model['nnv_2'][subject]), dtype=np.float32)
	else:
		synthetic_fmri_responses = np.zeros((len(images),
			encoding_model['nnv'][subject]), dtype=np.float32)

	# Preprocess the images
	transform = trn.Compose([
		trn.Resize((resize_px,resize_px))
	])
	images = torch.from_numpy(images)
	images = transform(images)
	images = np.asarray(images)
	images = image_feature_fn(images)

	# Synthesize the fMRI responses
	with torch.no_grad():
		if roi in ['lateral', 'ventral']:
			synthetic_fmri_responses_1 = subject_pred_pass(
				_pred_fn, encoding_model['shared_model_1'],
				encoding_model['subject_fwrfs_1'][subject], images,
				batch_size=100)
			synthetic_fmri_responses_2 = subject_pred_pass(
				_pred_fn, encoding_model['shared_model_2'],
				encoding_model['subject_fwrfs_2'][subject], images,
				batch_size=100)
			synthetic_fmri_responses = np.append(synthetic_fmri_responses_1,
				synthetic_fmri_responses_2, 1)
		else:
			synthetic_fmri_responses = subject_pred_pass(_pred_fn,
				encoding_model['shared_model'],
				encoding_model['subject_fwrfs'][subject], images,
				batch_size=100)

	# Convert synthetic fMRI responses to float 32
	synthetic_fmri_responses = synthetic_fmri_responses.astype(np.float32)

	### Output ###
	return synthetic_fmri_responses


def get_model_eeg_things_eeg_2_vit_b_32(ned_dir, subject, device):
	"""
	Load the vision-transformer-based (Dosovitskiy et al., 2020) linearizing
	encoding model.

	Parameters
	----------
	ned_dir : str
		Path to the "neural_encoding_dataset" folder.
	subject : int
		Subject number for which the encoding model was trained.
	device : str
		Whether the encoding model is stored on the 'cpu' or 'cuda'. If 'auto',
		the code will use GPU if available, and otherwise CPU.

	Returns
	-------
	encoding_model : dict
		Neural encoding model.
	"""

	### Get the EEG channels and time points dimensions ###
	metadata_dir = os.path.join(ned_dir, 'encoding_models', 'modality-eeg',
		'train_dataset-things_eeg_2', 'model-vit_b_32',
		'metadata', 'metadata_sub-'+format(subject,'02')+'.npy')
	metadata = np.load(metadata_dir, allow_pickle=True).item()
	ch_names = metadata['eeg']['ch_names']
	times = metadata['eeg']['times']

	### Load the vision transformer ###
	model = torchvision.models.vit_b_32(weights='DEFAULT')
	model.eval()

	# Select the used layers for feature extraction
	model_layers = ['encoder.layers.encoder_layer_0.add_1',
					'encoder.layers.encoder_layer_1.add_1',
					'encoder.layers.encoder_layer_2.add_1',
					'encoder.layers.encoder_layer_3.add_1',
					'encoder.layers.encoder_layer_4.add_1',
					'encoder.layers.encoder_layer_5.add_1',
					'encoder.layers.encoder_layer_6.add_1',
					'encoder.layers.encoder_layer_7.add_1',
					'encoder.layers.encoder_layer_8.add_1',
					'encoder.layers.encoder_layer_9.add_1',
					'encoder.layers.encoder_layer_10.add_1',
					'encoder.layers.encoder_layer_11.add_1']
	feature_extractor = create_feature_extractor(model,
		return_nodes=model_layers)
	feature_extractor.to(device)
	feature_extractor.eval()

	### Load the scaler and PCA weights ###
	# Scaler
	weights_dir = os.path.join(ned_dir, 'encoding_models', 'modality-eeg',
		'train_dataset-things_eeg_2', 'model-vit_b_32',
		'encoding_models_weights', 'StandardScaler_param.npy')
	scaler_weights = np.load(weights_dir, allow_pickle=True).item()
	scaler = StandardScaler()
	scaler.scale_ = scaler_weights['scale_']
	scaler.mean_ = scaler_weights['mean_']
	scaler.var_ = scaler_weights['var_']
	scaler.n_features_in_ = scaler_weights['n_features_in_']
	scaler.n_samples_seen_ = scaler_weights['n_samples_seen_']
	# PCA
	weights_dir = os.path.join(ned_dir, 'encoding_models', 'modality-eeg',
		'train_dataset-things_eeg_2', 'model-vit_b_32',
		'encoding_models_weights', 'pca_param.npy')
	pca_weights = np.load(weights_dir, allow_pickle=True).item()
	pca = PCA(n_components=1000, random_state=20200220)
	pca.components_ = pca_weights['components_']
	pca.explained_variance_ = pca_weights['explained_variance_']
	pca.explained_variance_ratio_ = pca_weights['explained_variance_ratio_']
	pca.singular_values_ = pca_weights['singular_values_']
	pca.mean_ = pca_weights['mean_']
	pca.n_components_ = pca_weights['n_components_']
	pca.n_samples_ = pca_weights['n_samples_']
	pca.noise_variance_ = pca_weights['noise_variance_']
	pca.n_features_in_ = pca_weights['n_features_in_']

	### Define the image preprocessing ###
	transform = torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.transforms()

	### Load the trained regression weights ###
	weights_dir = os.path.join(ned_dir, 'encoding_models', 'modality-eeg',
		'train_dataset-things_eeg_2', 'model-vit_b_32',
		'encoding_models_weights', 'LinearRegression_param_sub-'+
		format(subject, '02')+'.npy')
	reg_weights = np.load(weights_dir, allow_pickle=True).item()
	regression_weights = []
	for r in range(len(reg_weights)):
		reg = LinearRegression()
		reg.coef_ = reg_weights['rep-'+str(r+1)]['coef_']
		reg.intercept_ = reg_weights['rep-'+str(r+1)]['intercept_']
		reg.n_features_in_ = reg_weights['rep-'+str(r+1)]['n_features_in_']
		regression_weights.append(deepcopy(reg))
		del reg

	### Store the encoding model into a dictionary ###
	encoding_model = {}
	encoding_model['feature_extractor'] = feature_extractor
	encoding_model['scaler'] = scaler
	encoding_model['pca'] = pca
	encoding_model['transform'] = transform
	encoding_model['regression_weights'] = regression_weights
	encoding_model['ch_names'] = ch_names
	encoding_model['times'] = times

	### Output ###
	return encoding_model


def encode_eeg_things_eeg_2_vit_b_32(encoding_model, images, device):
	"""
	Synthesize EEG responses to images using a linear mapping of a
	pre-trained vision transformer (Dosovitskiy et al., 2020) image features.

	Parameters
	----------
	encoding_model : dict
		Neural encoding model.
	images : int
		Images for which the neural responses are synthesized. Must be a 4-D
		numpy array of shape (Batch size x 3 RGB Channels x Width x Height)
		consisting of integer values in the range [0, 255]. Furthermore, the images
		must be of square size (i.e., the width equals the height).
	device : str
		Whether to work on the 'cpu' or 'cuda'.

	Returns
	-------
	synthetic_eeg_responses : float
		Synthetic EEG responses for the input stimulus images, of shape:
		(Images x Repetitions x EEG Channels x EEG time points).
	"""

	### Extract model parameters ###
	ch_names = encoding_model['ch_names']
	times = encoding_model['times']

	### Preprocess the images ###
	images = encoding_model['transform'](torch.from_numpy(images))

	### Extract the feature maps ###
	# Input the images in batches
	batch_size = 100
	n_batches = int(np.ceil(len(images) / batch_size))
	progress_bar = tqdm(range(n_batches), desc='Encoding EEG responses')
	idx = 0
	with torch.no_grad():
		for b in progress_bar:
			# Update the progress bar
			encoded_images = batch_size * idx
			progress_bar.set_postfix(
				{'Encoded images': encoded_images, 'Total images': len(images)})
			# Image batch indices
			idx_start = b * batch_size
			idx_end = idx_start + batch_size
			# Extract the features
			ft = encoding_model['feature_extractor'](
				images[idx_start:idx_end].to(device))
			# Flatten the features
			ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
			ft = ft.detach().cpu().numpy()
			# Standardize the features
			ft = encoding_model['scaler'].transform(ft)
			# Apply PCA
			ft = encoding_model['pca'].transform(ft)
			# Store the features
			if b == 0:
				features = ft
			else:
				features = np.append(features, ft, 0)
			del ft
			# Update the progress bar
			idx += 1
			encoded_images = batch_size * idx
			progress_bar.set_postfix(
				{'Encoded images': encoded_images, 'Total images': len(images)})
	features = features.astype(np.float32)
	# The encoding models are trained using only the first 250 principal
	# components
	features = features[:,:250]

	### Synthesize the EEG responses ###
	synthetic_eeg_responses = []

	for reg in encoding_model['regression_weights']:

		# Synthesize the EEG responses
		synt_eeg = reg.predict(features)
		synt_eeg = synt_eeg.astype(np.float32)

		# Reshape the synthetic EEG data to (Images x Channels x Time)
		synt_eeg = np.reshape(synt_eeg, (len(synt_eeg), len(ch_names),
		len(times)))
		synthetic_eeg_responses.append(synt_eeg)
		del synt_eeg

	# Reshape to: (Images x Repeats x Channels x Time)
	synthetic_eeg_responses = np.swapaxes(np.asarray(synthetic_eeg_responses),
		0, 1)
	synthetic_eeg_responses = synthetic_eeg_responses.astype(np.float32)

	### Output ###
	return synthetic_eeg_responses
