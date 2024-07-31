import os
import numpy as np
import h5py
import torch

from ned.utils import get_model_fmri_nsd_fwrf
from ned.utils import get_model_eeg_things_eeg_2_vit_b_32
from ned.utils import encode_fmri_nsd_fwrf
from ned.utils import encode_eeg_things_eeg_2_vit_b_32


class NED():


	def __init__(self, ned_dir):
		"""
		Neural Encoding Dataset (NED) object.

		Parameters
		----------
		ned_dir : str
			Path to the "neural_encoding_dataset" folder.
		"""

		self.ned_dir = ned_dir


	def which_modalities(self):
		"""
		Return the neural data modalities available in NED.

		Returns
		-------
		modalities : list of str
			List of neural data modalities available in NED.
		"""

		### List modalities ###
		modalities = ['fmri', 'eeg']

		### Output ###
		return modalities


	def which_train_datasets(self, modality):
		"""
		For a given neural data modality, return the available datasets on which
		the NED encoding models are trained.

		Parameters
		-------
		modality : str
			Neural data modality.

		Returns
		-------
		train_datasets : list of str
			List of neural datasets on which the NED encoding models are
			trained.
		"""

		### Check input ###
		# modality
		if type(modality) != str:
			raise TypeError("'modality' must be of type str!")
		modalities = self.which_modalities()
		if modality not in modalities:
			raise ValueError(f"'modality' value must be one of the following: {modalities}!")

		### List training datasets ###
		if modality == 'fmri':
			train_datasets = ['nsd']
		elif modality == 'eeg':
			train_datasets = ['things_eeg_2']

		### Output ###
		return train_datasets


	def which_models(self, modality, train_dataset):
		"""
		For a given neural data modality and training dataset, return the
		encoding model types available in NED.

		Parameters
		-------
		modality : str
			Neural data modality.
		train_dataset : str
			Neural dataset on which the NED encoding models are trained.

		Returns
		-------
		models : list of str
			List of NED encoding models trained on a given neural data modality
			and neural dataset.
		"""

		### Check input ###
		# modality
		if type(modality) != str:
			raise TypeError("'modality' must be of type str!")
		modalities = self.which_modalities()
		if modality not in modalities:
			raise ValueError(f"'modality' value must be one of the following: {modalities}!")

		# train_dataset
		if type(train_dataset) != str:
			raise TypeError("'train_dataset' must be of type str!")
		train_datasets = self.which_train_datasets(modality)
		if train_dataset not in train_datasets:
			raise ValueError(f"For '{modality}' modality 'train_dataset' must be one of the following: {train_datasets}")

		### List models ###
		if modality == 'fmri':
			if train_dataset == 'nsd':
				models = ['fwrf']
		elif modality == 'eeg':
			if train_dataset == 'things_eeg_2':
				models = ['vit_b_32']

		### Output ###
		return models


	def which_subjects(self, modality, train_dataset):
		"""
		For a given neural data modality and training dataset, return the
		available subjects on which encoding models are trained.

		Parameters
		-------
		modality : str
			Neural data modality.
		train_dataset : str
			Neural dataset on which the NED encoding models are trained.

		Returns
		-------
		subjects : list of int
			List of available subjects for a given data modality and training
			dataset.
		"""

		### Check input ###
		# modality
		if type(modality) != str:
			raise TypeError("'modality' must be of type str!")
		modalities = self.which_modalities()
		if modality not in modalities:
			raise ValueError(f"'modality' value must be one of the following: {modalities}!")

		# train_dataset
		if type(train_dataset) != str:
			raise TypeError("'train_dataset' must be of type str!")
		train_datasets = self.which_train_datasets(modality)
		if train_dataset not in train_datasets:
			raise ValueError(f"For '{modality}' modality 'train_dataset' must be one of the following: {train_datasets}")

		### List subjects ###
		if modality == 'fmri':
			if train_dataset == 'nsd':
				subjects = [1, 2, 3, 4, 5, 6, 7, 8]
		elif modality == 'eeg':
			if train_dataset == 'things_eeg_2':
				subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

		### Output ###
		return subjects


	def which_rois(self, train_dataset):
		"""
		For a given fMRI training dataset, return the available ROIs on which
		encoding models are trained.

		Parameters
		-------
		train_dataset : str
			Neural dataset on which the NED encoding models are trained.

		Returns
		-------
		rois : list of str
			List of available rois for a given fMRI training dataset.
		"""

		### Check input ###
		# train_dataset
		if type(train_dataset) != str:
			raise TypeError("'train_dataset' must be of type str!")
		train_datasets = self.which_train_datasets('fmri')
		if train_dataset not in train_datasets:
			raise ValueError(f"For 'fmri' modality 'train_dataset' must be one of the following: {train_datasets}")

		### List ROIs ###
		if train_dataset == 'nsd':
			rois = ['V1', 'V2', 'V3', 'hV4', 'EBA', 'FBA-2', 'OFA', 'FFA-1',
				'FFA-2', 'PPA', 'RSC', 'OPA', 'OWFA', 'VWFA-1', 'VWFA-2',
				'mfs-words', 'early', 'midventral', 'midlateral', 'midparietal',
				'parietal', 'lateral', 'ventral']

		### Output ###
		return rois


	def get_encoding_model(self, modality, train_dataset, model, subject,
		roi=None, trained=True, device='auto'):
		"""
		Load the encoding model of interest.

		Parameters
		----------
		modality : str
			Neural data modality.
		train_dataset : str
			Name of the neural dataset used to train the encoding models.
		model : str
			Encoding model type used to synthesize the neural responses.
		subject : int
			Subject number for which the encoding model was trained.
		roi : str
			Only required if modality=='fmri'. Name of the Region of Interest
			(ROI) for which the fMRI encoding model was trained.
		trained : bool
			If True, load a trained encoding model, that is, a model with a
			trained backbone feature extractor. If False, the model's backbone
			feature extractor is randomly initialized.
		device : str
			Whether the encoding model is stored on the 'cpu' or 'cuda'. If
			'auto', the code will use GPU if available, and otherwise CPU.

		Returns
		-------
		encoding_model : dict
			Neural encoding model.
		"""

		### Check input ###
		# modality
		if type(modality) != str:
			raise TypeError("'modality' must be of type str!")
		modalities = self.which_modalities()
		if modality not in modalities:
			raise ValueError(f"'modality' value must be one of the following: {modalities}!")

		# train_dataset
		if type(train_dataset) != str:
			raise TypeError("'train_dataset' must be of type str!")
		train_dataset_options = self.which_train_datasets(modality)
		if train_dataset not in train_dataset_options:
			raise ValueError(f"'train_dataset' value must be one of the following: {train_dataset_options}!")

		# model
		if type(model) != str:
			raise TypeError("'model' must be of type str!")
		models = self.which_models(modality, train_dataset)
		if model not in models:
			raise ValueError(f"'model' value must be one of the following: {models}!")

		# subject
		if not(isinstance(subject, (int, np.integer))):
			raise TypeError("'subject' must be of type int!")
		subjects = self.which_subjects(modality, train_dataset)
		if subject not in subjects:
			raise ValueError(f"'subject' value must be one of the following: {subjects}!")

		# roi
		if modality == 'fmri':
			if type(roi) != str:
				raise TypeError("'roi' must be of type str!")
			rois = self.which_rois(train_dataset)
			if roi not in rois:
				raise ValueError(f"'roi' value must be one of the following: {rois}!")

		# trained
		if type(trained) != bool:
			raise TypeError("'trained' must be of type bool!")

		# device
		if type(device) != str:
			raise TypeError("'device' must be of type str!")
		device_options = ['cpu', 'cuda', 'auto']
		if device not in device_options:
			raise ValueError(f"'device' value must be one of the following: {device_options}!")

		### Select device ###
		if device == 'auto':
			device = 'cuda' if torch.cuda.is_available() else 'cpu'

		### Load the encoding models ###
		if modality == 'fmri':
			if train_dataset == 'nsd':
				if model == 'fwrf':
					# Load the feature-weighted receptive field (fwrf) encoding
					# model (St-Yves & Naselaris, 2018).
					encoding_model = get_model_fmri_nsd_fwrf(
						self.ned_dir,
						subject,
						roi,
						trained,
						device
						)

		elif modality == 'eeg':
			if train_dataset == 'things_eeg_2':
				if model == 'vit_b_32':
					# Load the vision-transformer-based (Dosovitskiy et al.,
					# 2020) linearizing encoding model.
					encoding_model = get_model_eeg_things_eeg_2_vit_b_32(
						self.ned_dir,
						subject,
						trained,
						device
						)

		### Add arguments to the model dictionary ###
		args = {}
		args['modality'] = modality
		args['train_dataset'] = train_dataset
		args['model'] = model
		args['subject'] = subject
		args['roi'] = roi
		args['trained'] = trained
		encoding_model['args'] = args

		### Output ###
		return encoding_model


	def encode(self, encoding_model, images, return_metadata=True,
			device='auto'):
		"""
		Synthesize neural responses for arbitrary stimulus images, and
		optionally return the synthetic fMRI metadata.

		Parameters
		----------
		encoding_model : list
			Neural encoding model.
		images : int
			Images for which the neural responses are synthesized. Must be a 4-D
			numpy array of shape (Batch size x 3 RGB Channels x Width x Height)
			consisting of integer values in the range 0/255. Furthermore, the
			images must be of square size (i.e., equal width and height).
		return_metadata : bool
			If True, return medatata along with the synthetic neural responses.
		device : str
			Whether to work on the 'cpu' or 'cuda'. If 'auto', the code will
			use GPU if available, and otherwise CPU.

		Returns
		-------
		synthetic_neural_responses : float
			Synthetic neural responses for the input stimulus images.
			If modality=='fmri', the neural response will be of shape:
			(Images x N ROI Voxels).
			If modality=='eeg', the neural response will be of shape:
			(Images x Repetitions x EEG Channels x EEG time points) if
		metadata : dict
			Synthetic neural responses metadata.
		"""

		### Extract model parameters ###
		modality = encoding_model['args']['modality']
		train_dataset = encoding_model['args']['train_dataset']
		model = encoding_model['args']['model']
		subject = encoding_model['args']['subject']
		roi = encoding_model['args']['roi']

		### Check input ###
		# images
		if not isinstance(images, np.ndarray) and np.issubdtype(images.dtype, np.integer):
			raise TypeError("'images' must be a numpy integer array, with values in the range 0/255!")
		if len(images.shape) != 4:
			raise ValueError("'images' must be a 4-D array of shape (Batch size x 3 RGB Channels x Width x Height)!")
		if images.shape[1] != 3:
			raise ValueError("'images' must have 3 RGB channels!")
		if images.shape[2] != images.shape[3]:
			raise ValueError("'images' must be squared (i.e., equal width and height)!")

		# modality
		if type(modality) != str:
			raise TypeError("'modality' must be of type str!")
		modalities = self.which_modalities()
		if modality not in modalities:
			raise ValueError(f"'modality' value must be one of the following: {modalities}!")

		# train_dataset
		if type(train_dataset) != str:
			raise TypeError("'train_dataset' must be of type str!")
		train_dataset_options = self.which_train_datasets(modality)
		if train_dataset not in train_dataset_options:
			raise ValueError(f"'train_dataset' value must be one of the following: {train_dataset_options}!")

		# model
		if type(model) != str:
			raise TypeError("'model' must be of type str!")
		models = self.which_models(modality, train_dataset)
		if model not in models:
			raise ValueError(f"'model' value must be one of the following: {models}!")

		# subject
		if not(isinstance(subject, (int, np.integer))):
			raise TypeError("'subject' must be of type int!")
		subjects = self.which_subjects(modality, train_dataset)
		if subject not in subjects:
			raise ValueError(f"'subject' value must be one of the following: {subjects}!")

		# roi
		if modality == 'fmri':
			if type(roi) != str:
				raise TypeError("'roi' must be of type str!")
			rois = self.which_rois(train_dataset)
			if roi not in rois:
				raise ValueError(f"'roi' value must be one of the following: {rois}!")

		# return_metadata
		if type(return_metadata) != bool:
			raise TypeError("'return_metadata' must be of type bool!")

		# device
		if type(device) != str:
			raise TypeError("'device' must be of type str!")
		device_options = ['cpu', 'cuda', 'auto']
		if device not in device_options:
			raise ValueError(f"'device' value must be one of the following: {device_options}!")

		### Select device ###
		if device == 'auto':
			device = 'cuda' if torch.cuda.is_available() else 'cpu'

		### Synthesize neural responses to the input images ###
		if modality == 'fmri':
			if train_dataset == 'nsd':
				if model == 'fwrf':
					# Synthesize fMRI responses to images using the
					# feature-weighted receptive field (fwrf) encoding model
					# (St-Yves & Naselaris, 2018).
					synthetic_neural_responses = encode_fmri_nsd_fwrf(
						encoding_model,
						images,
						device
						)

		elif modality == 'eeg':
			if train_dataset == 'things_eeg_2':
				if model == 'vit_b_32':
					# Synthesize EEG responses to images using a linear mapping
					# of a pre-trained vision transformer (Dosovitskiy et al.,
					# 2020) image features.
					synthetic_neural_responses = encode_eeg_things_eeg_2_vit_b_32(
						encoding_model,
						images,
						device
						)

		### Get the synthetic neural responses metadata ###
		if return_metadata == True:
			metadata = self.get_metadata(
				modality,
				train_dataset,
				model,
				subject,
				roi
				)

		### Output ###
		if return_metadata == False:
			return synthetic_neural_responses
		else:
			return synthetic_neural_responses, metadata


	def get_metadata(self, modality, train_dataset, model, subject, roi=None):
		"""
		Get the metadata, consisting in information on the neural data used to
		train the encoding models (e.g., the amount of fMRI voxels or EEG time
		points), and on the trained encoding models (e.g., how was the data
		split to train and test the models, and the models accuracy scores).

		Parameters
		----------
		modality : str
			Neural data modality.
		train_dataset : str
			Name of the neural dataset used to train the encoding models.
		model : str
			Encoding model type used to synthesize the neural responses.
		subject : int
			Subject number for which the metadata is loaded.
		roi : str
			Only required if modality=='fmri'. Name of the Region of Interest
			(ROI) for which the metadata is loaded.
	
		Returns
		-------
		metadata : dict
			Synthetic neural responses metadata.
		"""

		### Check input ###
		# modality
		if type(modality) != str:
			raise TypeError("'modality' must be of type str!")
		modalities = self.which_modalities()
		if modality not in modalities:
			raise ValueError(f"'modality' value must be one of the following: {modalities}!")

		# train_dataset
		if type(train_dataset) != str:
			raise TypeError("'train_dataset' must be of type str!")
		train_dataset_options = self.which_train_datasets(modality)
		if train_dataset not in train_dataset_options:
			raise ValueError(f"'train_dataset' value must be one of the following: {train_dataset_options}!")

		# model
		if type(model) != str:
			raise TypeError("'model' must be of type str!")
		models = self.which_models(modality, train_dataset)
		if model not in models:
			raise ValueError(f"'model' value must be one of the following: {models}!")

		# subject
		if not(isinstance(subject, (int, np.integer))):
			raise TypeError("'subject' must be of type int!")
		subjects = self.which_subjects(modality, train_dataset)
		if subject not in subjects:
			raise ValueError(f"'subject' value must be one of the following: {subjects}!")

		# roi
		if modality == 'fmri':
			if type(roi) != str:
				raise TypeError("'roi' must be of type str!")
			rois = self.which_rois(train_dataset)
			if roi not in rois:
				raise ValueError(f"'roi' value must be one of the following: {rois}!")

		### Metadata directories ###
		parent_dir = os.path.join(self.ned_dir, 'encoding_models',
			'modality-'+modality, 'train_dataset-'+train_dataset, 'model-'+
			model, 'metadata')

		if modality == 'fmri':
			file_name = 'metadata_sub-' + format(subject,'02') + '_roi-' + \
				roi + '.npy'

		elif modality == 'eeg':
			file_name = 'metadata_sub-' + format(subject,'02') + '.npy'

		### Load the metadata ###
		metadata = np.load(os.path.join(parent_dir, file_name),
			allow_pickle=True).item()

		### Output ###
		return metadata


	def load_synthetic_neural_responses(self, modality, train_dataset, model,
		imageset, subject, roi=None, return_metadata=True):
		"""
		Load NED's synthetic neural responses, and optionally their metadata.

		Parameters
		----------
		modality : str
			Neural data modality.
		train_dataset : str
			Name of the neural dataset used to train the encoding models.
		model : str
			Encoding model type used to synthesize the fMRI responses.
		imageset : str
			Imageset for which the fMRI responses are synthesized. Available
			options are 'nsd', 'imagenet_val' and 'things'.
			If 'nsd', load synthetic neural responses for the 73,000 NSD images
			(Allen et al., 2023).
			If 'imagenet_val', load synthetic neural responses for the 50,000
			ILSVRC-2012 validation images (Russakovsky et al., 2015).
			If 'things', load synthetic neural responses for the 26,107 images
			from the THINGS database (Hebart et al., 2019).
		subject : int
			Subject number for which the fMRI image responses are synthesized.
		roi : str
			Only required if modality=='fmri'. Name of the Region of Interest
			(ROI) for which the fMRI image responses are synthesized.
		return_metadata : bool
			If True, return fMRI medatata along with the synthetic fMRI
			responses.

		Returns
		-------
		synthetic_neural_responses : h5py
			Synthetic neural responses for the input stimulus images.
			If modality=='fmri', the neural response will be of shape:
			(Images x N ROI Voxels).
			If modality=='eeg', the neural response will be of shape:
			(Images x Repetitions x EEG Channels x EEG time points) if
		metadata : dict
			Synthetic neural responses metadata.
		"""

		### Check input ###
		# modality
		if type(modality) != str:
			raise TypeError("'modality' must be of type str!")
		modalities = self.which_modalities()
		if modality not in modalities:
			raise ValueError(f"'modality' value must be one of the following: {modalities}!")

		# train_dataset
		if type(train_dataset) != str:
			raise TypeError("'train_dataset' must be of type str!")
		train_dataset_options = self.which_train_datasets(modality)
		if train_dataset not in train_dataset_options:
			raise ValueError(f"'train_dataset' value must be one of the following: {train_dataset_options}!")

		# model
		if type(model) != str:
			raise TypeError("'model' must be of type str!")
		models = self.which_models(modality, train_dataset)
		if model not in models:
			raise ValueError(f"'model' value must be one of the following: {models}!")

		# imageset
		if type(imageset) != str:
			raise TypeError("'imageset' must be of type str!")
		imagesets = ['nsd', 'imagenet_val', 'things']
		if imageset not in imagesets:
			raise ValueError(f"'imageset' value must be one of the following: {imagesets}!")

		# subject
		if not(isinstance(subject, (int, np.integer))):
			raise TypeError("'subject' must be of type int!")
		subjects = self.which_subjects(modality, train_dataset)
		if subject not in subjects:
			raise ValueError(f"'subject' value must be one of the following: {subjects}!")

		# roi
		if modality == 'fmri':
			if type(roi) != str:
				raise TypeError("'roi' must be of type str!")
			rois = self.which_rois(train_dataset)
			if roi not in rois:
				raise ValueError(f"'roi' value must be one of the following: {rois}!")

		# return_metadata
		if type(return_metadata) != bool:
			raise TypeError("'return_metadata' must be of type bool!")

		### Synthetic neural responses directories ###
		parent_dir = os.path.join(self.ned_dir, 'synthetic_neural_responses',
			'modality-'+modality, 'train_dataset-'+train_dataset, 'model-'+
			model, 'imageset-'+imageset)
			
		if modality == 'fmri':
			file_name = 'synthetic_neural_responses_sub-' + \
				format(subject, '02') + '_roi-' + roi + '.h5'

		elif modality == 'eeg':
			file_name = 'synthetic_neural_responses_sub-' + \
				format(subject, '02') +'.h5'

		### Load NED's synthetic neural responses ###
		synthetic_neural_responses = h5py.File(os.path.join(parent_dir,
			file_name), 'r').get('synthetic_neural_responses')

		### Metadata directories ###
		if return_metadata == True:
			if modality == 'fmri':
				file_name = 'synthetic_neural_responses_metadata_sub-' + \
					format(subject,'02') + '_roi-' + roi + '.npy'
			elif modality == 'eeg':
				file_name = 'synthetic_neural_responses_metadata_sub-' + \
					format(subject,'02') + '.npy'

		### Load the metadata ###
		if return_metadata == True:
			metadata = np.load(os.path.join(parent_dir, file_name),
				allow_pickle=True).item()

		### Output ###
		if return_metadata == False:
			return synthetic_neural_responses
		else:
			return synthetic_neural_responses, metadata
