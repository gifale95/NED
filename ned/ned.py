import os
import numpy as np
import h5py
import torch

from ned.utils import synthesize_fmri_responses
from ned.utils import synthesize_eeg_responses
from ned.utils import get_fmri_metadata
from ned.utils import get_eeg_metadata


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
			rois = ['V1', 'V2', 'V3', 'V4', 'EBA', 'FBA-2', 'OFA', 'FFA-1',
				'FFA-2', 'PPA', 'RSC', 'OPA', 'OWFA', 'VWFA-1', 'VWFA-2',
				'mfs-words', 'early', 'midventral', 'midlateral', 'midparietal',
				'parietal', 'lateral', 'ventral']

		### Output ###
		return rois


	def encode_fmri(self, images, train_dataset, subject, roi, model,
		return_metadata=True, device='auto'):
		"""
		Synthesize fMRI responses for arbitrary stimulus images, and optionally
		return the synthetic fMRI metadata.

		Parameters
		----------
		images : int
			Images for which the neural responses are synthesized. Must be a 4-D
			numpy array of shape (Batch size x 3 RGB Channels x Width x Height)
			consisting of integer values in the range 0/255. Furthermore, the
			images must be of square size (i.e., equal width and height).
		train_dataset : str
			Name of the neural dataset used to train the encoding models.
		subject : int
			Subject number for which the fMRI image responses are synthesized.
		roi : str
			Name of the Region of Interest (ROI) for which the fMRI image
			responses are synthesized.
		model : str
			Encoding model type used to synthesize the fMRI responses.
		return_metadata : bool
			If True, return fMRI medatata along with the synthetic fMRI
			responses.
		device : str
			Whether to work on the 'cpu' or 'cuda'. If 'auto', the code will
			use GPU if available, and otherwise CPU.

		Returns
		-------
		synthetic_fmri_responses : float
			Synthetic fMRI responses for the input stimulus images, of shape:
			(Images x N ROI Voxels).
		synthetic_fmri_metadata : dict
			Synthetic fMRI responses metadata.
		"""

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

		# train_dataset
		if type(train_dataset) != str:
			raise TypeError("'train_dataset' must be of type str!")
		train_dataset_options = self.which_train_datasets('fmri')
		if train_dataset not in train_dataset_options:
			raise ValueError(f"'train_dataset' value must be one of the following: {train_dataset_options}!")

		# subject
		if type(subject) != int:
			raise TypeError("'subject' must be of type int!")
		subjects = self.which_subjects('fmri', train_dataset)
		if subject not in subjects:
			raise ValueError(f"'subject' value must be one of the following: {subjects}!")

		# roi
		if type(roi) != str:
			raise TypeError("'roi' must be of type str!")
		rois = self.which_rois(train_dataset)
		if roi not in rois:
			raise ValueError(f"'roi' value must be one of the following: {rois}!")

		# model
		if type(model) != str:
			raise TypeError("'model' must be of type str!")
		models = self.which_models('fmri', train_dataset)
		if model not in models:
			raise ValueError(f"'model' value must be one of the following: {models}!")

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

		### Synthesize fMRI responses to the input images ###
		synthetic_fmri_responses = synthesize_fmri_responses(
			self.ned_dir,
			images,
			train_dataset,
			subject,
			roi,
			model,
			device
			)

		### Get syntehtic fMRI responses metadata ###
		if return_metadata == True:
			synthetic_fmri_metadata = get_fmri_metadata(
				self.ned_dir,
				train_dataset,
				subject,
				roi,
				model
				)

		### Output ###
		if return_metadata == False:
			return synthetic_fmri_responses
		else:
			return synthetic_fmri_responses, synthetic_fmri_metadata


	def encode_eeg(self, images, train_dataset, subject, model,
		return_metadata=True, device='auto'):
		"""
		Synthesize EEG responses for arbitrary stimulus images, and optionally
		return the synthetic EEG metadata.

		Parameters
		----------
		images : int
			Images for which the neural responses are synthesized. Must be a 4-D
			numpy array of shape (Batch size x 3 RGB Channels x Width x Height)
			consisting of integer values in the range 0/255. Furthermore, the
			images must be of square size (i.e., equal width and height).
		train_dataset : str
			Name of the neural dataset used to train the encoding models.
		subject : int
			Subject number for which the EEG image responses are synthesized.
		model : str
			Encoding model type used to synthesize the EEG responses.
		return_metadata : bool
			If True, return fMRI medatata along with the synthetic EEG
			responses.
		device : str
			Whether to work on the 'cpu' or 'cuda'. If 'auto', the code will
			use GPU if available, and otherwise CPU.

		Returns
		-------
		synthetic_eeg_responses : float
			Synthetic EEG responses for the input stimulus images, of shape:
			(Images x Repetitions x EEG Channels x EEG time points).
		synthetic_eeg_metadata : dict
			Synthetic EEG responses metadata.
		"""

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

		# train_dataset
		if type(train_dataset) != str:
			raise TypeError("'train_dataset' must be of type str!")
		train_dataset_options = self.which_train_datasets('eeg')
		if train_dataset not in train_dataset_options:
			raise ValueError(f"'train_dataset' value must be one of the following: {train_dataset_options}!")

		# subject
		if type(subject) != int:
			raise TypeError("'subject' must be of type int!")
		subjects = self.which_subjects('eeg', train_dataset)
		if subject not in subjects:
			raise ValueError(f"'subject' value must be one of the following: {subjects}!")

		# model
		if type(model) != str:
			raise TypeError("'model' must be of type str!")
		models = self.which_models('eeg', train_dataset)
		if model not in models:
			raise ValueError(f"'model' value must be one of the following: {models}!")

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

		### Synthesize EEG responses to the input images ###
		synthetic_eeg_responses = synthesize_eeg_responses(
			self.ned_dir,
			images,
			train_dataset,
			subject,
			model,
			device
			)

		### Get syntehtic EEG responses metadata ###
		if return_metadata == True:
			synthetic_eeg_metadata = get_eeg_metadata(
				self.ned_dir,
				train_dataset,
				subject,
				model
				)

		### Output ###
		if return_metadata == False:
			return synthetic_eeg_responses
		else:
			return synthetic_eeg_responses, synthetic_eeg_metadata


	def load_ned_fmri(self, train_dataset, subject, roi, model, imageset,
		return_metadata=True):
		"""
		Load the NED's synthetic fMRI responses, and optionally their metadata.

		Parameters
		----------
		train_dataset : str
			Name of the neural dataset used to train the encoding models.
		subject : int
			Subject number for which the fMRI image responses are synthesized.
		roi : str
			Name of the Region of Interest (ROI) for which the fMRI image
			responses are synthesized.
		model : str
			Encoding model type used to synthesize the fMRI responses.
		imageset : str
			Imageset for which the fMRI responses are synthesized. Available
			options are 'nsd', 'imagenet_val' and 'things'.
		return_metadata : bool
			If True, return fMRI medatata along with the synthetic fMRI
			responses.

		Returns
		-------
		synthetic_fmri_responses : h5py
			Synthetic fMRI responses for the input stimulus images, of shape:
			(Images x N ROI Voxels).
		synthetic_fmri_metadata : dict
			Synthetic fMRI responses metadata.
		"""

		### Check input ###
		# train_dataset
		if type(train_dataset) != str:
			raise TypeError("'train_dataset' must be of type str!")
		train_dataset_options = self.which_train_datasets('fmri')
		if train_dataset not in train_dataset_options:
			raise ValueError(f"'train_dataset' value must be one of the following: {train_dataset_options}!")

		# subject
		if type(subject) != int:
			raise TypeError("'subject' must be of type int!")
		subjects = self.which_subjects('fmri', train_dataset)
		if subject not in subjects:
			raise ValueError(f"'subject' value must be one of the following: {subjects}!")

		# roi
		if type(roi) != str:
			raise TypeError("'roi' must be of type str!")
		rois = self.which_rois(train_dataset)
		if roi not in rois:
			raise ValueError(f"'roi' value must be one of the following: {rois}!")

		# model
		if type(model) != str:
			raise TypeError("'model' must be of type str!")
		models = self.which_models('fmri', train_dataset)
		if model not in models:
			raise ValueError(f"'model' value must be one of the following: {models}!")

		# imageset
		if type(imageset) != str:
			raise TypeError("'imageset' must be of type str!")
		imagesets = ['nsd', 'imagenet_val', 'things']
		if imageset not in imagesets:
			raise ValueError(f"'imageset' value must be one of the following: {imagesets}!")

		# return_metadata
		if type(return_metadata) != bool:
			raise TypeError("'return_metadata' must be of type bool!")

		### Load NED's synthetic fMRI responses ###
		data_dir = os.path.join(self.ned_dir, 'dataset', 'modality-fmri',
			'training_dataset-'+train_dataset, 'model-'+model,
			'synthetic_neural_responses', 'imageset-'+imageset,
			'synthetic_neural_responses_training_dataset-'+train_dataset+
			'_model-'+model+'_imageset-'+imageset+'_sub-'+format(subject, '02')+
			'_roi-'+roi+'.h5')
		synthetic_fmri_responses = h5py.File(
			data_dir, 'r').get('synthetic_neural_responses')
		
		### Load the metadata ###
		if return_metadata == True:
			metadata_dir = os.path.join(self.ned_dir, 'dataset',
				'modality-fmri', 'training_dataset-'+train_dataset, 'model-'+
				model, 'synthetic_neural_responses', 'imageset-'+train_dataset,
				'synthetic_neural_responses_metadata_training_dataset-'+
				train_dataset+'_model-'+model+'_imageset-'+train_dataset+
				'_sub-'+format(subject,'02')+'_roi-'+roi+'.npy')
			synthetic_fmri_metadata = np.load(metadata_dir,
				allow_pickle=True).item()

		### Output ###
		if return_metadata == False:
			return synthetic_fmri_responses
		else:
			return synthetic_fmri_responses, synthetic_fmri_metadata


	def load_ned_eeg(self, train_dataset, subject, model, imageset,
		return_metadata=True):
		"""
		Load the NED's synthetic EEG responses, and optionally their metadata.

		Parameters
		----------
		train_dataset : str
			Name of the neural dataset used to train the encoding models.
		subject : int
			Subject number for which the EEG image responses are synthesized.
		model : str
			Encoding model type used to synthesize the EEG responses.
		imageset : str
			Imageset for which the EEG responses are synthesized. Available
			options are 'nsd', 'imagenet_val' and 'things'.
		return_metadata : bool
			If True, return EEG medatata along with the synthetic EEG
			responses.

		Returns
		-------
		synthetic_eeg_responses : float
			Synthetic EEG responses for the input stimulus images, of shape:
			(Images x Repetitions x EEG Channels x EEG time points).
		synthetic_eeg_metadata : dict
			Synthetic EEG responses metadata.
		"""

		### Check input ###
		# train_dataset
		if type(train_dataset) != str:
			raise TypeError("'train_dataset' must be of type str!")
		train_dataset_options = self.which_train_datasets('eeg')
		if train_dataset not in train_dataset_options:
			raise ValueError(f"'train_dataset' value must be one of the following: {train_dataset_options}!")

		# subject
		if type(subject) != int:
			raise TypeError("'subject' must be of type int!")
		subjects = self.which_subjects('eeg', train_dataset)
		if subject not in subjects:
			raise ValueError(f"'subject' value must be one of the following: {subjects}!")

		# model
		if type(model) != str:
			raise TypeError("'model' must be of type str!")
		models = self.which_models('eeg', train_dataset)
		if model not in models:
			raise ValueError(f"'model' value must be one of the following: {models}!")

		# imageset
		if type(imageset) != str:
			raise TypeError("'imageset' must be of type str!")
		imagesets = ['nsd', 'imagenet_val', 'things']
		if imageset not in imagesets:
			raise ValueError(f"'imageset' value must be one of the following: {imagesets}!")

		# return_metadata
		if type(return_metadata) != bool:
			raise TypeError("'return_metadata' must be of type bool!")

		### Load NED's synthetic fMRI responses ###
		data_dir = os.path.join(self.ned_dir, 'dataset', 'modality-eeg',
			'training_dataset-'+train_dataset, 'model-'+model,
			'synthetic_neural_responses', 'imageset-'+imageset,
			'synthetic_neural_responses_training_dataset-'+train_dataset+
			'_model-'+model+'_imageset-'+imageset+'_sub-'+format(subject, '02')+
			'.h5')
		synthetic_eeg_responses = h5py.File(
			data_dir, 'r').get('synthetic_neural_responses')

		### Load the metadata ###
		if return_metadata == True:
			metadata_dir = os.path.join(self.ned_dir, 'dataset',
				'modality-eeg', 'training_dataset-'+train_dataset, 'model-'+
				model, 'synthetic_neural_responses', 'imageset-'+imageset,
				'synthetic_neural_responses_metadata_training_dataset-'+
				train_dataset+'_model-'+model+'_imageset-'+imageset+
				'_sub-'+format(subject,'02')+'.npy')
			synthetic_eeg_metadata = np.load(metadata_dir,
				allow_pickle=True).item()

		### Output ###
		if return_metadata == False:
			return synthetic_eeg_responses
		else:
			return synthetic_eeg_responses, synthetic_eeg_metadata

