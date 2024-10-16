def epoching(args, data_part, seed):
	"""This function first converts the EEG data to MNE raw format, and
	performs, channel selection, filtering, epoching, current source density
	transform, frequency downsampling and baseline correction. Then, it sorts
	the EEG data of each session based on the image conditions.

	Parameters
	----------
	args : Namespace
		Input arguments.
	data_part : str
		'test' or 'training' data partitions.
	seed : int
		Random seed.

	Returns
	-------
	epoched_data : list of float
		Epoched EEG data.
	img_conditions : list of int
		Unique image conditions of the epoched and sorted EEG data.
	ch_names : list of str
		EEG channel names.
	times : float
		EEG time points.

	"""

	import os
	import mne
	import numpy as np
	from sklearn.utils import shuffle

	### Loop across data collection sessions ###
	epoched_data = []
	img_conditions = []
	for s in range(args.n_ses):

		### Load the EEG data and convert it to MNE raw format ###
		eeg_dir = os.path.join('raw_data', 'sub-'+format(args.sub,'02'), 'ses-'+
			format(s+1,'02'), 'raw_eeg_'+data_part+'.npy')
		eeg_data = np.load(os.path.join(args.things_eeg_2_dir, eeg_dir),
			allow_pickle=True).item()
		ch_names = eeg_data['ch_names']
		sfreq = eeg_data['sfreq']
		ch_types = eeg_data['ch_types']
		eeg_data = eeg_data['raw_eeg_data']
		# Convert to MNE raw format
		info = mne.create_info(ch_names, sfreq, ch_types)
		raw = mne.io.RawArray(eeg_data, info)
		del eeg_data

		### Filter the data ###
		if args.highpass != None or args.lowpass != None:
			raw = raw.copy().filter(l_freq=args.highpass, h_freq=args.lowpass)

		### Get events, drop unused channels and reject target trials ###
		events = mne.find_events(raw, stim_channel='stim')
		# Drop the 'stim' channel (keep only the 'eeg' channels)
		raw.pick_types(eeg=True, stim=False)
		# Select only occipital (O) channels
#		chan_idx = np.asarray(mne.pick_channels_regexp(raw.info['ch_names'],
#			'^O *|^P *'))
#		new_chans = [raw.info['ch_names'][c] for c in chan_idx]
#		raw.pick_channels(new_chans)
		# Reject the target trials (event 99999)
		idx_target = np.where(events[:,2] == 99999)[0]
		events = np.delete(events, idx_target, 0)
		img_cond = np.unique(events[:,2])

		### Epoching ###
		epochs = mne.Epochs(raw, events, tmin=args.tmin, tmax=args.tmax,
			baseline=None, preload=True)
		del raw

		### Compute current source density ###
		if args.csd == 1:
			# Load the EEG info dictionary from the source data, and extract
			# the channels positions
			eeg_dir = os.path.join(args.things_eeg_2_dir, 'source_data', 'sub-'+
				format(args.sub,'02'), 'ses-'+format(s+1,'02'), 'eeg', 'sub-'+
				format(args.sub,'02')+'_ses-'+format(s+1,'02')+
				'_task-test_eeg.vhdr')
			source_eeg = mne.io.read_raw_brainvision(eeg_dir, preload=True)
			source_info = source_eeg.info
			del source_eeg
			# Create the channels montage file
			ch_pos = {}
			for c, dig in enumerate(source_info['dig']):
				if c > 2:
					if source_info['ch_names'][c-3] in epochs.info['ch_names']:
						ch_pos[source_info['ch_names'][c-3]] = dig['r']
			montage = mne.channels.make_dig_montage(ch_pos)
			# Apply the montage to the epoched data
			epochs.set_montage(montage)
			# Compute current source density
			epochs = mne.preprocessing.compute_current_source_density(epochs,
				lambda2=1e-05, stiffness=4)

		### Resample the epoched data ###
		if args.sfreq < 1000:
			epochs.resample(args.sfreq)
		ch_names = epochs.info['ch_names']
		times = epochs.times
		info = epochs.info
		epochs = epochs.get_data()

		### Baseline correction ###
		if args.baseline_correction == 1:
			epochs = mne.baseline.rescale(epochs, times, baseline=(None, 0),
				mode=args.baseline_mode)

		### Sort the data ###
		# Select only a maximum number of EEG repetitions
		if data_part == 'test':
			max_rep = 20
		else:
			max_rep = 2
		# Sorted data matrix of shape:
		# (Image conditions x EEG repetitions x EEG channels x EEG time points)
		sorted_data = np.zeros((len(img_cond),max_rep,epochs.shape[1],
			epochs.shape[2]))
		for i in range(len(img_cond)):
			# Find the indices of the selected image condition
			idx = np.where(events[:,2] == img_cond[i])[0]
			# Randomly select only the max number of EEG repetitions
			idx = shuffle(idx, random_state=seed, n_samples=max_rep)
			sorted_data[i] = epochs[idx]
		del epochs
		epoched_data.append(sorted_data)
		img_conditions.append(img_cond)
		del sorted_data

	### Output ###
	return epoched_data, img_conditions, ch_names, times


def zscore(args, epoched_test, epoched_train):
	"""z-score the EEG data at each recording session.

	Parameters
	----------
	args : Namespace
		Input arguments.
	epoched_test : list of floats
		Epoched test EEG data.
	epoched_train : list of floats
		Epoched training EEG data.

	Returns
	-------
	zscored_test : list of float
		zscored test EEG data.
	zscored_train : list of float
		zscored training EEG data.

	"""

	import numpy as np
	from sklearn.preprocessing import StandardScaler

	### z-score the data of each session ###
	zscored_test = []
	zscored_train = []
	for s in range(args.n_ses):
		test_data = epoched_test[s]
		train_data = epoched_train[s]
		test_data_shape = test_data.shape
		train_data_shape = train_data.shape
		test_samples = test_data_shape[0] * test_data_shape[1]
		train_samples = train_data_shape[0] * train_data_shape[1]
		test_data = np.reshape(test_data, (test_samples, -1))
		train_data = np.reshape(train_data, (train_samples, -1))
		all_data = np.append(test_data, train_data, 0)
		scaler = StandardScaler()
		all_data = scaler.fit_transform(all_data)
		test_data = all_data[:test_samples]
		train_data = all_data[test_samples:]
		test_data = np.reshape(test_data, test_data_shape)
		train_data = np.reshape(train_data, train_data_shape)
		zscored_test.append(test_data)
		zscored_train.append(train_data)

	### Output ###
	return zscored_test, zscored_train


def compute_ncsnr(zscored_test):
	"""Compute the noise ceiling as in the NSD paper, using the test split
	data.

	Parameters
	----------
	zscored_test : list of float
		zscored test EEG data.

	Returns
	-------
	ncsnr : float
		Noise ceiling SNR.

	"""

	import numpy as np
	from copy import copy
	from sklearn.preprocessing import StandardScaler

	### Standardize the data at each scan session ###
	for s in range(len(zscored_test)):
		data_shape = zscored_test[s].shape
		provv_data = np.reshape(copy(zscored_test[s]),
			(data_shape[0]*data_shape[1],-1))
		scaler = StandardScaler()
		provv_data = scaler.fit_transform(provv_data)
		if s == 0:
			zscored_data = np.reshape(provv_data, data_shape)
		else:
			zscored_data = np.append(zscored_data, np.reshape(
				provv_data, data_shape), 1)

	### Compute the ncsnr ###
	std_noise = np.sqrt(np.mean(np.var(zscored_data, axis=1, ddof=1), 0))
	std_signal = 1 - (std_noise ** 2)
	std_signal[std_signal<0] = 0
	std_signal = np.sqrt(std_signal)
	ncsnr = std_signal / std_noise

	### Output ###
	return ncsnr


def save_prepr(args, zscored_test, zscored_train, img_conditions_train,
	ch_names, times, seed, ncsnr):
	"""Merge the EEG data of all sessions together, shuffle the EEG repetitions
	across sessions and reshaping the data to the format:
	(Image conditions x EEG repetitions x EEG channels x EEG time points).
	Then, the data of both test and training EEG partitions is saved.

	Parameters
	----------
	args : Namespace
		Input arguments.
	zscored_test : list of float
		zscored test EEG data.
	zscored_train : list of float
		zscored training EEG data.
	img_conditions_train : list of int
		Unique image conditions of the epoched and sorted train EEG data.
	ch_names : list of str
		EEG channel names.
	times : float
		EEG time points.
	seed : int
		Random seed.
	ncsnr : float
		Noise ceiling SNR.

	"""

	import numpy as np
	from sklearn.utils import shuffle
	import os

	### Load the image metadata ###
	metadata_dir = os.path.join(args.things_eeg_2_dir, 'image_set',
		'image_metadata.npy')
	img_metadata = np.load(metadata_dir, allow_pickle=True).item()
	img_metadata_test = {
		'test_img_concepts': img_metadata['test_img_concepts'],
		'test_img_concepts_THINGS': img_metadata['test_img_concepts_THINGS'],
		'test_img_files': img_metadata['test_img_files']
		}
	img_metadata_train = {
		'train_img_concepts': img_metadata['train_img_concepts'],
		'train_img_concepts_THINGS': img_metadata['train_img_concepts_THINGS'],
		'train_img_files': img_metadata['train_img_files']
		}

	### Merge and save the test data ###
	for s in range(args.n_ses):
		if s == 0:
			merged_test = zscored_test[s]
		else:
			merged_test = np.append(merged_test, zscored_test[s], 1)
	del zscored_test
	# Shuffle the repetitions of different sessions
	idx = shuffle(np.arange(0, merged_test.shape[1]), random_state=seed)
	merged_test = merged_test[:,idx]
	# Convert to float32
	merged_test = merged_test.astype(np.float32)
	# Insert the data into a dictionary
	test_dict = {
		'args': args,
		'preprocessed_eeg_data': merged_test,
		'ch_names': ch_names,
		'times': times,
		'img_metadata': img_metadata_test,
		'ncsnr': ncsnr
	}
	del merged_test
	# Saving directories
	save_dir = os.path.join(args.ned_dir, 'model_training_datasets',
		'train_dataset-things_eeg_2', 'model-vit_b_32', 'neural_data')
	file_name_test = 'eeg_sub-'+format(args.sub,'02')+'_split-test.npy'
	file_name_train = 'eeg_sub-'+format(args.sub,'02')+'_split-train.npy'
	# Create the directory if not existing and save the data
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)
	np.save(os.path.join(save_dir, file_name_test), test_dict)
	del test_dict

	### Merge and save the training data ###
	for s in range(args.n_ses):
		if s == 0:
			white_data = zscored_train[s]
			img_cond = img_conditions_train[s]
		else:
			white_data = np.append(white_data, zscored_train[s], 0)
			img_cond = np.append(img_cond, img_conditions_train[s], 0)
	del zscored_train, img_conditions_train
	# Data matrix of shape:
	# (Image conditions x EEG repetitions x EEG channels x EEG time points)
	merged_train = np.zeros((len(np.unique(img_cond)), white_data.shape[1]*2,
		white_data.shape[2],white_data.shape[3]))
	for i in range(len(np.unique(img_cond))):
		# Find the indices of the selected category
		idx = np.where(img_cond == i+1)[0]
		for r in range(len(idx)):
			if r == 0:
				ordered_data = white_data[idx[r]]
			else:
				ordered_data = np.append(ordered_data, white_data[idx[r]], 0)
		merged_train[i] = ordered_data
	# Shuffle the repetitions of different sessions
	idx = shuffle(np.arange(0, merged_train.shape[1]), random_state=seed)
	merged_train = merged_train[:,idx]
	# Convert to float32
	merged_train = merged_train.astype(np.float32)
	# Insert the data into a dictionary
	train_dict = {
		'args': args,
		'preprocessed_eeg_data': merged_train,
		'ch_names': ch_names,
		'times': times,
		'img_metadata': img_metadata_train,
		'ncsnr': ncsnr
	}
	del merged_train
	# Create the directory if not existing and save the data
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)
	np.save(os.path.join(save_dir, file_name_train),
		train_dict)
	del train_dict
