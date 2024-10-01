"""Extract the features maps which will be later used to synthesize EEG
responses.

Parameters
----------
sub : int
	Number of sed subject.
imageset : str
	Name of imageset for which the neural responses are predicted.
tot_img_partitions : int
	Total amount of image partitions in which the feature maps were divided.
img_partition : int
	Image partition (from 1 to 20) for which the feature maps are created.
ned_dir : str
	Neural encoding dataset directory.
	https://github.com/gifale95/NED
nsd_dir : str
	Directory of the NSD.
	https://naturalscenesdataset.org/
imagenet_dir : str
	Directory of the ImageNet dataset.
	https://www.image-net.org/challenges/LSVRC/2012/index.php
things_dir : str
	Directory of the THINGS database.
	https://osf.io/jum2f/

"""

import argparse
import os
import numpy as np
import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from tqdm import tqdm
from PIL import Image
import h5py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default=1) # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
parser.add_argument('--imageset', type=str, default='nsd') # ['nsd', 'things', 'imagenet-val']
parser.add_argument('--tot_img_partitions', type=int, default=20)
parser.add_argument('--img_partition', type=int, default=1) # np.arange(1, 21)
parser.add_argument('--ned_dir', default='../neural_encoding_dataset', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
parser.add_argument('--imagenet_dir', default='../ILSVRC2012', type=str)
parser.add_argument('--things_dir', default='../things_database', type=str)
args = parser.parse_args()

print('>>> Extract feature maps <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Import the image datasets
# =============================================================================
if args.imageset == 'nsd':
	img_dir = os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli',
		'nsd', 'nsd_stimuli.hdf5')
	img_dataset = h5py.File(img_dir, 'r').get('imgBrick')

elif args.imageset == 'imagenet_val':
	img_dataset = torchvision.datasets.ImageNet(root=args.imagenet_dir,
		split='val')

elif args.imageset == 'things':
	image_concept_index = np.squeeze(pd.read_csv(os.path.join(args.things_dir,
		'02_object-level', 'image-concept-index_things.csv'),
		header=None).values) - 1
	image_paths_df = pd.read_csv(os.path.join(args.things_dir, '01_image-level',
		'image-paths_things.csv'), header=None)
	unique_id_df = pd.read_csv(os.path.join(args.things_dir, '02_object-level',
		'unique-id_things.csv'), header=None)
	image_paths = {}
	for i in range(len(image_paths_df)):
		image_paths[i] = image_paths_df[0][i][7:]
	unique_id = {}
	for i in range(len(unique_id_df)):
		unique_id[i] = unique_id_df[0][i]
	img_dataset = image_paths


# =============================================================================
# Image partitioning
# =============================================================================
tot_images = np.arange(len(img_dataset))
imgs_per_split = int(np.ceil(len(img_dataset) / args.tot_img_partitions))
idx_start = imgs_per_split * (args.img_partition - 1)
idx_end = idx_start + imgs_per_split
img_idxs = tot_images[idx_start:idx_end]


# =============================================================================
# Define the image preprocessing
# =============================================================================
transform = torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.transforms()


# =============================================================================
# Vision model
# =============================================================================
# Load the model
model = torchvision.models.vit_b_32(weights='DEFAULT')
model.eval()

# Select the used layers for feature extraction
#nodes, _ = get_graph_node_names(model)
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
feature_extractor = create_feature_extractor(model, return_nodes=model_layers)
feature_extractor.to(device)
feature_extractor.eval()


# =============================================================================
# Load the scaler and PCA weights
# =============================================================================
# Load the scaler weights
scaler = StandardScaler()
weights_dir = os.path.join(args.ned_dir, 'encoding_models', 'modality-eeg',
	'train_dataset-things_eeg_2', 'model-vit_b_32',
	'encoding_models_weights', 'StandardScaler_param.npy')
scaler_weights = np.load(weights_dir, allow_pickle=True).item()
scaler.scale_ = scaler_weights['scale_']
scaler.mean_ = scaler_weights['mean_']
scaler.var_ = scaler_weights['var_']
scaler.n_features_in_ = scaler_weights['n_features_in_']
scaler.n_samples_seen_ = scaler_weights['n_samples_seen_']

# Load the PCA weights
pca = PCA(n_components=1000, random_state=seed)
weights_dir = os.path.join(args.ned_dir, 'encoding_models', 'modality-eeg',
	'train_dataset-things_eeg_2', 'model-vit_b_32',
	'encoding_models_weights', 'pca_param.npy')
pca_weights = np.load(weights_dir, allow_pickle=True).item()
pca.components_ = pca_weights['components_']
pca.explained_variance_ = pca_weights['explained_variance_']
pca.explained_variance_ratio_ = pca_weights['explained_variance_ratio_']
pca.singular_values_ = pca_weights['singular_values_']
pca.mean_ = pca_weights['mean_']
pca.n_components_ = pca_weights['n_components_']
pca.n_samples_ = pca_weights['n_samples_']
pca.noise_variance_ = pca_weights['noise_variance_']
pca.n_features_in_ = pca_weights['n_features_in_']


# =============================================================================
# Extract the feature maps
# =============================================================================
features = []

with torch.no_grad():
	for i in tqdm(img_idxs):
		# Load the images
		if args.imageset == 'nsd':
			img = img_dataset[i]
			img = Image.fromarray(np.uint8(img))
		elif args.imageset == 'imagenet_val':
			img, _ = img_dataset.__getitem__(i)
		elif args.imageset == 'things':
			img = Image.open(os.path.join(args.things_dir, 'images',
				img_dataset[i])).convert('RGB')
		# Preprocess the image
		img = transform(img).unsqueeze(0)
		img = img.to(device)
		if device == 'cuda':
			img = img.cuda()
		# Extract the features
		ft = feature_extractor(img)
		# Flatten the features
		ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
		ft = np.squeeze(ft.detach().cpu().numpy())
		# Standardize the features
		ft = scaler.transform(np.reshape(ft, (1, -1)))
		# Apply PCA
		ft = pca.transform(ft)
		# Store the features
		features.append(ft)
		del img, ft

features = np.squeeze(np.asarray(features))
features = features.astype(np.float32)


# =============================================================================
# Save the features maps
# =============================================================================
save_dir = os.path.join(args.ned_dir, 'results', 'synthesize_neural_responses',
	'modality-eeg', 'train_dataset-things_eeg_2', 'model-vit_b_32',
	'imageset-'+args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'feature_maps_partition-' + format(args.img_partition, '02')

np.save(os.path.join(save_dir, file_name), features)

