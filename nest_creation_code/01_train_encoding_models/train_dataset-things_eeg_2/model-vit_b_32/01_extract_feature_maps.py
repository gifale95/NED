"""Use a CLIP vision transformer to generate feature maps for the THINGS EEG 2
train and test images.

https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_32.html

Parameters
----------
things_eeg_2_dir : str
	Directory of the THINGS EEG2 dataset.
	https://osf.io/3jk45/
ned_dir : str
	Neural encoding dataset directory.
	https://github.com/gifale95/NED

"""

import argparse
import torch
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--things_eeg_2_dir', default='../things_eeg_2', type=str)
parser.add_argument('--ned_dir', default='../neural_encoding_dataset', type=str)
args = parser.parse_args()

print('>>> Extract feature maps <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Define the image preprocessing
# =============================================================================
transform = torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.transforms()


# =============================================================================
# Vision model
# =============================================================================
# Load the model
model = torchvision.models.vit_b_32(weights='DEFAULT')
model.to(device)
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


# =============================================================================
# Extract the THINGS EEG 2 train image features
# =============================================================================
# Image directories
img_dir = os.path.join(args.things_eeg_2_dir, 'image_set', 'training_images')
image_list = []
for root, dirs, files in os.walk(img_dir):
	for file in files:
		if file.endswith(".jpg") or file.endswith(".JPEG"):
			image_list.append(os.path.join(root,file))
image_list.sort()

fmaps_train = []
with torch.no_grad():
	for i, img_dir in enumerate(tqdm(image_list, leave=False)):
		# Load the images
		img = Image.open(img_dir).convert('RGB')
		img = transform(img).unsqueeze(0)
		img.to(device)
		# Extract the features
		ft = feature_extractor(img)
		# Flatten the features
		ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
		fmaps_train.append(np.squeeze(ft.detach().numpy()))
		del ft
fmaps_train = np.asarray(fmaps_train)

# Standardize the features
scaler = StandardScaler()
scaler.fit(fmaps_train)
fmaps_train = scaler.transform(fmaps_train)
# Save the StandardScaler parameters
save_dir = os.path.join(args.ned_dir, 'encoding_models', 'modality-eeg',
	'train_dataset-things_eeg_2', 'model-vit_b_32',
	'encoding_models_weights')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
scaler_param = {
	'scale_': scaler.scale_,
	'mean_': scaler.mean_,
	'var_': scaler.var_,
	'n_features_in_': scaler.n_features_in_,
	'n_samples_seen_': scaler.n_samples_seen_
	}
np.save(os.path.join(save_dir, 'StandardScaler_param.npy'), scaler_param)

# Apply PCA
pca = PCA(n_components=1000, random_state=seed)
pca.fit(fmaps_train)
fmaps_train = pca.transform(fmaps_train)
fmaps_train = fmaps_train.astype(np.float32)
# Save the PCA parameters
pca_param = {
	'components_': pca.components_,
	'explained_variance_': pca.explained_variance_,
	'explained_variance_ratio_': pca.explained_variance_ratio_,
	'singular_values_': pca.singular_values_,
	'mean_': pca.mean_,
	'n_components_': pca.n_components_,
	'n_samples_': pca.n_samples_,
	'noise_variance_': pca.noise_variance_,
	'n_features_in_': pca.n_features_in_
	}
np.save(os.path.join(save_dir, 'pca_param.npy'), pca_param)

# Save the downsampled feature maps
np.save(os.path.join(save_dir, 'pca_feature_maps_train'), fmaps_train)
del fmaps_train


# =============================================================================
# Extract the THINGS EEG2 test image features
# =============================================================================
# Image directories
img_dir = os.path.join(args.things_eeg_2_dir, 'image_set', 'test_images')
image_list = []
for root, dirs, files in os.walk(img_dir):
	for file in files:
		if file.endswith(".jpg") or file.endswith(".JPEG"):
			image_list.append(os.path.join(root,file))
image_list.sort()

fmaps_test = []
for i, img_dir in enumerate(tqdm(image_list, leave=False)):
	# Load the images
	img = Image.open(img_dir).convert('RGB')
	img = transform(img).unsqueeze(0)
	img.to(device)
	# Extract the features
	ft = feature_extractor(img)
	# Flatten the features
	ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
	fmaps_test.append(np.squeeze(ft.detach().numpy()))
	del ft
fmaps_test = np.asarray(fmaps_test)

# Standardize the features
fmaps_test = scaler.transform(fmaps_test)

# Apply PCA
fmaps_test = pca.transform(fmaps_test)

# Save the downsampled feature maps
np.save(os.path.join(save_dir, 'pca_feature_maps_test'), fmaps_test)

