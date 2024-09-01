# NED

The `NED` toolbox provides utility functions and tutorials for using the [**Neural Encoding Dataset**][ned_website]: trained encoding models of fMRI and EEG responses to images of multiple subjects, which you can use to synthesize fMRI and EEG responses to any image of your choice.

The Neural Encoding Dataset also comes with pre-generated synthetic fMRI and EEG responses for ~150,000 naturalistic images coming from the [ImageNet 2012 Challenge][imagenet] ([*Russakovsky et al., 2015*][russakovsky]), the [THINGS Database][things] ([*Hebart et al., 2019*][hebart]), and the [Natural Scenes Dataset][nsd] ([*Allen et al., 2022*][allen]), which you can use for research purposes.

For additional information on the Neural Encoding Dataset you can check out the [website][ned_website].



## 🤝 Contribute to Expanding the Neural Encoding Dataset

Do you have encoding models with higher prediction accuracies than the ones currently available in the Neural Encoding Dataset, and would like to make them available to the community? Or maybe you have encoding models for new neural datasets, data modalities (e.g., MEG/ECoG/animal), or stimulus types (e.g., videos, language) that you would like to share? Or perhaps you have suggestions for improving the Neural Encoding Dataset? Then please get in touch with Ale (alessandro.gifford@gmail.com): all feedback and help is strongly appreciated!



## ⚙️ Installation

To install `NED` run the following command on your terminal:

```shell
pip install -U git+https://github.com/gifale95/NED.git
```

You will additionally need to install the Python dependencies found in [requirements.txt][requirements].



## 🕹️ How to use

### 🧰 Download the Neural Encoding Dataset

To use `NED` you first need to download the Neural Encoding Dataset from [here][ned_data]. Depending on how you want to use the Neural Encoding Dataset, you might need to download all of it, or only parts of it. For this please refer to the [data manual][data_manual], which describes how the Neural Encoding Dataset is structured.

The Neural Encoding dataset is stored in a Google Drive folder called `neural_encoding_dataset`. We recommend downloading the dataset directly from Google Drive via terminal using [Rclone][rclone]. [Here][guide] is a step-by-step guide for how to install and use Rclone to move files to and from your Google Drive. Before downloading NED via terminal you need to add a shortcut of the `neural_encoding_dataset` folder to your Google Drive. You can do this by right-clicking on the `neural_encoding_dataset` folder, and selecting `Organise` → `Add shortcut`. This will create a shortcut (without copying or taking space) of the folder to a desired path in your Google Drive, from which you can download its content.

### 🧠 Available Encoding Models

Following is a table with the encoding models available in the Neural Encoding Dataset. Each row corresponds to a different encoding model, and the columns indicate their *attributes*:

* **modality:** the neural data modality on which the encoding model was trained.
* **train_dataset:** the neural dataset on which the encoding model was trained.
* **model:** the type of encoding model used.
* **subject:** independent subjects on which encoding models were trained (a separate encoding model is trained for each subject).
* **roi:** independent Regions of Interest (ROIs) on which encoding models were trained (a separate encoding model is trained for each ROI). This only applies to fMRI neural data modality.

| modality | train_dataset | model | subject | roi |
|-------------|-----------------------|----------| ----------| ----|
| fmri | nsd | fwrf | 1, 2, 3, 4, 5, 6, 7, 8 | V1, V2, V3, hV4, EBA, FBA-2, OFA, FFA-1, FFA-2, PPA, RSC, OPA, OWFA, VWFA-1, VWFA-2, mfs-words, early, midventral, midlateral, midparietal, parietal, lateral, ventral|
| eeg | things_eeg_2 | vit_b_32 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10| – |
 
For more information on the encoding model's *attributes* (e.g., training dataset or model type) please see the [data manual][data_manual]. These *attributes* are required inputs when using `NED`'s functions (i.e., to select the encoding model you actually want to use).

### ✨ NED Functions

#### 🔹 Initialize the NED object

To use `NED`'s functions you need to import `NED` and create a `ned_object`.

```python
from ned.ned import NED

# The NED object requires as input the directory to the Neural Encoding Dataset
ned_dir = '../neural_encoding_dataset/'

# Create the NED object
ned_object = NED(ned_dir)
```
#### 🔹 Synthesize Neural Responses to any Image of Your Choice

Synthesizing neural responses for images involves two steps. You first need to load the neural encoding model of your choice using the `get_encoding_model` method.

```python
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
device : str
	Whether the encoding model is stored on the 'cpu' or 'cuda'. If
	'auto', the code will use GPU if available, and otherwise CPU.

Returns
-------
encoding_model : dict
	Neural encoding model.
"""

# Load the fMRI encoding model
fmri_encoding_model = ned_object.get_encoding_model(
	modality='fmri', # required
	train_dataset='nsd', # required
	model='fwrf', # required
	subject=1, # required
	roi='V1', # default is None, only required if modality=='fmri'
	device='auto' # default is 'auto'
	)

# Load the EEG encoding model
eeg_encoding_model = ned_object.get_encoding_model(
	modality='eeg', # required
	train_dataset='things_eeg_2', # required
	model='vit_b_32', # required
	subject=1, # required
	roi=None, # default is None, only required if modality=='fmri'
	device='auto' # default is 'auto'
	)

```

Next, with the `encode` method you can synthesize fMRI or EEG responses to any image of your choice, and optionally return the corresponding metadata (i.e., information on the neural data used to train the encoding models such as the amount of fMRI voxels or EEG time points, and on the trained encoding models, such as which data was used to train and test the models, or the models accuracy scores).

```python
"""
Synthesize neural responses for arbitrary stimulus images, and
optionally return the synthetic fMRI metadata.

Parameters
----------
encoding_model : dict
	Neural encoding model.
images : int
	Images for which the neural responses are synthesized. Must be a 4-D
	numpy array of shape (Batch size x 3 RGB Channels x Width x Height)
	consisting of integer values in the range [0, 255]. Furthermore, the
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

# Encode fMRI responses to images
encoded_fmri, fmri_metadata = ned_object.encode(
	encoding_model, # required
	images, # required
	return_metadata=True, # default is True
	device='auto' # default is 'auto'
	)

# Encode EEG responses to images
encoded_eeg, eeg_metadata = ned_object.encode(
	encoding_model, # required
	images, # required
	return_metadata=True, # default is True
	device='auto' # default is 'auto'
	)
```

#### 🔹 Load the Pre-Generated Synthetic Neural Responses

The `load_synthetic_neural_responses` method will pre-generated synthetic fMRI or EEG responses for ~150,000 naturalistic images (either 73,000 images from the Natural Scenes Dataset, 26,107 images from the THINGS Database, or 50,000 images from the ImageNet 2012 Challenge validation split), and optionally return the corresponding metadata.

```python
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
	If 'nsd', load synthetic neural responses for the 73,000 NSD images
	(Allen et al., 2023).
	If 'imagenet_val', load synthetic neural responses for the 50,000
	ILSVRC-2012 validation images (Russakovsky et al., 2015).
	If 'things', load synthetic neural responses for the 26,107 images
	from the THINGS database (Hebart et al., 2019).
	Imageset for which the fMRI responses are synthesized. Available
	options are 'nsd', 'imagenet_val' and 'things'.
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

# Load synthetic fMRI responses
synthetic_fmri, fmri_metadata = ned_object.load_synthetic_neural_responses(
	modality='fmri', # required
	train_dataset='nsd', # required
	model='fwrf', # required
	imageset='things', # required, one of ['nsd', 'things', 'imagenet_val']
	subject=1, # required
	roi='V1', # default is None, only required if modality=='fmri'
	return_metadata=True # default is True
	)

# Load synthetic EEG responses
synthetic_eeg, eeg_metadata = ned_object.load_synthetic_neural_responses(
	modality='eeg', # required
	train_dataset='things_eeg_2', # required
	model='vit_b_32', # required
	imageset='things', # required, one of ['nsd', 'things', 'imagenet_val']
	subject=1, # required
	roi=None, # default is None, only required if modality=='fmri'
	return_metadata=True # default is True
	)
```

### 💻 Tutorials

To familiarize with the Neural Encoding Dataset we created tutorials for both fMRI and EEG modalities. In these tutorial you will learn how to use `NED`'s functions, for example to synthesize fMRI and EEG responses for images of your choice, and you will also familiarize with the pre-generated synthetic fMRI and EEG responses for ~150,000 naturalistic images.

These tutorials are available on either Google Colab ([fMRI tutorial][fmri_tutorial_colab], [EEG tutorial][eeg_tutorial_colab]) or Jupyter Notebook ([fMRI tutorial][fmri_tutorial_jupyter], [EEG tutorial][eeg_tutorial_jupyter]).



## 📦 Neural Encoding Dataset Creation Code

The folder [`../NED/ned_creation_code/`][ned_creation_code] contains the code used to create the Neural Encoding Dataset, divided in the following sub-folders:

* **[`../00_prepare_data/`][prepare_data]:** prepare the data (i.e., images and corresponding neural responses) used to train the encoding models.
* **[`../01_train_encoding_models/`][train_encoding]:** train the encoding models, and save their weights.
* **[`../02_test_encoding_models/`][test_encoding]:** test the encoding models (i.e., compute and plot their encoding accuracy).
* **[`../03_create_metadata/`][metadata]:** create metadata files for the encoding models and their synthetic neural responses.
* **[`../04_synthesize_neural_responses/`][synthesize]:** use the trained encoding models to synthesize neural responses for ~150,000 naturalistic images.



## ❗ Issues

If you come across problems with the toolbox, please submit an issue!



## 📜 Citation

If you use the Neural Encoding Dataset, please cite:

> *Gifford AT, Cichy RM. 2024. In preparation. https://github.com/gifale95/NED*


[ned_website]: https://www.alegifford.com/projects/ned/
[imagenet]: https://www.image-net.org/challenges/LSVRC/2012/index.php
[russakovsky]: https://link.springer.com/article/10.1007/s11263-015-0816-y
[things]: https://things-initiative.org/
[hebart]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223792
[nsd]: https://naturalscenesdataset.org/
[allen]: https://www.nature.com/articles/s41593-021-00962-x
[requirements]: https://github.com/gifale95/NED/blob/main/requirements.txt
[rclone]: https://rclone.org/
[guide]: https://noisyneuron.github.io/nyu-hpc/transfer.html
[ned_data]: https://forms.gle/ZKxEcjBmdYL6zdrg9
[data_manual]: https://docs.google.com/document/d/1DeQwjq96pTkPEnqv7V6q9g_NTHCjc6aYr6y3wPlwgDE/edit?usp=drive_link
[encode_doc]: https://github.com/gifale95/NED/blob/main/ned/ned.py#L205
[load_synthetic_doc]: https://github.com/gifale95/NED/blob/main/ned/ned.py#L438
[fmri_tutorial_colab]: https://colab.research.google.com/drive/1W9Sroz2Y0eTYfyhVrAJwe50GGHHAGBdE?usp=drive_link
[eeg_tutorial_colab]: https://colab.research.google.com/drive/10NSRBrJ390vuaPyRWq5fDBIA4NNAUlTk?usp=drive_link
[fmri_tutorial_jupyter]: https://github.com/gifale95/NED/blob/main/tutorials/ned_fmri_tutorial.ipynb
[eeg_tutorial_jupyter]: https://github.com/gifale95/NED/blob/main/tutorials/ned_eeg_tutorial.ipynb
[ned_creation_code]: https://github.com/gifale95/NED/tree/main/ned_creation_code/
[prepare_data]: https://github.com/gifale95/NED/tree/main/ned_creation_code/00_prepare_data
[train_encoding]: https://github.com/gifale95/NED/tree/main/ned_creation_code/01_train_encoding_models
[test_encoding]: https://github.com/gifale95/NED/tree/main/ned_creation_code/02_test_encoding_models
[metadata]: https://github.com/gifale95/NED/tree/main/ned_creation_code/03_create_metadata
[synthesize]: https://github.com/gifale95/NED/tree/main/ned_creation_code/04_synthesize_neural_responses





