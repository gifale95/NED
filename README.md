# NED

The `NED` toolbox provides utility functions and tutorials for using the [**Neural Encoding Dataset**][ned_website]: trained encoding models of fMRI and EEG responses to images of multiple subjects, which you can use to synthesize fMRI and EEG responses to any image of your choice.

The Neural Encoding Dataset also comes with pre-generated synthetic fMRI and EEG responses for ~150,000 naturalistic images coming from the [ImageNet 2012 Challenge][imagenet] ([*Russakovsky et al., 2015*][russakovsky]), the [THINGS Database][things] ([*Hebart et al., 2019*][hebart]), and the [Natural Scenes Dataset][nsd] ([*Allen et al., 2022*][allen]).

For additional information on the Neural Encoding Dataset you can check out the [website][ned_website].



## 🤝 Contribute to Expanding the Neural Encoding Dataset

Do you have encoding models with higher prediction accuracies than the ones currently available in the Neural Encoding Dataset, and would like to make them available to the community? Or maybe you have encoding models for new neural datasets, data modalities (e.g., MEG/ECoG/animal), or stimulus types (e.g., videos, language) that you would like to share? Or perhaps you have suggestions for improving the Neural Encoding Dataset? Then please get in touch vith Ale (alessandro.gifford@gmail.com): all feedback and help is strongly appreciated!



## ⚙️ Installation

To install `NED` run the following command on your terminal:

```shell
pip install -U git+https://github.com/gifale95/NED.git
```

You will additionally need to install the Python dependencies found in [requirements.txt][requirements].



## 🕹️ How to use

### 🧰 Download the Neural Encoding Dataset

To use `NED` you first need to download the Neural Encoding Dataset from [here][ned_data]. Depending on how you want to use the Neural Encoding Dataset, you might need to download all of it, or only parts of it. For this please refer to the [data manual][data_manual], which describes how the Neural Encoding Dataset is structured.

### 🧠 Available Encoding Models

Following is a table with the encoding models available in the Neural Encoding Dataset. Each row corresponds to a different encoding model, and the columns indicate their *attributes*:

* **modality:** the neural data modality on which the encoding model was trained.
* **training_dataset:** the neural dataset on which the encoding model was trained.
* **model:** the type of encoding model used.
* **subjects:** amount of independent subjects on which encoding models were trained (a separate encoding model is trained for each subject).

| modality | training_dataset | model |
|-------------|-----------------------|----------|
| fmri | nsd | fwrf |
| eeg | things_eeg_2 | vit_b_32 |
 
For more information on the encoding model's *attributes* (e.g., training dataset or model type) please see the [data manual][data_manual]. These *attributes* are required inputs when using `NED`'s functions (i.e., to select the encoding model you actually want to use).

 !!!!!!!!!!!!!!!!!!!! ADD LIST OF FMRI ROIS and subjects !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


### ✨ NED Functions !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#### Initialize the NED object

To use `NED`'s functions you will first need to import `NED` and create a `ned_object`.

```python
from ned.ned import NED

# The NED object requires as input the directory to the Neural Encoding Dataset
ned_dir = '../neural_encoding_dataset/'

# Create the NED object
ned_object = NED(ned_dir)
```
#### Synthesize Neural Responses to any Image of Your Choice

The `encode` method will synthesize fMRI or EEG responses to any image of your choice, and optionally return the corresponding metadata (i.e., information on the neural data used to train the encoding models such as the amount of fMRI voxels or EEG time points, and on the trained encoding models, such as which data was used to train and test the models, or the models accuracy scores). You can find more information on the input parameters and output of the `encode` method in its [documentation string][encode_doc].

```python
# The 'images' input variable consists in the images for which the neural
# responses are synthesized. It must be a 4-D numpy array of shape
# (Batch size x 3 RGB Channels x Width x Height), consisting of integer values
# in the range 0/255. Furthermore, the images must be of square size
# (i.e., equal width and height).

# Encode fMRI responses to images
encoded_fmri, fmri_metadata = ned_object.encode(
	images, # required
	modality='fmri', # required
	train_dataset='nsd', # required
	model='fwrf', # required
	subject=1, # required
	roi='V1', # default is None, only required if modality=='fmri'
	return_metadata=True, # default is True
	device='auto' # default is 'auto'
	)

# Encode EEG responses to images
encode_eeg, eeg_metadata = ned_object.encode(
	images, # required
	modality='eeg', # required
	train_dataset='things_eeg_2', # required
	model='vit_b_32', # required
	subject=1, # required
	roi=None, # default is None, only required if modality='fmri'
	return_metadata=True, # default is True
	device='auto' # default is 'auto'
	)
```

#### !!!!!!!!!!!!!!!!!!!!!!!!!!!!

The `load_synthetic_neural_responses` method will pre-generated synthetic fMRI or EEG responses for ~150,000 naturalistic images (either 73,000 images from the Natural Scenes Dataset, 26,107 images from the THINGS Database, or 50,000 images from the ImageNet 2012 Challenge validation split), and optionally return the corresponding metadata. You can find more information on the input parameters and output of the `load_synthetic_neural_responses` method in its [documentation string][load_synthetic_doc].

```python
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

These tutorials are available on either [Colab][colab] or [Jupyter Notebook][jupyter].



## 📦 Neural Encoding Dataset Creation Code

The folder [`../NED/ned_creation_code/`][ned_creation_code] contains the code used to create the Neural Encoding Dataset, divided in the following sub-folders:

* **[`../00_prepare_data/`][prepare_data]:** prepare the data (i.e., images and corresponding neural responses) used to train the encoding models.
* **[`../01_train_encoding_models/`][train_encoding]:** train the encoding models, and save their weights.
* **[`../02_test_encoding_models/`][test_encoding]:** test the encoding models (i.e., compute and plot their encoding accuracy).
* **[`../03_create_metadata/`][test_encoding]:** create metadata files for the encoding models and their synthetic neural responses.
* **[`../04_synthesize_neural_responses/`][synthesize]:** use the trained encoding models to synthesize neural responses for ~150,000 naturalistic images.



## ❗ Issues

If you come across problems with the toolbox, please submit an issue!



## 📜 Citation

If you use the Neural Encoding Dataset, please cite:

> *Gifford AT, Cichy RM. 2024. In preparation.*


[ned_website]: https://www.alegifford.com/projects/ned/
[imagenet]: https://www.image-net.org/challenges/LSVRC/2012/index.php
[russakovsky]: https://link.springer.com/article/10.1007/s11263-015-0816-y
[things]: https://things-initiative.org/
[hebart]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223792
[nsd]: https://naturalscenesdataset.org/
[allen]: https://www.nature.com/articles/s41593-021-00962-x
[requirements]: https://github.com/gifale95/NED/blob/main/requirements.txt
[ned_data]: https://drive.google.com/drive/folders/1flyZw95cZGBTbePByWUKN6JQz1j1HoYh?usp=drive_link
[data_manual]: https://docs.google.com/document/d/1DeQwjq96pTkPEnqv7V6q9g_NTHCjc6aYr6y3wPlwgDE/edit?usp=drive_link
[encode_doc]: https://github.com/gifale95/NED/blob/main/ned/ned.py#L205
[load_synthetic_doc]: https://github.com/gifale95/NED/blob/main/ned/ned.py#L438
[colab]: https://drive.google.com/drive/folders/13aTI5eSK4yDosi63OfsyN20fLo6T5uNj?usp=drive_link
[jupyter]: https://github.com/gifale95/NED/tree/main/tutorials
[ned_creation_code]: https://github.com/gifale95/NED/tree/main/ned_creation_code/
[prepare_data]: https://github.com/gifale95/NED/tree/main/ned_creation_code/00_prepare_data
[train_encoding]: https://github.com/gifale95/NED/tree/main/ned_creation_code/01_train_encoding_models
[test_encoding]: https://github.com/gifale95/NED/tree/main/ned_creation_code/02_test_encoding_models
[synthesize]: https://github.com/gifale95/NED/tree/main/ned_creation_code/03_synthesize_neural_responses





