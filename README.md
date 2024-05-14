# NED

The ```NED``` toolbox provides utility functions and tutorials for using the [**Neural Encoding Dataset**][ned_website]: trained encoding models of fMRI and EEG responses to images of multiple subjects, which you can use to synthesize fMRI and EEG responses to any image of your choice.

The Neural Encoding Dataset also comes with pre-generated synthetic fMRI and EEG responses for ~150,000 naturalistic images coming from the [ImageNet 2012 Challenge][imagenet] ([*Russakovsky et al., 2015*][russakovsky]), the [THINGS database][things] ([*Hebart et al., 2019*][hebart]), and the [Natural Scenes Dataset][nsd] ([*Allen et al., 2022*][allen]).

For additional information on the Neural Encoding Dataset you can check out the [website][ned_website].



## 🤝 Contribute to Expanding the Neural Encoding Dataset

Do you have encoding models with higher prediction accuracies than the ones currently available in the Neural Encoding Dataset, and would like to make them available to the community? Or maybe you have encoding models for new neural datasets, data modalities (e.g., MEG/ECoG/animal), or stimulus types (e.g., videos, language) that you would like to share? Or perhaps you have suggestions for improving the Neural Encoding Dataset? Then please get in touch vith Ale (alessandro.gifford@gmail.com): all feedback and help is strongly appreciated!



## 🔧 Installation

To install ```NED``` run the following command on your terminal:

```shell
pip install -U git+https://github.com/gifale95/NED.git
```

You will additionally need to install the Python dependencies found in [requirements.txt][requirements].



## 📃 How to use !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

To use ```NED``` you first need to download the Neural Encoding Dataset from [here][ned_data]. Depending on how you want to use the Neural Encoding Dataset, you might not need to download all of it. For this please see the [data manual][data_manual], which describes how the Neural Encoding Dataset is structured.

Small example of utility functions (see frrsa & ncsnr repos)

Provide list of available modalities, training datasets, models.



## 💻 Tutorials

To familiarize with ```NED``` we created tutorials for both fMRI and EEG modalities. In these tutorial you will learn how to use ```NED```'s functions to synthesize fMRI and EEG responses for new images, as well as using the pre-generated synthetic fMRI and EEG responses for ~150,000 naturalistic images.

These tutorials are available on either [Colab][colab] or [Jupyter Notebook][jupyter].



## 📦 Neural Encoding Dataset Creation Code

The folder [```../NED/ned_creation_code/```][ned_creation_code] contains the code used to create the Neural Encoding Dataset, divided in the following sub-folders:

> * [```../00_prepare_data/```][prepare_data]: prepare the data (i.e., images and corresponding neural responses) used to train the encoding models.
> * [```../01_train_encoding_models/```][train_encoding]: train the encoding models, and save their weights.
> * [```../02_test_encoding_models/```][test_encoding]: test the encoding models (i.e., compute and plot their encoding accuracy).
> * [```../03_create_metadata/```][test_encoding]: create metadata files for the encoding models and their synthetic neural responses.
> * [```../04_synthesize_neural_responses/```][synthesize]: use the trained encoding models to synthesize neural responses for ~150,000 naturalistic images.



## ❗ Issues

If you come across problems with the toolbox, please submit an issue!



## 📜 Citation

If you use the Neural Encoding Dataset, please cite the following paper:

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
[colab]: https://drive.google.com/drive/folders/13aTI5eSK4yDosi63OfsyN20fLo6T5uNj?usp=drive_link
[jupyter]: https://github.com/gifale95/NED/tree/main/tutorials
[ned_creation_code]: https://github.com/gifale95/NED/tree/main/ned_creation_code/
[prepare_data]: https://github.com/gifale95/NED/tree/main/ned_creation_code/00_prepare_data
[train_encoding]: https://github.com/gifale95/NED/tree/main/ned_creation_code/01_train_encoding_models
[test_encoding]: https://github.com/gifale95/NED/tree/main/ned_creation_code/02_test_encoding_models
[synthesize]: https://github.com/gifale95/NED/tree/main/ned_creation_code/03_synthesize_neural_responses





