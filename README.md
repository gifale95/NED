# NED

The ```NED``` toolbox provides utility functions and tutorials for using the [**Neural Encoding Dataset**][ned_website]: trained encoding models of fMRI and EEG responses to images of multiple subjects, which you can use to synthesize fMRI and EEG responses to any image of your choice.

The Neural Encoding Dataset also comes with pre-generated synthetic fMRI and EEG responses for ~150,000 naturalistic images coming from the [ImageNet 2012 Challenge][imagenet] ([*Russakovsky et al., 2015*][russakovsky]), the [THINGS database][things] ([*Hebart et al., 2019*][hebart]), and the [Natural Scenes Dataset][nsd] ([*Allen et al., 2022*][allen]).

For additional information on the Neural Encoding Dataset you can check out the [website][ned_website].

!!!!!!!!!!!!!!!!!!!!!!!! Use emojis for titles !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



## Installation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

To run the code first install [Anaconda][conda], then create and activate a dedicated Conda environment by typing the following into your terminal:
```shell
curl -O https://raw.githubusercontent.com/gifale95/eeg_encoding_model/main/environment.yml
conda env create -f environment.yml
conda activate eeg_encoding
```
Alternatively, after installing Anaconda you can download the [environment.yml][env_file] file, open the terminal in the download directory and type:
```shell
conda env create -f environment.yml
conda activate eeg_encoding
```



## Documentation / How to use !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

The NED toolbox requires you to prior download the neural encoding dataset at ... You don't need to download all of it, but only the parts you fnd necessary.

Link to NED data manual.

Small example of utility functions (see frrsa & ncsnr repos)

Provide list of available modalities, training datasets, models.



## Tutorials !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Tutorials ("tutorials" folder, add colab tutorials there in jupyter notebook format, and say that the tutorial are also available on Colab):

[Here][colab] you will find a Colab interactive tutorial on how to load and visualize the preprocessed EEG data and the corresponding stimuli images.

[colab]: https://colab.research.google.com/drive/1i1IKeP4cK3ViscP4b4kNOVo4kRoL8tf6?usp=sharing



## ❗ Issues

If you come across problems or have suggestions please submit an issue!



## Citation

If you use the Neural Encoding Dataset, please cite the following paper:

*Gifford AT, Cichy RM. 2024. In preparation.*


[ned_website]: https://www.alegifford.com/projects/ned/
[imagenet]: https://www.image-net.org/challenges/LSVRC/2012/index.php
[russakovsky]: https://link.springer.com/article/10.1007/s11263-015-0816-y
[things]: https://things-initiative.org/
[hebart]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223792
[nsd]: https://naturalscenesdataset.org/
[allen]: https://www.nature.com/articles/s41593-021-00962-x

