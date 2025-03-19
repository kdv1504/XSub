# Extension of XSub (Explanation-Driven Adversarial Attack against Blackbox Classifiers via Feature Substitution) to Medical Image Classifiers

This repository contains the code to extend the results of the [Explanation-Driven Adversarial Attack against Blackbox Classifiers via Feature Substitution](https://ieeexplore.ieee.org/document/10825935) paper to medical image classifiers. 

![Overview of the attack](framework.png)

## Installation 

### Cloning and handling dependencies 

Clone the repo:

```
git clone https://github.com/kdv1504/XSub.git
```

Create a conda environment and activate it:

```
conda env create -f environment.yml
conda activate xsub
```

## Downloading and preparing data 

The code in this repository was written to work with the [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/data) as an example. You can download the data directly from the kaggle site, or from [Google Drive](https://drive.google.com/drive/folders/1oXBbcrzAxSqokRGvyruX98bcjnlyTmOf?usp=drive_link). Note that the dataset from Kaggle includes both the original images and their corresponding masks. In the Drive link, you can find a "cleaned" version of the dataset where only the original images are retained. This dataset is stored as ```breast_ultrasound```. You can find the dataset with the masks in ```breast_ultrasound_ori```.

When working with any other dataset, make sure dataset is stored in the following order:
```
data #root data directory
|
|__ class_0 
    |__ image_0
    |__ image_1
    ...
|__ class_1
    |__ image_0
    |__ image_1
    ...
...
|__ class_n
    |__ image_0
    |__ image_1
    ...
```
The [dataprep.py](dataprep.py) file contains code to prepare the data. The code has been written to be as general and applicable to as many datasets, but you can modify this file to suit your own needs and the specificity of your datasets. 

### Setting up classifier
You can use [this pretrained model](https://drive.google.com/drive/folders/1Iq29jV_vktPM2BNEEi98BM7YzOKZSnUE?usp=drive_link) to classify the Breast Ultrasound Images dataset. Remember to put the downloaded folder within the ```saved``` directory for the code to work.  

## Running the XSub code 
Running the [xsub.py](xsub.py) file should give you the results of the attack on a given dataset:
```
python3 xsub.py
```
[xsub_backdoor.py](xsub_backdoor.py) will be updated in the future.

Within [xsub.py](xsub.py), there are lines of code you can modify to change the hyperparameters of the attack. Note that the code in this repository is most suitable for attacks using only 1 feature (i.e., $K=1$). Attacks with more features will be updated soon. 

The [config.py](utils/config.py) file can also be modified if you need to:
- Resize images
- Adjust batch size
- Adjust hyperparameters for training a different classifier
- Change directories for saving your progress


