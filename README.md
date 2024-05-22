# Generative SSL

This is the PyTorch implemention of our paper **"Can Generative Models Improve Self-Supervised Representation Learning?"** submitted to Neurips 2024 for reproducing the experiments. 

## Requirements

We used solo-learn library for the implementation of SSL method. You can find the library in this [LINK](https://github.com/vturrisi/solo-learn).

To create the virtual environment for running the experiments please first:

`cd solo-learn`

Then install requirements based on solo-learn library documentation [here](https://github.com/vturrisi/solo-learn?tab=readme-ov-file#installation).


## Data Generation
**Note:** 
**You always need to set the proper path to the virtual environment and path to save generated data in generation scripts.**

To generate augmentations with ICGAN run:

`sbatch GenerativeSSL/scripts/generation_scripts/gen_img_icgan.slrm`

To generate augmentations with Stable Diffusion run:

`sbatch GenerativeSSL/scripts/generation_scripts/gen_img_stablediff.slrm`

## Training and Evaluation 
**Note:** 
**You always need to set the proper path to the virtual environment in solo-learn slrm files. We pretrained our models on train split of Imagenet. Here are the options for the evaluation datasets and models that we used in our experiments:**

- **Datasets:** ImageNet, iNaturalist2018, Food101, Places365, CIFAR10/100
- **Models:** Simclr (Baseline, ICGAN, Stablediff), SimSiam (Baseline, ICGAN, Stablediff), MoCo (Baseline, ICGAN, Stablediff), BYOL (Baseline, ICGAN, Stablediff), Barlow Twins (Baseline, ICGAN, Stablediff)

### Training 

Configs for training are in the `solo-learn/scripts/pretrain` folder. You can find the config files for each model and dataset in the respective folders. You need to set **path for the dataset** and **dir to save model** in each respective config file before submitting the job. By choosing the desired config you can train the methods on the ImageNet, run:

`sbatch GenerativeSSL/solo_learn/train_solo_learn.slrm`



### Evaluation

Configs for training are in the `solo-learn/scripts/linear` folder. You can find the config files for each model and dataset in the respective folders. You need to set **path for the dataset**, **dir to save model** and **path to pretrained feature extractor** in each respective config file before submitting the job. By choosing the desired config you can train the methods on the ImageNet, run:

`sbatch GenerativeSSL/solo_learn/train_solo_learn.slrm`


