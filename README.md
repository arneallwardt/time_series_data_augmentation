# Introduction
This repository contains Code for my bachelors thesis "**Analyse von Machine-Learning-Methoden zur synthetischen Erzeugung von Zeitreihen**".

## Kurzfassung
In der Domäne der Zeitreihen besteht ein Mangel an qualitativ hochwertigen Daten, was die Leistungsfähigkeit von Deep-Learning-Modellen beeinträchtigen kann. Diese Arbeit untersucht Data-Augmentation-Methoden im Bereich von Zeitreihen, welche dieses Problem adressieren. Dabei werden insbesondere Machine-Learning-Verfahren (Autoencoder (AE), Variational Autoencoder (VAE), TimeGAN) untersucht und mit algorithmischen Ansätzen (Jittering, Time Warping) verglichen. Die Methoden werden anhand der Kriterien Ähnlichkeit, Diversität und Nützlichkeit bewertet. Die Ergebnisse zeigen, dass algorithmische Verfahren konsistent bessere Ergebnisse liefern als Machine-Learning-Methoden. Zudem wird festgestellt, dass der Einsatz generativer Machine-Learning-Verfahren nur dann sinnvoll ist, wenn das generative Modell weniger komplex ist als das Modell, welches die synthetischen Daten letztlich verwenden soll.

## Abstract
In the domain of time series, there is a lack of high-quality data, which can negatively affect the performance of deep learning models. This thesis examines data augmentation methods in the field of time series that address this issue. In particular, machine learning approaches (Autoencoders (AEs), Variational Autoencoders (VAEs), TimeGAN) are investigated and compared with algorithmic techniques (Jittering, Time Warping). The methods are evaluated using discriminative-, visual- and predictive evaluation. The results show that algorithmic approaches consistently yield better outcomes than machine learning methods. Furthermore, it is noted that the use of generative machine learning techniques is only meaningful when the generative model is less complex than the model that will ultimately utilize the synthetic data.

## Existing Implementations
An existing implementation of TimeGAN was used in this repository: https://github.com/jsyoon0823/TimeGAN ([Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)) (it uses TensorFlow 1.15.0, which requires a python version <= 3.7; To install the necassary packages, you can use [this file](./data_generation/GANs/TimeGAN/requirements.txt).)

The architecture of the Variational Autoencoder is based on this example from PyTorch: https://github.com/pytorch/examples/blob/main/vae/main.py ([BSD 3-Clause License](https://opensource.org/license/bsd-3-clause))


## Dataset
All results are based on this dataset: [Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)

# How to use
You'll first need to install the required dependencies by executing `pip install -r requirements.txt` inside the root directory

This repository contains 3 important directories: [data](./data/), [data_evaluation](./data_evaluation/) and [data_generation](./data_generation/) which you can use to do the following
- plot results for discriminative- (**Ähnlichkeit**), visual- (**Diversität**), predictive evaluation (**Nützlichkeit**) aswell as the field test (**Praxistest**)
- train new AE, VAE and TimeGAN models
- generate new data using Jittering, Time Warping, AE, VAE, and TimeGAN
- look at real and synthetic data

## data_evaluation
Here you can plot results for the discriminative- (**Ähnlichkeit**), visual- (**Diversität**) and predictive evaluation (**Nützlichkeit**) aswell as the field test (**Praxistest**) by running the [evaluation_pipeline](./data_evaluation/evaluation_pipeline.ipynb). This notebook will utilize the [evaluation_data](./data/evaluation_data/) to plot the results for the discriminative- and predictive evaluation aswell as the field test. It will also calculate PCA and t-SNE for the visual evaluation.

The architectures aswell as training files for the discriminative model (**Ähnlichkeit**) and predictive model (**Nützlichkeit**) are also provided:
- [Discriminative model training](./data_evaluation/discriminative/discriminative_model_training.ipynb)
- [Discriminative model architecture](./data_evaluation/discriminative/discriminative_model.py)
- [Predictive model training](./data_evaluation/predictive/predictive_model_training.ipynb) (Saves results aswell. You can change the type of synthetic data used for the evaluation by changing `syn_data_type`)
- [Predictive model architecture](./data_evaluation/predictive/LSTM.py) 

## data_generation
Here you can use the data augmentation methods discussed in the thesis to generate synthetic data.

### Random Transformations
- [Jittering implementation](./data_generation/random_transformations/jittering.py)
- [Time Warping implementation](./data_generation/random_transformations/time_warping.py)
- [Applying random transformations to original data](./data_generation/random_transformations/apply_random_transformations.ipynb)

### Autoencoder
- [AE Architecture](./data_generation/AE/AE.py)
- [AE Training](./data_generation/AE/AE_training.ipynb) (You can train a new AE or test the saved model by setting `TEST_SAVED_MODEL=True`)

### Variational Autoencoder
- [VAE Architecture](./data_generation/VAE/VAE.py)
- [VAE Training](./data_generation/VAE/VAE_training.ipynb) (You can train a new VAE or test the saved model by setting `TEST_SAVED_MODEL=True`)

### TimeGAN
- [TimeGAN Architecture](./data_generation/GANs/TimeGAN/timegan.py), [TimeGAN Training](./data_generation/GANs/TimeGAN/tutorial_timegan.ipynb)
- NOTE: you'll need a python version <= 3.7 to run the TimeGAN training since it is based on TensorFlow 1.15.0. You can install dependencies by using [this file](./data_generation/GANs/TimeGAN/requirements.txt). The [TimeGAN README](./data_generation/GANs/TimeGAN/README_TimeGAN.md) will provide more information on how to use it.

## data
This directory contains [real](./data/real/) and [synthetic](./data/synthetic/) data (preprocessed aswell as not preprocessed).

It also contains [evaluation_data](./data/evaluation_data/), which is all of the needed data to generate the results for **Ähnlichkeit** (discriminative), **Diversität** (visual) and **Nützlichkeit** (predictive).

For every kind of data augmentation method the following kinds of data are provided inside the evaluation_data folder:
- *discriminative_test_xy*: Test data for the discriminative model to evaluate the **Ähnlichkeit**
- *results_xy*: Results for the predictive evaluation (**Nützlichkeit**) aswell as results for the field test (**Praxistest**) (Running the predictive evaluation and the field test takes up to 4 hours, therefore the results (MAEs and MSEs) are already saved in this file)