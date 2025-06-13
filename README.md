
ML models for Charge Predicition
================================

This repository contains the code and dataset to train a convulotional neural network (CNN) and a SOAP based neural network (NN).

### ML models
We used two models for charge density prediction. 

#### CNN
A UNet architecture with some modifications has been trained to learn the final density $\rho$ from initial density $\rho_0$. The effect of using the local potential $v_{loc}$ as an input feature has aslo been studied. 

#### SOAP based
Features have been calcuted using SOAP as implmeneted in the [DScribe](www.github.com/). A NN is then used to learn the density map. Please see notebooks for details.

#### Dataset
The dataset for verb[MgO] and verb[Cu] have been produced using Castep by outputing the input and output density. A sample of the dataset can be found in this repository. However, you can download the full dataset from [the Google drive](www.google.com).

#### Reproducing the results
You can create a python environment using
```bash
python3 -m venv env_name
```
or in conda using
```bash
conda create -n env_name
```
After activating the environment, you can install these requirements by running
```bash
pip install -r requirements.txt
```
You should be able to run the jupyter notebooks in this repository.
