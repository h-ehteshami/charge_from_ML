
ML models for Charge Prediction
================================

This repository contains the code and dataset to train a convolutional neural network (CNN) and a SOAP-based neural network (NN).

### ML models
We used two models for charge density prediction. 

#### CNN
A UNet architecture with some modifications has been trained to learn the final density $\rho$ from the initial density $\rho_0$. The effect of using the local potential $v_{loc}$ as an input feature has also been studied. 

#### SOAP-based
Features have been calculated using SOAP as implemented in the [DScribe](https://singroup.github.io/dscribe/latest/index.html). A NN is then used to learn the density map. Please take a look at the notebooks for details.

#### Dataset
The dataset for verb[MgO] and verb[Cu] was produced using CASTEP by outputting the input and output density. A sample of the dataset is available in this repository. However, you can download the full dataset from [the Google Drive]( www.google.com).

#### Reproducing the results
You can create a Python environment using
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
