# random-nonwoven-fibers
Code for the paper "Graph-Based Tensile Strength Approximation of Random Nonwoven Materials by Interpretable Regression" by Dario Antweiler, Marc Harmening, Nicole Marheineke, Andre Schmeißer, Raimund Wegener, and Pascal Welke.

If you want to access the exact code that we used and submitted with the above publication, please use Release v1.0.0 .

The current version v2.0.0 differs as follows:
- it provides the code that is necessary to generate fiber graphs 
- it provides the code to simulate the stress-strain curves
- it stores the trained surrogate model, instead of just reporting its quality
- it contains a small script ```predict.py``` to apply the learned surrogate model to novel fiber graphs to predict stress-strain curves 
- it expects to run in a docker container


## Dependencies

The code expects to run in a docker container. A docker file is provided in ```/environment/Dockerfile```.

The fiber __graph generation__ and subsequent stress-strain curve __simulation__ are written in Matlab and hence require a recent Matlab version (and license).

The __surrogate model__ training, evaluation, and application are written in Python.
All python packages that we depend on are listed in ```environment.yaml```. 
You may create a virtual environment e.g. using [anaconda](https://anaconda.org) via the command

```conda env create -n nwf-stretch -f environment.yaml```

After activating the environment, you can follow the steps below.


## Complete Run of the Whole Pipeline
The entry point of the full pipeline is the bash script ```/code/run```.
The script expects to be run in a docker container in the absolute path ```/code/run```.
You can choose, by setting the parameter within ```/code/run``` how much generation and simulation you want to run by setting the ```generate_data``` variable. By default, ```generate_data="none"```.

You can choose from the following parameters:

#### ```all```
Create a dataset similar to the one used in 

Antweiler, Harmening, Marheineke, Schmeißer, Wegener, Welke (2022):

Graph-Based Tensile Strength Approximation of Random Nonwoven Materials by Interpretable Regression.

Machine Learning with Applications (8), Elsevier.

https://doi.org/10.1016/j.mlwa.2022.100288 

Generate 25 graphs each for 
  - 5 random parameter combinations and simulate stress-strain curves for all 25 graphs
  - 50 random parameter combinations and simulate only one stress-strain curve 

#### ```single```
A 'short' example run that generates a single graph and simulates its stress-strain curve
(takes roughly 30h on current CodeOcean instances)
Note that this setting is just to showcase that everything works in time that a free-tier CodeOcean user has. 
It should not be used to train any production model.


#### ```none```
No graph is generated, only the few graphs that are already in the repository are used for a very small training run.


## Run with Existing Dataset
If graph generation and subsequent stress-strain curve simulation takes too long, or you want to reproduce the results reported in our paper, you may download the datasets used for our experiments using ```download_data.py```. 
It downloads two (large) archives to the base folder of the repository.
In particular
- ```labeled.tar.gz``` contains 295 graphs in GML format and 295 corresponding CSV files containing the strain/stress curves of the corresponding samples (as computed by an ordinary differential equation solver) 
- ```unlabeled.tar.gz``` contains 739 graphs in GML format without corresponding strain/stress curves.


### Full Usage Example with zipped files
```
conda activate nwf-stretch

python download_data.py
python feature_generation.py -f /results/labeled.tar.gz
python ansatzfitting.py -f /results/labeled.tar.gz
python train_final_model.py /results/labeled.tar.gz
```

### Full Usage Example with unzipped files
```
conda activate nwf-stretch

python feature_generation.py /results/input_data_labelled
python ansatzfitting.py /results/input_data_labelled
python train_final_model.py /results/input_data_labelled
```

### Full Reproduction of Results in the Paper
```
conda activate nwf-stretch

python download_data.py
python feature_generation.py -f /results/labeled.tar.gz
python ansatzfitting.py -f /results/labeled.tar.gz
python train_validate.py /results/labeled.tar.gz /results/unlabeled.tar.gz -f -p
```

## Detailed Description of Individual Steps

### Feature generation
Calculate graph and stretch features for a set of graphml files from a folder or a gzipped file:

Folder: (some toy data is included in this git repository, without the need to download all data)
```python feature_generation.py /results/input_data_labelled```

File:
```python feature_generation.py -f /results/labeled.tar.gz```
```python feature_generation.py -f /results/unlabeled.tar.gz```


Results are placed in ```/results/features/``` with a subfolder corresponding to the folder/filename. 
!!! Please note, only folders/files in the base directory of this repository work. Please consider creating a symlink if you have to store the data somewhere else !!! 

### Ansatz fitting
Calculate alpha, beta the best fitting parameters to a set of given strain-stress curves in a folder.

Folder:
```python ansatzfitting.py /results/input_data_labelled```

File:
```python ansatzfitting.py -f /results/labeled.tar.gz```
```python ansatzfitting.py -f /results/unlabeled.tar.gz```

Results are placed in ```/results/polyfit/``` with a subfolder corresponding to the folder/filename. 
!!! Please note, only folders/files in the base directory of this repository work. Please consider creating a symlink if you have to store the data somewhere else !!! 

### Train and validate

To reproduce the results in the paper you can run the ```train_validate.py``` script. For each parameter combination in the labelled dataset this function trains two models (for alpha, beta) on all data points except for the chosen combination and tests them on the rest.

Assuming that you have labeled input data in folder or file ```/results/input_data_labelled``` and unlabeled input data in ```input_data_graphonly``` and have already run ```feature_generation.py``` and ```ansatzfitting.py``` for both datasets, you can run

```python train_validate.py /results/input_data_labelled /results/input_data_graphonly```

You can activate plotting with the "-p" flag:

```python train_validate.py /results/input_data_labelled /results/input_data_graphonly -p```

Plots are placed in the folder "visuals".

To run the function on zipped input, call

```python train_validate.py /results/labeled.tar.gz /results/unlabeled.tar.gz -f -p```


### Training final model

Train linear regression models for predicting alpha and beta respectively.

```python train_final_model.py /results/input_data_labelled``` 

Results are placed in ```/results/trained_models``` as pickled scikit-learn models.

!!! Note that even for zipped input data the call for this step does not include the ```-f``` flag, i.e. you should call

```python train_final_model.py /results/labeled.tar.gz```

### Making predictions

Using the trained models to predict alpha and beta from graph samples.

```predict.py trained_models/pickle_final_linreg_alpha trained_models/pickle_final_linreg_beta input_data_graphonly```

Results are placed in ```/results/predictions/input_data_graphonly```