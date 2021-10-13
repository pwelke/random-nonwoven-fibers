# random-nonwoven-fibers
Code for the paper "Graph-Based Tensile Strength Approximation of Random Nonwoven Materials by Interpretable Regression" by Dario Antweiler, Marc Harmening, Nicole Marheineke, Andre  Schmeißer, Raimund Wegener, and Pascal Welke.


## Dependencies
All packages that we depend on are listed in ```environment.yaml```. 
You may create a virtual environment e.g. using [anaconda](https://anaconda.org) via the command

```conda env create -n nwf-stretch -f environment.yaml```

After activating the environment, you can follow the steps below.

## Dataset
You may download the datasets used for our experiments using ```download_data.py```. 
It downloads two (large) archives to the base folder of the repository.
In particular
- ```labeled.tar.gz``` contains 295 graphs in GML format and 295 corresponding CSV files containing the strain/stress curves of the corresponding samples (as computed by an ordinary differential equation solver) 
- ```unlabeled.tar.gz``` contains 739 graphs in GML format without corresponding strain/stress curves.

## Usage

### Full Usage Example with zipped files
```
conda activate nwf-stretch

python download_data.py
python feature_generation.py -f data_labeled.tar.gz
python ansatzfitting.py -f data_labeled.tar.gz
python train_final_model.py data_labeled.tar.gz
```

### Full Usage Example with unzipped files
```
conda activate nwf-stretch

python feature_generation.py input_data_labelled
python ansatzfitting.py input_data_labelled
python train_final_model.py input_data_labelled
```

### Full Reproduction of Results in the Paper
```
conda activate nwf-stretch

python download_data.py
python feature_generation.py -f data_labeled.tar.gz
python ansatzfitting.py -f data_labeled.tar.gz
python train_final_model.py data_labeled.tar.gz
```

## Detailed Description of Individual Steps

### Feature generation
Calculate graph and stretch features for a set of graphml files from a folder or a gzipped file:

Folder: (some toy data is included in this git repository, without the need to download all data)
```python feature_generation.py input_data_labelled```

File:
```python feature_generation.py -f nonwoven-fiber-graphs_labeled.tar.gz```
```python feature_generation.py -f nonwoven-fiber-graphs_unlabeled.tar.gz```


Results are placed in ```features/``` with a subfolder corresponding to the folder/filename. 
!!! Please note, only folders/files in the base directory of this repository work. Please consider creating a symlink if you have to store the data somewhere else !!! 

### Ansatz fitting
Calculate alpha, beta the best fitting parameters to a set of given strain-stress curves in a folder.

Folder:
```python ansatzfitting.py input_data_labelled```

File:
```python ansatzfitting.py -f nonwoven-fiber-graphs_labeled.tar.gz```
```python ansatzfitting.py -f nonwoven-fiber-graphs_unlabeled.tar.gz```

Results are placed in ```polyfit/``` with a subfolder corresponding to the folder/filename. 
!!! Please note, only folders/files in the base directory of this repository work. Please consider creating a symlink if you have to store the data somewhere else !!! 

### Train and validate

To reproduce the results in the paper you can run the ```train_validate.py``` script. For each parameter combination in the labelled dataset this function trains two models (for alpha, beta) on all data points except for the chosen combination and tests them on the rest.

Assuming that you have labeled input data in folder or file ```input_data_labelled``` and unlabeled input data in ```input_data_graphonly``` and have already run ```feature_generation.py``` and ```ansatzfitting.py``` for both datasets, you can run

```python train_validate.py input_data_labelled input_data_graphonly```

You can activate plotting with the "-p" flag:

```python train_validate.py input_data_labelled input_data_graphonly -p```

Plots are placed in the folder "visuals".



### Training final model

Train linear regression models for predicting alpha and beta respectively.

```python train_final_model.py input_data_labelled``` 

Results are placed in ```trained_models``` as pickled scikit-learn models.