# random-nonwoven-fibers
Code for the paper "Graph-Based Tensile Strength Approximation of Random Nonwoven Materials by Interpretable Regression"


## Dataset
You may download the datasets used for our experiments using ```download_data.py```. 
It downloads two (large) archives to the base folder of the repository.
In particular
- ```labeled.tar.gz``` contains 295 graphs in GML format and 295 corresponding CSV files containing the strain/stress curves of the corresponding samples (as computed by an ordinary differential equation solver) 
- ```unlabeled.tar.gz``` contains 739 graphs in GML format without corresponding strain/stress curves.

## Usage
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

### Training final model
Train linear regression models for predicting alpha and beta respectively.

```python train_final_model.py input_data_labelled``` 

Results are placed in ```trained_models``` as pickled scikit-learn models.