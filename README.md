# random-nonwoven-fibers
Code for the paper "Graph-Based Tensile Strength Approximation of Random Nonwoven Materials by Interpretable Regression"


## Dataset
You may download the datasets used for our experiments using ```download_data.py```. 
It downloads two (large) archives to the ```data``` folder.
In particular
- ```labeled.tar.gz``` contains 295 graphs in GML format and 295 corresponding CSV files containing the strain/stress curves of the corresponding samples (as computed by an ordinary differential equation solver) 
- ```unlabeled.tar.gz``` contains 739 graphs in GML format without corresponding strain/stress curves.

## Usage
### Feature generation
Calculate graph and stretch features for a set of graphml files from a folder.

```python feature_generation.py input_data_labelled```

Results are placed in ```features/input_data_labelled```

### Ansatz fitting
Calculate alpha, beta the best fitting parameters to a set of given strain-stress curves in a folder.

```python ansatzfitting.py input_data_labelled```

Results are placed in ```polyfit/input_data_labelled```

### Training final model
Train linear regression models for predicting alpha and beta respectively.

```python train_final_model.py input_data_labelled``` 

Results are placed in ```trained_models``` as pickled scikit-learn models.