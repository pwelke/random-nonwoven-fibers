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
Calculate features for set of graphml files from a folder.

```python feature_generation.py [Folder]``` e.g. ```python feature_generation.py [input_data_labelled]```

Results are placed in ```features/[folder]```