## Imports
# Data related libraries
import pandas as pd

# System
from os import listdir, remove
import os.path
from os.path import isfile, join, splitext
from os import listdir
from os.path import isfile, join
import sys

# Persistance
import pickle

# Modeling
from sklearn.linear_model import LinearRegression

# Ansatz function
def f(x, a, b):
    """
    Constant-quadratic ansatz-function with two parameters
    """
    return (x >= a).astype(int)*(b*((x-a)**2))
	
def reduceFilePath(filepath):
    return os.path.split(filepath)[1]
	
def readGraphFeatures(folder):
    """
    Reads in calculated graph features placed in 'folder'
    """
    # Get all the filenames in the directory
    path = join("features", folder)
    files = [f for f in listdir(path) if isfile(join(path, f)) and "_graph" in f]
    
    # Creat empty list
    li = []
    
    # Iterate over files
    for f in files:
        file = join(path, f)
        data = pd.read_csv(file, delimiter=',', encoding='utf-8', index_col = 0)
        li.append(data)
        
    # Combine into dataframe and return
    data_graph = pd.concat(li, axis = 0, ignore_index = False)
    data_graph.index = data_graph.index.str.replace("_Microstructure.graphml", "")
    data_graph.index = data_graph.index.map(reduceFilePath)
    data_graph = data_graph.sort_index()
	
    print(f"LOG: data_graph:")
    print(data_graph)
    
    return data_graph
	
def readStretchFeatures(folder):
    """
    Reads in calculated stretch features placed in 'folder'
    """
    # Get all the filenames in the directory
    path = join("features", folder)
    files = [f for f in listdir(path) if isfile(join(path, f)) and "_stretch" in f]
    
    # Creat empty list
    li = []
    
    # Iterate over files
    for f in files:
        file = join(path, f)
        data = pd.read_csv(file, delimiter=',', encoding='utf-8', index_col = 0)
        li.append(data)
        
    # Combine into dataframe and return
    data_stretch = pd.concat(li, axis=0, ignore_index=False)
    data_stretch.index = data_stretch.index.str.replace("_Microstructure.graphml", "")
    data_stretch.index = data_stretch.index.map(reduceFilePath)
    data_stretch = data_stretch.sort_index()
	
    print(f"LOG: data_stretch:")
    print(data_stretch)
    
    return data_stretch
	
def readPolyfitTargets(path):
    """
    Reads in alpha, beta targets based on the ansatzfitting from 'path'
    """
    # Get all the filenames in the directory
    #path = f"labels/"
    files = [f for f in listdir(path) if isfile(join(path, f)) and "_polyfit" in f]
    
    # Creat empty list
    li = []
    
    # Iterate over files
    for f in files:
        file = join(path, f)
        data = pd.read_csv(file, delimiter=',', encoding='utf-8', index_col = 0)
        data = data.T
        data.index = [f]
        li.append(data)
        
    # Combine into dataframe and return
    #data_polyfit = pd.concat(li, axis=0, ignore_index=False)
    data_polyfit = pd.concat(li, ignore_index=False)
    data_polyfit.index = data_polyfit.index.str.replace("_StressStrainCurve.csv_polyfit.csv", "")
    #data_polyfit = data_polyfit.sort_index()
	
    print(f"LOG: data_polyfit:")
    print(data_polyfit)	
    
    return data_polyfit
	
def combineInputData(data_graph, data_stretch, data_polyfit, deduplicate = True):
    """
    Combines the read in data into a single dataframe
    """
    # Check if same samples inside both
    #print(f"LOG: Check if combined data contains same samples: {data_polyfit.index.equals(data_graph.index)}, {data_polyfit.index.equals(data_stretch.index)}")

    data_joined = data_graph.join(data_polyfit)
    data_joined = data_joined.join(data_stretch)
    
    # Select only the ones inside a certain threshold
    data_joined = data_joined[data_joined.sld >= 1040]
    
    # Remove duplicated entries
    if deduplicate:
        data_joined = data_joined[data_joined.duplicated() == False]
    
    #print(f"LOG: Input samples: {len(data_joined)}")
    #print(f"LOG: Number of missing values: {data_joined.isnull().sum().sum()}")
    
    return data_joined
	
def getFeatureCombinations():
    """
    This function groups the columns into feature sets togeter and returns a dict
    """
    param_features = ["kappa", "sigRamp", "sigSde", "sld"]
    graph_features = list(data_graph.columns[4:])
    stretch_features = list(data_stretch.columns)
    
    feature_combinations = {'param':param_features, 
                            'stretch':stretch_features, 
                            'graph':graph_features, 
                            'param+graph':param_features + graph_features, 
                            'param+stretch':param_features + stretch_features, 
                            'graph+stretch':graph_features + stretch_features, 
                            'param+graph+stretch':param_features + graph_features + stretch_features,
                            }
    
    return feature_combinations
	
def trainFinalModel(data_joined, features):
    """
    Reads in the graph and stretch features and trains separate linear regression models 
    to predict alpha resp. beta. 
    """

    feature_comb = getFeatureCombinations()[features]
    
    final_linreg_alpha = LinearRegression().fit(data_joined[feature_comb], data_joined['alpha'])
    final_linreg_beta = LinearRegression().fit(data_joined[feature_comb], data_joined['beta'])
            
    return final_linreg_alpha, final_linreg_beta
	
if __name__ == "__main__":

	folder = sys.argv[1]
	
	
    # Get full command-line arguments
	full_cmd_arguments = sys.argv
    
    # Keep all but the first
	argument_list = full_cmd_arguments[1:]
	
    # Evaluate given options
	zipped = False
	
	for current_argument in argument_list:
		if current_argument in ("-f", "--file"):
			print ("Enabling zipped mode")
			zipped = True
	
	data_graph = readGraphFeatures(folder)
	data_stretch = readStretchFeatures(folder)
	polyfit_path = join("polyfit", folder)
	data_polyfit = readPolyfitTargets(polyfit_path)
	data_joined = combineInputData(data_graph, data_stretch, data_polyfit, deduplicate = True)
	features = 'graph+stretch'
	feature_comb = getFeatureCombinations()[features]
	
	final_linreg_alpha, final_linreg_beta = trainFinalModel(data_joined, features = features)
	
	# Serialization
	filename = 'trained_models/pickle_final_linreg_alpha'
	outfile = open(filename,'wb')
	pickle.dump(final_linreg_alpha,outfile)
	outfile.close()
	
	filename = 'trained_models/pickle_final_linreg_beta'
	outfile = open(filename,'wb')
	pickle.dump(final_linreg_beta,outfile)
	outfile.close()
	
	print("Done training models!")
	