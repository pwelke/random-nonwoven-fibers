# Data related libraries
import numpy as np
import pandas as pd

# System
from os import listdir
from os.path import isfile, join
import sys, getopt
from pathlib import Path

# Persistance
import pickle

# Modeling
from sklearn.preprocessing import minmax_scale, StandardScaler, MinMaxScaler
from sklearn import preprocessing, linear_model
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ansatz function
def f(x, a, b):
    return (x >= a).astype(int)*(b*((x-a)**2))
	
def readGraphFeatures(folder):

    # Get all the filenames in the directory
    path = f"features/{folder}/"
    files = [f for f in listdir(path) if isfile(join(path, f)) and "_graph" in f]
    
    # Creat empty list
    li = []
    
    # Iterate over files
    for f in files:
        file = f"{path}/{f}"
        data = pd.read_csv(file, delimiter=',', encoding='utf-8', index_col = 0)
        li.append(data)
        
    # Combine into dataframe and return
    data_graph = pd.concat(li, axis=0, ignore_index=False)
    data_graph.index = data_graph.index.str.replace("_Microstructure.graphml", "")
    data_graph.index = data_graph.index.str.replace(folder, '')
    data_graph.index = data_graph.index.str.replace("\\",'')
    data_graph = data_graph.sort_index()
    
    return data_graph
	
def readStretchFeatures(folder):

    # Get all the filenames in the directory
    path = f"features/{folder}/"
    files = [f for f in listdir(path) if isfile(join(path, f)) and "_stretch" in f]
    
    # Creat empty list
    li = []
    
    # Iterate over files
    for f in files:
        file = f"{path}/{f}"
        data = pd.read_csv(file, delimiter=',', encoding='utf-8', index_col = 0)
        li.append(data)
        
    # Combine into dataframe and return
    data_stretch = pd.concat(li, axis=0, ignore_index=False)
    data_stretch.index = data_stretch.index.str.replace("_Microstructure.graphml", "")
    data_stretch.index = data_graph.index.str.replace(folder, '')
    data_stretch.index = data_graph.index.str.replace("\\",'')
    data_stretch = data_stretch.sort_index()
    
    return data_stretch
	
def readPolyfitTargets(path):

    # Get all the filenames in the directory
    #path = f"labels/"
    files = [f for f in listdir(path) if isfile(join(path, f)) and "_polyfit" in f]
    
    # Creat empty list
    li = []
    
    # Iterate over files
    for f in files:
        file = f"{path}/{f}"
        data = pd.read_csv(file, delimiter=',', encoding='utf-8', index_col = 0)
        data = data.T
        data.index = [f]
        li.append(data)
        
    # Combine into dataframe and return
    #data_polyfit = pd.concat(li, axis=0, ignore_index=False)
    data_polyfit = pd.concat(li, ignore_index=False)
    data_polyfit.index = data_polyfit.index.str.replace("_StressStrainCurve.csv_polyfit.csv", "")
    #data_polyfit = data_polyfit.sort_index()
    
    return data_polyfit
	
def combineInputData(data_graph, data_stretch, data_polyfit, deduplicate = True):
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
    df_predictions = []
    
    final_linreg_alpha = LinearRegression().fit(data_joined[feature_comb], data_joined['alpha'])
    final_linreg_beta = LinearRegression().fit(data_joined[feature_comb], data_joined['beta'])
            
    return final_linreg_alpha, final_linreg_beta
	
if __name__ == "__main__":

	folder = sys.argv[1]
	
	data_graph = readGraphFeatures(folder)
	data_stretch = readStretchFeatures(folder)
	data_polyfit = readPolyfitTargets(f"polyfit/{folder}/")
	data_joined = combineInputData(data_graph, data_stretch, data_polyfit, deduplicate = True)
	features = 'graph+stretch'
	feature_comb = getFeatureCombinations()[features]
	
	#print(f"Länge von data_polyfit: {len(data_polyfit)}")
	#print(f"data_polyfit: {data_polyfit.columns}")
	#print(data_polyfit)
	
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
	