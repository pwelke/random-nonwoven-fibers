## Imports
# Data related libraries
import numpy as np
import pandas as pd

# System
from os import listdir, remove, mkdir, makedirs
import os.path
from os.path import isfile, join, splitext
import sys, getopt
from pathlib import Path
from datetime import date
import tarfile
from genericpath import exists

# Persistance
import pickle

# Plotting
import matplotlib.pyplot as plt

# Validation
import ot
import ot.plot
from bisect import bisect_left

# Modeling
from sklearn.preprocessing import minmax_scale, StandardScaler, MinMaxScaler
from sklearn import preprocessing, linear_model
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def reduceFilePath(filepath):
    return os.path.split(filepath)[1]

def readGraphOnlyData(folder):
    """
    Reads in the graph features from unlabelled data points (without stress-strain curve)
    """

    # Get all the filenames in the directory
    path = join("features", folder)
    files = [f for f in listdir(path) if isfile(join(path, f)) and "_graph" in f]
    
    li = []
    
    # Iterate over files
    for f in files:
        file = join(path, f)
        data = pd.read_csv(file, delimiter=',', encoding='utf-8', index_col = 0)
        li.append(data)
        
    # Combine into dataframe and return
    data_graphonly = pd.concat(li, axis=0, ignore_index=False)
    data_graphonly.index = data_graphonly.index.str.replace("_Microstructure.graphml", "")
	
    data_graphonly.index = data_graphonly.index.map(reduceFilePath)
	
    data_graphonly = data_graphonly.sort_index()
    
    return data_graphonly
	
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
    
    return data_stretch

if __name__ == "__main__":

    # Get full command-line arguments
	full_cmd_arguments = sys.argv
	
	# Load trained models
	file_alpha = sys.argv[1]
	file_beta = sys.argv[2]
	
	## Alpha
	print(f"LOG: Loading file: {file_alpha}")
	with open(file_alpha, 'rb') as f:
		final_linreg_alpha2 = pickle.load(f)
	#print(final_linreg_alpha2.coef_)
	
	## Beta
	print(f"LOG: Loading file: {file_beta}")
	with open(file_beta, 'rb') as f:
		final_linreg_beta2 = pickle.load(f)
	#print(final_linreg_beta2.coef_)
	
	print(f"LOG: Loaded models successfully")
	
	## Read in Graph_only data
	folder_graphonly_source = sys.argv[3]
	data_graphonly = readGraphOnlyData(folder_graphonly_source)
	
	## Read in stretch data
	data_stretch = readStretchFeatures(folder_graphonly_source)
	
	data_joined = data_graphonly.join(data_stretch)
	#print(data_joined)
	print(f"LOG: Loaded data successfully")
	
	pred_alpha = final_linreg_alpha2.predict(data_joined[data_joined.columns[4:]])
	pred_beta = final_linreg_beta2.predict(data_joined[data_joined.columns[4:]])
	
	pred = pd.DataFrame([data_joined.index, pred_alpha, pred_beta]).T
	pred.columns = ["file", "alpha", "beta"]
	pred = pred.set_index('file')
	print(f"LOG: Calculated predictions")
	
	
	#if not exists(join('../results/predictions/', folder_graphonly_source)):
	#	mkdir(join('../results/predictions/', folder_graphonly_source))
		
	if not exists(f'../results/predictions/{folder_graphonly_source}'):
		makedirs(f'../results/predictions/{folder_graphonly_source}')
	
	for i in pred.index:
		pred.loc[i].to_csv(f"../results/predictions/{folder_graphonly_source}/{i}.csv")
	
	print(f"LOG: Exported predictions")