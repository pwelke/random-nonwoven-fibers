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
	
def getAdjR2(r2, n, p):
    return 1-(1-r2)*(n-1)/(n-p-1)
	
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
	
def readGraphOnlyData(folder):

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
    data_graphonly = pd.concat(li, axis=0, ignore_index=False)
    data_graphonly.index = data_graphonly.index.str.replace("_Microstructure.graphml", "")
    data_graphonly.index = data_graphonly.index.str.replace(folder, '')
    data_graphonly.index = data_graphonly.index.str.replace("\\",'')
    data_graphonly = data_graphonly.sort_index()
    
    return data_graphonly
	
def getFeatureCombinations():
    param_features = ["kappa", "sigRamp", "sigSde", "sld"]
    graph_features = list(data_graph.columns[4:])
    stretch_features = list(data_stretch.columns)
    
    feature_combinations = {#'param':param_features, 
                            #'stretch':stretch_features, 
                            #'graph':graph_features, 
                            #'param+graph':param_features + graph_features, 
                            #'param+stretch':param_features + stretch_features, 
                            'graph+stretch':graph_features + stretch_features, 
                            #'param+graph+stretch':param_features + graph_features + stretch_features,
                            }
    
    return feature_combinations
	
def getParamCombinations(data_joined):

    # Identify parameter combinations
    para_combs = []
    for i in data_joined.index:
        kappa = data_joined.loc[i, 'kappa']
        sigRamp = data_joined.loc[i, 'sigRamp']
        sigSde = data_joined.loc[i, 'sigSde']
        sld = data_joined.loc[i, 'sld']

        if not {'kappa': kappa, 'sigRamp': sigRamp, 'sigSde': sigSde, 'sld': sld} in para_combs:
            para_combs.append({'kappa': kappa, 'sigRamp': sigRamp, 'sigSde': sigSde, 'sld': sld}) 

    #print(f"{len(para_combs)} verschiedene Parameter-Kombinationen im Datensatz")
    
    return para_combs
	
def getParamCombData(data, fix_param):
    fix_param_data = data[(data['kappa'] == fix_param['kappa']) 
                        & (data['sigRamp'] == fix_param['sigRamp'])
                        & (data['sigSde'] == fix_param['sigSde'])
                        & (data['sld'] == fix_param['sld'])]
    return fix_param_data
	
def getMultipleParamCombData(data, fix_params):
    
    #Create empty dataframe
    fix_multiple_param_data = pd.DataFrame()
    
    for fix_param in fix_params:
        fix_param_data = data_joined[(data_joined['kappa'] == fix_param['kappa']) 
                                    & (data_joined['sigRamp'] == fix_param['sigRamp'])
                                    & (data_joined['sigSde'] == fix_param['sigSde'])
                                    & (data_joined['sld'] == fix_param['sld'])]
        #print(len(test_data_fix_param))
        fix_multiple_param_data = fix_multiple_param_data.append(fix_param_data)
        
    return fix_multiple_param_data
	
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
	
def train_validate(data_joined, df_graphonly):
	"""
	This function takes the labelled and unlabelled data and performs a k-fold cross-validation of linear regression training and testing, where k is the number of parameter combinations within the labelled data
	"""
	# Parameters
	verbose = False
	scale = True
	lasso = False
	
	# Data structures
	trained_models = []
	predictions = []
	
	# Parameter and feature combinations
	para_combs = getParamCombinations(data_joined)
	feature_combs = getFeatureCombinations()
	
	print("LOG: Starting training")
	print(f"LOG: Para Combs: {len(para_combs)}")
	print(f"LOG: Feature Combs: {len(feature_combs)}")

	# Check every feature combination seperately
	for feature_comb in feature_combs:
			
		if verbose:
			print("___")
			print(f"LOG: Feature combination: {feature_comb}")

		for fix_param in para_combs:
			number_of_curves = len(getParamCombData(data_joined, fix_param))
			number_of_graphonly = len(getParamCombData(df_graphonly, fix_param))

			if verbose:
				print(f"#LOG: Graphen+Kurven: {number_of_curves}, Graphen: {number_of_graphonly}")

			# Parameter Combinations where every graph has a curve
			if number_of_graphonly == 0:

				# Remove the selected test param comb
				train_para_combs = [x for x in para_combs if x != fix_param]

				# Retrieve the train/test data
				train_data = getMultipleParamCombData(data_joined, train_para_combs)
				test_data = getParamCombData(data_joined, fix_param)
				#if verbose:
					#print(f"LOG: Train: {len(train_data)} in {len(train_para_combs)} combs, Test: {len(test_data)}")

				# Split feature data from labels
				X_train = train_data[feature_combs[feature_comb]]
				y_train_alpha = train_data['alpha']
				y_train_beta = train_data['beta']

				X_test = test_data[feature_combs[feature_comb]]
				y_test_alpha = test_data['alpha']
				y_test_beta = test_data['beta']
				
				# Scale data
				if scale == True:
					#scaler = StandardScaler()
					scaler = MinMaxScaler(feature_range = (-1,1))
					# Fit on train data, transform both train and test data
					#print(f"LOG: Scale data on columns: {X_train.columns}")
					scaler.fit(X_train)
					X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
					X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
				else: 
					X_train_scaled = X_train
					X_test_scaled = X_test

				# Train nodel
				
				#reg_alpha = LinearRegression().fit(X_train_scaled, y_train_alpha)
				#reg_beta = LinearRegression().fit(X_train_scaled, y_train_beta)
				
				if lasso:
					reg_alpha = Lasso(alpha = 0.005).fit(X_train_scaled, y_train_alpha)
					reg_beta = Lasso(alpha = 0.005).fit(X_train_scaled, y_train_beta)
				else:
					reg_alpha = LinearRegression().fit(X_train_scaled, y_train_alpha)
					reg_beta = LinearRegression().fit(X_train_scaled, y_train_beta)

				trained_models.append({'feature_comb':feature_comb, 
									   'fix_param':fix_param, 
									   'label':'alpha', 
									   'fitted_linreg_model':reg_alpha})
				trained_models.append({'feature_comb':feature_comb, 
									   'fix_param':fix_param, 
									   'label':'beta', 
									   'fitted_linreg_model':reg_beta})

				# Make predictions
				df_predictions = pd.DataFrame()
				df_predictions["predicted_alpha_linreg"] = reg_alpha.predict(X_test_scaled)
				df_predictions["predicted_beta_linreg"] = reg_beta.predict(X_test_scaled)
				df_predictions.index = test_data.index

				predictions.append({'feature_comb':feature_comb, 
									'fix_param':fix_param, 
									'predictions':df_predictions})


			# Parameter Combinations where there are some graphs without a curve
			else:
				# Remove the selected test param comb
				train_para_combs = [x for x in para_combs if x != fix_param]
				
				# Retrieve the train/test data
				train_data = getMultipleParamCombData(data_joined, train_para_combs)
				test_data = getParamCombData(df_graphonly, fix_param)
				if verbose:
					print(f"LOG: Train: {len(train_data)} in {len(train_para_combs)} combs, Test: {len(test_data)}")
				
				# Split feature data from labels
				X_train = train_data[feature_combs[feature_comb]]
				y_train_alpha = train_data['alpha']
				y_train_beta = train_data['beta']

				X_test = test_data[feature_combs[feature_comb]]
				
				# Scale data
				if scale == True:
					#scaler = StandardScaler()
					scaler = MinMaxScaler(feature_range = (-1,1))
					# Fit on train data, transform both train and test data
					#print(f"LOG: Scale data on columns: {X_train.columns}")
					scaler.fit(X_train)
					X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
					X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
				else: 
					X_train_scaled = X_train
					X_test_scaled = X_test
				
				# Train nodel
				#reg_alpha = LinearRegression().fit(X_train_scaled, y_train_alpha)
				#reg_beta = LinearRegression().fit(X_train_scaled, y_train_beta)
				
				if lasso:
					reg_alpha = Lasso(alpha = 0.005).fit(X_train_scaled, y_train_alpha)
					reg_beta = Lasso(alpha = 0.005).fit(X_train_scaled, y_train_beta)
				else:
					reg_alpha = LinearRegression().fit(X_train_scaled, y_train_alpha)
					reg_beta = LinearRegression().fit(X_train_scaled, y_train_beta)

				trained_models.append({'feature_comb':feature_comb, 
									   'fix_param':fix_param, 
									   'label':'alpha', 
									   'fitted_linreg_model':reg_alpha})
				trained_models.append({'feature_comb':feature_comb, 
									   'fix_param':fix_param, 
									   'label':'beta', 
									   'fitted_linreg_model':reg_beta})

				# Make predictions
				df_predictions = pd.DataFrame()
				df_predictions["predicted_alpha_linreg"] = reg_alpha.predict(X_test_scaled)
				df_predictions["predicted_beta_linreg"] = reg_beta.predict(X_test_scaled)
				df_predictions.index = test_data.index
				
				predictions.append({'feature_comb':feature_comb, 
									'fix_param':fix_param, 
									'predictions':df_predictions})
									
	print(f"LOG: Number of trained models: {len(trained_models)}")
	
if __name__ == "__main__":

	folder = sys.argv[1]
	folder_graphonly = sys.argv[2]
	
	# Read data in
	data_graph = readGraphFeatures(folder)
	data_stretch = readStretchFeatures(folder)
	data_polyfit = readPolyfitTargets(f"polyfit/{folder}/")
	data_joined = combineInputData(data_graph, data_stretch, data_polyfit, deduplicate = True)
	data_graphonly = readGraphOnlyData(folder_graphonly)
	
	# Features
	features = 'graph+stretch'
	feature_comb = getFeatureCombinations()[features]
	
	# Parameters
	para_combs = getParamCombinations(data_joined)
	
	# Logging
	print(f"#Labelled: {len(data_graph)}, #unlabelled: {len(data_graphonly)}")
	
	#print(f"Länge von data_polyfit: {len(data_polyfit)}")
	#print(f"data_polyfit: {data_polyfit.columns}")
	#print(data_polyfit)
	
	train_validate(data_joined, data_graphonly)
	
	#final_linreg_alpha, final_linreg_beta = trainFinalModel(data_joined, features = features)
	
	# Serialization
	#filename = 'trained_models/pickle_final_linreg_alpha'
	#outfile = open(filename,'wb')
	#pickle.dump(final_linreg_alpha,outfile)
	#outfile.close()
	#
	#filename = 'trained_models/pickle_final_linreg_beta'
	#outfile = open(filename,'wb')
	#pickle.dump(final_linreg_beta,outfile)
	#outfile.close()
	
	print("Done training models!")
	