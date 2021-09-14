# Data related libraries
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

# Network
import networkx as nx
from networkx.algorithms import bipartite

# Plotting
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
from IPython.display import display, HTML

# Convenient helpers
import csv
import math
import re
from copy import copy
from time import time
from datetime import date
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import r2_score

from os import listdir
from os.path import isfile, join
from pathlib import Path
import sys, getopt

# Printing libraries and settings
# import warnings; warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.float_format','{0:.2f}'.format)

def f(x, a, b):
    #print(f"Called with xdata:{x}, alpha:{a}, beta:{b}")
    return (x >= a).astype(int)*(b*((x-a)**2))
	
def run_fitting(file):
	data = pd.read_csv(file, delimiter=',', encoding='utf-8')
	
	popt, pcov = curve_fit(f, 
						xdata = data.Strain.values, 
						ydata = data.Stress.values, 
						maxfev = 500,
						bounds = ([0, -20], [1, 500]))
	results = [popt[0], popt[1]]
	
	results_df = pd.DataFrame(results)
	results_df.index = ['alpha', 'beta']
	filename_start = file.rfind('\\')
	results_df.index.name = f"{file[filename_start + 1:]}"
	Path("polyfit/").mkdir(parents=True, exist_ok=True)
	results_df.to_csv(f"polyfit/{file}_polyfit.csv")
		   
		   
if __name__ == "__main__":

	folder = sys.argv[1]
	
	# Get all files from that folder
	files = [f for f in listdir(folder) if isfile(join(folder, f)) and "StressStrainCurve.csv" in f]
	
	print(f"Folder = {folder} with {len(files)} Files")

	for file in files:
		run_fitting(join(folder, file))
	print("Done!")