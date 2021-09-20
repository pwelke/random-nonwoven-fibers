# Data related libraries
import pandas as pd
from scipy.optimize import curve_fit

from os import listdir, mkdir, remove
from os.path import isfile, join, exists
from pathlib import Path
import sys
import tarfile

# Printing libraries and settings
# import warnings; warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.float_format','{0:.2f}'.format)

def f(x, a, b):
    #print(f"Called with xdata:{x}, alpha:{a}, beta:{b}")
    return (x >= a).astype(int)*(b*((x-a)**2))
	
def fit_ansatz_function(filename, fileobject):

	if fileobject is None:
		fileobject = filename

	data = pd.read_csv(fileobject, delimiter=',', encoding='utf-8')
	
	popt, pcov = curve_fit(f, 
						xdata = data.Strain.values, 
						ydata = data.Stress.values, 
						maxfev = 500,
						bounds = ([0, -20], [1, 500]))
	results = [popt[0], popt[1]]
	
	results_df = pd.DataFrame(results)
	results_df.index = ['alpha', 'beta']
	filename_start = filename.rfind('\\')
	results_df.index.name = f"{filename[filename_start + 1:]}"
	Path("polyfit/").mkdir(parents=True, exist_ok=True)
	results_df.to_csv(join('polyfit', f"{filename}_polyfit.csv"))
		   
		   
if __name__ == "__main__":

	folder = sys.argv[1]

	if folder == '-f':
		zipping = True
		folder = sys.argv[2]
	else:
		zipping = False

	# create output folder if necessary
	if not exists(join('polyfit', folder)):
		mkdir(join('polyfit', folder))

	if zipping:
		with tarfile.open(folder, 'r:*') as tarredfiles:
			for tarinfo in tarredfiles:
				if tarinfo.isreg() and "StressStrainCurve.csv" in tarinfo.name:
					tarredfiles.extract(tarinfo)
					fit_ansatz_function(join(folder, tarinfo.name), tarinfo.name)
					remove(tarinfo.name)
	else:
		# Get all files from that folder
		files = [f for f in listdir(folder) if isfile(
		    join(folder, f)) and "StressStrainCurve.csv" in f]

		print(f"Folder = {folder} with {len(files)} Files")

		for file in files:
			fit_ansatz_function(join(folder, file), join(folder, file))
		print("Done!")
