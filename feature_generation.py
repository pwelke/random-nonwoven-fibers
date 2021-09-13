# System
from os import listdir
from os.path import isfile, join
import sys, getopt
from pathlib import Path

from graph_features_single import run_analysis
from stretch import run

if __name__ == "__main__":
	
	folder = sys.argv[1]
	
	# Get all files from that folder
	files = [f for f in listdir(folder) if isfile(join(folder, f)) and ".graphml" in f]
	
	print(f"Folder = {folder} with {len(files)} Files")
	
	#print(files)
	
	for file in files:
	
		# Calculate graph features
		## Check if already exists
		my_file = Path(f"features/{join(folder, file)}_graph.csv")
		if my_file.is_file():
			print("Graph features already calculated")
		else:
			Path("features/").mkdir(parents=True, exist_ok=True)
			print("Starting graph feature calculation")
			run_analysis(join(folder, file))
		
		# Calculate stretch features
		
		# Check if already exists
		my_file = Path(f"features/{join(folder, file)}_stretch.csv")
		if my_file.is_file():
			print("Stretch features already calculated")
		else:
			Path("features/").mkdir(parents=True, exist_ok=True)
			print("Starting stretch feature calculation")
			run(join(folder, file))
			
	print("Done!")