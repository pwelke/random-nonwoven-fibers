# System
from genericpath import exists
from os import listdir, mkdir, remove
from os.path import isfile, join, splitext
import sys, getopt
from pathlib import Path
import tarfile

from graph_features_single import compute_standard_graph_features, fileImportToNetworkX
from stretch import compute_stretch_graph_features

print("LOG: Starting feature_generation.py")

def compute_all_graph_features(filename, file):

	if file is None:
		file = filename

	# Calculate graph features
	## Check if already exists
	my_file = Path(f"../results/features/{filename}_graph.csv")
	## CodeOcean
	#my_file = Path(f"results/features/{filename}_graph.csv")

	if my_file.is_file():
		print(f"Graph features already computed for {filename}")
	else:
		Path("../results/features/").mkdir(parents=True, exist_ok=True)
		## CodeOcean
		#Path("results/features/").mkdir(parents=True, exist_ok=True)

		print(f"Graph feature calculation for {filename}")
		compute_standard_graph_features(filename, file)
		## CodeOcean
		#compute_standard_graph_features(filename, join("../results/", file))

	# Calculate stretch features

	# Check if already exists
	my_file = Path(f"../results/features/{filename}_stretch.csv")
	## CodeOcean
	#my_file = Path(f"results/features/{filename}_stretch.csv")

	if my_file.is_file():
		print(f'Stretch features already computed for {filename}')
	else:
		Path("../results/features/").mkdir(parents=True, exist_ok=True)
		## CodeOcean
		#Path("results/features/").mkdir(parents=True, exist_ok=True)

		print(f'Stretch feature calculation for {filename}')
		compute_stretch_graph_features(filename, file)


if __name__ == "__main__":

	print("LOG: Starting main in feature_generation.py")

	folder = sys.argv[1]

	if folder == '-f':
		zipping = True
		folder = sys.argv[2]
	else:
		zipping = False
		
	# new because of structure change
	folder_source = folder.replace("../data/", "")
	folder_source = folder.replace("../results/", "")	

	print(f"Folder source: {folder_source}")
	
	# create output folder if necessary
	if not exists(join('features', folder_source)):
		mkdir(join('features', folder_source))

	## New because of CodeOceans structure
	#if not exists("../results/features"):
	#	mkdir("../results/features")

	if not zipping:
		# Get all files from that folder
		files = [f for f in listdir(folder) if isfile(join(folder, f)) and ".graphml" in f]
		print(f"Folder = {folder} with {len(files)} Files")

		if not exists(join('../results/', 'features', folder_source)):
			mkdir(join('../results/features', folder_source))
		
		for filename in files:
			compute_all_graph_features(filename=join(folder, filename), file=None)
				
		print("Done!")

	else:
		with tarfile.open(folder, 'r:*') as tarredfiles:
			for tarinfo in tarredfiles:
				if tarinfo.isreg():
					if splitext(tarinfo.name)[1] == '.graphml':
						# have a folder called like the tarfile and a file named after the current file in the tarfile
						# content of tarfile is streamed
						tarredfiles.extract(tarinfo)
						compute_all_graph_features(filename=join(folder_source, tarinfo.name), file=tarinfo.name)
						remove(tarinfo.name)
						
	print("LOG: Finishing main in feature_generation.py")
