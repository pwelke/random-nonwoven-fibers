# System
from genericpath import exists
from os import listdir, mkdir, remove
from os.path import isfile, join, splitext
import sys, getopt
from pathlib import Path
import tarfile

from graph_features_single import run_analysis
from stretch import run

def process_file(filename, file):

	if file is None:
		file = filename

	# Calculate graph features
	## Check if already exists
	my_file = Path(f"features/{filename}_graph.csv")
	if my_file.is_file():
		print(f"Graph features already computed for {filename}")
	else:
		Path("features/").mkdir(parents=True, exist_ok=True)
		print(f"Graph feature calculation for {filename}")
		run_analysis(filename, file)

	# Calculate stretch features

	# Check if already exists
	my_file = Path(f"features/{filename}_stretch.csv")
	if my_file.is_file():
		print(f'Stretch features already computed for {filename}')
	else:
		Path("features/").mkdir(parents=True, exist_ok=True)
		print(f'Stretch feature calculation for {filename}')
		run(filename, file)

if __name__ == "__main__":
		
	folder = sys.argv[1]

	if folder == '-f':
		zipping = True
		folder = sys.argv[2]
	else:
		zipping = False
	
	# create output folder if necessary
	if not exists(join('features', folder)):
		mkdir(join('features', folder))

	if not zipping:
		# Get all files from that folder
		files = [f for f in listdir(folder) if isfile(join(folder, f)) and ".graphml" in f]
		print(f"Folder = {folder} with {len(files)} Files")

		if not exists(join('features', folder)):
			mkdir(join('features', folder))
				
		for filename in files:
			process_file(filename=join(folder, filename), file=None)
				
		print("Done!")

	else:
		with tarfile.open(folder, 'r:*') as tarredfiles:
			for tarinfo in tarredfiles:
				if tarinfo.isreg():
					if splitext(tarinfo.name)[1] == '.graphml':
						# have a folder called like the tarfile and a file named after the current file in the tarfile
						# content of tarfile is streamed
						tarredfiles.extract(tarinfo)
						process_file(filename=join(folder, tarinfo.name), file=tarinfo.name)
						remove(tarinfo.name)
						
