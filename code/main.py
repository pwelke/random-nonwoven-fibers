import os

# python download_data.py
# python feature_generation.py -f labeled.tar.gz
# python ansatzfitting.py -f labeled.tar.gz
# python train_validate.py labeled.tar.gz unlabeled.tar.gz -f -p

print("LOG: Starting main.py script")

# os.system("python download_data.py")

os.system("python feature_generation.py -f ../data/toy_data_labeled.tar.gz")
os.system("python feature_generation.py -f ../data/toy_data_unlabeled.tar.gz")
print("Finished feature_generation")
os.system("python ansatzfitting.py -f ../data/toy_data_labeled.tar.gz")
print("Finished ansatzfitting")
os.system("python train_validate.py ../data/toy_data_labeled.tar.gz ../data/toy_data_unlabeled.tar.gz -f")
print("Finished train_validate")

os.system("python predict.py trained_models\pickle_final_linreg_alpha trained_models\pickle_final_linreg_beta input_data_graphonly")
print("Finished predict")