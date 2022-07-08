#!/usr/bin/env bash
set -ex


# Create directories for intermediate results
mkdir /results/input_data_graphonly
mkdir /results/input_data_labelled
mkdir /results/features
mkdir /results/polyfit
mkdir /results/trained_models

## Matlab Part
# Goal: Create x Graph-csv-pairs and y graphs and store them in results/input_data_graphonly and results/input_data_labelled

 if [ "$1" == "all" ] ; then
    # Create a dataset similar to the one used in 
    # Antweiler, Harmening, Marheineke, Schmeißer, Wegener, Welke (2022):
    # Graph-Based Tensile Strength Approximation of Random Nonwoven Materials by Interpretable Regression.
    # Machine Learning with Applications (8), Elsevier.
    # https://doi.org/10.1016/j.mlwa.2022.100288 
    #
    # Generate 25 graphs each for 
    #   - 5 random parameter combinations and simulate stress-strain curves for all 25 graphs
    #   - 50 random parameter combinations and simulate only one stress-strain curve 

    echo "Running fiber graph generator for all data sets."
    matlab -nodisplay -r "addpath(genpath('.')); runDataBaseGeneration(5,50,25)"

 else
    # Use the provided dataset (precomputed instances) for training and validation

    if [ "$1" == "single" ] ; then   
       # A 'short' example run that generates a single graph and simulates its stress-strain curve
       # (takes roughly 30h on current CodeOcean instances)
       #
       # Note that this setting is just to showcase that everything works in time that a free-tier CodeOcean user has. 
       # It should not be used to train any production model.

       echo "Running fiber graph generator for one data set."
       matlab -nodisplay -r "addpath(genpath('.')); runSampleAndSimulate(1,0)"
    else
       echo "No additional data is generated."
    fi

     # Copy files from data to results (2 graph-curve pairs and 1 single-graph)
    cp /data/input_data_graphonly/2021_03_10_Sld1062_SigRamp1p135_SigSde2p206_Kappa0p029839_N25_Microstructure.graphml /results/input_data_graphonly/2021_03_10_Sld1062_SigRamp1p135_SigSde2p206_Kappa0p029839_N25_Microstructure.graphml
    cp /data/input_data_labelled/2021_02_05_Sld1109_SigRamp1p350_SigSde4p116_Kappa0p029540_N1_Microstructure.graphml /results/input_data_labelled/2021_02_05_Sld1109_SigRamp1p350_SigSde4p116_Kappa0p029540_N1_Microstructure.graphml
    cp /data/input_data_labelled/2021_02_05_Sld1109_SigRamp1p350_SigSde4p116_Kappa0p029540_N1_StressStrainCurve.csv /results/input_data_labelled/2021_02_05_Sld1109_SigRamp1p350_SigSde4p116_Kappa0p029540_N1_StressStrainCurve.csv

    cp /data/input_data_labelled/2021_02_19_Sld1107_SigRamp3p689_SigSde3p910_Kappa0p029708_N1_Microstructure.graphml /results/input_data_labelled/2021_02_19_Sld1107_SigRamp3p689_SigSde3p910_Kappa0p029708_N1_Microstructure.graphml
    cp /data/input_data_labelled/2021_02_19_Sld1107_SigRamp3p689_SigSde3p910_Kappa0p029708_N1_StressStrainCurve.csv /results/input_data_labelled/2021_02_19_Sld1107_SigRamp3p689_SigSde3p910_Kappa0p029708_N1_StressStrainCurve.csv
 fi



## Python Part
# Goal: Train a surrogate model for the costly stress-strain curve simulation that took so long.
python3 -u feature_generation.py ../results/input_data_labelled 
python3 -u feature_generation.py ../results/input_data_graphonly 
python3 -u ansatzfitting.py ../results/input_data_labelled 
python3 -u train_validate.py ../results/input_data_labelled ../results/input_data_graphonly 
python3 -u train_final_model.py ../results/input_data_labelled
python3 -u predict.py trained_models/pickle_final_linreg_alpha trained_models/pickle_final_linreg_beta input_data_graphonly
