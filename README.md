# Projecting Data onto Clinical Constraints for Improved Learning

This project translates clinical constraints into high dimensional mathematical constraints and uses projections to correct erroneous data as well as engineer new "distance-to-normal" features that help improve sepsis predictions.

## Setting up the Environment

1. Create your virtual environment using the requirements.txt file
2. Get the Gurobi license (Academic license is free) and follow the steps to activate it : [Gurobi License](https://www.gurobi.com/academia/academic-program-and-licenses/)
3. Install gurobipy

## Running the Code

### Step 1: Imputing Missing Data & generating subpatients

1. Patient data would need to be in the form of .psv files similar to the [Physionet Dataset](https://physionet.org/content/challenge-2019/1.0.0/)
2. Change the data paths in *get_imputations.py* to point to this folder
3. Run:

  $python get_imputations.py

### Step 2: Projections onto Physical and Normal Clinical Constraints

1. Requires the file: constraints_wo_calcium.txt
2. Change the data paths to point to your imputed data in step 1 in *get_projections.py*
3. Set the right subpatient lengths in *get_projections.py*
4. Run:

  $python get_projections.py
  
### Step 3: Training Sepsis Prediction Models (with bootstrapping and cross-validation)

1. Change the config file to point to your data and (optional) clustering objects, specify other parameters
2. Run the python script with the path to the config file as cl argument

  $python train_with_parser.py --<path_to_config>
  
### TO DO:

1. Create a single inference script that runs on any form of new data 
2. Add branch for clustering analysis

