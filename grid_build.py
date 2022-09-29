#This script creates a manual grid search for tensorflow keras iwht use of .json files for hyperparameter optimistaion. Each hyperaparameter combination is run as a separate job so
#care must be taken with the number of jobs thats will be created


import pathlib
import os
import subprocess
from itertools import product
import json
import pandas as pd

#define feature set to be used, options are "rdkit", "mordred", "MOE"
FEATSET = "FEATSET"

#defining hyperparameters to optimise using .json file
def config_writer(layer1, layer2, layer3, learning_rate, batch_size, loss):
    config = {
    "layer1": layer1,
    "layer2": layer2,
    "layer3": layer3,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "loss": loss    
    }
    with open("config.json", "w+") as f:
        json.dump(config, f, indent = 4)

#creating submission scripts for HPC
#NN_model.py is called in the submission script
def submit_writer(n):
    job_name = f"grid_{n}"
    submit = f"""
    INSERT HPC SUBMISSION SCRIPT HERE
"""
    with open("submit", "w+") as f:
        print(submit, file = f)

ROOT_DIR = pathlib.Path(__file__).parent.resolve()

SCRIPT = pathlib.Path("/PATH/TO/NN_model.py")

#defining ranges over which hyperparameters will be optimised
#empty lists can be populated with custom ranges
#Note: total number of jobs is equal to the number of selected values for each hyperparameter multiplied together
L1 = []
L2 = []
L3 = []
LR = []
BS = []
LOSS = []

#creating a list of all hyperparameter combinations
combos = list(product(L1, L2, L3, LR, BS, LOSS))

#creating list for tracking jobs
jobname_encode = []

#creating a separate job for each hyperparameter combination and submitting it to HPC
#each job has a different name (line 67)
for n, combo in enumerate(combos):
    NEW_DIR = ROOT_DIR/FEATSET/("_".join(map(str, combo)))
    print(f"Creating model {n} in {NEW_DIR.name}")
    NEW_DIR.mkdir(exist_ok = True, parents = True)
    os.chdir(NEW_DIR)
    (NEW_DIR/"Results/Plots").mkdir(exist_ok = True, parents = True)
    config_writer(*combo)
    submit_writer(n)
    jobname_encode.append({"Run": NEW_DIR.name, "Job_Name": f"grid_{n}"})
    subprocess.run(["sbatch", "submit"])
    
#creates a csv which shows which hyperparameter combination was used per job name
os.chdir(ROOT_DIR/FEATSET)
df = pd.DataFrame(jobname_encode)
df.to_csv("job_dir_match.csv", index = False)
