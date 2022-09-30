## Requirements for training supplied models

### System requirements

Software requirements:
All models were developed on a Linux CentOS 8 operating system.
These models have not been trained on MacOS or Windows, but as they are developed using Python, no compatibility issues are expected.

### Installation guide

These models require the Python programming language. 
These models were developed using various versions of Python as they were developed at various times. 
Each model has been tested using Python v3.8.1 which can be installed at https://www.python.org/downloads/release/python-381/

Additional libraries are required to run these models. The versions used are as follows:

For the RF and NN models:
scikit-learn 0.21.3\
numpy v1.18.1\
pandas v1.0.3\
matplotlib v3.1.3\
tensorflow v1.13.1

For GraphConv, DAG and Weave models:
scikit-learn v0.24.1\
numpy v1.18.5\
pandas v1.3.1\
matplotlib v3.3.4\
deepchem v2.4.0

These libraries can be installed using pip or, if available, anaconda. Examples of installations are:
'pip install matplotlib' and 'conda install pandas'

### Running the scripts

To run these models, only the training dataset (e.g. "D300" in Training_sets.csv) and the selected model script (e.g. "NN_model.py") are necessary. 
These may be in a single directory or paths may be defined if they are saved in separate directories.
The models are each developed such that they import required libraries, read in specified datasets, train the model over a number of resamples, perform hyperparameter optimisation, and produce both prediction outputs and performance statistics.

All alterable variables are defined and well commented within the scripts. Some variables, such as number of resamples, has been kept at 50, but this is easily changed. Some variables, such as "DATASET" must be changed to match input files.

NN_model.py makes use of a manual grid search for hyperparameter optimisation, developed in the script grid_build.py. The hyperparameters to be optimised and value ranges over which to optimise are defined in grid_build.py and an additional .json file called as config.json. The .json is as follows:
```
{
    "layer1": a,
    "layer2": b,
    "layer3": c,
    "learning_rate": d,
    "batch_size": e,
    "loss": "f"
}
```

The file grid_build.py defines ranges over which each hyperparameter is optimised and a config.json file is created for each unique combination of hyperparameter values. This config.json file is then used in the building of models with those hyperparameter values, where 1 model is trained per unique combination. This results in a large number of models being trained where the highest performing model may be identified along with the optimal hyperparameters. This is due to the lack of in-built grid search in TensorFlow. Care should be taken when defining value ranges to optimise over. Total model count can be calculated by multiplying the value count of each hyperparameter. Also, grid_build.py was developed for use with a HPC cluster and amendments will be required to account for this.

To run these models, from a terminal, enter the following command:
`python chosen_model.py`, where "chosen_model" is the replaced with the model of interest.

Outputs vary between models: \
RF_model.py outputs 2 .csv files, one of which is for the tight set and, the other, the loose set. Each .csv contains the predicted solubility value per resample ran, with the average prediction
value and statistics for each molecule in the given testing set.
grid_build.py, which calls upon NN_model.py, outputs 1 directory for each hyperparameter combination. Each directory contains .csv files which correspond to the tight set predictions, loose set
predictions, and each of the validation set predictions, per resample ran. Error plots for the validation predictions are also output.
