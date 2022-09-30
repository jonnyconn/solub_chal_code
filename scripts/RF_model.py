import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn import metrics

from sklearn import datasets

import time 

#define root directory to shorten paths for inputs and outputs
ROOT_DIR = "PATH/TO/ROOT_DIRECTORY"

#define feature set to be used, options are "rdkit", "mordred", "MOE"
#define dataset to be used, options are "D300", "D2999", "D5697"
FEATSET = "RDKit"
DATASET = "D300"

#randomise initial seed with a numpy random number generator for subsequent seeds
seed = 8434
np.random.seed(seed)

#set number of resamples to 50 and parallelisation to run on 40 cpu cores
nresamples = 50
njobs = 40

#separately read in training set, tight testing set, and loose testing set
df_int = pd.read_csv(f"{ROOT_DIR}/datasets/Training_sets.csv",
                     header=[0, 1], index_col=0)
df_tight = pd.read_csv(f"{ROOT_DIR}/datasets/Tight_set.csv",
                       header=[0, 1], index_col=0)
df_loose = pd.read_csv(f"{ROOT_DIR}/datasets/Loose_set.csv",
                       header=[0, 1], index_col=0)

#define a set of columns as metadata that will not be used for training
meta_cols = ["Name", "SMILES", "Solubility"]

#select training dataset
df_int = df_int.loc[df_int[('Dataset', DATASET)]]
df_int.drop(['Dataset', 'Reference'], axis=1, level=0, inplace=True)

#define features as full dataset after dropping metadata columns
#"Solubility" column is dropped here to ensure it is not used as a training feature
features_int = df_int.drop(meta_cols, axis=1, level=0)[FEATSET].values
features_tight = df_tight.drop(meta_cols, axis=1, level=0)[FEATSET].values
features_loose = df_loose.drop(meta_cols, axis=1, level=0)[FEATSET].values

#define column "Solubility" as targets
targets_int = df_int["Solubility"].values
targets_tight = df_tight["Solubility"].values
targets_loose = df_loose["Solubility"].values

#retain metadata for later use in output
meta_int = df_int[meta_cols].values
meta_tight = df_tight[meta_cols].values
meta_loose = df_loose[meta_cols].values

#define empty lists for r2 and rmse scores, to be appended per resample
r2_scores_int = []
rmse_scores_int = []

#define empty lists for predicted values for both tight and loose sets, to be appended per resample
preds_list1 = []
preds_list2 = []

#assigning labels to training and testing data
xte_t = features_tight
xte_l = features_loose

yte_t = targets_tight
yte_l = targets_loose

#begin timer for job runtime
start = time.time() 

#resample loop
for n in range(nresamples):
    print(f'Performing resample {n + 1} of {nresamples}...')
    #defining train/test split of 70/30
    xtr, xte, ytr, yte = train_test_split(features_int, targets_int, test_size=0.3, random_state=3*n)

    #defining model using seed and njobs defined earlier, can use ExtraTreesRegressor or RandomForestRegressor
    mymodel = ExtraTreesRegressor(random_state=seed, n_jobs=njobs)

    #hyperparameter optimisation, each (x, y, z) is treated independently and defines range of values to optimise over, from x to y in increments of z
    #5 fold cross validation is used for optimisation with an r2 scoring function
    param_grid =  {"n_estimators": np.arange(x, y, z), "min_samples_leaf": np.arange(x, y, z)}
    optimiser = GridSearchCV(mymodel, param_grid, verbose=2, cv=5, n_jobs=njobs, scoring="r2")
    optimiser.fit(xtr, ytr)

    #using the optimal parameter combination for model
    best_params = optimiser.best_params_
    print(f'Best parameters were: {best_params}') 

    #fitting training data to model
    mymodel.fit(xtr, ytr)

    #applying model for predictions
    preds = mymodel.predict(xte)
    preds1 = mymodel.predict(xte_t)
    preds2 = mymodel.predict(xte_l)

    #calculating r2 and rmse of testing data and predicted values
    r2_int = metrics.r2_score(yte, preds)
    rmse_int = np.sqrt(metrics.mean_squared_error(yte, preds))

    #appending calculated values to lists defined previously
    r2_scores_int.append(r2_int)
    rmse_scores_int.append(rmse_int)

    #appending predcition values to lists defined previously
    preds_list1.append(preds1)
    preds_list2.append(preds2)

#creating pandas dataframe from metadata
df_tight_meta = pd.DataFrame(meta_tight, columns=meta_cols)
df_loose_meta = pd.DataFrame(meta_loose, columns=meta_cols)

#creating dataframe for tight set prediction values
df_tight_preds = pd.DataFrame(preds_list1).transpose()
pred_names = [f"Prediction {n+1}" for n in range(nresamples)]
df_tight_preds.columns = pred_names

#creating dataframe for loose set prediction values
df_loose_preds = pd.DataFrame(preds_list2).transpose()
pred_names = [f"Prediction {n+1}" for n in range(nresamples)]
df_loose_preds.columns = pred_names

#creating additional columns in tight predictions dataframe for various stats
df_tight_preds["Mean"] = df_tight_preds[pred_names].mean(axis=1).values
df_tight_preds["Median"] = df_tight_preds[pred_names].median(axis=1).values
df_tight_preds["Std"] = df_tight_preds[pred_names].std(axis=1).values
df_tight_preds["Min"] = df_tight_preds[pred_names].min(axis=1).values
df_tight_preds["Max"] = df_tight_preds[pred_names].max(axis=1).values

#creating additional columns in loose predictions dataframe for various stats
df_loose_preds["Mean"] = df_loose_preds[pred_names].mean(axis=1).values
df_loose_preds["Median"] = df_loose_preds[pred_names].median(axis=1).values
df_loose_preds["Std"] = df_loose_preds[pred_names].std(axis=1).values
df_loose_preds["Min"] = df_loose_preds[pred_names].min(axis=1).values
df_loose_preds["Max"] = df_loose_preds[pred_names].max(axis=1).values

#concatenating metadata and predicted values (along with stats) for both tight and loose sets (separately) 
df_tight_results = pd.concat([df_tight_meta, df_tight_preds], axis=1)
df_loose_results = pd.concat([df_loose_meta, df_loose_preds], axis=1)

#writing outputs to file
df_tight_results.to_csv(f"{ROOT_DIR}Results/RF_tight_{DATASET}_{FEATSET}.csv", index=False)
df_loose_results.to_csv(f"{ROOT_DIR}Results/RF_loose_{DATASET}_{FEATSET}.csv", index=False)

#print statements for time taken and mean r2 and rmse values
print(f"Time taken to complete = {time.time() - start}")

print(f"Mean for int set r2 score: {np.mean(r2_scores_int)}, with SD {np.std(r2_scores_int)}")
print(f"Mean for int set rmse score: {np.mean(rmse_scores_int)}, with SD {np.std(rmse_scores_int)}")
