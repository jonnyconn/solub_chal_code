from __future__ import absolute_import, division, print_function

import json
import pathlib

import matplotlib as mpl

mpl.use("pdf")

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

#open config.json file for use in manual grid search
with open("config.json", "r") as f:
    config = json.load(f)

#define feature set to be used, options are "rdkit", "mordred", "MOE"
#define dataset to be used, options are "D300", "D2999", "D5697"
FEATSET = "FEATSET"
DATASET = "DATASET"

#define root directory to shorten paths for inputs and outputs
ROOT_DIR = "PATH/TO/ROOT_DIRECTORY"

#separately read in training set, tight testing set, and loose testing set
df_int = pd.read_csv(f"{ROOT_DIR}/datasets/Training_sets.csv", 
                     header=[0, 1], index_col=0)
df_tight = pd.read_csv(f"{ROOT_DIR}/datasets/Tight_set.csv")
                       header=[0, 1], index_col=0)
df_loose = pd.read_csv(f"{ROOT_DIR}/datasets/Loose_set.csv")
                       header=[0, 1], index_col=0)

#define a set of columns as metadata that will not be used for training
meta_cols = ["Names", "SMILES", "Solubility"]

#select training dataset
df_int = df_int.loc[df_int[('Dataset', DATASET)]]
df_int.drop(['Dataset', 'Reference'], axis=1, level=0, inplace=True)

#define features as full dataset after dropping metadata columns
#"Solubility" column is dropped here to ensure it is not used as a training feature
features_int = df_int.drop(meta_cols, axis=1, level=0)['FEATSET']
features_tight = df_tight.drop(meta_cols, axis=1, level=0)['FEATSET']
features_loose = df_loose.drop(meta_cols, axis=1, level=0)['FEATSET']

#define column "Solubility" as targets
targets_int = df_int["Solubility"]
targets_tight = df_tight["Solubility"]
targets_loose = df_loose["Solubility"]

#retain metadata for later use in output
meta_int = df_int[meta_cols].values
meta_tight = df_tight[meta_cols].values
meta_loose = df_loose[meta_cols].values

ncols = features_int.shape[1]

#set random initial seed, 50 resamples, jobs to run on 1 cpu core, 100 epochs
seed = 8434
nresamples = 50
n_jobs = 1
EPOCHS = 100

#define parameter for shape of data
in_shape = (ncols, )

#define empty lists for r2 and rmse scores, to be appended per resample
r2_scores_int = []
rmse_scores_int = []

#define empty lists for predicted values for both tight and loose sets, to be appended per resample
preds_list1 = []
preds_list2 = []

#build network in keras.sequential with arguments to be optimised
#number of nodes in hidden layers 1, 2, 3, learning rate, loss funciton all optimised
#activation can also be optimised but "relu" was used 
def build_model(layer1, layer2, layer3, learning_rate, loss, **kwargs):
    model = keras.Sequential([
        layers.Dense(layer1, activation = "relu", input_shape = in_shape),
        layers.Dense((layer2), activation = "relu"),
        layers.Dense((layer3), activation = "relu"),
        layers.Dense(1)
    ])

    #setting optimiser to "Adam"
    optimiser = tf.keras.optimizers.Adam(learning_rate)

    #setting loss functin, optimiser, and loss metrics 
    #loss metrics optimised over "mae" and "mse" 
    model.compile(loss = loss,
                optimizer = optimiser,
                metrics = ["mae", "mse"],
                )
    return model

#defining model based on built network
#batch_size optimised
#number of epochs can be optimised but was kept as 100
def train_model(model, nxtr, ytr, batch_size, epochs = 100, **kwargs):
    HISTORY = model.fit(
    nxtr, ytr,
    batch_size = batch_size, epochs = epochs, validation_split = 0.3, 
    )
    return HISTORY

#resample loop
for n in range(nresamples):

    print(f"Performing resamples {n + 1} of {nresamples}.")

    #lines 117-136, train/test splitof 70/30, defining features and targets, and formatting
    xtr = features_int.sample(frac = 0.7, random_state = 2 * n)
    xte = features_int.drop(xtr.index)

    ytr = targets_int.sample(frac = 0.7, random_state = 2 * n)
    yte = targets_int.drop(ytr.index)
    ytr = ytr.values.squeeze()
    yte = yte.values.squeeze()

    mtr = meta_int.sample(frac = 0.7, random_state = 2 * n)
    mte = meta_int.drop(mtr.index)

    xte_t = features_tight
    xte_l = features_loose

    yte_t = targets_tight
    yte_l = targets_loose

    xtr_stats = xtr.describe()
    xtr_stats = xtr_stats.transpose()

    #normalising feature data using sklearn and fitting normalised data to training and testing data
    norm_props = StandardScaler()
    norm_props.fit(xtr)
    nxtr = norm_props.transform(xtr)
    nxte = norm_props.transform(xte)
    nxte_t = norm_props.transform(xte_t)
    nxte_l = norm_props.transform(xte_l)

    #building the model
    model = build_model(**config)

    #training the model
    HISTORY = train_model(model, nxtr, ytr, epochs = EPOCHS, **config)

    #making predictions on testing split
    preds = model.predict(nxte)
    preds = preds.squeeze()

    mte["Prediction"] = preds

    #writing out a file consisting of predicted values per resample
    mte.to_csv(f"Results/{FEATSET}_int_{n+1}.csv")
   
    #making predictions on tight and loose sets
    preds1 = model.predict(nxte_t)
    preds2 = model.predict(nxte_l)

    #calculating r2 and rmse for prediction on testing split
    r2_int = metrics.r2_score(yte, preds)
    rmse_int = np.sqrt(metrics.mean_squared_error(yte, preds))

    #appending to list with stat values
    r2_scores_int.append(r2_int)
    rmse_scores_int.append(rmse_int)

    #appending to list of prediciton values for tight and loose sets
    preds_list1.append(preds1)
    preds_list2.append(preds2)

    #plotting loss vs epoch
    plt.figure()
    plt.plot(HISTORY.history["mae"], label = "mae")
    plt.plot(HISTORY.history["loss"], label = "loss")
    plt.plot(HISTORY.history["val_mae"], label = "val_mae")
    plt.plot(HISTORY.history["val_loss"], label = "val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc = "best")
    plt.savefig(f"Results/Plots/{FEATSET}_error_{n+1}.png")

#creating dataframe of metadata
df_tight_meta = pd.DataFrame(meta_tight, columns = meta_cols)
df_loose_meta = pd.DataFrame(meta_loose, columns = meta_cols)

#creating empty dictionaries for tight and loose predictions
preds_dict_t = {}
preds_dict_l = {}

#adding predicitons to dictionaries
for i, preds in enumerate(preds_list_t):
    preds_dict1[f"Prediction {i+1}"] = preds.T[0]

for i, preds in enumerate(preds_list_l):
    preds_dict2[f"Prediction {i+1}"] = preds.T[0]

#creating dataframe of predicitons
df_tight_preds = pd.DataFrame(preds_dict_t)
df_loose_preds = pd.DataFrame(preds_dict_l)

pred_names = df_tight_preds.columns

#calculating stats for tight set predictions
df_tight_preds["Mean"] = df_tight_preds[pred_names].mean(axis = 1).values
df_tight_preds["Median"] = df_tight_preds[pred_names].median(axis = 1).values
df_tight_preds["Std"] = df_tight_preds[pred_names].std(axis = 1).values
df_tight_preds["Min"] = df_tight_preds[pred_names].min(axis = 1).values
df_tight_preds["Max"] = df_tight_preds[pred_names].max(axis = 1).values

#calculating stats for loose set predictions
df_loose_preds["Mean"] = df_loose_preds[pred_names].mean(axis = 1).values
df_loose_preds["Median"] = df_loose_preds[pred_names].median(axis = 1).values
df_loose_preds["Std"] = df_loose_preds[pred_names].std(axis = 1).values
df_loose_preds["Min"] = df_loose_preds[pred_names].min(axis = 1).values
df_loose_preds["Max"] = df_loose_preds[pred_names].max(axis = 1).values

#concatenating metadata and predictions for tight and loose sets
df_tight_results = pd.concat([df_tight_meta, df_tight_preds], axis = 1)
df_loose_results = pd.concat([df_loose_meta, df_loose_preds], axis = 1)

#writing out prediction results to csv
df_tight_results.to_csv(f"Results/{FEATSET}_tight_results.csv", index = False)
df_loose_results.to_csv(f"Results/{FEATSET}_loose_results.csv", index = False)

print("\n")
print(f"Mean for int set r2 score: {np.mean(r2_scores_int)}, with SD {np.std(r2_scores_int)}")
print(f"Mean for int set rmse score: {np.mean(rmse_scores_int)}, with SD {np.std(rmse_scores_int)}")
