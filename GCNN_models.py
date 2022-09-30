import deepchem as dc
from itertools import product
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import math

# Metrics:
r2 = dc.metrics.Metric(metric=r2_score, name='r2')
rmsd = dc.metrics.Metric(metric=dc.metrics.rms_score, name='rmsd')
bias = dc.metrics.Metric(metric=lambda t, p: np.mean(p - t), name='bias', mode='regression')
sdep = dc.metrics.Metric(metric=lambda t, p: math.sqrt(np.mean(((p - t) - np.mean(p - t))**2)), name='sdep', mode='regression')
metrics_ls = [r2, rmsd, bias, sdep]

# Filenames:
ROOT_DIR = 'datasets/'
dataset_file = ROOT_DIR+'Training_sets.csv'
dataset_name = 'D300' #DATASET
tight_set_file = ROOT_DIR+'Tight_set.csv'
loose_set_file = ROOT_DIR+'Loose_set.csv'

# Model name (GraphConv, DAG or Weave):
model_name = 'GraphConv'

# Possible hyperparameters:
hyperparams = {
    'GraphConv' : {
        'batch_normalize' : [True],
        'batch_size' : [50, 100],
        'dense_layer_size' : [128, 256, 64],
        'graph_conv_layers' : [[64], [256], [64, 64], [64, 128], [128, 64]],
        'dropout' : [0.0, 0.1]}, 
    'Weave' : {
        'batch_normalize' : [True],
        'batch_size' : [50, 100],
        'n_hidden' : [50, 25, 100],
        'fully_connected_layer_sizes' : [[2000, 100], [2000], [2000, 1000, 100]],
        'dropouts' : [0.0, 0.25, 0.1]}, 
    'DAG' : {
        'batch_normalize' : [True],
        'batch_size' : [100],
        'layer_sizes' : [[100]],
        'layer_sizes_gather' : [[100]],
        'dropout' : [0.0]}}

hyperparams = hyperparams[model_name]

# Load training dataset:
featurizer = dc.feat.ConvMolFeaturizer()
if model_name == 'Weave':
    featurizer = dc.feat.WeaveFeaturizer()
df = pd.read_csv(dataset_file, header=[0, 1])
df = df.loc[df[('Dataset', dataset_name)]]\
       .drop(['Dataset', 'Reference', 'RDKit', 'Mordred'], axis=1, level=0)\
       .droplevel(0, axis=1)\
       .to_csv(dataset_name+'.csv')
loader = dc.data.CSVLoader(tasks=['Solubility'],
                           feature_field='SMILES',
                           id_field='ID',
                           featurizer=featurizer)
dataset = loader.create_dataset(dataset_name+'.csv')

# Load tight and loose datasets:
df = pd.read_csv(tight_set_file, header=[0, 1])
df = df.drop(['RDKit', 'Mordred'], axis=1, level=0)\
       .droplevel(0, axis=1)\
       .to_csv('Tight_set.csv')
tight_set = loader.create_dataset('Tight_set.csv')
df = pd.read_csv(loose_set_file, header=[0, 1])
df = df.drop(['RDKit', 'Mordred'], axis=1, level=0)\
       .droplevel(0, axis=1)\
       .to_csv('Loose_set.csv')
loose_set = loader.create_dataset('Loose_set.csv')

# Split dataset:
splitter = dc.splits.RandomSplitter()
train_set, test_set = \
    splitter.train_test_split(dataset,
                              frac_train=0.7)

# Transformers:
if model_name == 'DAG':
    max_atoms = max([mol.get_num_atoms() for mol in dataset.X] + \
                    [mol.get_num_atoms() for mol in tight_set.X] + \
                    [mol.get_num_atoms() for mol in loose_set.X])
    transformer = dc.trans.DAGTransformer(max_atoms=max_atoms)
    train_set = transformer.transform(train_set)
    test_set = transformer.transform(test_set)
    tight_set = transformer.transform(tight_set)
    loose_set = transformer.transform(loose_set)

transformer = dc.trans.NormalizationTransformer(transform_y=True,
                                                dataset=train_set)
train_set = transformer.transform(train_set)
test_set = transformer.transform(test_set)
tight_set = transformer.transform(tight_set)
loose_set = transformer.transform(loose_set)

# Train model:
all_results = []
hp_names = hyperparams.keys()
hp_vals = [hyperparams[hp] for hp in hp_names]
for i, hp_combination in enumerate(product(*hp_vals)):
    hp_dict = dict(zip(hp_names, hp_combination))
    
    if model_name == 'GraphConv':
        model = dc.models.GraphConvModel(n_tasks=1, 
                                         mode='regression', 
                                         **hp_dict)
    elif model_name == 'DAG':
        model = dc.models.DAGModel(n_tasks=1, 
                                   max_atoms=max_atoms, 
                                   mode='regression', 
                                   **hp_dict)
    elif model_name == 'Weave':
        model = dc.models.WeaveModel(n_tasks=1, 
                                     mode='regression', 
                                     **hp_dict)

    model.fit(train_set, nb_epoch=2)

    # Get predictions on test sets:
    test_preds = model.predict(test_set, transformers=[transformer]).squeeze()
    tight_preds = model.predict(tight_set, transformers=[transformer]).squeeze()
    loose_preds = model.predict(loose_set, transformers=[transformer]).squeeze()

    # Save predictions to file:
    pd.DataFrame(data=test_preds, 
                 columns=['Predicted_solubility'], 
                 index=test_set.ids)\
      .to_csv('test_preds_'+str(i)+'.csv')
    pd.DataFrame(data=tight_preds, 
                 columns=['Predicted_solubility'], 
                 index=tight_set.ids)\
      .to_csv('tight_preds_'+str(i)+'.csv')
    pd.DataFrame(data=loose_preds, 
                 columns=['Predicted_solubility'], 
                 index=loose_set.ids)\
      .to_csv('loose_preds_'+str(i)+'.csv')

    # Get statistics:
    test_scores = model.evaluate(test_set, metrics=[r2, rmsd, sdep, bias], transformers=[transformer])
    tight_scores = model.evaluate(tight_set, metrics=[r2, rmsd, sdep, bias], transformers=[transformer])
    loose_scores = model.evaluate(loose_set, metrics=[r2, rmsd, sdep, bias], transformers=[transformer])

    test_scores = {'test_'+n : s for n, s in test_scores.items()}
    tight_scores = {'tight_'+n : s for n, s in tight_scores.items()}
    loose_scores = {'loose_'+n : s for n, s in loose_scores.items()}

    # Convert list to str before saving to DataFrame:
    for hp_name, hp_val in hp_dict.items():
        if isinstance(hp_val, list):
            hp_dict[hp_name] = str(hp_val)

    # Save results:
    all_results.append(pd.DataFrame(data={'model' : model_name,
                                          **hp_dict,
                                          **test_scores,
                                          **tight_scores,
                                          **loose_scores},
                                    index=[i]))

# Combine results from separate runs:
df_results = pd.concat(all_results)
df_results.to_csv('results.csv')
