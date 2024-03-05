import numpy as np
from hyperopt import hp

'''
Funzione che restituisce lo spazio degli iperparametri da testare con l'algoritmo Grid Search.
'''


def get_sklearn_hyperparameters(ol_units_input):
    params = {
        'batch_size': [32, 64],
        'epochs': [20],
        'fl_filter': [16, 32, 64],
        'ol_units': [ol_units_input],
        'n_dropout': [1],
        'drop_value': [0.5],
        'n_layer': [2, 3, 4],
        'lr': [0.0001],
        'patience': [6]
    }
    return params


'''
Funzione che restituisce gli iperparametri da dare in input alla Random Search e alla Tree of Parzen Estimators.
'''


def get_hyperopt_hyperparameters(ol_units_input):
    space = {
        'batch_size': hp.choice('batch_size', [8, 16, 32, 64]),
        'epochs': hp.choice('epochs', [10, 20, 30, 40, 50]),
        'fl_filter': hp.choice('fl_filter', [8, 16, 32, 64]),
        'ol_units': hp.choice('ol_units', [ol_units_input]),
        'n_dropout': hp.choice('n_dropout', [1, 2, 3]),
        'drop_value': hp.uniform('drop_value', 0.3, 0.7),
        'n_layer': hp.choice('n_layer', [2, 3, 4, 5]),
        'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.001)),
        'patience': hp.choice('patience', [6])
    }
    return space


'''
Funzione che restituisce lo spazio degli iperparametri da valutare per la libreria optuna.
'''


def get_optuna_hyperparameters(trial):
    params = {
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
        'epochs': trial.suggest_int('epochs', 10, 50),
        'fl_filter': trial.suggest_categorical('fl_filter', [8, 16, 32, 64]),
        'n_dropout': trial.suggest_int('n_dropout', 1, 3),
        'drop_value': trial.suggest_float('drop_value', 0.2, 0.8),
        'n_layer': trial.suggest_int('n_layer', 2, 5),
        'lr': trial.suggest_float('lr', 0.0001, 0.001, log=True),
        'patience': trial.suggest_int('patience', 5, 20)
    }
    return params
