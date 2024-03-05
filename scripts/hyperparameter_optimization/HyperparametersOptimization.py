import csv
import datetime
import os
from functools import partial
from pathlib import Path

import optuna
import pandas as pd
from hyperopt import fmin, tpe, Trials, rand, STATUS_OK
from sklearn.model_selection import ParameterGrid

from Tesi.scripts.cnn.CNN import CNN
from Tesi.scripts.hyperparameter_optimization.HyperparameterSpace import get_hyperopt_hyperparameters, \
    get_sklearn_hyperparameters, get_optuna_hyperparameters
from Tesi.scripts.hyperparameter_optimization.MetricsHyperparameterPlotter import plot_hyperopt_al_graphs, \
    plot_hyperopt_apr_graphs, \
    plot_grid_graphs, plot_bayesian_graphs
from Tesi.scripts.hyperparameter_optimization.Utility import clean_file, print_results

source_path = Path(__file__).resolve()
source_dir = Path(source_path.parent.parent.parent)

'''
La libreria Hyperopt non ha messo a disposizione l'ottimizzazione degli iperparametri con la tecnica della Grid Search.
Si utilizza la libreria Sklearn e in particolare ParameterGrid, per la ricerca della lista degli iperparametri migliori
tramite la tecnica Grid Search.
'''


def grid_search(dataset_directory, ol_units_input):
    params = get_sklearn_hyperparameters(ol_units_input)
    param_list = list(ParameterGrid(params))

    results = []
    histories = []
    for params in param_list:
        print(f"Current parameters: {params}")
        cnn = CNN(dataset=dataset_directory, model_mk='model_mk', **params)
        model = cnn.create_model()
        datagen_list = cnn.datagen()
        history = cnn.train(model, datagen_list[0], datagen_list[1])

        val_loss = history.history['val_loss'][-1]
        val_acc = history.history['val_accuracy'][-1]
        val_precision = history.history['val_precision'][-1]
        val_recall = history.history['val_recall'][-1]
        val_auc = history.history['val_auc'][-1]

        results.append((params, val_loss, val_acc, val_precision, val_recall, val_auc))
        histories.append(history)
    best_params, best_val_loss, best_val_acc, best_val_precision, best_val_recall, best_val_auc = max(results, key=lambda x: x[1])
    print(f"Best parameters: {best_params}")
    print(f"Best validation loss: {best_val_loss}")
    print(f"Best validation accuracy: {best_val_acc}")
    print(f"Best validation precision: {best_val_precision}")
    print(f"Best validation recall: {best_val_recall}")
    print(f"Best validation AUC: {best_val_auc}")

    now = datetime.datetime.now()
    folder_name = f"{dataset_directory}"
    folder_path = os.path.join(source_dir, "Hyperparameters_Optimization/GridSearchResults", folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    csv_folder_path = os.path.join(folder_path, "csv")
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)

    filename = os.path.join(csv_folder_path, f"{dataset_directory}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    with open(filename, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['batch_size', 'epochs', 'fl_filter', 'ol_units', 'n_dropout', 'drop_value', 'n_layer',
                         'lr', 'patience', 'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_auc'])
        for params, val_loss, val_acc, val_precision, val_recall, val_auc in results:
            writer.writerow(
                [params['batch_size'], params['epochs'], params['fl_filter'], params['ol_units'], params['n_dropout'],
                 params['drop_value'], params['n_layer'], params['lr'], params['patience'], val_loss, val_acc,
                 val_precision, val_recall, val_auc])

    plot_grid_graphs(histories, dataset_directory, metric='val_loss')
    plot_grid_graphs(histories, dataset_directory, metric='val_accuracy')
    plot_grid_graphs(histories, dataset_directory, metric='val_precision')
    plot_grid_graphs(histories, dataset_directory, metric='val_recall')
    plot_grid_graphs(histories, dataset_directory, metric='val_auc')
    with open(filename, mode='r') as results_file:
        lines = results_file.readlines()
    with open(filename, mode='w') as results_file:
        results_file.writelines(line for line in lines if line.strip())


'''
Funzione di sviluppo dell'ottimizzazione degli iperparametri
'''


def optimize(dataset_directory, ol_units_input, optimization_method, n_eval):
    print(f"Stai eseguendo l'ottimizzazione degli iperparametri tramite {optimization_method.upper()}")
    space = get_hyperopt_hyperparameters(ol_units_input)
    trials = Trials()
    now = datetime.datetime.now()
    folder_name = f"{dataset_directory}"
    folder_path = os.path.join(
        source_dir, f"Hyperparameters_Optimization/{optimization_method.capitalize()}_Results", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    csv_folder_path = os.path.join(folder_path, "csv")
    os.makedirs(csv_folder_path, exist_ok=True)
    filename = os.path.join(csv_folder_path, f"{dataset_directory}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    with open(filename, mode='w') as results_file:
        fieldnames = ['batch_size', 'epochs', 'fl_filter', 'ol_units', 'n_dropout', 'drop_value', 'n_layer', 'lr',
                      'patience', 'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_auc']
        writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        writer.writeheader()
        val_losses = []
        val_accs = []
        val_precs = []
        val_recs = []
        val_aucs = []
        best_loss = float('inf')
        best_acc = 0.0
        best_prec = 0.0
        best_rec = 0.0
        best_auc = 0.0
        algo = rand.suggest if optimization_method == 'random_search' else tpe.suggest
        for i in range(n_eval):
            result = fmin(fn=partial(objective_hyperopt, dataset_directory=dataset_directory, writer=writer, i=i + 1,
                                     val_losses=val_losses, val_accs=val_accs, val_precs=val_precs,
                                     val_recs=val_recs, val_aucs=val_aucs), space=space,
                          algo=algo, max_evals=i + 1, trials=trials, verbose=False)
            print("Best hyperparameters found in evaluation ", i + 1, ":")
            print(result)
            best_loss = min(val_losses) if min(val_losses) < best_loss else best_loss
            best_acc = max(val_accs) if max(val_accs) > best_acc else best_acc
            best_prec = max(val_precs) if max(val_precs) > best_prec else best_prec
            best_rec = max(val_recs) if max(val_recs) > best_rec else best_rec
            best_auc = max(val_aucs) if max(val_aucs) > best_auc else best_auc
        print_results(best_loss, best_acc, best_prec, best_rec, best_auc)
        plot_hyperopt_al_graphs(val_losses, val_accs, now, dataset_directory, optimization_method)
        plot_hyperopt_apr_graphs(val_precs, val_recs, val_aucs, now, dataset_directory, optimization_method)
        clean_file(filename)


'''
Sviluppo della tecnica dell'ottimizzazione degli iperparametri "Random Search" con la libreria Hyperopt
'''


def random_search(dataset_directory, ol_units_input, n_eval):
    optimize(dataset_directory, ol_units_input, 'random_search', n_eval)


'''
Sviluppo della tecnica dell'ottimizzazione degli iperparametri "Tree of Parzen Estimators" con la libreria Hyperopt
'''


def tree_of_parzen_estimators(dataset_directory, ol_units_input, n_eval):
    optimize(dataset_directory, ol_units_input, 'Tree_of_Parzen_Estimators', n_eval)


'''
Sviluppo dell'ottimizzazione degli iperparametri tramite la Bayesian Optimization con la libreria Optuna
'''


def bayesian_optimization(dataset_directory, ol_units_input, n_eval):
    now = datetime.datetime.now()
    folder_name = f"{dataset_directory}"
    folder_path_dir = os.path.join(source_dir, "Hyperparameters_Optimization/Bayesian_optimization_Results")
    if not os.path.exists(folder_path_dir):
        os.makedirs(folder_path_dir)
    folder_path = os.path.join(source_dir, "Hyperparameters_Optimization/Bayesian_optimization_Results", folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    csv_folder_path = os.path.join(folder_path, "csv")
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)

    filename = os.path.join(csv_folder_path, f"{dataset_directory}{now.strftime('%Y-%m-%d%H-%M-%S')}.csv")
    study = optuna.create_study(direction='minimize')
    headers = ['batch_size', 'epochs', 'fl_filter', 'n_dropout', 'drop_value', 'n_layer', 'lr', 'patience', 'val_loss',
               'val_accuracy', 'val_precision', 'val_recall', 'val_auc']
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(n_eval):
            trial = study.ask()
            result = objective_bayesian(dataset_directory, ol_units_input, trial)
            study.tell(trial, result['val_loss'])
            row = [trial.params['batch_size'], trial.params['epochs'], trial.params['fl_filter'],
                   trial.params['n_dropout'],
                   trial.params['drop_value'], trial.params['n_layer'], trial.params['lr'], trial.params['patience'],
                   result['val_loss'], result['val_accuracy'], result['val_precision'], result['val_recall'],
                   result['val_auc']]
            writer.writerow(row)
        print(f"Results saved to {filename}")

    df = pd.read_csv(filename)
    plot_bayesian_graphs(df, folder_path, now)



'''
Funzione obiettivo della Bayesian Optimization
'''


def objective_bayesian(dataset_directory, ol_units_input, trial):
    hyperparameters = get_optuna_hyperparameters(trial)
    cnn = CNN(dataset=dataset_directory, model_mk='model_mk', batch_size=hyperparameters['batch_size'], epochs=hyperparameters['epochs'], fl_filter=hyperparameters['fl_filter'],
              ol_units=ol_units_input, n_dropout=hyperparameters['n_dropout'], drop_value=hyperparameters['drop_value'], n_layer=hyperparameters['n_layer'], lr=hyperparameters['lr'],
              patience=hyperparameters['patience'])
    model = cnn.create_model()
    datagen_list = cnn.datagen()
    history = cnn.train(model, datagen_list[0], datagen_list[1])
    try:
        val_loss = history.history['val_loss'][-1]
        val_accuracy = history.history['val_accuracy'][-1]
        val_precision = history.history['val_precision'][-1]
        val_recall = history.history['val_recall'][-1]
        val_auc = history.history['val_auc'][-1]
    except KeyError:
        print(
            "KeyError: 'val_loss', 'val_accuracy', 'val_precision', 'val_recall' or 'val_auc' not found in "
            "history.history")
        val_loss = float('inf')
        val_accuracy = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_auc = 0.0

    return {'val_loss': val_loss, 'val_accuracy': val_accuracy, 'val_precision': val_precision, 'val_recall': val_recall,
            'val_auc': val_auc, 'status': 'ok', 'params': trial.params}


'''
La funzione obiettivo crea il modello per la CNN con i parametri dati. Successivamente, la funzione addestra il modello
utilizzando i dati di addestramento e restituisce i valori di loss e accuracy per i dati di validazione. 
Infine, la funzione restituisce un dizionario contenente le loss, le accuracy e uno stato (STATUS_OK) che indica che la
funzione è stata eseguita correttamente.
'''


def objective_hyperopt(params, dataset_directory, writer, i, val_losses, val_accs, val_precs, val_recs, val_aucs):
    print("Evaluation ", i)
    print(params)
    cnn = CNN(dataset=dataset_directory, model_mk='model_mk', **params)
    model = cnn.create_model()
    datagen_list = cnn.datagen()
    history = cnn.train(model, datagen_list[0], datagen_list[1])
    try:
        val_loss = history.history['val_loss'][-1]
        val_accuracy = history.history['val_accuracy'][-1]
        val_precision = history.history['val_precision'][-1]
        val_recall = history.history['val_recall'][-1]
        val_auc = history.history['val_auc'][-1]
    except KeyError:
        print(
            "KeyError: 'val_loss', 'val_accuracy', 'val_precision', 'val_recall' or 'val_auc' not found in "
            "history.history")
        val_loss = float('inf')
        val_accuracy = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_auc = 0.0
    val_losses.append(val_loss)
    val_accs.append(val_accuracy)
    val_precs.append(val_precision)
    val_recs.append(val_recall)
    val_aucs.append(val_auc)

    writer.writerow({
        'batch_size': params['batch_size'],
        'epochs': params['epochs'],
        'fl_filter': params['fl_filter'],
        'ol_units': params['ol_units'],
        'n_dropout': params['n_dropout'],
        'drop_value': params['drop_value'],
        'n_layer': params['n_layer'],
        'lr': params['lr'],
        'patience': params['patience'],
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_auc': val_auc
    })
    return {'loss': val_loss, 'val_accuracy': val_accuracy, 'val_precision': val_precision, 'val_recall': val_recall,
            'val_auc': val_auc, 'status': STATUS_OK}
