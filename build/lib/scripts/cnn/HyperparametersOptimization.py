import numpy as np

from hyperopt import hp, rand, STATUS_OK
from matplotlib import pyplot as plt
from hyperopt import fmin, tpe, Trials
from functools import partial
import datetime
from pathlib import Path
from Tesi.scripts.cnn.CNN import CNN

source_path = Path(__file__).resolve()
source_dir = Path(source_path.parent.parent.parent)

'''
La libreria Hyperopt non ha messo a disposizione l'ottimizzazione degli iperparametri con la tecnica della Grid Search.
Si utilizza la libreria sklearn e in particolare ParameterGrid, per la ricerca della lista degli iperparametri migliori.
'''
import csv
from sklearn.model_selection import ParameterGrid

def grid_search(dataset_directory, ol_units_input):
    params = {
        'batch_size': [8, 16],
        'epochs': [20, 30],
        'fl_filter': [8, 16],
        'ol_units': [ol_units_input],
        'n_dropout': [1],
        'drop_value': [0.5, 0.6],
        'n_layer': [2, 3, 4],
        'lr': [0.0001, 0.001],
        'patience': [5]
    }
    param_list = list(ParameterGrid(params))
    results = []
    for params in param_list:
        cnn = CNN(dataset=dataset_directory, model_mk='model_mk', **params)
        model = cnn.create_model()
        datagen_list = cnn.datagen()
        history = cnn.train(model, datagen_list[0], datagen_list[1])
        val_acc = history.history['val_accuracy'][-1]
        results.append((params, val_acc))

    # Trova la combinazione migliore di iperparametri
    best_params, best_val_acc = max(results, key=lambda x: x[1])
    print(f"Best params: {best_params}")
    print(f"Best validation accuracy: {best_val_acc}")

    now = datetime.datetime.now()
    filename = Path(str(source_dir) + "/Hyperparameters_Optimization/GridSearchResults/csv/grid_{}_{}.csv".
                    format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S')))
    with open(filename, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['batch_size', 'epochs', 'fl_filter', 'ol_units', 'n_dropout', 'drop_value', 'n_layer',
                         'lr', 'patience', 'val_accuracy'])
        for params, val_acc in results:
            writer.writerow([params['batch_size'], params['epochs'], params['fl_filter'], params['ol_units'], params['n_dropout'],
                             params['drop_value'], params['n_layer'], params['lr'], params['patience'], val_acc])

'''
Funzione che restituisce gli iperparametri da dare in input alla Random Search e alla Bayesian Optimation.
'''
def get_hyperparameters(ol_units_input):
    space = {
        'batch_size': hp.choice('batch_size', [8, 16]),
        'epochs': hp.choice('epochs', [2, 3, 4]),
        'fl_filter': hp.choice('fl_filter', [8, 16, 32, 64]),
        'ol_units': hp.choice('ol_units', [ol_units_input]),
        'n_dropout': hp.choice('n_dropout', [1, 2, 3]),
        'drop_value': hp.uniform('drop_value', 0.1, 0.7),
        'n_layer': hp.choice('n_layer', [2, 3, 4]),
        'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.001)),
        'patience': hp.choice('patience', [5, 10, 15])
    }
    return space
'''
Sviluppo della tecnica dell'ottimizzazione degli iperparametri "Random Search" con la libreria Hyperopt
'''
def random_search(dataset_directory, ol_units_input):
    print("Stai eseguendo l'ottimizzazione degli iperparametri tramite RANDOM SEARCH")
    space = get_hyperparameters(ol_units_input)
    trials = Trials()
    now = datetime.datetime.now()
    filename = Path(str(source_dir) + "/Hyperparameters_Optimization/RandomSearchResults/csv/random_{}_{}.csv".
                    format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S')))
    # Apriamo il file CSV in modalità di scrittura
    with open(filename, mode='w') as results_file:
        fieldnames = ['batch_size', 'epochs', 'fl_filter', 'ol_units', 'n_dropout', 'drop_value', 'n_layer', 'lr',
                      'patience',
                      'val_loss', 'val_acc']
        writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        writer.writeheader()

        val_losses = []  # Inizializziamo una lista vuota per tenere traccia delle loss di validazione
        val_accs = []

        for i in range(20):  # Eseguiamo tot valutazioni massime
            result = fmin(fn=partial(objective, dataset_directory=dataset_directory, writer=writer, i=i + 1,
                                     val_losses=val_losses,
                                     val_accs=val_accs, verbose=False), space=space,
                          algo=rand.suggest, max_evals=i + 1, trials=trials)
            print("Best hyperparameters found in evaluation ", i + 1, ":")
            print(result)

        plt.plot(val_losses)
        plt.title('Validation Loss')
        plt.xlabel('Evaluation')
        plt.ylabel('Loss')
        plt.savefig(Path(str(source_dir) + "/Hyperparameters_Optimization/RandomSearchResults/losses/loss_{}_{}.png".
                         format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))))
        plt.close()

        plt.plot(val_accs)
        plt.title('Validation Accuracy')
        plt.xlabel('Evaluation')
        plt.ylabel('Accuracy')
        plt.savefig(Path(str(source_dir) + "/Hyperparameters_Optimization/RandomSearchResults/accs/acc_{}_{}.png".
                         format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))))
        plt.close()

    print("Optimization completed")


def bayesian_optimization(dataset_directory, ol_units_input):
    print("Stai eseguendo l'ottimizzazione degli iperparametri tramite BAYESIAN OPTIMIZATION")
    space = get_hyperparameters(ol_units_input)
    trials = Trials()
    # Generiamo un nome di file unico basato sulla data e sull'ora di esecuzione del programma
    now = datetime.datetime.now()
    filename = Path(
        str(source_dir) + "/Hyperparameters_Optimization/BayesianOptimizationResults/csv/bayesian_{}_{}.csv".
        format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S')))
    # Apriamo il file CSV in modalità di scrittura
    with open(filename, mode='w') as results_file:
        fieldnames = ['batch_size', 'epochs', 'fl_filter', 'ol_units', 'n_dropout', 'drop_value', 'n_layer', 'lr',
                      'patience',
                      'val_loss', 'val_acc']
        writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        writer.writeheader()

        val_losses = []  # Inizializziamo una lista vuota per tenere traccia delle loss di validazione
        val_accs = []

        for i in range(20):  # Eseguiamo tot valutazioni massime
            result = fmin(fn=partial(objective, dataset_directory=dataset_directory, writer=writer, i=i + 1,
                                     val_losses=val_losses,
                                     val_accs=val_accs), space=space, algo=tpe.suggest, max_evals=i + 1, trials=trials, verbose=False)
            print("Best hyperparameters found in evaluation ", i + 1, ":")
            print(result)

        plt.plot(val_losses)
        plt.title('Validation Loss')
        plt.xlabel('Evaluation')
        plt.ylabel('Loss')
        plt.savefig(
            Path(str(source_dir) + "/Hyperparameters_Optimization/BayesianOptimizationResults/losses/loss_{}_{}.png".
                 format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))))
        plt.close()

        plt.plot(val_accs)
        plt.title('Validation Accuracy')
        plt.xlabel('Evaluation')
        plt.ylabel('Accuracy')
        plt.savefig(
            Path(str(source_dir) + "/Hyperparameters_Optimization/BayesianOptimizationResults/accs/acc_{}_{}.png".
                 format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))))
        plt.close()

    print("Optimization completed")


def objective(params, dataset_directory, writer, i, val_losses, val_accs):
    print("Evaluation ", i)
    print(params)
    cnn = CNN(dataset=dataset_directory, model_mk='model_mk', **params)
    model = cnn.create_model()
    datagen_list = cnn.datagen()
    history = cnn.train(model, datagen_list[0], datagen_list[1])

    try:
        val_loss = history.history['val_loss'][-1]
        val_accuracy = history.history['val_accuracy'][-1]
    except KeyError:
        print("KeyError: 'val_loss' or 'val_accuracy' not found in history.history")
        val_loss = float('inf')
        val_accuracy = 0.0

    val_losses.append(val_loss)
    val_accs.append(val_accuracy)

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
        'val_acc': val_accuracy
    })
    return {'loss': val_loss, 'val_accuracy': val_accuracy, 'status': STATUS_OK}
