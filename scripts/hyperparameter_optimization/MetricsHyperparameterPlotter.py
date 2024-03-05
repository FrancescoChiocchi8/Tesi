import datetime
import os
from pathlib import Path

from matplotlib import pyplot as plt

source_path = Path(__file__).resolve()
source_dir = Path(source_path.parent.parent.parent)

'''
Metodo utilizzato da grid_search(...) per la costruzione di grafici per alcune metriche.
'''


def plot_grid_graphs(histories, dataset_directory, metric='val_loss'):
    now = datetime.datetime.now()
    if metric == 'val_loss':
        filename = os.path.join(source_dir, "Hyperparameters_Optimization/GridSearchResults", dataset_directory,
                                "losses",
                                "loss{}_{}.png".format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S')))
    elif metric == 'val_accuracy':
        filename = os.path.join(source_dir, "Hyperparameters_Optimization/GridSearchResults", dataset_directory, "accs",
                                "accuracy_{}_{}.png".format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S')))
    elif metric == 'val_precision':
        filename = os.path.join(source_dir, "Hyperparameters_Optimization/GridSearchResults", dataset_directory,
                                "precs",
                                "precision_{}_{}.png".format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S')))
    elif metric == 'val_recall':
        filename = os.path.join(source_dir, "Hyperparameters_Optimization/GridSearchResults", dataset_directory, "recs",
                                "recall_{}_{}.png".format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S')))
    elif metric == 'val_auc':
        filename = os.path.join(source_dir, "Hyperparameters_Optimization/GridSearchResults", dataset_directory, "aucs",
                                "auc_{}_{}.png".format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S')))
    else:
        print("Metrica non valida.")
        return
    folder_name = f"{dataset_directory}"
    folder_path = os.path.join(source_dir, "Hyperparameters_Optimization/GridSearchResults", folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.figure(figsize=(10, 6))
    for i, history in enumerate(histories):
        plt.plot(history.history[metric], label=f'Trial {i + 1}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.title(f'{metric.capitalize()} per ogni trial')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()


'''
Creo i grafici e imposto un valore massimo alla loss uguale a 3 SOLO per un fatto visivo.
'''


def plot_hyperopt_al_graphs(val_losses, val_accs, now, dataset_directory, optimization_method):
    global loss_dir, acc_dir

    if optimization_method == 'random_search':
        folder_name = f"{dataset_directory}"
        folder_path = os.path.join(source_dir, "Hyperparameters_Optimization/Random_search_Results", folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        loss_dir = os.path.join(source_dir, "Hyperparameters_Optimization/Random_search_Results", dataset_directory,
                                "losses/")
        acc_dir = os.path.join(source_dir, "Hyperparameters_Optimization/Random_search_Results", dataset_directory,
                               "accs/")
    elif optimization_method == 'Tree_of_Parzen_Estimators':
        folder_name = f"{dataset_directory}"
        folder_path = os.path.join(source_dir, "Hyperparameters_Optimization/Tree_of_Parzen_Estimators_Results",
                                   folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        loss_dir = os.path.join(source_dir, "Hyperparameters_Optimization/Tree_of_Parzen_Estimators_Results",
                                dataset_directory, "losses/")
        acc_dir = os.path.join(source_dir, "Hyperparameters_Optimization/Tree_of_Parzen_Estimators_Results",
                               dataset_directory, "accs/")

    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(acc_dir, exist_ok=True)

    val_losses = [min(loss, 3) for loss in val_losses]

    plt.plot(val_losses)
    plt.title('Validation Loss')
    plt.xlabel('Evaluation')
    plt.ylabel('Loss')
    plt.savefig(
        os.path.join(loss_dir, "loss_{}_{}.png".format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))))
    plt.close()

    plt.plot(val_accs)
    plt.title('Validation Accuracy')
    plt.xlabel('Evaluation')
    plt.ylabel('Accuracy')
    plt.savefig(
        os.path.join(acc_dir, "accuracy_{}_{}.png".format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))))
    plt.close()


'''
Funzione che crea grafici per le metriche utilizzate dagli algoritmi Random Search e Tree of Parzen Estimators.
'''


def plot_hyperopt_apr_graphs(val_precs, val_recs, val_aucs, now, dataset_directory, optimization_method):
    global precs_dir, recs_dir, aucs_dir

    if optimization_method == 'random_search':
        folder_name = f"{dataset_directory}"
        folder_path = os.path.join(source_dir, "Hyperparameters_Optimization/Random_search_Results", folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        precs_dir = os.path.join(source_dir, "Hyperparameters_Optimization/Random_search_Results", dataset_directory,
                                 "precs/")
        recs_dir = os.path.join(source_dir, "Hyperparameters_Optimization/Random_search_Results", dataset_directory,
                                "recs/")
        aucs_dir = os.path.join(source_dir, "Hyperparameters_Optimization/Random_search_Results", dataset_directory,
                                "aucs/")
    elif optimization_method == 'Tree_of_Parzen_Estimators':
        folder_name = f"{dataset_directory}"
        folder_path = os.path.join(source_dir, "Hyperparameters_Optimization/Tree_of_Parzen_Estimators_Results",
                                   folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        precs_dir = os.path.join(source_dir, "Hyperparameters_Optimization/Tree_of_Parzen_Estimators_Results",
                                 dataset_directory, "precs/")
        recs_dir = os.path.join(source_dir, "Hyperparameters_Optimization/Tree_of_Parzen_Estimators_Results",
                                dataset_directory, "recs/")
        aucs_dir = os.path.join(source_dir, "Hyperparameters_Optimization/Tree_of_Parzen_Estimators_Results",
                                dataset_directory, "aucs/")

    os.makedirs(precs_dir, exist_ok=True)
    os.makedirs(recs_dir, exist_ok=True)
    os.makedirs(aucs_dir, exist_ok=True)

    plt.plot(val_precs)
    plt.title('Validation Precision')
    plt.xlabel('Evaluation')
    plt.ylabel('Precision')
    plt.savefig(
        os.path.join(precs_dir, "precision_{}_{}.png".format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))))
    plt.close()

    plt.plot(val_recs)
    plt.title('Validation Recall')
    plt.xlabel('Evaluation')
    plt.ylabel('Recall')
    plt.savefig(
        os.path.join(recs_dir, "recall_{}_{}.png".format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))))
    plt.close()

    plt.plot(val_aucs)
    plt.title('Validation Auc')
    plt.xlabel('Evaluation')
    plt.ylabel('Auc')
    plt.savefig(
        os.path.join(aucs_dir, "auc_{}_{}.png".format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))))
    plt.close()


'''
Funzione che crea grafici per le metriche utilizzate dall'algoritmo Bayesian Optimization.
'''


def plot_bayesian_graphs(df, folder_path, now):
    global loss_dir, accuracy_dir, precs_dir, recs_dir, aucs_dir

    loss_path = os.path.join(folder_path, 'losses')
    accuracy_path = os.path.join(folder_path, 'accs')
    precs_path = os.path.join(folder_path, 'precs')
    recs_path = os.path.join(folder_path, 'recs')
    aucs_path = os.path.join(folder_path, 'aucs')

    os.makedirs(loss_path, exist_ok=True)
    os.makedirs(accuracy_path, exist_ok=True)
    os.makedirs(precs_path, exist_ok=True)
    os.makedirs(recs_path, exist_ok=True)
    os.makedirs(aucs_path, exist_ok=True)

    plt.plot(df.val_loss)
    plt.title('Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(loss_path, "loss_{}_{}.png".format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))))
    plt.close()

    plt.plot(df.val_accuracy)
    plt.title('Validation Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.savefig(
        os.path.join(accuracy_path, "accuracy_{}_{}.png".format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))))
    plt.close()

    plt.plot(df.val_precision)
    plt.title('Validation Precision')
    plt.xlabel('Iteration')
    plt.ylabel('Precision')
    plt.savefig(
        os.path.join(precs_path, "precision_{}_{}.png".format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))))
    plt.close()

    plt.plot(df.val_recall)
    plt.title('Validation Recall')
    plt.xlabel('Iteration')
    plt.ylabel('Recall')
    plt.savefig(os.path.join(recs_path, "recall_{}_{}.png".format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))))
    plt.close()

    plt.plot(df.val_auc)
    plt.title('Validation AUC')
    plt.xlabel('Iteration')
    plt.ylabel('AUC')
    plt.savefig(os.path.join(aucs_path, "auc_{}_{}.png".format(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))))
    plt.close()
