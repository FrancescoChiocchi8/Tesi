import sys

from Tesi.scripts.hyperparameter_optimization.HyperparametersOptimization import grid_search, random_search, \
    tree_of_parzen_estimators, bayesian_optimization

def main(argv):
    if len(argv) != 4:
        print("Numero di argomenti non valido. Assicurati di fornire 4 argomenti.")
        return

    dataset_directory = argv[0]
    label = argv[1]
    technique = argv[2]
    n_eval = int(argv[3])

    match technique:
        case "1":
            grid_search(dataset_directory, label)
        case "2":
            random_search(dataset_directory, label, n_eval)
        case "3":
            tree_of_parzen_estimators(dataset_directory, label, n_eval)
        case "4":
            bayesian_optimization(dataset_directory, label, n_eval)

if __name__ == "__main__":
    main(sys.argv[1:])
