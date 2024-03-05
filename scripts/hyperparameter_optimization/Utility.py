'''
Stampa a video i risultati migliori raggiunti durante le varie evalutazioni.
'''
def print_results(best_loss, best_acc, best_prec, best_rec, best_auc):
    print("Best loss:", best_loss)
    print("Best accuracy:", best_acc)
    print("Best precision:", best_prec)
    print("Best recall:", best_rec)
    print("Best auc:", best_auc)


'''
Metodo che scorre tutto il file passato in input ed elimina eventuali righe vuote generate dopo l'ottimizzazione degli iperparametri.
'''


def clean_file(filename):
    with open(filename, mode='r') as results_file:
        lines = results_file.readlines()
    with open(filename, mode='w') as results_file:
        results_file.writelines(line for line in lines if line.strip())