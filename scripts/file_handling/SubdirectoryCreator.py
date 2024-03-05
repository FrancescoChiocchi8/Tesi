import csv
import os
from pathlib import Path

import pandas as pd

'''
Questo script consente di scorrere una determinata colonna di un file excel e di creare delle cartelle aventi il nome
della riga che al momento si sta considerando.
'''



# Percorsi dei files excel
source_path = Path(__file__).resolve()
source_dir = Path(source_path.parent.parent.parent)

classification_name = "ENA"

path = Path(str(source_dir) + "/.16S_%s/%s.xlsx" % (classification_name, classification_name))
csv_path = Path(str(source_dir) + "/.16S_%s/%s.csv" % (classification_name, classification_name))
destination_path = Path(str(source_dir) + "/Classification/.16S_%s/all_phylum/" % classification_name)

df = pd.read_excel(path)

# Funzione che prende il percorso di un file csv, il path dove andare a creare le subdirectory, la colonna dell'id delle molecole
# e quella del phylum presenti sul file csv. Poi le crea con il nome della riga di riferimento al phylum, se non esiste ancora.
def create_subdirectory(csv_filepath, dest_path, benchmark_id_csv, col_classifier):
    with open(csv_filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        subdirectory_names = []
        for row in reader:
            if row[benchmark_id_csv] != '' and row[col_classifier] != '':
                if row[benchmark_id_csv] in df['Benchmark ID'].values:
                    if row[col_classifier] not in subdirectory_names:
                        os.mkdir(os.path.join(dest_path, row[col_classifier]))
                        subdirectory_names.append(row[col_classifier])


# Crea, se non sono state già create, tutte le cartella in base al nome del classificatore
create_subdirectory(csv_path, destination_path, 2, 4)

