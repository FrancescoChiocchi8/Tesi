import os
import shutil
from pathlib import Path

import pandas as pd

'''
Questo script consente la copia dei files fasta presenti in una determinata cartella, in un'altra, verificando se
il nome del file fasta preso in esame è presente nel file excel. Se la condizione è vera, allora avviene la copia
del file in un altro determinato path.
'''



# Percorsi del file excel e della cartella dei files fasta
source_path = Path(__file__).resolve()
source_dir = Path(source_path.parent.parent.parent)

classification_name = "ENA"

path = Path(str(source_dir) + "/16S_2_csv/Results/notnull_phylum/%s.xlsx" % classification_name)
path_fasta_folder = Path(str(source_dir) + "/Fasta_nH_16S/")
path_new_fasta_folder = Path(str(source_dir) + "/.16S_%s/Fasta_16S_%s/" % (classification_name, classification_name))

df = pd.read_excel(path)

benchmark_id_xlsx = 'Benchmark ID'
molecule_id = set(df[benchmark_id_xlsx] + '_nH')

'''
Scorro la cartella contenente i files fasta. Per ogni elemento prendo il nome, elimino l'estensione del file,
controllo se la nuova stringa è contenuta nel file excel confrontandola con l'id univoco delle molecole.
Se è presente avviene la copia dell'intero file fasta in un'altra directory, altrimenti vado a confrontare il prossimo file
'''
for file_fasta in os.listdir(path_fasta_folder):
    filename = os.path.splitext(file_fasta)[0].replace('_nH.fasta', '')
    if filename in molecule_id:
        if not os.path.exists(path_new_fasta_folder):
            os.makedirs(path_new_fasta_folder)
        shutil.copy2(os.path.join(path_fasta_folder, file_fasta), os.path.join(path_new_fasta_folder, file_fasta))

