'''
Questo semplice script, prende un file excel, elimina i valori di una determinata colonna che sono nulli, elimina i duplicati
e rimuove le righe i cui parametri non sono soddisfatti (length < 100 || length > 130). Salva poi il risultato su un nuovo
file excel.
'''

from pathlib import Path

import pandas as pd

source_path = Path(__file__).resolve()
source_dir = Path(source_path.parent.parent.parent)
path = Path(str(source_dir) + "/16S_complete/notnull_phylum/join_notnull_phylum.xlsx")
path_results = Path(str(source_dir) + "/16S_complete/notnull_phylum/noduplicates/join_notnull_noduplicates_phylum.xlsx")

# Caricamento del file 5s.xlsx
df = pd.read_excel(path)

# Rimozione dei duplicati nella colonna Organism name
df.drop_duplicates(subset='Organism name', keep='first', inplace=True)

# Elimino i valori nulli della colonna Organism name
#df.dropna(subset=['Organism Name'], inplace=True)

# Rimuovo le righe con lunghezza inferiore a 100 o maggiore di 130
#df = df[(df['Length'] >= 100) & (df['Length'] <= 130)]

# Salvataggio del risultato in un nuovo file xlsx nella cartella results
df.to_excel(path_results, index=False)