import pandas as pd
import os
from pathlib import Path
'''
Questo script serve per solamente per eliminare righe, il cui contenuto, filtrato sulla colonna, è vuoto.
Viene poi salvato il risultato su un nuovo file excel/csv, a seconda dell'occorrenza.
'''



source_path = Path(__file__).resolve()
source_dir = Path(source_path.parent.parent.parent)
path = Path(str(source_dir) + "/.tRNA/ENA.xlsx")
path_results = Path(str(source_dir) + "/.tRNA/ENA_notnull_phylum/ENA.xlsx")

df = pd.read_excel(path)

df.dropna(subset=['Taxonomy.ENA.phylum'], inplace=True)

# Salvataggio del risultato in un nuovo file xlsx nella cartella results
df.to_excel(path_results, index=False)

#df.to_csv(path_results, index=False)