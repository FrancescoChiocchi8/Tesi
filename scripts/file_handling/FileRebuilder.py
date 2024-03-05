from pathlib import Path

import pandas as pd

'''
Questo script consente di prendere più files per un certo database, e ricostruirlo sotto forma di un unico file.
È stato usato nell'esperimento riguardante il tRNA e per il 16S per ricostruire i files.
'''


# Percorso dei file excel
source_path = Path(__file__).resolve()
source_dir = Path(source_path.parent.parent.parent)

classification_name = "ENA"

# Nuovo file unico da generare
result_path = Path(str(source_dir) + "/16S_complete/%s.csv" % classification_name)

# File excel da cui prendere i dati
# in caso di più file si può contemplare un ciclo che scorre i file all'interno di una cartella
data_path = Path(str(source_dir) + "/16S_2_csv/Results/%s.csv" % classification_name)


def rebuild_file(file_path, columns, df_results=None):
    df = pd.read_csv(file_path)
    df = df[columns]
    if df_results is None:
        df_results = df
    else:
        df_results = pd.concat([df_results, df])
    return df_results


df_results = None
columns = ['Organism name', 'Length', 'Benchmark ID', 'Taxonomy.GTDB.domain', 'Taxonomy.GTDB.phylum']

df_results = rebuild_file(data_path, columns, df_results)
df_results.to_csv(result_path, index=False)
