import pandas as pd
import os
from pathlib import Path
'''
Questo script consente di effettuare un join per due o più file excel che si vogliono prendere in esame.
È stato usato per l'esperimento riguardante il tRNA, dove si andava a verificare se in tutti i file considerati era
presente l'id della molecola. Quindi, se la verifica era positiva, allora si aggiungeva la riga (contenente la colonna
del nome del file, dell'identificativo, del superkingdom e del phylum) in un altro file excel.
Altrimenti saltava al prossimo id della molecola da confrontare.
'''


# Percorsi dei files excel
source_path = Path(__file__).resolve()
source_dir = Path(source_path.parent.parent.parent)

classification_name = "ENA"
second_classification_name = "NCBI"

path_origin_1 = Path(str(source_dir) + "/16S_complete/notnull_phylum/%s.xlsx" % classification_name)
path_origin_2 = Path(str(source_dir) + "/16S_complete/notnull_phylum/%s.xlsx" % second_classification_name)
result_path = Path(str(source_dir) + "/16S_complete/notnull_phylum/join_notnull_phylum.csv")

# Carica i file in DataFrames
df1 = pd.read_excel(path_origin_1)
df2 = pd.read_excel(path_origin_2)

# Seleziona la colonna desiderata da ogni file
df1_col = df1["Benchmark ID"]
df2_col = df2["Benchmark ID"]

df_result = pd.DataFrame(columns=df1.columns)

for index, value in df1_col.iteritems():
    if value in df1_col.values and value in df2_col.values and value:
        df_result = df_result.append(df1.loc[index], ignore_index=True)

result_df = pd.DataFrame(df_result)
result_df.to_csv(result_path, index=False)
