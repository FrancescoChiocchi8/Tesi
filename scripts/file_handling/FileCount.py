from pathlib import Path

import pandas as pd

'''
Questo script consente di contare il numero di occorrenze per un certo dato, contenuto in una colonna di un file excel.
Salva poi il risultato in un altro file.
'''

source_path = Path(__file__).resolve()
source_dir = Path(source_path.parent.parent.parent)

excel_path = Path(str(source_dir) + "/tRNA_csv/tRNA_csv/Results/NCBI.xlsx")
result_excel_path = Path(str(source_dir) + "/tRNA_csv/tRNA_csv/Tassonomie_numero_occorrenze/NCBI_superkingdom_occorrenze.xlsx")

df = pd.read_excel(excel_path)
colonna = df['Taxonomy.NCBI.superkingdom']
occorrenze = {}

for stringa in colonna:
    if not pd.isna(stringa):
        if stringa in occorrenze:
            occorrenze[stringa] += 1
        else:
            occorrenze[stringa] = 1

risultati_df = pd.DataFrame({'Taxonomy.NCBI.superkingdom': list(occorrenze.keys()), 'n° occorrenze': list(occorrenze.values())})

risultati_df.to_excel(result_excel_path, index=False)
