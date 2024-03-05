import shutil
import re
import pandas as pd
import os
from pathlib import Path
'''
Questo script serve per copiare le immagini presenti in un dataset in un determinato percorso, a seconda del nome
del classificatore (in questo caso nome_superkingdom o nome_phylum).
'''



# Percorsi dei files excel, del dataset e delle subdirectory.
source_path = Path(__file__).resolve()
source_dir = Path(source_path.parent.parent.parent)

classification_name = "ENA"

path = Path(str(source_dir) + "/.16S_%s/%s.xlsx" % (classification_name, classification_name))
path_images = Path(str(source_dir) + "/.16S_%s/16S_%s_Dataset/" % (classification_name, classification_name))

path_origin = Path(str(source_dir) + "/.16S_%s/%s.csv" % (classification_name, classification_name))
path_destination = Path(str(source_dir) + "/Classification/.16S_%s/all_phylum/" % classification_name)

df = pd.read_excel(path)
image_names = os.listdir(path_images)


# Metodo per ordinare le immagini di un determinato Dataset. La funzione, restituisce il primo numero intero
# presente nella stringa s, permettendo quindi l'ordinamento del tipo -> CGR_RNA1, CGR_RNA2, CGR_RNA3, ...
# invece che l'ordinamento -> CGR_RNA1, CGR_RNA10, CGR_RNA100, ...
def sort_key(s):
    return int(re.findall(r'\d+', s)[0])


# Elimino la stringa '.png' alla fine di ogni file del Dataset
image_names = [file_name.replace('.png', '') for file_name in image_names]
sorted_image_names = sorted(image_names, key=sort_key)


# Copia immagini in un determinato percorso in base al nome del classificatore (superkingdom_class1, ..., phylum_class1, ...)
def copy_image(csv_filepath, dest_path, benchmark_id_csv, col_classificator):
    df_csv = pd.read_csv(csv_filepath)
    count = 0
    for index, row in df.iterrows():
        if row['Benchmark ID'] in df_csv[benchmark_id_csv].values:
            count = count + 1
            corresponding_row = df_csv.loc[df_csv[benchmark_id_csv] == row['Benchmark ID']]
            if not pd.isna(corresponding_row[benchmark_id_csv].values[0]) and not pd.isna(
                    corresponding_row[col_classificator].values[0]):
                print(row['Benchmark ID'], count, corresponding_row[col_classificator].values[0])
                src = os.path.join(path_images, f"{sorted_image_names[count - 1]}.png")
                dest = os.path.join(dest_path, corresponding_row[col_classificator].values[0],
                                    f"{corresponding_row[col_classificator].values[0]}_{count}.png")
                shutil.copy2(src, dest)


# Chiamate per la copia d'immagini
copy_image(path_origin, path_destination, 'Benchmark ID', 'Taxonomy.ENA.phylum')
