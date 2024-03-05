import os
import random
import shutil

source_dir = 'C:/Users/fchio/OneDrive/Desktop/Tesi/Tesi/Classification/DATASET/IMMAGINI_DA_DIVIDERE/28s_dataset_90%/eukaryota'
dest_dir_40 = 'C:/Users/fchio/OneDrive/Desktop/Tesi/Tesi/Classification/DATASET/28S_SUPERKINGDOM_DATASET_40%/eukaryota/'
dest_dir_50 = 'C:/Users/fchio/OneDrive/Desktop/Tesi/Tesi/Classification/DATASET/28S_SUPERKINGDOM_DATASET_50%/eukaryota/'

files = os.listdir(source_dir)

# Seleziona casualmente il 10% dei file
n_40 = int(len(files) * 0.4)
selected_files = random.sample(files, n_40)

# Copia i file selezionati nella cartella di destinazione dest_dir_40
for file_name in selected_files:
    source_path = os.path.join(source_dir, file_name)
    dest_path = os.path.join(dest_dir_40, file_name)
    shutil.copy(source_path, dest_path)

# Copia i file non selezionati nella cartella di destinazione dest_dir_50
for file_name in files:
    if file_name not in selected_files:
        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir_50, file_name)
        shutil.copy(source_path, dest_path)
