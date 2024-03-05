import os

path = "/Tesi/Classification/DATASET/IMMAGINI_DA_DIVIDERE/28S_SUPERKINGDOM_DATASET_90%/bacteria"
prefix = "bacteria_"

# Iteriamo sui file nella cartella
for filename in os.listdir(path):
    if filename.startswith("CGR_RNA") and filename.endswith(".png"):
        # Estraiamo il numero dal nome del file
        num = filename.split("_")[2].split(".")[0]

        # Rinominiamo il file
        new_filename = prefix + num + ".png"
        os.rename(os.path.join(path, filename), os.path.join(path, new_filename))

