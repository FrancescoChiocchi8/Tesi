import math
import os
from pathlib import Path
import shutil
import random
import glob

source_path = Path(__file__).resolve()
source_dir = Path(source_path.parent.parent.parent)

database = "Superkingdom"
livello = "16S"
completo = "%s_%s" % (database, livello)
sottoclasse = "Eukaryota"

dataset_dir = Path(str(source_dir) + "/Classification/%s_DATASET/" % completo.upper())
data_dir = Path(str(source_dir) + "/Classification/DATASET/IMMAGINI DA DIVIDERE/28s_dataset_10%/archaea" )
train_dir = Path(str(source_dir) + "/Classification/DATASET/IMMAGINI DA DIVIDERE/28s_dataset_10%/archaea/train/")
valid_dir = Path(str(source_dir) + "/Classification/DATASET/IMMAGINI DA DIVIDERE/28s_dataset_10%/archaea/valid/")
test_dir = Path(str(source_dir) + "/Classification/DATASET/IMMAGINI DA DIVIDERE/28s_dataset_10%/archaea/test/")

# Controlla se le cartelle esistono, altrimenti creale
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

filelist = os.listdir(data_dir)
sub_dir_path = []
'''
cerca tutte le sottocartelle relative al phylum(o superkingdom) e poi itera al loro interno per spostare i file secondo
la proporzione 70/20/10 nelle rispettive cartelle train/valid/test
'''
if len(os.listdir(train_dir)) == 0:
    for x in filelist:
        if x != "DATASET":
            sub_dir_path.append(data_dir + x + '/')
    for x in sub_dir_path:
        print(str(x))

    for i in sub_dir_path:
        filelist2 = os.listdir(i)
        train_part = math.floor(len(filelist2) * 0.7)
        valid_part = math.floor(len(filelist2) * 0.2)
        test_part = math.floor(len(filelist2) * 0.1)
        print("phylum: " + str(i))
        print("70: " + str(train_part))
        print("20: " + str(valid_part))
        print("10: " + str(test_part))

        os.chdir(i)
        for j in random.sample(glob.glob('*.png'), train_part):
            source_path = i + j
            shutil.move(source_path, train_dir)
        for j in random.sample(glob.glob('*.png'), valid_part):
            source_path = i + j
            shutil.move(source_path, valid_dir)
        for j in random.sample(glob.glob('*.png'), test_part):
            source_path = i + j
            shutil.move(source_path, test_dir)


'''
controlla se è presente la cartella "Actinobacteria"(può variare) comune a tutte e 3 le cartelle(train, valid, test) e quindi sicuramente presente
se si va avanti e crea le sottocartelle(labels) con i nomi dei phylum presenti.
Per creare le cartelle dei labels vengono letti tutti i file .png, viene estratta la prima parte del nome relativa al phylum
e poi ci viene spostato il file
'''
if os.path.isdir(os.path.abspath(os.path.join(train_dir, "archaea/"))) is False:
    os.chdir(train_dir)
    for file in os.listdir():
        if file.endswith('.png'):
            phylum = file.split('_')[0]
            if os.path.isdir(phylum) is False:
                os.makedirs(phylum)
            dir_path = train_dir + phylum
            shutil.move(file, dir_path)

if os.path.isdir(os.path.abspath(os.path.join(valid_dir, "archaea/"))) is False:
    os.chdir(valid_dir)
    for file in os.listdir():
        if file.endswith('.png'):
            phylum = file.split('_')[0]
            if os.path.isdir(phylum) is False:
                os.makedirs(phylum)
            dir_path = valid_dir + phylum
            shutil.move(file, dir_path)

if os.path.isdir(os.path.abspath(os.path.join(test_dir, "archaea/"))) is False:
    os.chdir(test_dir)
    for file in os.listdir():
        if file.endswith('.png'):
            phylum = file.split('_')[0]
            if os.path.isdir(phylum) is False:
                os.makedirs(phylum)
            dir_path = test_dir + phylum
            shutil.move(file, dir_path)