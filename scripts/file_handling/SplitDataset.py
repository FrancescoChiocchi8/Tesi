import os
import random
import shutil

main_directory = "C:/Users/fchio/OneDrive/Desktop/Tesi/Tesi/Classification/DATASET/28S_SUPERKINGDOM_DATASET_50%"

subdirectories = ["archaea", "bacteria", "eukaryota"]

train_directory = os.path.join(main_directory, "train")
valid_directory = os.path.join(main_directory, "valid")
test_directory = os.path.join(main_directory, "test")

if not os.path.exists(train_directory):
    os.makedirs(train_directory)
if not os.path.exists(valid_directory):
    os.makedirs(valid_directory)
if not os.path.exists(test_directory):
    os.makedirs(test_directory)

for subdir in subdirectories:
    sub_directory_path = os.path.join(main_directory, subdir)
    images = os.listdir(sub_directory_path)
    num_images = len(images)
    random.shuffle(images)

    train_count = int(0.7 * num_images)
    valid_count = int(0.2 * num_images)
    test_count = num_images - train_count - valid_count

    # Si scorrono tutte le immagini
    for i, image in enumerate(images):
        image_path = os.path.join(sub_directory_path, image)
        if i < train_count:
            destination_folder = os.path.join(train_directory, subdir)
        elif i < train_count + valid_count:
            destination_folder = os.path.join(valid_directory, subdir)
        else:
            destination_folder = os.path.join(test_directory, subdir)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        shutil.copy(image_path, destination_folder)

