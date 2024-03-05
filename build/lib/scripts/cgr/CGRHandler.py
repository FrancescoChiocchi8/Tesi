import os
from pathlib import Path
from Bio.Seq import MutableSeq

from Tesi.scripts.cgr import CGRepresentation

'''
Questa classe ha il compito di gestire la generazione di immagini CGR tramite l'utilizzo del codice della libreria
adkwazar/CGR
'''


class CGRHandler:
    source_path = Path(__file__).resolve()
    source_dir = Path(source_path.parent.parent.parent)

    # in futuro cambiare il costruttore per poter scegliere le modalità della CGR
    def __init__(self, CGR_type, outer_representation, rna_2structure, data_dir, save_dir):
        self.CGR_type = CGR_type
        self.outer_representation = outer_representation
        self.rna_2structure = rna_2structure
        self.sequence = None
        self.source = data_dir
        self.data_dir = data_dir
        self.save_dir = save_dir


    '''
    Inizializza le cartelle utili alla CNN
    '''
    def init_dirs(self):
        # cartelle urilizzate per l'esperimento
        self.data_dir = Path(str(self.source_dir) + "/FASTA/%s/" % self.data_dir)
        image_dir = Path(str(self.source_dir) + "/IMMAGINI_CGR")
        self.save_dir = Path(str(image_dir) + "/%s" % self.save_dir)
        #self.save_dir = Path(str(self.source_dir) + "/IMMAGINI_CGR/%s" % self.save_dir)
        os.chdir(image_dir)
        if os.path.isdir(self.save_dir) is False:
            os.makedirs(self.save_dir)

    '''
    Questo metodo scorre i file fasta nella cartella selezionata e ne estrae la sequenza
    '''

    def read_files(self):

        self.init_dirs()

        # Folder Path
        source_path = Path(__file__).resolve()
        source_dir = Path(source_path.parent.parent.parent)
        path = Path(str(source_dir) + "/FASTA/%s" % self.source)
        counter = 1
        # Change the directory
        os.chdir(path)

        # Read text File

        def read_fasta_file(file_path):
            with open(file_path, 'r') as f:
                line = f.readline()
                self.sequence = line.replace("\n", "")

        # iterate through all file
        for file in os.listdir():
            # Check whether file is in text format or not
            if file.endswith(".fasta"):
                file_path = f"{path}\{file}"

                # call read text file function
                read_fasta_file(file_path)
                print(file_path)
                self.generate_dataset(counter)
                counter += 1

    '''
    Questo metodo filtra le sequenze rimuovendo i caratteri che causano ambiguità
    '''

    def filter_sequence(self, sequence):

        filtered_sequence = MutableSeq(str(sequence))
        chars = ["Y", "N", "R", "M", "S", "W", "K", "D", "V", "B", "H", "P", "O"]

        if any(x in sequence for x in chars):

            char_count = self.count_char(sequence)

            for i in range(0, char_count):

                if filtered_sequence.find("Y") != -1:
                    filtered_sequence = filtered_sequence.replace("Y", "")

                elif filtered_sequence.find("N") != -1:
                    filtered_sequence = filtered_sequence.replace("N", "")

                elif filtered_sequence.find("R") != -1:
                    filtered_sequence = filtered_sequence.replace("R", "")

                elif filtered_sequence.find("M") != -1:
                    filtered_sequence = filtered_sequence.replace("M", "")

                elif filtered_sequence.find("S") != -1:
                    filtered_sequence = filtered_sequence.replace("S", "")

                elif filtered_sequence.find("W") != -1:
                    filtered_sequence = filtered_sequence.replace("W", "")

                elif filtered_sequence.find("D") != -1:
                    filtered_sequence = filtered_sequence.replace("D", "")

                elif filtered_sequence.find("B") != -1:
                    filtered_sequence = filtered_sequence.replace("B", "")

                elif filtered_sequence.find("H") != -1:
                    filtered_sequence = filtered_sequence.replace("H", "")

                elif filtered_sequence.find("V") != -1:
                    filtered_sequence = filtered_sequence.replace("V", "")

                elif filtered_sequence.find("K") != -1:
                    filtered_sequence = filtered_sequence.replace("K", "")

                elif filtered_sequence.find("P") != -1:
                    filtered_sequence = filtered_sequence.replace("P", "")

                elif filtered_sequence.find("O") != -1:
                    filtered_sequence = filtered_sequence.replace("O", "")

                else:
                    pass

        return filtered_sequence

    '''
    metodo di supporto a filter_sequence, ha il compito di contare il numero di caratteri ambigui presenti nella sequenza
    '''

    def count_char(self, sequence):
        tmp_sequence = MutableSeq(sequence)
        char_count = 0

        char_count += tmp_sequence.count("Y")
        char_count += tmp_sequence.count("N")
        char_count += tmp_sequence.count("R")
        char_count += tmp_sequence.count("M")
        char_count += tmp_sequence.count("S")
        char_count += tmp_sequence.count("W")
        char_count += tmp_sequence.count("K")
        char_count += tmp_sequence.count("D")
        char_count += tmp_sequence.count("H")
        char_count += tmp_sequence.count("V")
        char_count += tmp_sequence.count("B")
        char_count += tmp_sequence.count("P")
        char_count += tmp_sequence.count("O")

        return char_count

    '''Questo metodo ha il compito di generare le immagini CGR sfruttando la libreria adkwazar/CGR e filter_sequence 
    per filtrare le sequenze
    '''

    def generate_dataset(self, counter):

        bio_sequence = self.filter_sequence(self.sequence)
        print("SEQUENZA UTILIZZATA: " + bio_sequence)
        print("COUNTER: " + str(counter))
        drawer = CGRepresentation.CGR(bio_sequence, self.CGR_type, self.outer_representation, self.rna_2structure)
        drawer.representation()
        path = Path(str(self.save_dir) + "/CGR_RNA_" + str(counter) + ".png")
        drawer.plot(counter, path)
