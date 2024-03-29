from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class CGR:
    def __init__(self, seq, seq_type, outer_representation, rna_2structure):
        self.seq = seq
        self.seq_type = seq_type
        self.outer_representation = outer_representation
        self.rna_2structure = rna_2structure

    def representation(self):
        if self.outer_representation is not False:
            coordinates = self.outer_representation
        else:
            if self.seq_type == "DNA":
                coordinates = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
            elif self.seq_type == "RNA" and self.rna_2structure:
                coordinates = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1], [0, 0]])
            elif self.seq_type == "RNA" and not self.rna_2structure:
                coordinates = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
            elif self.seq_type == "protein":
                hydrob = [0.62, 0.29, -0.9, -0.74, 1.19, 0.48, -0.4, 1.38, -1.5, 1.06, 0.64, -0.78, 0.12, -0.85, -2.53,
                          -0.18, -0.05, 1.08, 0.81, 0.26]
                hydrof = [-0.5, -1, 3, 3, -2.5, 0, -0.5, -1.8, 3, -1.8, -1.3, 0.2, 0, 0.2, 3, 0.3, -0.4, -1.5, -3.4,
                          -2.3]
                coordinates = np.array((hydrob, hydrof)).T
            else:
                return "Invalid type of sequence"

        if True:
            if self.seq_type == "DNA":
                residues = ["A", "C", "T", "G"]
                start_x, start_y = coordinates.mean(axis=0)
                sequence = np.array([start_x, start_y])
                for i in range(len(self.seq)):
                    x = 0.5 * (sequence[len(sequence) - 1] + coordinates[residues.index(self.seq[i])])
                    sequence = np.vstack((sequence, x))
                return sequence

            elif self.seq_type == "protein":
                residues = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V",
                            "W", "Y"]
                start_x, start_y = coordinates.mean(axis=0)
                sequence = np.array([start_x, start_y])
                for i in range(len(self.seq)):
                    x = 0.5 * (sequence[len(sequence) - 1] + coordinates[residues.index(self.seq[i])])
                    sequence = np.vstack((sequence, x))
                return sequence

            elif self.seq_type == "RNA" and not self.rna_2structure:
                residues = ["A", "C", "U", "G"]
                start_x, start_y = coordinates.mean(axis=0)
                sequence = np.array([start_x, start_y])
                for i in range(len(self.seq)):
                    x = 0.5 * (sequence[len(sequence) - 1] + coordinates[residues.index(self.seq[i])])
                    sequence = np.vstack((sequence, x))
                return sequence

            elif self.seq_type == "RNA" and self.rna_2structure:
                residues = ["A", "C", "U", "G"]
                start_x, start_y = 0, 0
                sequence = np.array([start_x, start_y])
                for i in range(len(self.seq)):
                    if self.rna_2structure[i] == "(" or self.rna_2structure[i] == ")":
                        x = 0.5 * (sequence[len(sequence) - 1] + coordinates[4])
                    else:
                        x = 0.5 * (sequence[len(sequence) - 1] + coordinates[residues.index(self.seq[i])])
                    sequence = np.vstack((sequence, x))
                return sequence

    def save_representation(self, name="seq.txt"):
        r = CGR.representation(self)
        f = open(name, "w")
        for elem in r:
            f.write(str(elem[0]) + ", " + str(elem[1]) + "\n")
        f.close()


    def plot(self, counter, path):
        r = CGR.representation(self)
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(r[:, 0], r[:, 1])

        source_path = Path(__file__).resolve()
        source_dir = Path(source_path.parent.parent.parent)
        plt.savefig(path)
        plt.close(fig)
        #plt.show()