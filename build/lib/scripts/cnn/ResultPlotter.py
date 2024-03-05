import matplotlib.pyplot as plt
import os
from pathlib import Path


class ResultPlotter:
    def __init__(self, history, dataset, model_filename):
        self.history = history
        self.dataset = dataset
        self.model_filename = model_filename
        self.save_fig_dir = ""

        '''
        variabili comuni per la creazione dei grafici, estratte dalla storia del metodo fit
        '''
        self.actual_epochs = len(history.history['loss'])
        self.xc = range(self.actual_epochs)

    source_path = Path(__file__).resolve()
    source_dir = Path(source_path.parent.parent.parent)

    '''
    Inizializza le cartelle dove salvare i grafici
    '''

    def init_graph_dir(self):

        graph_dir = Path(str(self.source_dir) + "/Grafici/")
        self.save_fig_dir = Path(str(self.source_dir) + "/Grafici/%s" % self.dataset)

        if os.path.isdir(self.save_fig_dir) is False:
            os.makedirs(self.save_fig_dir)
            print("SOTTOCARTELLA SALVATAGGIO GRAFICI CREATA")

        '''
        Controlla se la cartella dei grafici del modello esiste e se necessario la crea
        '''
        if os.path.isdir(os.path.abspath(os.path.join(self.save_fig_dir, "GRAPHS_" + self.model_filename))) is False:
            os.chdir(self.save_fig_dir)
            os.makedirs("GRAPHS_" + self.model_filename)
            print("CARTELLA SALVATAGGIO GRAFICI CREATA")

        '''
        aggiorna la variabile del salvataggio dei grafici e si sposta in quella cartella
        '''
        self.save_fig_dir = os.path.abspath(
            os.path.abspath(os.path.join(self.save_fig_dir, "GRAPHS_" + self.model_filename)))
        os.chdir(self.save_fig_dir)

    '''
    Disegna tutti i grafici relativi a test e validation e li salva su disco
    '''

    def drawall_graphs(self):
        self.init_graph_dir()
        self.loss_graph()
        self.accuracy_graph()
        self.precision_graph()
        self.recall_graph()
        self.auroc_graph()

    '''
    Crea il grafico della loss per train e validation
    '''

    def loss_graph(self):
        train_loss = self.history['loss']
        val_loss = self.history['val_loss']
        os.chdir(self.save_fig_dir)

        plt.figure()
        plt.grid()
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.xlabel("epoche")
        plt.ylabel("loss")
        plt.title("LOSS GRAPH")
        plt.legend(["train", "val"], loc="upper left")
        plt.savefig("LOSS_GRAPH.png")
        plt.show()

    '''
    Crea il grafico della accuracy per train e validation
    '''

    def accuracy_graph(self):
        train_acc = self.history['accuracy']
        val_acc = self.history['val_accuracy']
        os.chdir(self.save_fig_dir)

        plt.grid()
        plt.plot(train_acc)
        plt.plot(val_acc)
        plt.xlabel("epoche")
        plt.ylabel("accuracy")
        plt.title("ACCURACY GRAPH")
        plt.legend(["train", "val"], loc="upper left")
        plt.savefig("ACCURACY_GRAPH.png")
        plt.show()

    '''
    Crea il grafico della precision per train e validation
    '''

    def precision_graph(self):
        train_precision = self.history['precision']
        val_precision = self.history['val_precision']
        os.chdir(self.save_fig_dir)

        plt.grid()
        plt.plot(train_precision)
        plt.plot(val_precision)
        plt.xlabel("epoche")
        plt.ylabel("precision")
        plt.title("PRECISION GRAPH")
        plt.legend(["train", "val"], loc="upper left")
        plt.savefig("PRECISION_GRAPH.png")
        plt.show()

    '''
    Crea il grafico del recall per train e validation
    '''

    def recall_graph(self):
        train_recall = self.history['recall']
        val_recall = self.history['val_recall']
        os.chdir(self.save_fig_dir)

        plt.grid()
        plt.plot(train_recall)
        plt.plot(val_recall)
        plt.xlabel("epoche")
        plt.ylabel("recall")
        plt.title("RECALL GRAPH")
        plt.legend(["train", "val"], loc="upper left")
        plt.savefig("RECALL_GRAPH.png")
        plt.show()

    '''
    Crea il grafico dell'AUC per train e validation
    '''

    def auroc_graph(self):
        train_auc = self.history['auc']
        val_auc = self.history['val_auc']
        os.chdir(self.save_fig_dir)

        plt.grid()
        plt.plot(train_auc)
        plt.plot(val_auc)
        plt.xlabel("epoche")
        plt.ylabel("auc")
        plt.title("AUC GRAPH")
        plt.legend(["train", "val"], loc="upper left")
        plt.savefig("AUC_GRAPH.png")
        plt.show()

    '''
    Disegna tutti i grafici del test e li salva su disco
    '''

    def drawall_test_graphs(self):
        self.init_graph_dir()
        self.test_loss_graph()
        self.test_accuracy_graph()
        self.test_precision_graph()
        self.test_recall_graph()
        self.test_auroc_graph()
        self.test_f1_graph()

    '''
    Crea il grafico della loss per il test
    '''

    def test_loss_graph(self):
        test_loss = self.history[0]
        os.chdir(self.save_fig_dir)

        plt.grid()
        plt.scatter([1], test_loss, zorder=5)
        plt.ylabel("loss")
        plt.title("LOSS GRAPH")
        plt.savefig("LOSS(test)_GRAPH.png")
        plt.show()

    '''
    Crea il grafico della accuracy per il test
    '''

    def test_accuracy_graph(self):
        test_accuracy = self.history[1]
        os.chdir(self.save_fig_dir)

        plt.grid()
        plt.scatter([1], test_accuracy, zorder=5)
        plt.ylabel("accuracy")
        plt.title("ACCURACY GRAPH")
        plt.savefig("ACCURACY(test)_GRAPH.png")
        plt.show()

    '''
    Crea il grafico della precision per il test
    '''

    def test_precision_graph(self):
        test_precision = self.history[2]
        os.chdir(self.save_fig_dir)

        plt.grid()
        plt.scatter([1], test_precision, zorder=5)
        plt.ylabel("precision")
        plt.title("PRECISION GRAPH")
        plt.savefig("PRECISION(test)_GRAPH.png")
        plt.show()

    '''
    Crea il grafico del recall per il test
    '''

    def test_recall_graph(self):
        test_recall = self.history[3]
        os.chdir(self.save_fig_dir)

        plt.grid()
        plt.scatter([1], test_recall, zorder=5)
        plt.ylabel("recall")
        plt.title("RECALL GRAPH")
        plt.savefig("RECALL(test)_GRAPH.png")
        plt.show()

    '''
    Crea il grafico dell'AUC per il test
    '''

    def test_auroc_graph(self):
        test_auroc = self.history[4]
        os.chdir(self.save_fig_dir)

        plt.grid()
        plt.scatter([1], test_auroc, zorder=5)
        plt.ylabel("auc metric")
        plt.title("AUC GRAPH")
        plt.savefig("AUC(test)_GRAPH.png")
        plt.show()

    '''
    Crea il grafico dell'F1 per il test
    '''

    def test_f1_graph(self):
        test_f1 = 2 * (self.history[3] * self.history[2]) / (self.history[3] + self.history[2])
        os.chdir(self.save_fig_dir)

        plt.grid()
        plt.scatter([1], test_f1, zorder=5)
        plt.ylabel("F1 metric")
        plt.title("F1 GRAPH")
        plt.savefig("F1(test)_GRAPH.png")
        plt.show()
