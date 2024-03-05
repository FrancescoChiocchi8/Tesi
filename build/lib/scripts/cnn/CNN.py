from hyperopt import STATUS_OK
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, History
import os
from pathlib import Path

class CNN:

    def __init__(self, dataset, model_mk, batch_size, epochs, fl_filter, ol_units, n_dropout, drop_value, n_layer, lr, patience):
        # variabili per creazione e salvataggio del modello
        self.dataset = dataset
        self.model_mk = model_mk
        self.batch_size = batch_size
        self.epochs = epochs
        self.fl_filter = fl_filter
        self.ol_units = ol_units
        self.n_dropout = n_dropout
        self.drop_value = drop_value
        self.n_layer = n_layer
        self.lr = lr
        self.patience = patience

        # cartelle utili per la CNN
        self.model_filename = ""
        self.train_dir = ""
        self.valid_dir = ""
        self.test_dir = ""

    source_path = Path(__file__).resolve()
    source_dir = Path(source_path.parent.parent.parent)
    # indica la dimensione delle immagini date in input alla CNN
    img_width, img_height = 150, 150

    '''
    Inizializza le cartelle utili alla CNN
    '''
    def init_dirs(self):
        # cartelle utilizzate per l'esperimento
        data_dir = Path(str(self.source_dir) + "/Classification/DATASET/%s/" % self.dataset)
        self.train_dir = Path(str(self.source_dir) + "/Classification/DATASET/%s/train/" % self.dataset)
        self.valid_dir = Path(str(self.source_dir) + "/Classification/DATASET/%s/valid/" % self.dataset)
        self.test_dir = Path(str(self.source_dir) + "/Classification/DATASET/%s/test/" % self.dataset)
        '''
        models_dir = Path(str(self.source_dir) + "/CNN_models/")
        save_model_dir = Path(str(self.source_dir) + "/CNN_models/%s/" % self.dataset)

        self.model_filename = "%s_model_%s_LR%s_batch%s_%sDropout(0.5)_%slayer(FL=%s)_epochs(%s)" % (
            self.dataset,
            self.model_mk,
            self.lr,
            self.batch_size,
            self.n_dropout,
            self.n_layer,
            self.fl_filter,
            self.epochs)'''
        # Chiama il metodo create_model_dir per creare effettivamente la directory models_dir e la sottodirectory, se non
        # esistono ancora.
        #self.create_model_dir(save_model_dir, self.model_filename)

    '''
    Crea la cartella dove salvare il modello di CNN
    
    def create_model_dir(self, save_model_dir, model_filename):
        if os.path.isdir(save_model_dir) is False:
            os.chdir(self.source_dir)
            os.makedirs(save_model_dir)
            print("\nCARTELLA SALVATAGGIO MODELLO CREATA")

        self.model_save_name = os.path.abspath(os.path.join(save_model_dir, model_filename + ".h5"))
    '''
    '''
    Metodo che crea i callback per la funzione fit, i callback implementati sono:
    early_stopping: ferma prematuramente il training se non ci sono progressi per un numero specifico di epoche(patience)
    reduce_lr: riduce il learning rate in caso di appiattimento della curva di apprendimento
    model_checkpoint: salva automaticamente il modello con il valore indicato migliore(monitor)
    '''
    def create_callbacks(self):
        early_stopping = EarlyStopping(patience=self.patience, monitor='val_loss', verbose=1)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_lr=0.001, patience=self.patience, mode='min', verbose=1)

        #model_checkpoint = ModelCheckpoint(monitor='val_loss', filepath=self.model_save_name, save_best_only=True, verbose=1)
        hist = History()

        callbacks = [early_stopping, reduce_lr, hist]# model_checkpoint, hist]

        return callbacks

    '''
    Metodo che crea il modello della rete neurale
    '''
    def create_model(self):
        last_n_filter = self.fl_filter
        model = Sequential()
        for i in range(self.n_layer):
            if i == 0:
                model.add(Conv2D(filters=last_n_filter, activation='relu', kernel_size=(3, 3),
                                 input_shape=(self.img_width, self.img_height, 3)))
                model.add(MaxPooling2D(pool_size=(2, 2)))
            else:
                last_n_filter = last_n_filter * 2
                model.add(Conv2D(filters=last_n_filter, activation='relu', kernel_size=(3, 3),
                                 input_shape=(self.img_width, self.img_height, 3)))
                model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(units=last_n_filter * 4, activation='relu', ))

        for i in range(self.n_dropout):
            model.add(Dropout(self.drop_value))

        model.add(Dense(units=self.ol_units, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=self.lr),
                      metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

        self.init_dirs()
        return model

    '''
    I batch per train/valid/test sono creati con l'ausilio di ImageDataGenerator che permette di applicare
    delle modifiche alle immagini. Nel nostro caso per il train viene applicata la data augmentation
    '''
    def datagen(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        validation_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(self.train_dir, target_size=(self.img_width, self.img_height),
                                                            batch_size=self.batch_size, class_mode='categorical')

        validation_generator = validation_datagen.flow_from_directory(self.valid_dir, target_size=(self.img_width, self.img_height),
                                                                      batch_size=self.batch_size, class_mode='categorical')

        test_generator = test_datagen.flow_from_directory(self.test_dir, target_size=(self.img_width, self.img_height),
                                                          batch_size=self.batch_size, class_mode='categorical')

        result_list = [train_generator, validation_generator, test_generator]
        return result_list

    '''
    Metodo fit per istruire la rete neurale, i risultati vengono salvati nella variabile history
    per eventuali utilizzi futuri(i.e. grafici)
    '''
    def train(self, model, train_generator, validation_generator):
        history = model.fit(train_generator, steps_per_epoch=train_generator.n // self.batch_size, epochs=self.epochs,
                            validation_data=validation_generator,
                            validation_steps=validation_generator.n // self.batch_size,
                            callbacks=self.create_callbacks(), verbose=2)
        return history

    def create_and_train(self):
        model = self.create_model()
        datagen_list = self.datagen()
        self.train(model, datagen_list[0], datagen_list[1])

    '''
    Al termine del training viene chiamato il metodo evaluate sul batch di test per verificare il training della rete
    vengono stampate le metriche e viene calcolata la metrica F1
    '''
    def test_evaluate(self, model, test_generator):
        score = model.evaluate(test_generator, verbose=2)
        return score
