# Tesi Francesco Chiocchi
Repository per la tesi.

La precedente versione, "_Studio della filogenesi di molecole di RNA attraverso l’uso della rappresentazione Chaos Game_" è disponibile al seguente 
[link](https://github.com/MartiniMichele/GP-Chiocchi-Filipponi-Martini).

L'applicazione serve per generare immagini CGR da file fasta contenenti sequenze di RNA
e poi classificarle tramite una rete neurale convoluzionale(CNN).
Il progetto è composto di 4 package:
- cgr: contiene i file relativi alla generazione delle immagini CGR
- cnn: contiene i file relativi alla CNN e alla gestione del dataset
- file_handling: contiene tutti i file relativi alla gestione dei file csv e
  l'organizzazione delle immagini
- hyperparameter_optimization: contiene tutti i file relativi alle tecniche 
  per la ricerca degli iperparametri ottimi
- main: il file main in cui è presente la CLI non è contenuto in nessun package

## ISTRUZIONE PER L'USO

una volta eseguito il file Main.py viene eseguita la CLI, a questo punto
si hanno 4 scelte:

## GENERAZIONE IMMAGINI CGR DA SEQUENZE RNA
- permette di generare immagini CGR dai file fasta
  - vanno inseriti il nome della cartella da cui prendere i fasta
    e la cartella dove salvare le immagini
  - il programma non è case sensitive
  - in caso la cartella di salvataggio non esista sarà creata in automatico

## CLASSIFICAZIONE IMMAGINI TRAMITE CNN
- permette di classificare le immagini appositamente organizzate in un dataset
  - va inserito il nome della cartella in cui è presente il dataset
    (creato con il file DatasetHandler.py)
  - va inserito il numero di label
  - viene restituito un riepilogo della rete creata
  - viene creata la cartella per il salvataggio dei modelli se non è presente
    - N.B. i modelli occupano molto spazio nel disco
  - viene chiesto all'utente se vuole salvare i grafici relativi alle
    prestazioni della rete, in caso positivo sono create anche le cartelle necessarie
  - infine viene effettuato il training

## GENERAZIONE IMMAGINI E CLASSIFICAZIONE
- l'unione dei due casi precedenti

## ESEGUIRE OTTIMIZZAZIONE DEGLI IPERPARAMETRI
- permette di eseguire l'ottimizzazione degli iperparametri presenti in uno spazio con la possibilità di scelta
di tre diverse tecniche: Grid Search, Random Search, Bayesian Optimization.
  - restituisce grafici per tutte le metriche utilizzate 
  - inoltre, crea un file .csv dove sono presenti tutti i parametri presi nelle varie valutazioni con relative metriche
