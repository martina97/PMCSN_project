import csv
import os

from configurations import NUM_BLOCKS


def CreatePaths():
    file_paths = []

    # Creazione dei file in ordine
    for i in range(1, NUM_BLOCKS + 1):
        file_path = f"./data/finite/node{i}.csv"
        file =CreateCsvFiles(file_path)
        file_paths.append(file)

    for i in range(1, NUM_BLOCKS + 1):
        file_path = f"./data/infinite/node{i}.csv"
        file = CreateCsvFiles(file_path)
        file_paths.append(file)

    for i in range(1, NUM_BLOCKS + 1):
        file_path = f"./data/finite/samplingnode{i}.csv"
        file =CreateCsvFiles(file_path)
        file_paths.append(file)
    return file_paths


def CreateCsvFiles(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Crea il file .csv vuoto
    return open(file_path, 'w',newline="")

def create_statistic_files():
    print("sto increate stat file")
    """
    Nel codice Python, ho utilizzato la libreria os per gestire le operazioni sui file e sulle cartelle.
    La funzione create_statistic_files() replica il comportamento della funzione C originale createStatisticFiles().
    All'interno della funzione, utilizziamo os.path.exists() per verificare se le cartelle "./data", "./data/finite"
    e "./data/infinite" esistono. Se non esistono, vengono create utilizzando os.mkdir().
    Successivamente, utilizziamo una lista file_paths per memorizzare i percorsi dei file che vengono creati.
    Utilizziamo un ciclo for per generare i nomi dei file in base all'indice i e creare i file utilizzando la funzione open()
    con la modalità "w" (scrittura).
    Se si verifica un errore durante l'apertura del file, viene stampato un messaggio di errore e il programma viene terminato.
    :return: lista file_paths contenente i file aperti
    """

    if not os.path.exists("./data"):
        try:
            os.mkdir("./data")
        except OSError:
            exit(-4)

    if not os.path.exists("./data/finite"):
        try:
            os.mkdir("./data/finite")
        except OSError:
            exit(-5)

    if not os.path.exists("./data/infinite"):
        try:
            os.mkdir("./data/infinite")
        except OSError:
            exit(-6)

    file_paths = []
    for i in range(12):
        if i < 4:
            filename = "./data/finite/node{}.dat".format(i + 1)
        elif 4 <= i < 8:
            filename = "./data/infinite/node{}.dat".format(i - 3)
        else:
            filename = "./data/finite/samplingnode{}.dat".format(i - 7)

        try:
            fp = open(filename, "w")
        except IOError:
            print("ERRORE: impossibile creare il file {}.".format(filename))
            exit(-5)

        file_paths.append(fp)

    return file_paths
