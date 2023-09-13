import numpy as np

from configurations import REPLICATIONS, NUM_BLOCKS, INTERVALS, STATISTICS, STOP, SAMPLING_INTERVAL


def print_size_3dim(array):
    """
       stampa le dimensioni di un array tridimensionale
       :param array_m: array tridimensionale
       :return: DIM_1 x DIM_2 x DIM_3
       """
    dimensions = []
    while isinstance(array, list):
        dimensions.append(len(array))
        array = array[0] if array else None

    return " x ".join(map(str, dimensions))


def print_size_2dim(array_m):
    """
    stampa le dimensioni di un array bidimensionale
    :param array_m: array bidimensionale
    :return: #righe x #colonne
    """
    num_rows = len(array_m)  # Numero di righe
    num_cols = len(array_m[0])  # Numero di colonne
    return "{} righe x {} colonne".format(num_rows, num_cols)


class CarNumberCenter:
    def __init__(self):
        self.car_arrivals = 0
        self.car_loss = 0
        self.prev_car_arrivals = 0
        self.prev_car_loss = 0
        self.car_compl = 0
        self.prev_compl_f = 0


class ArrivalTime:
    def __init__(self, time_value, is_old):
        self.time_value = time_value
        self.is_old_car = is_old
        self.prev = None
        self.next = None


class Arrivals:
    def __init__(self):
        self.head3 = None
        self.tail3 = None
        self.head4 = None
        self.tail4 = None

    def __str__(self):
        return f"Arrivals: head3 ={self.head3}, tail3={self.tail3}"


def scrivi_su_file(nome_file, contenuto, start):
    if start:
        with open(nome_file, "w") as file:
            print(contenuto, file=file)
    else:
        with open(nome_file, "a") as file:
            print(contenuto, file=file)

class ResultFinite:
    def __init__(self):
        self.nsim = np.empty((REPLICATIONS, NUM_BLOCKS, INTERVALS, STATISTICS), dtype=np.float64)
        self.samplingTime = np.empty((REPLICATIONS, NUM_BLOCKS, STATISTICS, int(STOP/SAMPLING_INTERVAL)), dtype=np.float64)
        self.numMedioServentiAttivi = np.empty((REPLICATIONS, NUM_BLOCKS, int(STOP/SAMPLING_INTERVAL)), dtype=np.float64)