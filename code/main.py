import csv
import locale
import math
import sys
from time import sleep

import numpy as np

from DES_Python.rngs import plantSeeds
from DES_Python.rvms import idfStudent
from DES_Python.ssq3 import track, time, START
from configurations import NUM_TIME_SLOT, num_servers, NUM_BLOCKS, STATISTICS, K, SAMPLING_INTERVAL, B, STOP, \
    Q_INTERIOR, CAR_TO_WAIT_H, CAR_TO_WAIT_T, REPLICATIONS, \
    INTERVALS, LOC, HAND_WASH_PRICE, T, CLEANSER_COST, CAR_WASHER_COST, TOUCHLESS_WASH_PRICE, TUNNEL_COST, CAR_WAX_COST, \
    INTERIOR_WASH_PRICE, INTERIOR_MACHINE_COST
from dataClasses.calendar import Calendar, getOldCarArrivalTouchless, \
    getNewCarArrivalHandWash, getOldCarArrivalHandWash, NextAbandon, NextCompletion, GetServiceHandWash, Job, \
    getAbandonHandWash, GetServiceTouchless, getAbandonTouchless, GetServicePolishing, GetServiceInteriorWash, \
    GetInteriorWashProbability
from dataClasses.stateVariable import StateVariables
from fileWriter import CreatePaths, create_statistic_files
from plotter import plotFiniteHorizon
from utilities import CarNumberCenter, Arrivals, scrivi_su_file, ArrivalTime, ResultFinite

# variabili iniziali
file_paths = []
file_path2 = []
config_matrix = [[0] * NUM_BLOCKS for _ in range(NUM_TIME_SLOT)]  # array 6 righe x 5 colonne (6 è numero di time slot)
interTime = 0
server_conf = []  # vettore con elementi pari al numero di blocchi, e per valore ha il numero di server per blocco
server_configuration_max = []
statistics = ["Utilizzazione", "Popolazione media in coda", "Popolazione media nel centro", "Tempo medio di servizio",
              "Tempo medio di attesa", "Tempo medio di risposta", "Tempo medio di interarrivo",
              "Numero arrivi macchine", "Numero abbandoni", "Probabilità abbandono"]

# CLASSI
calendar = Calendar()  # sarebbe la lista degli eventi (Event List)
clock = time()
area = [track() for _ in range(NUM_BLOCKS)]  # track in python sarebbe struct area in C
# a = lista che contiene 5 oggetti istanziati dalla classe track di ssq3.py
areaSampling = [track() for _ in range(NUM_BLOCKS)]
state_variable = []
centers_num = [CarNumberCenter() for _ in range(NUM_BLOCKS)]  # lista al contenente 5 oggetti ArrivalLoss istanziati.
arrivals = Arrivals()

# variabili utili per simulazione
old_car_arrival_state_handWash = True
new_car_arrival_state_handWash = True
old_car_arrival_state_touchless = True

type_simulation_horizon = None


def main():
    global type_simulation_horizon
    type_simulation_horizon = input("Scegliere tra FINITE e INFINITE:\n")

    if len(sys.argv) < 2:
        print("INSERIRE UN NUMERO DA 1 a 2 RELATIVO ALLA FASCIA ORARIA.")
        sys.exit(-1)

    time_slot = int(sys.argv[1])

    for i in range(NUM_TIME_SLOT):
        if len(num_servers[i]) != NUM_BLOCKS:
            print("ERRORE: inserire il numero di blocchi corretto!")
            sys.exit()
        for j in range(NUM_BLOCKS):
            config_matrix[i][j] = num_servers[i][j]



    # creo files
    global file_paths, file_path2
    file_paths = CreatePaths()
    file_path2 = create_statistic_files()

    # ********************************** SIMULAZIONE A ORIZZONTE FINITO ********************************** #
    if type_simulation_horizon.upper() == "FINITE":
        result_finite = finite_horizon_simulation()
        ConfidenceIntervalsCalculationPerDay(result_finite)




    # ********************************** SIMULAZIONE A ORIZZONTE INFINITO ********************************** #
    elif type_simulation_horizon.upper() == "INFINITE":
        global interTime
        if time_slot == 1:
            interTime = 240.0
        elif time_slot == 2:
            interTime = 120.0
        else:
            print("ERRORE: la fascia oraria specificata non è valida. Fornire un valore intero compreso tra 1 e 2.")
            exit(-7)

        global server_conf, server_configuration_max
        server_conf = config_matrix[time_slot - 1]
        server_configuration_max = server_conf  # nella simulazione a orizzonte infinito non serve

        result_infinite = infinite_horizon_sim()
        locale.setlocale(locale.LC_ALL, 'it_IT.UTF-8')

        for center in range(NUM_BLOCKS):
            fp = file_paths[center + NUM_BLOCKS]
            writer = csv.writer(fp, delimiter=";")
            header = ["Utilizzazione", "Popolazione media in coda", "Popolazione media nel centro",
                      "Tempo medio di servizio",
                      "Tempo medio di attesa", "Tempo medio di risposta", "Tempo medio di interarrivo",
                      "Numero arrivi macchine",
                      "Numero abbandoni", "Probabilità abbandono"]
            writer.writerow(header)

            for row in zip(*result_infinite[center]):
                if all(value == "" for value in row):
                    continue
                row_formatted = [locale.format_string("%.15f", value) for value in row]
                writer.writerow(row_formatted)


def GainCalculationPerBlock(center, loss, arrivals):
    """
    Profitto mensile del centro. Dipende quindi dal centro che sto considerando. Poi devo fare la somma.
    """
    n = arrivals - loss
    profit = 0
    if center == 0:  # HAND WASH
        profit = n * HAND_WASH_PRICE * T - (CLEANSER_COST + 2 * CAR_WASHER_COST) * server_conf[0]
        print(f"profit centro HAND WASH = {profit}")

    elif center == 1:  # TOUCHLESS
        profit = n * TOUCHLESS_WASH_PRICE * T - (TUNNEL_COST + CLEANSER_COST) * server_conf[1]
        print(f"profit centro TOUCHLESS = {profit}")

    elif center == 2:  # POLISHING
        profit = - (CAR_WAX_COST + 2 * CAR_WASHER_COST) * server_conf[2]  # non ho profitto, ma solo in meno
        print(f"profit centro POLISHING = {profit}")

    elif center == 3:  # INTERIOR WASH
        profit = n * INTERIOR_WASH_PRICE * T - INTERIOR_MACHINE_COST * server_conf[3]
        print(f"profit centro INTERIOR WASH = {profit}")

    return profit


def ConfidenceIntervalsCalculationPerDay(result):
    diff = 0
    mean = np.zeros((NUM_BLOCKS, STATISTICS), dtype=np.float64)
    sum = np.zeros((NUM_BLOCKS, STATISTICS), dtype=np.float64)
    w = np.zeros((NUM_BLOCKS, STATISTICS), dtype=np.float64)

    for center in range(NUM_BLOCKS):
        for stat in range(STATISTICS):
            mean[center][stat] = 0.0
            sum[center][stat] = 0.0
            w[center][stat] = 0.0

    for centro in range(NUM_BLOCKS):
        for stat in range(STATISTICS):
            for replica in range(REPLICATIONS):
                diff = result.samplingTime[replica][centro][stat][107] - mean[centro][stat]
                sum[centro][stat] += diff * diff * ((replica + 1) - 1.0) / (replica + 1)
                mean[centro][stat] += diff / (replica + 1)

    import math

    r = REPLICATIONS
    u, t, stdv = 0.0, 0.0, 0.0

    for centro in range(NUM_BLOCKS):
        for stat in range(STATISTICS):
            stdv = math.sqrt(sum[centro][stat] / r)
            u = 1.0 - 0.5 * (1.0 - LOC)  # interval parameter
            t = idfStudent(r - 1, u)  # critical value of t
            w[centro][stat] = t * stdv / math.sqrt(r - 1)  # interval half width

            print(
                f"INTERVALLO-FINITO--{statistics[stat]}-centro-{centro} ------ {mean[centro][stat]:.6f} +/- {w[centro][stat]:.6f}")
            # If you're using Python 2.x, use: print("INTERVALLO-FINITO-statistica-%d-centro-%d ------ %10.6f +/- %6.6f" % (stat, centro, mean[centro][stat], w[centro][stat]))
            # If you're using Python 3.6+, you can use f-strings: print(f"INTERVALLO-FINITO-statistica-{stat}-centro-{centro} ------ {mean[centro][stat]:.6f} +/- {w[centro][stat]:.6f}")


def IsEmptySystem():
    if state_variable[0].number_in_center > 0 or state_variable[1].number_in_center > 0 or \
            state_variable[2].number_in_center > 0 or state_variable[3].number_in_center > 0:
        return False

    for i in range(server_configuration_max[0]):
        if state_variable[0].server_state[i] != 0 and state_variable[0].server_state[i] != -3:
            return False

    for i in range(server_configuration_max[1]):
        if state_variable[1].server_state[i] != 0 and state_variable[1].server_state[i] != -3:
            return False

    for i in range(server_configuration_max[2]):
        if state_variable[2].server_state[i] != 0 and state_variable[2].server_state[i] != -3:
            return False

    for i in range(server_configuration_max[3]):
        if state_variable[3].server_state[i] != 0 and state_variable[3].server_state[i] != -3:
            return False

    if arrivals.head3 is not None or arrivals.head4 is not None:
        return False

    return True


def AveragesCalculationPerReplica(interval, i, nsim, m):
    diff = 0.0

    if interval == 1:
        diff = 0.0
    elif interval == 2:
        diff = 16200.0

    # per ogni centro calcolo le medie
    for center in range(NUM_BLOCKS):

        # ************************* CALCOLO UTILIZZAZIONE ************************* #
        utilization = area[center].service / ((clock.current - diff) * m[center]) if (clock.current - diff) * m[
            center] != 0 else 0
        nsim[i][center][interval - 1][0] = utilization

        # ************************* CALCOLO POPOLAZIONE MEDIA NELLA CODA ************************* #
        queue_population = area[center].queue / (clock.current - diff) if clock.current - diff != 0 else 0
        nsim[i][center][interval - 1][1] = queue_population

        # ************************* CALCOLO POPOLAZIONE MEDIA NEL CENTRO ************************* #
        center_population = area[center].node / (clock.current - diff) if clock.current - diff != 0 else 0
        nsim[i][center][interval - 1][2] = center_population

        # ************************* CALCOLO TEMPO MEDIO DI SERVIZIO ************************* #
        service_time = area[center].service / (centers_num[center].car_compl - centers_num[center].prev_compl_f) \
            if centers_num[center].car_compl - centers_num[center].prev_compl_f != 0 else 0
        nsim[i][center][interval - 1][3] = service_time

        # ************************* CALCOLO TEMPO MEDIO DI ATTESA ************************* #
        wait_time = area[center].queue / (centers_num[center].car_compl - centers_num[center].prev_compl_f) \
            if centers_num[center].car_compl - centers_num[center].prev_compl_f != 0 else 0
        nsim[i][center][interval - 1][4] = wait_time

        # ************************* CALCOLO TEMPO MEDIO DI RISPOSTA ************************* #
        response_time = area[center].node / (centers_num[center].car_compl - centers_num[center].prev_compl_f) \
            if centers_num[center].car_compl - centers_num[center].prev_compl_f != 0 else 0
        nsim[i][center][interval - 1][5] = response_time

        # ************************* CALCOLO TEMPO MEDIO DI INTERARRIVO ************************* #
        interarrival_time = (clock.last[center] - diff) / (
                centers_num[center].car_arrivals - centers_num[center].prev_car_arrivals) \
            if centers_num[center].car_arrivals - centers_num[center].prev_car_arrivals != 0 else 0
        nsim[i][center][interval - 1][6] = interarrival_time

        # ************************* NUMERO ARRIVI MACCHINE ************************* #
        nsim[i][center][interval - 1][7] = (centers_num[center].car_arrivals - centers_num[center].prev_car_arrivals)

        # ************************* NUMERO ABBANDONI ************************* #
        nsim[i][center][interval - 1][8] = (centers_num[center].car_loss - centers_num[center].prev_car_loss)

        # ************************* PROBABILITA' DI ABBANDONO ************************* #
        if ((centers_num[center].car_arrivals - centers_num[center].prev_car_arrivals) + (
                centers_num[center].prev_car_arrivals - centers_num[center].prev_compl_f - centers_num[
            center].prev_car_loss) == 0):
            nsim[i][center][interval - 1][9] = 0
        else:
            nsim[i][center][interval - 1][9] = (centers_num[center].car_loss - centers_num[center].prev_car_loss) / (
                    (centers_num[center].car_arrivals - centers_num[center].prev_car_arrivals) + (
                    centers_num[center].prev_car_arrivals -
                    centers_num[center].prev_compl_f - centers_num[center].prev_car_loss))


def Reconfig(center_num_old, difference, index):
    if state_variable[index].number_in_center - center_num_old > 0:
        number_in_center = min(difference, state_variable[index].number_in_center - center_num_old)

        for i in range(number_in_center):

            if index == 0:  # centro HAND-WASH
                calendar.completions_handWash[center_num_old + i] = GetServiceHandWash(clock.current)
                state_variable[index].server_state[center_num_old + i] = 1

                if calendar.head_handWash is not None:
                    # Removal of the head node from the abandonment list
                    to_delete = calendar.head_handWash
                    if to_delete.next is None:
                        calendar.head_handWash = None
                        calendar.tail_handWash = None
                    else:
                        calendar.head_handWash = to_delete.next
                        calendar.head_handWash.prev = None

            if index == 1:  # centro TOUCHLESS
                calendar.completions_touchless[center_num_old + i] = GetServiceTouchless(clock.current)
                state_variable[index].server_state[center_num_old + i] = 1

                if calendar.head_touchless is not None:
                    # Removal of the head node from the abandonment list
                    to_delete = calendar.head_touchless
                    if to_delete.next is None:
                        calendar.head_touchless = None
                        calendar.tail_touchless = None
                    else:
                        calendar.head_touchless = to_delete.next
                        calendar.head_touchless.prev = None

            if index == 2:  # centro POLISHING:
                calendar.completions_polishing[center_num_old + i] = GetServicePolishing(clock.current)
                state_variable[index].server_state[center_num_old + i] = 1


def ServerConfigurationBetweenTimeSlot(old_configuration, new_configuration):
    """
    se nello slot 2 ho piu server nel centro rispetto allo slot 1, pongo a 0 lo stato
    dei server in più.
    """

    for c in range(3):  # scorro i centri
        if new_configuration[0] - old_configuration[c] > 0:
            for i in range(old_configuration[c], new_configuration[c]):
                state_variable[c].server_state[i] = 0
            Reconfig(old_configuration[c], new_configuration[c] - old_configuration[c], c)
        elif new_configuration[c] - old_configuration[c] < 0:
            for i in range(new_configuration[c], old_configuration[c]):
                if state_variable[c].server_state[i] == 0:
                    state_variable[c].server_state[i] = -3
                elif state_variable[c].server_state[i] == 1:
                    state_variable[c].server_state[i] = -1
                elif state_variable[c].server_state[i] == 2:
                    state_variable[c].server_state[i] = -2


def SamplingAverageCalculation(count, replica, samplingTime, interval, numMedioServentiAttivi):
    interval_size = [16200.0, 16200.0]
    diff = [0.0, 16200.0]

    for center in range(NUM_BLOCKS):
        denominatore = sum(config_matrix[i][center] * interval_size[i] for i in range(interval - 1))
        denominatore += (clock.current - diff[interval - 1]) * config_matrix[interval - 1][center]
        numMedioServentiAttivi[replica][center][count] = denominatore

        # ************************* UTILIZZAZIONE ************************* #
        samplingTime[replica][center][0][count] = areaSampling[
                                                      center].service / denominatore if denominatore != 0 else 0

        # ************************* POPOLAZIONE MEDIA NELLA CODA ************************* #
        samplingTime[replica][center][1][count] = areaSampling[
                                                      center].queue / clock.current if clock.current != 0 else 0

        # ************************* POPOLAZIONE MEDIA NEL CENTRO ************************* #
        samplingTime[replica][center][2][count] = areaSampling[center].node / clock.current if clock.current != 0 else 0

        # ************************* TEMPO MEDIO DI SERVIZIO ************************* #
        samplingTime[replica][center][3][count] = areaSampling[center].service / centers_num[center].car_compl \
            if centers_num[center].car_compl != 0 else 0

        # ************************* TEMPO MEDIO DI ATTESA ************************* #
        samplingTime[replica][center][4][count] = areaSampling[center].queue / centers_num[center].car_compl \
            if centers_num[center].car_compl != 0 else 0

        # ************************* TEMPO MEDIO DI RISPOSTA ************************* #
        samplingTime[replica][center][5][count] = areaSampling[center].node / centers_num[center].car_compl \
            if centers_num[center].car_compl != 0 else 0

        # ************************* TEMPO MEDIO DI INTERARRIVO ************************* #
        samplingTime[replica][center][6][count] = clock.last[center] / centers_num[center].car_arrivals \
            if centers_num[center].car_arrivals != 0 else 0

        # ************************* NUMERO ARRIVI MACCHINE ************************* #
        samplingTime[replica][center][7][count] = float(centers_num[center].car_arrivals)

        # ************************* NUMERO ABBANDONI ************************* #
        samplingTime[replica][center][8][count] = float(centers_num[center].car_loss)

        # ************************* PROBABILITA' DI ABBANDONO ************************* #
        samplingTime[replica][center][9][count] = float(centers_num[center].car_loss) / float(
            centers_num[center].car_arrivals) \
            if centers_num[center].car_arrivals != 0 else 0


def simulation(config_matrix, i, nsim, samplingTime, numMedioServentiAttivi):
    global interTime

    interval = 0
    count_sampling = 0
    m = config_matrix[0]

    while old_car_arrival_state_handWash or new_car_arrival_state_handWash or \
            old_car_arrival_state_touchless or not IsEmptySystem():

        clock.next = GetMinTime(server_configuration_max)
        UpdateTrack(server_configuration_max)
        clock.current = clock.next

        next_abandon_handWash = GetMinAbandon(calendar.head_handWash)
        next_abandon_touchless = GetMinAbandon(calendar.head_touchless)

        next_completion_handWash = GetMinCompletion(server_configuration_max[0], state_variable[0].server_state, 1)
        next_completion_touchless = GetMinCompletion(server_configuration_max[1], state_variable[1].server_state, 2)
        next_completion_polishing = GetMinCompletion(server_configuration_max[2], state_variable[2].server_state, 3)
        next_completion_interiorWash = GetMinCompletion(server_configuration_max[3], state_variable[3].server_state, 4)

        if clock.current == calendar.change_time_slot:
            scrivi_su_file("output.txt", "EVENTO: cambio intervallo, intervallo = " + interval.__str__(), False)
            if interval != 0:
                AveragesCalculationPerReplica(interval, i, nsim, m)
                for k in range(NUM_BLOCKS):
                    area[k].service = 0.0
                    area[k].queue = 0.0
                    area[k].node = 0.0
                    centers_num[k].prev_car_arrivals = centers_num[k].car_arrivals
                    centers_num[k].prev_car_loss = centers_num[k].car_loss
                    centers_num[k].prev_compl_f = centers_num[k].car_compl

            interval += 1
            if calendar.change_time_slot == 0.0:
                interTime = 240.0
                calendar.change_time_slot = 16200.0
            elif calendar.change_time_slot == 16200.0:
                interTime = 120.0
                calendar.change_time_slot = math.inf
                m = config_matrix[1]
                ServerConfigurationBetweenTimeSlot(config_matrix[0], m)


        elif clock.current == calendar.sampling:
            SamplingAverageCalculation(count_sampling, i, samplingTime, interval, numMedioServentiAttivi)
            if clock.current + SAMPLING_INTERVAL <= STOP:
                calendar.sampling += SAMPLING_INTERVAL
                count_sampling += 1
            else:
                calendar.sampling = math.inf

        # ***************      CENTRO HAND-WASH         ***************
        # arrivo newCar
        elif clock.current == calendar.newCar_arrivalTime_handWash:
            NewCarArrival(server_configuration_max[0], m[0], True)

        # arrivo oldCar
        elif clock.current == calendar.oldCar_arrivalTime_handWash:
            OldCarArrivalHandWash(server_configuration_max[0], m[0], True)

        # partenza newCar
        elif clock.current == next_completion_handWash.completionTime and not next_completion_handWash.is_old:
            NewCarDepartureHandWash(next_completion_handWash.server_offset, m[0])

        # partenza oldCar
        elif clock.current == next_completion_handWash.completionTime and next_completion_handWash.is_old:
            OldCarDepartureHandWash(next_completion_handWash.server_offset, m[0])

        # abbandono macchina
        elif clock.current == next_abandon_handWash.abandon_time:
            AbandonHandWash(next_abandon_handWash.job_id)


        # ***************      CENTRO TOUCHLESS-WASH         ***************

        # arrivo oldCar
        elif clock.current == calendar.oldCar_arrivalTime_touchless:
            OldCarArrivalTouchless(server_configuration_max[1], m[1], True)

        # partenza oldCar
        elif clock.current == next_completion_touchless.completionTime:
            OldCarDepartureTouchless(next_completion_touchless.server_offset, m[1])

        # abbandono oldCar
        elif clock.current == next_abandon_touchless.abandon_time:
            AbandonTouchless(next_abandon_touchless.job_id)

        # ***************      CENTRO POLISHING         ***************
        # arrivo newCar
        elif clock.current == calendar.newCar_arrivalTime_polishing:
            NewCarArrivalPolishing(server_configuration_max[2])

        # arrivo oldCar
        elif clock.current == calendar.oldCar_arrivalTime_polishing:
            OldCarArrivalPolishing(server_configuration_max[2])

        # partenza newCar
        elif clock.current == next_completion_polishing.completionTime and not next_completion_polishing.is_old:
            NewCarDeparturePolishing(next_completion_polishing.server_offset, m[2])

        # partenza oldCar
        elif clock.current == next_completion_polishing.completionTime and next_completion_polishing.is_old:
            OldCarDeparturePolishing(next_completion_polishing.server_offset, m[2])

            # ***************      CENTRO INTERIOR-WASH         ***************
        # arrivo newCar
        elif clock.current == calendar.newCar_arrivalTime_interiorWash:
            NewCarArrivalInteriorWash(server_configuration_max[3])


        # arrivo oldCar
        elif clock.current == calendar.oldCar_arrivalTime_interiorWash:
            OldCarArrivalInteriorWash(server_configuration_max[3])


        # partenza macchina
        elif clock.current == next_completion_interiorWash.completionTime:
            CarDepartureInteriorWash(next_completion_interiorWash.server_offset)

    AveragesCalculationPerReplica(interval, i, nsim, m)


def ResetGlobalVar():
    global centers_num, arrivals, clock, area, areaSampling, calendar, state_variable, old_car_arrival_state_handWash, \
        new_car_arrival_state_handWash, old_car_arrival_state_touchless
    centers_num = [CarNumberCenter() for _ in range(NUM_BLOCKS)]
    arrivals = Arrivals()
    clock = time()
    area = [track() for _ in range(NUM_BLOCKS)]
    areaSampling = [track() for _ in range(NUM_BLOCKS)]
    calendar = Calendar()

    state_variable = []
    old_car_arrival_state_handWash = True
    new_car_arrival_state_handWash = True

    old_car_arrival_state_touchless = True


def finite_horizon_simulation():
    plantSeeds(123456789)

    # uso un dizionario per salvarmi i valori della simulazione a orizzonte finito
    result_finite = {
        "simulation_number": None,
        "sampling_time": None,
        "medium_active_servers": None
    }

    samplingSize = int(STOP / SAMPLING_INTERVAL)
    nsim = np.zeros((REPLICATIONS, NUM_BLOCKS, INTERVALS, STATISTICS), dtype=np.float64)
    samplingTime = np.zeros((REPLICATIONS, NUM_BLOCKS, STATISTICS, samplingSize), dtype=np.float64)
    numMedioServentiAttivi = np.zeros((REPLICATIONS, NUM_BLOCKS, samplingSize), dtype=np.float64)

    for i in range(REPLICATIONS):
        init(config_matrix, True)
        simulation(config_matrix, i, nsim, samplingTime, numMedioServentiAttivi)
        ResetGlobalVar()

    result_finite["simulation_number"] = nsim
    result_finite["sampling_time"] = samplingTime
    result_finite["medium_active_servers"] = numMedioServentiAttivi



    ret = ResultFinite()
    ret.nsim = nsim
    ret.samplingTime = samplingTime
    ret.numMedioServentiAttivi = numMedioServentiAttivi





    for center in range(NUM_BLOCKS):
        fp = file_path2[center]
        for replica in range(REPLICATIONS):
            for interval in range(INTERVALS):
                for stat in range(STATISTICS):
                    stat_value = str(ret.nsim[replica][center][interval][stat]) + ";"
                    fp.write(stat_value)
                fp.write("\n")
            fp.write("\n")

    n = int(STOP / SAMPLING_INTERVAL)
    for center in range(NUM_BLOCKS):
        fp = file_path2[center + 8]
        for replica in range(REPLICATIONS):
            for stat in range(STATISTICS):
                for count in range(n):
                    stat_value = str(ret.samplingTime[replica][center][stat][count]) + ";"
                    fp.write(stat_value)
                fp.write("\n")
            fp.write("\n")


    return ret


def BatchMean(index_center, count, data, sum, mean, batch):
    # *************************         UTILIZZAZIONE               ************************* #
    utilization = area[index_center].service / ((clock.current - batch) * server_conf[index_center]) if \
        ((clock.current - batch) * server_conf[index_center]) != 0 else 0
    data[index_center][0][(count[index_center] // B) - 1] = utilization

    # *************************      POPOLAZIONE MEDIA NELLA CODA   ************************* #
    queue_population = area[index_center].queue / (clock.current - batch) if (clock.current - batch) != 0 else 0
    data[index_center][1][(count[index_center] // B) - 1] = queue_population

    # *************************      POPOLAZIONE MEDIA NEL CENTRO   ************************* #
    center_population = area[index_center].node / (clock.current - batch) if (clock.current - batch) != 0 else 0
    data[index_center][2][(count[index_center] // B) - 1] = center_population

    # *************************         TEMPO MEDIO DI SERVIZIO     ************************* #
    service_time = area[index_center].service / (
            centers_num[index_center].car_compl - centers_num[index_center].prev_compl_f) \
        if (centers_num[index_center].car_compl - centers_num[index_center].prev_compl_f) != 0 else 0
    data[index_center][3][(count[index_center] // B) - 1] = service_time

    # *************************         TEMPO MEDIO DI ATTESA       ************************* #
    wait_time = area[index_center].queue / (
            centers_num[index_center].car_compl - centers_num[index_center].prev_compl_f) \
        if (centers_num[index_center].car_compl - centers_num[index_center].prev_compl_f) != 0 else 0
    data[index_center][4][(count[index_center] // B) - 1] = wait_time

    # *************************         TEMPO MEDIO DI RISPOSTA     ************************* #
    response_time = area[index_center].node / (
            centers_num[index_center].car_compl - centers_num[index_center].prev_compl_f) \
        if (centers_num[index_center].car_compl - centers_num[index_center].prev_compl_f) != 0 else 0
    data[index_center][5][(count[index_center] // B) - 1] = response_time

    # *************************         TEMPO MEDIO DI INTERARRIVO  ************************* #
    interarrival_time = (clock.last[index_center] - batch) / (
            centers_num[index_center].car_arrivals - centers_num[index_center].prev_car_arrivals) \
        if (centers_num[index_center].car_arrivals - centers_num[index_center].prev_car_arrivals) != 0 else 0
    data[index_center][6][(count[index_center] // B) - 1] = interarrival_time

    # *************************         NUMERO ARRIVI MACCHINE      ************************* #

    data[index_center][7][(count[index_center] // B) - 1] = float(
        centers_num[index_center].car_arrivals - centers_num[index_center].prev_car_arrivals)

    # data[index_center][8][(count[index_center]//B)-1] = float(centers_num[index_center].index_a - centers_num[index_center].prev_index_a)

    # *************************         NUMERO ABBANDONI            ************************* #
    data[index_center][8][(count[index_center] // B) - 1] = float(
        centers_num[index_center].car_loss - centers_num[index_center].prev_car_loss)

    # *************************     PROBABILITA' DI ABBANDONO       ************************* #
    data[index_center][9][(count[index_center] // B) - 1] = float(
        centers_num[index_center].car_loss - centers_num[index_center].prev_car_loss) / \
                                                            float(centers_num[index_center].car_arrivals - centers_num[
                                                                index_center].prev_car_arrivals)

    diff = 0.0
    n = count[index_center] / B

    for i in range(STATISTICS):
        diff = data[index_center][i][(count[index_center] // B) - 1] - mean[index_center][i]
        sum[index_center][i] += diff * diff * (n - 1.0) / n
        mean[index_center][i] += diff / n

        data[index_center][i][(count[index_center] // B) - 1] = mean[index_center][i]

    area[index_center].service = 0.0
    area[index_center].queue = 0.0
    area[index_center].node = 0.0
    centers_num[index_center].prev_car_arrivals = centers_num[index_center].car_arrivals
    centers_num[index_center].prev_car_loss = centers_num[index_center].car_loss
    centers_num[index_center].prev_compl_f = centers_num[index_center].car_compl


def infinite_horizon_sim():
    plantSeeds(123456789)

    # data ha dimensione [NUM_BLOCKS][STATISTICS][K] --> array a 3 dimensioni
    data = []
    for i in range(NUM_BLOCKS):
        data.append([])
        for j in range(STATISTICS):
            data[i].append([0.0] * K)

    # sum e mean hanno dimensione [NUM_BLOCKS][STATISTICS] --> array a 2 dimensioni
    sum = []
    mean = []
    for i in range(NUM_BLOCKS):
        sum.append([0.0] * STATISTICS)
        mean.append([0.0] * STATISTICS)

    # initialization()
    init(server_conf, False)

    count = [0] * NUM_BLOCKS
    startBatch = [0.0] * NUM_BLOCKS

    while True:
        remaining = 0
        for i in range(NUM_BLOCKS):
            if count[i] < B * K:
                remaining += 1
        if remaining == 0:
            break

        clock.next = GetMinTime(server_conf)

        # UpdateTrack()
        UpdateTrack(server_conf)
        clock.current = clock.next

        next_abandon_handWash = GetMinAbandon(calendar.head_handWash)
        next_abandon_touchless = GetMinAbandon(calendar.head_touchless)

        next_completion_handWash = GetMinCompletion(server_conf[0], state_variable[0].server_state, 1)
        next_completion_touchless = GetMinCompletion(server_conf[1], state_variable[1].server_state, 2)
        next_completion_polishing = GetMinCompletion(server_conf[2], state_variable[2].server_state, 3)
        next_completion_interiorWash = GetMinCompletion(server_conf[3], state_variable[3].server_state, 4)

        # ***************      CENTRO HAND-WASH         ***************

        # arrivo newCar
        if clock.current == calendar.newCar_arrivalTime_handWash:
            NewCarArrival(server_conf[0], server_conf[0], False)
            if count[0] < B * K:
                count[0] += 1
                # L'espressione count[0] % B == 0 controlla se il resto è uguale a zero,
                # indicando che count[0] è un multiplo intero di B
                if count[0] % B == 0:
                    BatchMean(0, count, data, sum, mean, startBatch[0])
                    startBatch[0] = clock.current



        # arrivo oldCar
        elif clock.current == calendar.oldCar_arrivalTime_handWash:
            OldCarArrivalHandWash(server_conf[0], server_conf[0], False)
            if count[0] < B * K:
                count[0] += 1
                if count[0] % B == 0:
                    BatchMean(0, count, data, sum, mean, startBatch[0])
                    startBatch[0] = clock.current

        # partenza newCar
        elif clock.current == next_completion_handWash.completionTime and not next_completion_handWash.is_old:
            NewCarDepartureHandWash(next_completion_handWash.server_offset, server_conf[0])

        # partenza oldCar
        elif clock.current == next_completion_handWash.completionTime and next_completion_handWash.is_old:
            OldCarDepartureHandWash(next_completion_handWash.server_offset, server_conf[0])

        # abbandono macchina
        elif clock.current == next_abandon_handWash.abandon_time:
            AbandonHandWash(next_abandon_handWash.job_id)


        # ***************      CENTRO TOUCHLESS-WASH         ***************

        # arrivo oldCar
        elif clock.current == calendar.oldCar_arrivalTime_touchless:
            OldCarArrivalTouchless(server_conf[1], server_conf[1], False)
            if count[1] < B * K:
                count[1] += 1
                if count[1] % B == 0:
                    BatchMean(1, count, data, sum, mean, startBatch[1])
                    startBatch[1] = clock.current

        # partenza oldCar
        elif clock.current == next_completion_touchless.completionTime:
            OldCarDepartureTouchless(next_completion_touchless.server_offset, server_conf[1])



        # abbandono oldCar
        elif clock.current == next_abandon_touchless.abandon_time:
            AbandonTouchless(next_abandon_touchless.job_id)


        # ***************      CENTRO POLISHING         ***************
        # arrivo newCar
        elif clock.current == calendar.newCar_arrivalTime_polishing:
            NewCarArrivalPolishing(server_conf[2])
            if count[2] < B * K:
                count[2] += 1
                if count[2] % B == 0:
                    BatchMean(2, count, data, sum, mean, startBatch[2])
                    startBatch[2] = clock.current

        # arrivo oldCar
        elif clock.current == calendar.oldCar_arrivalTime_polishing:
            OldCarArrivalPolishing(server_conf[2])
            if count[2] < B * K:
                count[2] += 1
                if count[2] % B == 0:
                    BatchMean(2, count, data, sum, mean, startBatch[2])
                    startBatch[2] = clock.current

        # partenza newCar
        elif clock.current == next_completion_polishing.completionTime and not next_completion_polishing.is_old:
            NewCarDeparturePolishing(next_completion_polishing.server_offset, server_conf[2])

        # partenza oldCar
        elif clock.current == next_completion_polishing.completionTime and next_completion_polishing.is_old:
            OldCarDeparturePolishing(next_completion_polishing.server_offset, server_conf[2])

        # ***************      CENTRO INTERIOR-WASH         ***************
        # arrivo newCar
        elif clock.current == calendar.newCar_arrivalTime_interiorWash:
            NewCarArrivalInteriorWash(server_conf[3])
            if count[3] < B * K:
                count[3] += 1
                if count[3] % B == 0:
                    BatchMean(3, count, data, sum, mean, startBatch[3])
                    startBatch[3] = clock.current

        # arrivo oldCar
        elif clock.current == calendar.oldCar_arrivalTime_interiorWash:
            OldCarArrivalInteriorWash(server_conf[3])
            if count[3] < B * K:
                count[3] += 1
                if count[3] % B == 0:
                    BatchMean(3, count, data, sum, mean, startBatch[3])
                    startBatch[3] = clock.current

        # partenza macchina
        elif clock.current == next_completion_interiorWash.completionTime:
            CarDepartureInteriorWash(next_completion_interiorWash.server_offset)

    res = [[0.0] * STATISTICS for _ in range(NUM_BLOCKS)]

    profit_system = 0
    for i in range(NUM_BLOCKS):
        num_car = 0
        num_loss = 0
        for j in range(STATISTICS):
            n = count[i] / B
            stdv = math.sqrt(sum[i][j] / n)
            u = 1.0 - 0.5 * (1.0 - LOC)  # interval parameter
            t = idfStudent(n - 1, u)  # critical value of t
            res[i][j] = t * stdv / math.sqrt(n - 1)  # interval half width

            print(f"INTERVALLO-INFINITO--{statistics[j]}-centro-{i} ------ {mean[i][j]} +/- {res[i][j]}")
            if j == 7:
                num_car = int(mean[i][j])
            if j == 8:
                num_loss = mean[i][j]

        profit_block = GainCalculationPerBlock(i, num_loss, num_car)

        profit_system += profit_block

    print(f"profit_system == {profit_system}")

    return data


def UpdateTrack(server_configuration):
    for i in range(NUM_BLOCKS):
        busy = CountBusyServers(server_configuration[i], state_variable[i].server_state)

        if i != 3:  # todo: quando elimino ultimo blocco, che devo fare col blocco 4  ?????
            area[i].service += (clock.next - clock.current) * busy
            area[i].queue += (clock.next - clock.current) * (state_variable[i].number_in_center - busy)
            area[i].node += (clock.next - clock.current) * state_variable[i].number_in_center

            areaSampling[i].service += (clock.next - clock.current) * busy
            areaSampling[i].queue += (clock.next - clock.current) * (state_variable[i].number_in_center - busy)
            areaSampling[i].node += (clock.next - clock.current) * state_variable[i].number_in_center
        else:
            area[i].service += (clock.next - clock.current) * busy
            area[i].node += (clock.next - clock.current) * busy

            areaSampling[i].service += (clock.next - clock.current) * busy
            areaSampling[i].node += (clock.next - clock.current) * busy


def CountBusyServers(num_servers, server_list):
    """
    riceve il numero di server preso dalla configurazione iniziale, e la lista di server da sv*, e conta quanti server
    sono occupati.
    Un server è considerato occupato se il suo valore nella lista non è 0 e non è -3.
    :return: il numero di server occupati
    """
    count = 0
    for i in range(num_servers):
        if server_list[i] != 0 and server_list[i] != -3:
            count += 1

    return count


def GetMinTime(server_configuration):
    """
    La funzione getMinimumTime calcola i tempi minimi di abbandono minAbandon1 e minAbandon2 chiamando
    la funzione getMinAbandon rispettivamente su events.head1 e events.head2. Gli oggetti nextAb1 e nextAb2 contengono
    i risultati restituiti dalla funzione getMinAbandon.
    """
    min_abandon1 = math.inf
    min_abandon2 = math.inf

    if calendar.head_handWash is not None:
        # almeno un job che deve abbandonare
        abandon_handWash = GetMinAbandon(calendar.head_handWash)
        min_abandon1 = abandon_handWash.abandon_time

    if calendar.head_touchless is not None:
        abandon_touchless = GetMinAbandon(calendar.head_touchless)
        min_abandon2 = abandon_touchless.abandon_time

    min_service_handWash = GetMinCompletion(server_configuration[0], state_variable[0].server_state, 1).completionTime
    min_service_touchless = GetMinCompletion(server_configuration[1], state_variable[1].server_state, 2).completionTime
    min_service_polishing = GetMinCompletion(server_configuration[2], state_variable[2].server_state, 3).completionTime
    min_service_interiorWash = GetMinCompletion(server_configuration[3], state_variable[3].server_state,
                                                4).completionTime

    t = [0.0] * 15
    t[0] = min_abandon1
    t[1] = min_abandon2
    t[2] = min_service_handWash
    t[3] = min_service_touchless
    t[4] = min_service_polishing
    t[5] = min_service_interiorWash
    t[6] = calendar.newCar_arrivalTime_handWash
    t[7] = calendar.oldCar_arrivalTime_handWash
    t[8] = calendar.oldCar_arrivalTime_touchless
    t[9] = calendar.newCar_arrivalTime_polishing
    t[10] = calendar.oldCar_arrivalTime_polishing
    t[11] = calendar.newCar_arrivalTime_interiorWash
    t[12] = calendar.oldCar_arrivalTime_interiorWash
    t[13] = calendar.change_time_slot
    t[14] = calendar.sampling

    return GetMinValue(t)


def GetMinAbandon(head_job):
    min_abandon = NextAbandon(-1, math.inf)

    if head_job is not None:
        min_abandon.job_id = head_job.id
        min_abandon.abandon_time = head_job.abandon_time

        current = head_job

        while current is not None:
            if current.abandon_time < min_abandon.abandon_time:
                min_abandon.job_id = current.id
                min_abandon.abandon_time = current.abandon_time
            current = current.next
    return min_abandon


def GetMinCompletion(num_servers, x, index):
    min_completion = NextCompletion(0, True, float('inf'))
    completion_times = None

    if index == 1:
        completion_times = calendar.completions_handWash
    elif index == 2:
        completion_times = calendar.completions_touchless
    elif index == 3:
        completion_times = calendar.completions_polishing
    elif index == 4:
        completion_times = calendar.completions_interiorWash
    else:
        print(
            "ERRORE: il terzo parametro della funzione getMinCompletion() deve essere un valore intero compreso tra 1 e 4.")
        exit(-8)

    for i in range(num_servers):
        if completion_times[i] < min_completion.completionTime:
            min_completion.server_offset = i
            min_completion.is_old = x[i] == 1 or x[i] == -1
            min_completion.completionTime = completion_times[i]

    return min_completion


def GetMinValue(values):
    """
    Restituisce il valore più piccolo all'interno di una lista di valori numerici.
    :param values:
    :return: Minimo valore della lista values
    """
    min_value = float("inf")  # assicura che qualsiasi valore nella lista sia inferiore a min_value al primo confronto.
    for i in range(len(values)):
        if values[i] < min_value:
            min_value = values[i]

    return min_value


def GetMaxServerNumber(array):
    return [max(array[i][j] for i in range(NUM_TIME_SLOT)) for j in range(NUM_BLOCKS)]


def InitializeStateVariable(max_configuration, server_configuration):
    global state_variable

    """
    il vettore x di ogni centro rappresenta lo stato di ogni server del centro.
    quindi, ogni x è un array che avrà dimensione pari al numero di server impostati nella configurazione
    """

    state_variable = [
        StateVariables(0, [0] * max_configuration[0]),
        StateVariables(0, [0] * max_configuration[1]),
        StateVariables(0, [0] * server_configuration[2] + [-3] * (max_configuration[2] - server_configuration[2])),
        StateVariables(0, [0] * server_configuration[3] + [-3] * (max_configuration[3] - server_configuration[3]))
    ]


def init(server_configuration, type_simulation):
    global server_configuration_max
    if type_simulation:
        server_configuration_max = GetMaxServerNumber(server_configuration)
        InitializeStateVariable(server_configuration_max, server_configuration[0])
        InitializeCalendar(server_configuration_max, True)

    else:
        server_configuration_max = server_configuration
        InitializeStateVariable(server_configuration_max, server_configuration)
        InitializeCalendar(server_configuration_max, False)


def InitializeCalendar(max_configuration, simulation_type):
    global interTime
    if simulation_type:
        interTime = 240.0

    # ----- INIZIALIZZO LISTA EVENTI ------ #
    calendar.newCar_arrivalTime_handWash = getNewCarArrivalHandWash(START, interTime)

    calendar.oldCar_arrivalTime_handWash = getOldCarArrivalHandWash(START, interTime)

    calendar.oldCar_arrivalTime_touchless = getOldCarArrivalTouchless(START, interTime)

    calendar.oldCar_arrivalTime_polishing = math.inf
    calendar.newCar_arrivalTime_polishing = math.inf

    calendar.oldCar_arrivalTime_interiorWash = math.inf
    calendar.newCar_arrivalTime_interiorWash = math.inf

    calendar.oldCar_arrivalTime_5 = math.inf
    calendar.newCar_arrivalTime_5 = math.inf

    calendar.completions_handWash = [math.inf] * max_configuration[0]
    calendar.completions_touchless = [math.inf] * max_configuration[1]
    calendar.completions_polishing = [math.inf] * max_configuration[2]
    calendar.completions_interiorWash = [math.inf] * max_configuration[3]

    if simulation_type:
        calendar.change_time_slot = 0.0
        calendar.sampling = SAMPLING_INTERVAL
    else:
        calendar.change_time_slot = math.inf
        calendar.sampling = math.inf


def initialization():
    initialize_state_variables()
    initialize_calendar(False)


def initialize_calendar(simulation_type):
    global interTime
    if simulation_type:
        interTime = 240.0

    # ----- INIZIALIZZO LISTA EVENTI ------ #
    calendar.newCar_arrivalTime_handWash = getNewCarArrivalHandWash(START, interTime)

    calendar.oldCar_arrivalTime_handWash = getOldCarArrivalHandWash(START, interTime)

    calendar.oldCar_arrivalTime_touchless = getOldCarArrivalTouchless(START, interTime)

    calendar.oldCar_arrivalTime_polishing = math.inf
    calendar.newCar_arrivalTime_polishing = math.inf

    calendar.oldCar_arrivalTime_interiorWash = math.inf
    calendar.newCar_arrivalTime_interiorWash = math.inf

    calendar.oldCar_arrivalTime_5 = math.inf
    calendar.newCar_arrivalTime_5 = math.inf

    calendar.completions_handWash = [math.inf] * server_conf[0]
    calendar.completions_touchless = [math.inf] * server_conf[1]
    calendar.completions_polishing = [math.inf] * server_conf[2]
    calendar.completions_interiorWash = [math.inf] * server_conf[3]

    if simulation_type:
        calendar.change_time_slot = 0.0
        calendar.sampling = SAMPLING_INTERVAL
    else:
        calendar.change_time_slot = math.inf
        calendar.sampling = math.inf


def initialize_state_variables():
    global state_variable

    """
    il vettore x di ogni centro rappresenta lo stato di ogni server del centro.
    quindi, ogni x è un array che avrà dimensione pari al numero di server impostati nella configurazione
    """

    state_variable = [
        StateVariables(0, [0] * server_configuration_max[0]),
        StateVariables(0, [0] * server_configuration_max[1]),
        StateVariables(0, [-3] * (server_configuration_max[2] - server_conf[2]) + [0] * server_conf[2]),
        StateVariables(0, [-3] * (server_configuration_max[3] - server_conf[3]) + [0] * server_conf[3])
    ]


# ***************      CENTRO HAND-WASH         ***************
def NewCarArrival(max_len, curr_len, simulation_type):
    """
    arrivo macchina nuova al centro hand-wash
    :return:
    """
    centers_num[0].car_arrivals += 1  # incremento il numero delle famiglie che arrivano al centro
    state_variable[0].number_in_center += 1  # incremento la popolazione di 1

    calendar.newCar_arrivalTime_handWash = getNewCarArrivalHandWash(clock.current, interTime)
    clock.last[0] = clock.current

    if calendar.newCar_arrivalTime_handWash > STOP and simulation_type:
        calendar.newCar_arrivalTime_handWash = math.inf
        global new_car_arrival_state_handWash
        new_car_arrival_state_handWash = False  # serve per simulazione finita

    idle_server = GetIdleServer(max_len, state_variable[0])
    if idle_server >= 0:
        state_variable[0].server_state[idle_server] = 2
        calendar.completions_handWash[idle_server] = GetServiceHandWash(clock.current)

    elif state_variable[0].number_in_center > CAR_TO_WAIT_H + curr_len:
        # Inserimento in coda di un nuovo nodo all'interno della lista degli abbandoni
        tailJob = Job(centers_num[0].car_arrivals, getAbandonHandWash(clock.current))
        tailJob.next = None
        tailJob.prev = calendar.tail_handWash

        if calendar.tail_handWash is not None:
            calendar.tail_handWash.next = tailJob
        else:
            calendar.head_handWash = tailJob

        calendar.tail_handWash = tailJob


def OldCarArrivalHandWash(max_len, curr_len, simulation_type):
    """
    arrivo macchina vecchia al centro hand-wash
    :param max_len:
    :param curr_len:
    :param simulation_type:
    :return:
    """
    centers_num[0].car_arrivals += 1  # incremento il numero delle famiglie che arrivano al centro
    state_variable[0].number_in_center += 1  # incremento la popolazione di 1

    # genero l'istante di tempo del prossimo arrivo di una macchina vecchia
    calendar.oldCar_arrivalTime_handWash = getOldCarArrivalHandWash(clock.current, interTime)
    clock.last[0] = clock.current

    if calendar.oldCar_arrivalTime_handWash > STOP and simulation_type:
        calendar.oldCar_arrivalTime_handWash = math.inf
        global old_car_arrival_state_handWash
        old_car_arrival_state_handWash = False  # per sim finita

    idle_server = GetIdleServer(max_len, state_variable[0])
    if idle_server >= 0:
        state_variable[0].server_state[idle_server] = 1
        calendar.completions_handWash[idle_server] = GetServiceHandWash(clock.current)
    elif state_variable[0].number_in_center > CAR_TO_WAIT_H + curr_len:
        # Inserimento in coda di un nuovo nodo all'interno della lista degli abbandoni
        tailJob = Job(centers_num[0].car_arrivals, getAbandonHandWash(clock.current))
        tailJob.next = None
        tailJob.prev = calendar.tail_handWash

        if calendar.tail_handWash is not None:
            calendar.tail_handWash.next = tailJob
        else:
            calendar.head_handWash = tailJob

        calendar.tail_handWash = tailJob


def NewCarDepartureHandWash(offset, max_len):
    state_variable[0].number_in_center -= 1
    centers_num[0].car_compl += 1

    if calendar.head_handWash is not None:
        # rimuovo nodo da testa della lista abbandoni
        to_remove = calendar.head_handWash
        if to_remove.next is None:
            calendar.head_handWash = None
            calendar.tail_handWash = None
        else:
            calendar.head_handWash = to_remove.next
            calendar.head_handWash.prev = None

    if state_variable[0].server_state[offset] < 0:
        calendar.completions_handWash[offset] = math.inf
        state_variable[0].server_state[offset] = -3
    elif state_variable[0].number_in_center >= max_len:
        # almeno un job in coda --> genero il suo tempo di completamento
        calendar.completions_handWash[offset] = GetServiceHandWash(clock.current)
    else:
        calendar.completions_handWash[offset] = math.inf
        state_variable[0].server_state[offset] = 0

    # Inserimento in coda di un nuovo nodo all'interno della lista degli arrivi al centro 3 (polishing)
    tail_arr = ArrivalTime(clock.current, False)
    tail_arr.next = None
    tail_arr.prev = arrivals.tail3

    if arrivals.tail3 is not None:
        arrivals.tail3.next = tail_arr
    else:
        # scrivi_su_file("output.txt", "arrivals.tail3 is  None", False)
        arrivals.head3 = tail_arr
        calendar.newCar_arrivalTime_polishing = clock.current
    arrivals.tail3 = tail_arr


def OldCarDepartureHandWash(offset, max_len):
    state_variable[0].number_in_center -= 1
    centers_num[0].car_compl += 1

    if calendar.head_handWash is not None:
        # rimuovo nodo da testa della lista abbandoni
        to_remove = calendar.head_handWash
        if to_remove.next is None:
            calendar.head_handWash = None
            calendar.tail_handWash = None
        else:
            calendar.head_handWash = to_remove.next
            calendar.head_handWash.prev = None

    if state_variable[0].server_state[offset] < 0:
        calendar.completions_handWash[offset] = math.inf
        state_variable[0].server_state[offset] = -3
    elif state_variable[0].number_in_center >= max_len:
        # almeno un job in coda --> genero il suo tempo di completamento
        calendar.completions_handWash[offset] = GetServiceHandWash(clock.current)
    else:
        calendar.completions_handWash[offset] = math.inf
        state_variable[0].server_state[offset] = 0

    # Inserimento in coda di un nuovo nodo all'interno della lista degli arrivi al centro 3 (polishing)
    tail_arr = ArrivalTime(clock.current, True)
    tail_arr.next = None
    tail_arr.prev = arrivals.tail3

    if arrivals.tail3 is not None:
        arrivals.tail3.next = tail_arr

    else:
        arrivals.head3 = tail_arr
        calendar.oldCar_arrivalTime_polishing = clock.current
    arrivals.tail3 = tail_arr


def AbandonHandWash(id_job):
    head_job = calendar.head_handWash
    while head_job is not None:
        if head_job.id == id_job:
            break
        head_job = head_job.next

    prev_job = head_job.prev
    next_job = head_job.next

    if prev_job is not None:
        prev_job.next = next_job
    else:
        calendar.head_handWash = next_job

    if next_job is not None:
        next_job.prev = prev_job
    else:
        calendar.tail_handWash = prev_job

    centers_num[0].car_loss += 1
    state_variable[0].number_in_center -= 1


# ***************      CENTRO TOUCHLESS    N.2     ***************
def OldCarArrivalTouchless(max_len, curr_len, simulation_type):
    """
    arrivo macchina vecchia al centro touchless
    """
    centers_num[1].car_arrivals += 1
    state_variable[1].number_in_center += 1

    calendar.oldCar_arrivalTime_touchless = getOldCarArrivalTouchless(clock.current, interTime)
    clock.last[1] = clock.current

    if calendar.oldCar_arrivalTime_touchless > STOP and simulation_type:
        calendar.oldCar_arrivalTime_touchless = math.inf
        global old_car_arrival_state_touchless
        old_car_arrival_state_touchless = False  # per sim finita

    idle_server = GetIdleServer(max_len, state_variable[1])
    if idle_server >= 0:
        state_variable[1].server_state[idle_server] = 1
        calendar.completions_touchless[idle_server] = GetServiceTouchless(clock.current)
    elif state_variable[1].number_in_center > CAR_TO_WAIT_T + curr_len:
        tailJob = Job(centers_num[1].car_arrivals, getAbandonTouchless(clock.current))
        tailJob.next = None
        tailJob.prev = calendar.tail_touchless

        if calendar.tail_touchless is not None:
            calendar.tail_touchless.next = tailJob
        else:
            calendar.head_touchless = tailJob

        calendar.tail_touchless = tailJob


def OldCarDepartureTouchless(offset, max_len):
    state_variable[1].number_in_center -= 1
    centers_num[1].car_compl += 1

    if calendar.head_touchless is not None:
        to_remove = calendar.head_touchless
        if to_remove.next is None:
            calendar.head_touchless = None
            calendar.tail_touchless = None
        else:
            calendar.head_touchless = to_remove.next
            calendar.head_touchless.prev = None

    if state_variable[1].server_state[offset] < 0:
        calendar.completions_touchless[offset] = math.inf
        state_variable[1].server_state[offset] = -3
    elif state_variable[1].number_in_center >= max_len:
        calendar.completions_touchless[offset] = GetServiceTouchless(clock.current)
    else:
        calendar.completions_touchless[offset] = math.inf
        state_variable[1].server_state[offset] = 0

    tail_arr = ArrivalTime(clock.current, True)
    tail_arr.next = None
    tail_arr.prev = arrivals.tail3

    if arrivals.tail3 is not None:
        arrivals.tail3.next = tail_arr
    else:
        arrivals.head3 = tail_arr
        calendar.oldCar_arrivalTime_polishing = clock.current

    arrivals.tail3 = tail_arr


def AbandonTouchless(id_job):
    head_job = calendar.head_touchless
    while head_job is not None:
        if head_job.id == id_job:
            break
        head_job = head_job.next

    prev_job = head_job.prev
    next_job = head_job.next

    if prev_job is not None:
        prev_job.next = next_job
    else:
        calendar.head_touchless = next_job

    if next_job is not None:
        next_job.prev = prev_job
    else:
        calendar.tail_touchless = prev_job

    centers_num[1].car_loss += 1
    state_variable[1].number_in_center -= 1


# ***************      CENTRO POLISHING   N.3      ***************

def NewCarArrivalPolishing(max_len):
    """
    arrivo macchina nuova al centro polishing
    :return:
    """
    centers_num[2].car_arrivals += 1  # incremento num famiglie che arrivano al centro
    state_variable[2].number_in_center += 1

    to_remove = arrivals.head3

    clock.last[2] = clock.current
    if to_remove.next is None:
        arrivals.head3 = None
        arrivals.tail3 = None
        calendar.newCar_arrivalTime_polishing = math.inf
    else:
        arrivals.head3 = to_remove.next
        arrivals.head3.prev = None
        calendar.newCar_arrivalTime_polishing = arrivals.head3.time_value

    idle_server = GetIdleServer(max_len, state_variable[2])
    if idle_server >= 0:
        state_variable[2].server_state[idle_server] = 2
        calendar.completions_polishing[idle_server] = GetServicePolishing(clock.current)


def OldCarArrivalPolishing(max_len):
    centers_num[2].car_arrivals += 1  # incremento num famiglie che arrivano al centro
    state_variable[2].number_in_center += 1

    to_remove = arrivals.head3
    clock.last[2] = clock.current

    if to_remove.next is None:
        arrivals.head3 = None
        arrivals.tail3 = None
        calendar.oldCar_arrivalTime_polishing = math.inf
    else:
        arrivals.head3 = to_remove.next
        arrivals.head3.prev = None
        calendar.oldCar_arrivalTime_polishing = arrivals.head3.time_value

    idle_server = GetIdleServer(max_len, state_variable[2])
    if idle_server >= 0:
        state_variable[2].server_state[idle_server] = 1
        calendar.completions_polishing[idle_server] = GetServicePolishing(clock.current)


def NewCarDeparturePolishing(offset, max_len):
    state_variable[2].number_in_center -= 1
    centers_num[2].car_compl += 1

    if state_variable[2].server_state[offset] < 0:
        calendar.completions_polishing[offset] = math.inf
        state_variable[2].server_state[offset] = -3  # -3 = idle
    elif state_variable[2].number_in_center >= max_len:
        calendar.completions_polishing[offset] = GetServicePolishing(clock.current)
        state_variable[2].server_state[offset] = 2
    else:
        calendar.completions_polishing[offset] = math.inf
        state_variable[2].server_state[offset] = 0  # 0 = idle

    # se probab di fare interni < Q_INTERIOR, allora aggiungo nodo alla lista
    # degli arrivi al centro INTERIOR WASH

    if GetInteriorWashProbability() < Q_INTERIOR:
        tail_arr = ArrivalTime(clock.current, False)
        tail_arr.next = None
        tail_arr.prev = arrivals.tail4

        if arrivals.tail4 is not None:
            arrivals.tail4.next = tail_arr
        else:
            arrivals.head4 = tail_arr
            calendar.newCar_arrivalTime_interiorWash = clock.current

        arrivals.tail4 = tail_arr


def OldCarDeparturePolishing(offset, max_len):
    state_variable[2].number_in_center -= 1
    centers_num[2].car_compl += 1

    if state_variable[2].server_state[offset] < 0:
        calendar.completions_polishing[offset] = math.inf
        state_variable[2].server_state[offset] = -3  # -3 = idle
    elif state_variable[2].number_in_center >= max_len:
        calendar.completions_polishing[offset] = GetServicePolishing(clock.current)
    else:
        calendar.completions_polishing[offset] = math.inf
        state_variable[2].server_state[offset] = 0  # 0 = idle

    # se probab di fare interni < Q_INTERIOR, allora aggiungo nodo alla lista
    # degli arrivi al centro INTERIOR WASH

    if GetInteriorWashProbability() < Q_INTERIOR:
        tail_arr = ArrivalTime(clock.current, True)
        tail_arr.next = None
        tail_arr.prev = arrivals.tail4

        if arrivals.tail4 is not None:
            arrivals.tail4.next = tail_arr
        else:
            arrivals.head4 = tail_arr
            calendar.oldCar_arrivalTime_interiorWash = clock.current

        arrivals.tail4 = tail_arr


# ***************      CENTRO INTERIOR-WASH   N.4  ***************
def OldCarArrivalInteriorWash(max_len):
    centers_num[3].car_arrivals += 1  # incremento num famiglie che arrivano al centro

    to_remove = arrivals.head4

    clock.last[3] = clock.current

    if to_remove.next is None:
        arrivals.head4 = None
        arrivals.tail4 = None
        calendar.oldCar_arrivalTime_interiorWash = math.inf
    else:
        arrivals.head4 = to_remove.next
        arrivals.head4.prev = None
        calendar.oldCar_arrivalTime_interiorWash = arrivals.head4.time_value

    idle_server = GetIdleServer(max_len, state_variable[3])
    if idle_server >= 0:
        state_variable[3].server_state[idle_server] = 1
        calendar.completions_interiorWash[idle_server] = GetServiceInteriorWash(clock.current)
    else:
        centers_num[
            3].car_loss += 1  # se non trovano un servente libero, le macchine abbandonano, non si mettono in fila!


def NewCarArrivalInteriorWash(max_len):
    centers_num[3].car_arrivals += 1  # incremento num famiglie che arrivano al centro

    to_remove = arrivals.head4

    clock.last[3] = clock.current

    if to_remove.next is None:
        arrivals.head4 = None
        arrivals.tail4 = None
        calendar.newCar_arrivalTime_interiorWash = math.inf
    else:
        arrivals.head4 = to_remove.next
        arrivals.head4.prev = None
        calendar.newCar_arrivalTime_interiorWash = arrivals.head4.time_value

    idle_server = GetIdleServer(max_len, state_variable[3])
    if idle_server >= 0:
        state_variable[3].server_state[idle_server] = 2
        calendar.completions_interiorWash[idle_server] = GetServiceInteriorWash(clock.current)
    else:
        centers_num[
            3].car_loss += 1  # se non trovano un servente libero, le macchine abbandonano, non si mettono in fila!


def CarDepartureInteriorWash(offset):
    centers_num[3].car_compl += 1
    calendar.completions_interiorWash[offset] = float('inf')
    state_variable[3].server_state[offset] = 0


def GetIdleServer(num_server, center):
    for i in range(num_server):
        if center.server_state[i] == 0:
            return i
    return -1


if __name__ == '__main__':
    main()
