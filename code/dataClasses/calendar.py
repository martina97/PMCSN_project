from DES_Python.rngs import random, selectStream
from DES_Python.rvgs import Exponential, TruncatedNormal, Normal
from configurations import P_NEW, P_OLD_H, P_OLD_T, SERVICE_TIME_HANDWASH, SERVICE_TIME_INTERIORWASH, \
    SERVICE_TIME_TOUCHLESS, SERVICE_TIME_POLISHING, ABANDON_TIME_1, ABANDON_TIME_2


class Job:
    def __init__(self, id, abandon_time, prev=None, next=None):
        self.id = id  # int
        self.abandon_time = abandon_time  # double
        self.prev = prev  # job
        self.next = next  # job


class Calendar:
    def __init__(self):
        # lavaggio a mano
        self.newCar_arrivalTime_handWash = 0.0  # tempo arrivo new_car al centro 1
        self.oldCar_arrivalTime_handWash = 0.0  # tempo arrivo old_car al centro 1
        self.completions_handWash = None  # completamenti di ogni server del centro 1
        self.head_handWash = None  # tipo dato Job
        self.tail_handWash = None  # tipo dato Job
        # lavaggio con rulli
        self.oldCar_arrivalTime_touchless = 0.0  # tempo arrivo old_car al centro 2
        self.completions_touchless = None  # completamenti di ogni server del centro 2
        self.head_touchless = None
        self.tail_touchless = None
        # lucidatura
        self.newCar_arrivalTime_polishing = 0.0  # tempo arrivo new_car al centro 3
        self.oldCar_arrivalTime_polishing = 0.0  # tempo arrivo old_car al centro 3
        self.completions_polishing = None  # completamenti di ogni server del centro 3
        # lavaggio interni
        self.newCar_arrivalTime_interiorWash = 0.0  # tempo arrivo new_car al centro 4
        self.oldCar_arrivalTime_interiorWash = 0.0  # tempo arrivo old_car al centro 4
        self.completions_interiorWash = None  # completamenti di ogni server del centro 4
        self.change_time_slot = 0.0
        self.sampling = 0.0

    def __str__(self):
        return f"eventList:\tnewCar_arrivalTime_handWash=[{self.newCar_arrivalTime_handWash}]," \
               f"\n\t\toldCar_arrivalTime_handWash=[{self.oldCar_arrivalTime_handWash}]," \
               f"\n\t\tcompletionTimes1={self.completions_handWash}" \
               f"\n\t\thead_handWash=[{self.head_handWash}]" \
               f"\n\t\ttail_handWash=[{self.tail_handWash}]" \
               f"\n\t\toldCar_arrivalTime_touchless=[{self.oldCar_arrivalTime_touchless}]" \
               f"\n\t\tcompletions_touchless={self.completions_touchless}" \
               f"\n\t\thead_touchless=[{self.head_touchless}]" \
               f"\n\t\ttail_touchless=[{self.tail_touchless}]" \
               f"\n\t\tnewCar_arrivalTime_polishing=[{self.newCar_arrivalTime_polishing}]" \
               f"\n\t\toldCar_arrivalTime_polishing=[{self.oldCar_arrivalTime_polishing}]" \
               f"\n\t\tcompletions_polishing={self.completions_polishing}" \
               f"\n\t\tnewCar_arrivalTime_interiorWash=[{self.newCar_arrivalTime_interiorWash}]" \
               f"\n\t\toldCar_arrivalTime_interiorWash={self.oldCar_arrivalTime_interiorWash}" \
               f"\n\t\tcompletions_interiorWash={self.completions_interiorWash}" \
               f"\n\t\tchangeInterval=[{self.change_time_slot}]" \
               f"\n\t\tsampling=[{self.sampling}]"


class NextAbandon:
    def __init__(self, job_id, abandon_time):
        self.job_id = job_id  # int
        self.abandon_time = abandon_time  # double


class NextCompletion:
    def __init__(self, server_offset, is_old, completion_time):
        self.server_offset = server_offset  # int
        self.is_old = is_old  # bool
        self.completionTime = completion_time  # double


def getNewCarArrivalHandWash(arrival, inter_time):
    selectStream(0)  # rngs.py
    arrival += Exponential(inter_time / P_NEW)
    return arrival


def getOldCarArrivalHandWash(arrival, inter_time):
    selectStream(1)  # rngs.py
    arrival += Exponential(inter_time / P_OLD_H)
    return arrival


def getOldCarArrivalTouchless(arrival, inter_time):
    selectStream(2)  # rngs.py
    arrival += Exponential(inter_time / P_OLD_T)
    return arrival



def GetServiceHandWash(start):
    selectStream(3)
    mean = SERVICE_TIME_HANDWASH    # Media in secondi (30 min)
    std_dev = 5.0 * 60.0            # Deviazione standard in secondi
    a = 0.0                         # tempi di servizio non possono essere negativi
    b = 43.0 * 60.0
    return start + TruncatedNormal(mean, std_dev, a, b)


def GetServiceTouchless(start):
    selectStream(4)
    mean = SERVICE_TIME_TOUCHLESS
    std_dev = 5.0 * 60.0
    a = 0.0
    b = 25.0 * 60.0
    return start + TruncatedNormal(mean, std_dev, a, b)


def GetServicePolishing(start):
    selectStream(5)
    mean = SERVICE_TIME_POLISHING
    std_dev = 1.0 * 60.0
    a = 0.0
    b = 4.0 * 60.0
    return start + TruncatedNormal(mean, std_dev, a, b)


def GetServiceInteriorWash(start):
    selectStream(6)
    mean = SERVICE_TIME_INTERIORWASH
    std_dev = 3.0 * 60.0
    a = 0.0
    b = 10.0 * 60.0
    return start + TruncatedNormal(mean, std_dev, a, b)


def getAbandonHandWash(start):
    selectStream(7)  # rngs.py
    return start + Exponential(ABANDON_TIME_1)


def getAbandonTouchless(start):
    selectStream(8)  # rngs.py
    return start + Exponential(ABANDON_TIME_2)


def GetInteriorWashProbability():
    selectStream(9)
    return random()
