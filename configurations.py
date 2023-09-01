num_servers = [
    [7, 6, 1, 10],
    [7, 4, 4, 10]
]

STOP = 32400.0
NUM_BLOCKS = 4
NUM_TIME_SLOT = 2
STATISTICS = 10
B = 1024
K = 128
LOC = 0.99
SAMPLING_INTERVAL = 300
T = 30  # finestra temporale (1 mese)

P_OLD_H = 0.252  # macchine vecchie che usano lavaggio a mano
P_OLD_T = 0.84 - P_OLD_H  # macchine vecchie che usano lavaggio con rulli (t sta per "touchless")
P_NEW = 0.16  # macchine nuove --> fanno solo lavaggio con rulli
Q_INTERIOR = 0.8
CAR_TO_WAIT_H = 3
CAR_TO_WAIT_T = 4
# tempi di servizio
SERVICE_TIME_HANDWASH = 1800
SERVICE_TIME_TOUCHLESS = 600
SERVICE_TIME_POLISHING = 120
SERVICE_TIME_INTERIORWASH = 300
ABANDON_TIME_1 = 360
ABANDON_TIME_2 = 360

# FINITE SIMULATION
REPLICATIONS = 128
INTERVALS = 2  # num time slot

# Costi mensili
CAR_WASHER_COST = 1080  # costo lavaggista
TUNNEL_COST = 1000  # costo tunnel per lavaggio con rulli (touchless)
INTERIOR_MACHINE_COST = 300  # costo macchinario per lavaggio interno
CLEANSER_COST = 200  # costo detersivi (sia per lavaggio a mano che rulli)
CAR_WAX_COST = 100  # costo cera per lucidatura auto + cerchioni

# Profitti per singolo centro
HAND_WASH_PRICE = 25  # lavaggio singola auto
TOUCHLESS_WASH_PRICE = 10  # lavaggio singola auto
INTERIOR_WASH_PRICE = 2  # lavaggio singola auto
