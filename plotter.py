# coding=utf-8
from datetime import time
from time import sleep

import matplotlib.pyplot as plt

BATCHNUM = 128
STATISTICS = 10


def plotInfiniteHorizon(centro):
    if centro < 1 | centro > 5:
        print("Centro non valido...\n")
        return 1

    utilizzazioni = []
    numeroCoda = []
    numeroCentro = []
    servizi = []
    attese = []
    risposte = []
    interarrivo = []
    arriviFam = []
    # arriviAuto = []
    nmPerdite = []
    probDiPerdita = []
    statistiche = []

    statistiche.append(utilizzazioni)
    statistiche.append(numeroCoda)
    statistiche.append(numeroCentro)
    statistiche.append(servizi)
    statistiche.append(attese)
    statistiche.append(risposte)
    statistiche.append(interarrivo)
    statistiche.append(arriviFam)
    # statistiche.append(arriviAuto)
    statistiche.append(nmPerdite)
    statistiche.append(probDiPerdita)

    filename = "./data/infinite/node" + str(centro) + ".dat"
    f = open(filename, "r")

    numeroBatch = 0
    stat = 0
    item1 = 0
    item2 = 0
    item3 = 0
    item4 = 0
    item5 = 0
    item6 = 0
    item7 = 0
    # item8 = 0
    item9 = 0
    item10 = 0
    item11 = 0
    count = 0

    for line in f:
        if line != "\n":

            if stat == 0:
                rigaSplittata = line.split(";")
                for rho in rigaSplittata:
                    if rho != "\n":
                        item1 = item1 + 1
                        statistiche[stat].append(float(rho))
            elif stat == 1:
                rigaSplittata = line.split(";")
                for q in rigaSplittata:
                    if q != "\n":
                        item2 = item2 + 1
                        statistiche[stat].append(float(q))
            elif stat == 2:
                rigaSplittata = line.split(";")
                for n in rigaSplittata:
                    if n != "\n":
                        item3 = item3 + 1
                        statistiche[stat].append(float(n))
            elif stat == 3:
                rigaSplittata = line.split(";")
                for s in rigaSplittata:
                    if s != "\n":
                        item4 = item4 + 1
                        statistiche[stat].append(float(s))
            elif stat == 4:
                rigaSplittata = line.split(";")
                for d in rigaSplittata:
                    if d != "\n":
                        item5 = item5 + 1
                        statistiche[stat].append(float(d))
            elif stat == 5:
                rigaSplittata = line.split(";")
                for w in rigaSplittata:
                    if w != "\n":
                        item6 = item6 + 1
                        statistiche[stat].append(float(w))
            elif stat == 6:
                rigaSplittata = line.split(";")
                for r in rigaSplittata:
                    if r != "\n":
                        item7 = item7 + 1
                        statistiche[stat].append(float(r))
            elif stat == 8:
                rigaSplittata = line.split(";")
                for a in rigaSplittata:
                    if a != "\n":
                        item9 = item9 + 1
                        statistiche[stat].append(float(a))
            elif stat == 9:
                rigaSplittata = line.split(";")
                for np in rigaSplittata:
                    if np != "\n":
                        item10 = item10 + 1
                        statistiche[stat].append(float(np))
            elif stat == 10:
                rigaSplittata = line.split(";")
                for p in rigaSplittata:
                    if p != "\n":
                        item11 = item11 + 1
                        statistiche[stat].append(float(p))
            count = count + 1
            numeroBatch = numeroBatch + 1
            if numeroBatch % BATCHNUM == 0:
                stat = stat + 1

    fontLabel = {'color': 'black', 'size': 28}
    fontTitle = {'color': 'black', 'size': 36}
    FONTNUM = 36

    y = statistiche[0]
    plt.plot(y, color='r')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("rho", fontdict=fontLabel)
    plt.xlabel("Batch", fontdict=fontLabel)
    plt.title("UTILIZZAZIONE", fontdict=fontTitle)
    plt.show()

    y = statistiche[1]
    plt.plot(y, color='r')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("Nq", fontdict=fontLabel)
    plt.xlabel("Batch", fontdict=fontLabel)
    plt.title("NUMERO JOBS IN CODA", fontdict=fontTitle)
    plt.show()

    y = statistiche[2]
    plt.plot(y, color='r')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("Ns", fontdict=fontLabel)
    plt.xlabel("Batch", fontdict=fontLabel)
    plt.title("NUMERO JOBS NEL CENTRO", fontdict=fontTitle)
    plt.show()

    y = statistiche[3]
    plt.plot(y, color='r')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("S", fontdict=fontLabel)
    plt.xlabel("Batch", fontdict=fontLabel)
    plt.title("TEMPO DI SERVIZIO", fontdict=fontTitle)
    plt.show()

    y = statistiche[4]
    plt.plot(y, color='r')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("Tq", fontdict=fontLabel)
    plt.xlabel("Batch", fontdict=fontLabel)
    plt.title("TEMPO DI ATTESA", fontdict=fontTitle)
    plt.show()

    y = statistiche[5]
    plt.plot(y, color='r')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("Ts", fontdict=fontLabel)
    plt.xlabel("Batch", fontdict=fontLabel)
    plt.title("TEMPO DI RISPOSTA", fontdict=fontTitle)
    plt.show()

    y = statistiche[6]
    plt.plot(y, color='r')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("r", fontdict=fontLabel)
    plt.xlabel("Batch", fontdict=fontLabel)
    plt.title("INTERARRIVO", fontdict=fontTitle)
    plt.show()

    y = statistiche[7]
    plt.plot(y, color='r')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("F", fontdict=fontLabel)
    plt.xlabel("Batch", fontdict=fontLabel)
    plt.title("ARRIVI FAMIGLIE", fontdict=fontTitle)
    plt.show()

    y = statistiche[8]
    plt.plot(y, color='r')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("A", fontdict=fontLabel)
    plt.xlabel("Batch", fontdict=fontLabel)
    plt.title("ARRIVI AUTOMOBILI", fontdict=fontTitle)
    plt.show()

    y = statistiche[9]
    plt.plot(y, color='r')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("perdite", fontdict=fontLabel)
    plt.xlabel("Batch", fontdict=fontLabel)
    plt.title("NUMERO DI PERDITE NEL SISTEMA", fontdict=fontTitle)
    plt.show()

    y = statistiche[10]
    plt.plot(y, color='r')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("P(perdita)", fontdict=fontLabel)
    plt.xlabel("Batch", fontdict=fontLabel)
    plt.title("PROBABILITA' DI AVERE UNA PERDITA NEL SISTEMA", fontdict=fontTitle)
    plt.show()

    return 0


REPLICATIONS = 128
SAMPLESIZE = 108


def plotFiniteHorizon(centro):
    if centro < 1 | centro > 5:
        print("Centro non valido...\n")
        return 1

    utilizzazioni = []
    numeroCoda = []
    numeroCentro = []
    servizi = []
    attese = []
    risposte = []
    interarrivo = []
    arriviFam = []
    # arriviAuto = []
    nmPerdite = []
    probDiPerdita = []
    statistiche = []

    statistiche.append(utilizzazioni)
    statistiche.append(numeroCoda)
    statistiche.append(numeroCentro)
    statistiche.append(servizi)
    statistiche.append(attese)
    statistiche.append(risposte)
    statistiche.append(interarrivo)
    statistiche.append(arriviFam)
    # statistiche.append(arriviAuto)
    statistiche.append(nmPerdite)
    statistiche.append(probDiPerdita)

    # costruzione struttura dati per la manipolazione dei dati
    for stat in range(0, STATISTICS):
        statistiche[stat] = []
        for replica in range(0, REPLICATIONS):
            statistiche[stat].append([])

    filename = "./data/finite/samplingnode" + str(centro) + ".dat"
    f = open(filename, "r")

    replica = []
    for i in range(0, REPLICATIONS):
        replica.append(0)

    count = 0
    s = 0.0
    item1 = 0
    item2 = 0
    item3 = 0
    item4 = 0
    item5 = 0
    item6 = 0
    item7 = 0
    item8 = 0
    item9 = 0
    item10 = 0

    for line in f:
        if line != "\n":
            if count % STATISTICS == 0:
                rigaSplittata = line.split(";")
                for rho in rigaSplittata:
                    if rho != "\n":
                        item1 = item1 + 1
                        statistiche[0][replica[0]].append(float(rho))
                replica[0] = replica[0] + 1
            elif count % STATISTICS == 1:
                rigaSplittata = line.split(";")
                for q in rigaSplittata:
                    if q != "\n":
                        item2 = item2 + 1
                        statistiche[1][replica[1]].append(float(q))
                replica[1] = replica[1] + 1
            elif count % STATISTICS == 2:
                rigaSplittata = line.split(";")
                for n in rigaSplittata:
                    if n != "\n":
                        item3 = item3 + 1
                        statistiche[2][replica[2]].append(float(n))
                replica[2] = replica[2] + 1
            elif count % STATISTICS == 3:
                rigaSplittata = line.split(";")
                for s in rigaSplittata:
                    if s != "\n":
                        item4 = item4 + 1
                        statistiche[3][replica[3]].append(float(s))
                replica[3] = replica[3] + 1
            elif count % STATISTICS == 4:
                rigaSplittata = line.split(";")
                for d in rigaSplittata:
                    if d != "\n":
                        item5 = item5 + 1
                        statistiche[4][replica[4]].append(float(d))
                replica[4] = replica[4] + 1
            elif count % STATISTICS == 5:
                rigaSplittata = line.split(";")
                for w in rigaSplittata:
                    if w != "\n":
                        item6 = item6 + 1
                        statistiche[5][replica[5]].append(float(w))
                replica[5] = replica[5] + 1
            elif count % STATISTICS == 6:
                rigaSplittata = line.split(";")
                for r in rigaSplittata:
                    if r != "\n":
                        item7 = item7 + 1
                        statistiche[6][replica[6]].append(float(r))
                replica[6] = replica[6] + 1
            elif count % STATISTICS == 7:
                rigaSplittata = line.split(";")
                for f in rigaSplittata:
                    if f != "\n":
                        item8 = item8 + 1
                        statistiche[7][replica[7]].append(float(f))
                replica[7] = replica[7] + 1
            elif count % STATISTICS == 8:
                rigaSplittata = line.split(";")
                for np in rigaSplittata:
                    if np != "\n":
                        item9 = item9 + 1
                        statistiche[8][replica[8]].append(float(np))
                replica[8] = replica[8] + 1
            elif count % STATISTICS == 9:
                rigaSplittata = line.split(";")
                #print("riga splittata === ", rigaSplittata)
                for p in rigaSplittata:
                    if p != "\n":
                        item10 = item10 + 1
                        statistiche[9][replica[9]].append(float(p))
                replica[9] = replica[9] + 1
            count = count + 1
    #sleep(60)
    print(" print dimensioni statistiche ----- ")
    print(len(statistiche[0]))
    print(len(statistiche))
    print(" print dimensioni statistiche 2----- ")
    print((statistiche[0][0]))
    print(" print dimensioni statistiche 3----- ")
    print((statistiche[0][0][0]))

    print("----------------------------")
    #print(statistiche[9])
    print("----------------------")
    # campioni
    rho = []
    q = []
    n = []
    s = []
    tq = []
    ts = []
    r = []
    f = []
    np = []
    p = []
    stats = []
    stats.append(rho)
    stats.append(q)
    stats.append(n)
    stats.append(s)
    stats.append(tq)
    stats.append(ts)
    stats.append(r)
    stats.append(f)
    stats.append(np)
    stats.append(p)

    count = 0

    # costruzione dei campioni
    for stat in range(0, STATISTICS):
        for value in range(0, SAMPLESIZE):
            sommatoria = 0.0
            for rep in range(0, REPLICATIONS):
                # print(f"stat = {stat}, rep = {rep}, value = {value}")
                sommatoria = sommatoria + statistiche[stat][rep][value]
            media = float(sommatoria / REPLICATIONS)
            if stat == 4:
                stats[stat].append(-media)
            else:
                stats[stat].append(media)
    for i in range(0, STATISTICS):
        count = 0
        for j in stats[i]:
            count = count + 1

    # costruisco il grafico

    fontLabel = {'color': 'black', 'size': 10}
    fontTitle = {'color': 'black', 'size': 20}
    FONTNUM = 8

    y = stats[0]
    # ticks = [time(8, 30), time(10,10), time(11,50), time(13,30), time(15,10), time(17,30)]
    plt.plot(y, color='b')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("rho", fontdict=fontLabel)
    plt.xlabel("Campione", fontdict=fontLabel)
    # plt.xticks(ticks)
    plt.title("UTILIZZAZIONE", fontdict=fontTitle)
    plt.show()

    y = stats[1]
    plt.plot(y, color='b')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("Nq", fontdict=fontLabel)
    plt.xlabel("Campione", fontdict=fontLabel)
    plt.title("NUMERO JOBS IN CODA", fontdict=fontTitle)
    plt.show()

    y = stats[2]
    plt.plot(y, color='b')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("Ns", fontdict=fontLabel)
    plt.xlabel("Campione", fontdict=fontLabel)
    plt.title("NUMERO JOBS NEL CENTRO", fontdict=fontTitle)
    plt.show()

    y = stats[3]
    plt.plot(y, color='b')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("S", fontdict=fontLabel)
    plt.xlabel("Campione", fontdict=fontLabel)
    plt.title("TEMPO DI SERVIZIO", fontdict=fontTitle)
    plt.show()

    y = stats[4]
    y2 = 300
    x1 = 54  # time slot
    plt.plot(y, color='b', label="Tempo di attesa")
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.axhline(y2, linestyle="dashed", color="r", label="QoS")
    plt.axvline(x1, linestyle=":", color="g", label="Cambio fascia oraria")
    plt.xticks()
    plt.ylabel("Tempo di attesa", fontdict=fontLabel)
    plt.xlabel("Tempo", fontdict=fontLabel)
    plt.grid(axis="y")
    plt.title("TEMPO DI ATTESA", fontdict=fontTitle)
    plt.legend(loc="upper left")


    max_y = max(y)
    xpos = y.index(max_y)
    print("xpos ==", xpos)
    print("max_y ==", max_y)
    text= "y={:.3f}".format(max_y)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    #plt.annotate(text, xy=(xpos, max_y),xytext=(0.80,0.99), **kw)
    plt.annotate(text, xy=(xpos, max_y),xytext=(0.7,0.5), **kw)

    plt.show()

    y = stats[5]
    y2 = 2700
    x1 = 54
    plt.plot(y, color='b', label  = "Tempo di risposta")
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.axhline(y2, linestyle="dashed", color="r", label="QoS")
    plt.axvline(x1, linestyle=":", color="g", label="Cambio fascia oraria")
    plt.ylabel("Tempo di risposta", fontdict=fontLabel)
    plt.xlabel("Campione", fontdict=fontLabel)
    plt.grid(axis="y")
    plt.title("TEMPO DI RISPOSTA", fontdict=fontTitle)
    plt.legend(loc="upper left")

    plt.show()

    y = stats[6]
    plt.plot(y, color='b')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("r", fontdict=fontLabel)
    plt.xlabel("Campione", fontdict=fontLabel)
    plt.title("INTERARRIVO", fontdict=fontTitle)
    plt.show()

    y = stats[7]
    plt.plot(y, color='b')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("F", fontdict=fontLabel)
    plt.xlabel("Campione", fontdict=fontLabel)
    plt.title("ARRIVI FAMIGLIE", fontdict=fontTitle)
    plt.show()

    y = stats[8]
    plt.plot(y, color='b')
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("perdite", fontdict=fontLabel)
    plt.xlabel("Campione", fontdict=fontLabel)
    plt.title("NUMERO DI PERDITE NEL SISTEMA", fontdict=fontTitle)
    plt.show()

    y = stats[9]
    #print("y = ",y)
    y2 = 0.2
    x1 = 54
    plt.plot(y, color='b', label = "Probabilità perdita")
    plt.xticks(fontsize=FONTNUM)
    plt.yticks(fontsize=FONTNUM)
    plt.ylabel("Probabilità perdita", fontdict=fontLabel)
    plt.xlabel("Campione", fontdict=fontLabel)
    plt.axhline(y2, linestyle="dashed", color="r", label="QoS")
    plt.axvline(x1, linestyle=":", color="g", label="Cambio fascia oraria")
    plt.grid(axis="y")
    plt.title("PROBABILITA' DI AVERE UNA PERDITA NEL CENTRO", fontdict=fontTitle)
    plt.legend(loc="upper left")
    plt.show()

    return 0

#plotFiniteHorizon(4)

"""
plotFiniteHorizon(1)
plotFiniteHorizon(2)
plotFiniteHorizon(3)
plotFiniteHorizon(4)
plotFiniteHorizon(5)
plotInfiniteHorizon(1)
plotInfiniteHorizon(2)
plotInfiniteHorizon(3)
plotInfiniteHorizon(4)
plotInfiniteHorizon(5)

"""
if __name__ == '__main__':
    plotFiniteHorizon(1)