from DES_Python.rvgs import Exponential, Normal, Uniform, TruncatedNormal
from DES_Python.rvms import idfNormal, cdfNormal
from configurations import SERVICE_TIME_HANDWASH

if __name__ == '__main__':
    print("holaaa")

    mean = 30.0 * 60.0 # Media in secondi
    std_dev = 3.0  * 60 # Deviazione standard in secondi
    print(mean)
    print(std_dev)

    a = 0.0 # tempi di servizio non possono essere negativi
    b = 35.0 * 60 # 30+3= 33 minuti --> tempi di servizio sono minori di 35 --> 35*60 secondi
    print(b)
    alpha = cdfNormal(mean, std_dev, a)
    beta = 1.0 - cdfNormal(mean,std_dev,b)

    """
    print("alpha =" ,alpha)
    print("beta",beta)
    u = Uniform(alpha, 1.0-beta)
    print("u =", u)
    distr = idfNormal(mean, std_dev, u)

    print("distr == ", distr)

    multiplo_di_5 = ((mean+std_dev) // 5.0 + 1.0) * 5.0
    print("multiplo_di_5 = ", multiplo_di_5)

    print(Exponential(1800.0))

    
    """

    TruncatedNormal(mean, std_dev, a, b)