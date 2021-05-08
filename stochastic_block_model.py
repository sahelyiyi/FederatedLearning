from numpy.random import normal
from graspy.simulations import sbm
import numpy as np


def get_B_and_weight_vec(n, pin=0.5, pout=0.01, mu_in=8, mu_out=2):
    p = []
    wt = []
    wtargs = []
    for i in range(len(n)):
        sub_p = []
        sub_wt = []
        sub_wtargs = []
        for j in range(len(n)):
            sub_wt.append(normal)
            if i == j:
                sub_p.append(pin)
                sub_wtargs.append(dict(loc=mu_in, scale=1))
            else:
                sub_p.append(pout)
                sub_wtargs.append(dict(loc=mu_out, scale=1))

        wt.append(sub_wt)
        p.append(sub_p)
        wtargs.append(sub_wtargs)

    G = sbm(n=n, p=p, wt=wt, wtargs=wtargs)

    N = len(G)
    E = int(len(np.argwhere(G > 0))/2)
    B = np.zeros((E, N))
    weight_vec = np.zeros(E)
    cnt = 0
    for item in np.argwhere(G > 0):
        i, j = item
        if i > j:
            continue
        if i == j:
            print ('nooooo')
        B[cnt, i] = 1
        B[cnt, j] = -1

        weight_vec[cnt] = abs(G[i, j])
        cnt += 1

    return B, weight_vec
