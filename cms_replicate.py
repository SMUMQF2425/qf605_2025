import numpy as np
import matplotlib.pylab as plt


def IRR(x, N):
    return 1.0/x * (1.0 - 1.0/(1+x)**N)


no_of_swaptions = 2
N = 30  # tenor
Ks = np.linspace(0.05, 0.15, no_of_swaptions+1)
weights = []
for j, K in enumerate(Ks[:-1]):
    if j == 0:
        weights.append(1/IRR(Ks[1], N))
    elif j == 1:
        weights.append(2/IRR(Ks[2], N) - 2/IRR(Ks[1], N))
    else:
        weights.append((j+1)/IRR(Ks[j+1], N) - 2*j/IRR(Ks[j], N)
                       + (j-1)/IRR(Ks[j-1], N))


def caplet(F, K):
    return max(F-K, 0)


def swaption(F, K):
    return IRR(F, N) * (max(F-K, 0))


def replication(F, K):
    res = 0.0
    for w, rK in zip(weights, Ks[:-1]):
        res += w*swaption(F, rK)

    return res


cmsreplicate = []
cmscaplet = []
full_strikes = np.linspace(0.001, 0.15, 1000)
cms_strike = 0.05
for F in full_strikes:
    cmsreplicate.append(replication(F, cms_strike))
    cmscaplet.append(caplet(F, cms_strike))


plt.plot(full_strikes, cmsreplicate, label='replication')
plt.plot(full_strikes, cmscaplet, label='caplet')
plt.xlabel('Swap Rate')
plt.ylabel('Payoff')
for i in np.arange(len(Ks[:-1])):
    plt.plot(full_strikes,
             weights[i]*np.array(list(map(lambda x: swaption(x, Ks[i]),
                                          full_strikes))), 'k:')
plt.legend()
plt.show()
