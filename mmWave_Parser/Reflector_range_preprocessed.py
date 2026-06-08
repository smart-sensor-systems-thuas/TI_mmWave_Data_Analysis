import mmWave_Utils as utils
import numpy as np 
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import re
from scipy.stats import linregress
import scipy.stats as stats

def line(x, a, b, da, db):
    y = a*x + b
    ymin = (a - da)*x + b-db
    ymax = (a + da)*x + b+db
    return y, ymin, ymax



fac = 3*np.sqrt(3)/8

ICop = fac*np.load(r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\reflector_power\PCopper.npy")
xCop = fac*np.load(r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\reflector_power\AdivRCopper.npy")

ICon = fac*np.load(r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Conductive_reflector_power\Pconductive.npy")
xCon = fac*np.load(r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Conductive_reflector_power\AdivRConductive.npy")

copReg = linregress(xCop[:,:-1].flatten(), ICop[:,:-1].flatten())
conReg = linregress(xCon[:,:-1].flatten(), ICon[:,:-1].flatten())

cop, copMin, copMax = line(1.01*xCon[:,:-1].flatten(),copReg.slope, copReg.intercept, 3*copReg.stderr, 3*copReg.intercept_stderr)
con, conMin, conMax = line(1.01*xCon[:,:-1].flatten(),conReg.slope, conReg.intercept, 3*conReg.stderr, 3*conReg.intercept_stderr)
print(copReg.rvalue**2, conReg.rvalue**2)

print(copReg.rvalue**2, conReg.rvalue**2)
print(stats.pearsonr(xCop[:,:-1].flatten(), ICop[:,:-1].flatten()))
print(stats.pearsonr(xCon[:,:-1].flatten(), ICon[:,:-1].flatten()))
cuDis = f"$y = ({int(np.round(copReg.slope))}\\pm{int(np.round(3*copReg.stderr))})x + {np.round(copReg.intercept,3)} \\pm {np.round(3*copReg.intercept_stderr,3)}$"
coDis = f"$y = ({np.round(conReg.slope,1)}\\pm{np.round(3*conReg.stderr,1)})x + {np.round(conReg.intercept,3)} \\pm {np.round(3*conReg.intercept_stderr,3)}$"

fig,ax = plt.subplots(1,2,sharex=True,sharey=True)

for i in ax:
    i.grid(c="k", alpha=0.25, ls="--")
    i.set_xlabel(r"$\frac{A}{R^4}$ [m$^{-2}$]")
ax[0].scatter(xCop[:,:-1], ICop[:,:-1],c="firebrick",marker="+",label="copper Lined")
ax[0].plot(1.01*xCon[:,:-1].flatten(), cop, c="firebrick",label=cuDis)
ax[0].plot(1.01*xCon[:,:-1].flatten(), copMin, c="firebrick", ls="--")
ax[0].plot(1.01*xCon[:,:-1].flatten(), copMax, c="firebrick", ls="--")
ax[0].fill_between(1.01*xCon[:,:-1].flatten(),copMin, copMax, color="k", alpha=0.1)

ax[1].scatter(1.01*xCon[:,:-1], ICon[:,:-1],c="k",marker="x",label="conductive")
ax[1].plot(1.01*xCon[:,:-1].flatten(), con, c="k",label=coDis)
ax[1].plot(1.01*xCon[:,:-1].flatten(), conMin, c="k", ls="--")
ax[1].plot(1.01*xCon[:,:-1].flatten(), conMax, c="k", ls="--")
ax[1].fill_between(1.01*xCon[:,:-1].flatten(),conMin, conMax, color="k", alpha=0.1)

custom_xlim = (0, np.max(1.01*xCon[:,:-1]))
custom_ylim = (0, np.max(cop))
plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
ax[0].set_ylabel(r"$\frac{I}{I_0}$ [-]")
fig.legend()
fig.set_figheight(4.7)
fig.set_figwidth(10)
plt.savefig("refRange.pdf",format="pdf")
plt.show()