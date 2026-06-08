import mmWave_Utils as utils
import numpy as np 
import matplotlib.pyplot as plt
import os
from pathlib import Path
import re
from scipy.stats import linregress
from scipy.optimize import curve_fit as cf
import scipy.stats as stats
import pandas as pd

def line(x, a, b, da, db):
    y = a*x + b
    ymin = (a - da)*x + b-db
    ymax = (a + da)*x + b+db
    return y, ymin, ymax

def gainAcc(x):
    return 10**(-0.5 * np.log10(x)/10)

def gainFunc(f,f0,a,c):
    return a / (np.sqrt((1-(-f/f0)**2)**2 + 4*(-c*f/f0)**2)*f0)

Antenna = 0

rootdir = r'C:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Accelerometer_Data'
vibDir = r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Trillingsmetingen"


for files in os.walk(rootdir):
    for file in files:
        fnames = file
freq = []
data = np.array([])
for i,name in enumerate(fnames):
    if name[-4:] == ".npy":
        continue

    freq.append(float(name.split("-")[1].split("Hz")[0]))
    """
    df = pd.read_table(f"{rootdir}\\{name}",delimiter=",",header=None).to_numpy()[:,1:] 
    if data.size == 0:
        data = np.array([df])
    else:
        data = np.vstack((data,[df]))
    """
#np.save(f"{rootdir}\\numpydata", data)
diams = []
Il = []
Fl = []
for i,files in enumerate(os.walk(vibDir)):
    if i == 0:
        continue

    root = files[0]
    print(root)
    diams.append(int(root[-4:]))
    pow = files[2][-1]
    freqen = files[2][-2]
    Il.append(np.load(f"{root}\\{pow}"))
    Fl.append(np.load(f"{root}\\{freqen}"))
    
# for i in Il:
#     print(i.shape)
print(diams)
freqFile = f"{vibDir}\\frequencies.npy"
powFile = f"{vibDir}\\Power.npy"

bigPrun = []
bigV = []
bigF = []
bigI = []

Antennas = [1,1,2,1,2]

for i, val in enumerate(Il):

    I = val
    F = Fl[i]
    print(I.shape)

    data = np.load(f"{rootdir}\\numpydata.npy")

    Fs = 25e3
    Ts = 10
    res = 1/Ts
    amplitudes = np.array([])
    freq = np.array(freq)
    pruned = []
    for i, disp in enumerate(data):
        index = int(freq[i]/res)
        size = disp[0].size//2
        fft0 = np.fft.fftshift(np.fft.fft(disp[0]))[size:]
        fft1 = np.fft.fftshift(np.fft.fft(disp[1]))[size:]

        
        indexes = [np.argmax(np.abs(fft0)), np.argmax(np.abs(fft0))]
        if True: #np.abs(index - indexes[0]) < 100:
            pruned.append(freq[i])
            if amplitudes.size == 0:
                amplitudes = np.array([fft0[index],fft1[index]]) #
            else:
                amplitudes = np.vstack((amplitudes,[fft0[index],fft1[index]]))

    #I *= (F > 3)
    #F *= (F > 3)


    pruned = np.array(pruned)

    fBool = np.isin(F, pruned)
    F = F*fBool
    I *= fBool
    F = F[F != 0].reshape((4,-1))
    I = I[I != 0].reshape((4,-1))

    fBool = np.isin(pruned, F)
    pruned *= fBool
    amplitudes *= np.array([fBool, fBool]).T
    pruned = pruned[pruned != 0]#.reshape((2, -1))

    pruned = np.concatenate((pruned[:11], pruned[12:]))
    amplitudes = np.concatenate((amplitudes[:11], amplitudes[12:]))

    amplitudes = np.abs(amplitudes)

    amplitudes = amplitudes[amplitudes != 0].reshape((-1,2))


    for i in range(F.shape[0]):
        F[i] = F[i,np.argsort(F[i])]
        I[i] = I[i,np.argsort(F[i])]

    ind = np.argsort(pruned)
    pruned = pruned[np.argsort(pruned)]

    amplitudes = amplitudes.T
    for i in range(amplitudes.shape[0]):
        amplitudes[i] = amplitudes[i,ind]

    amplitudes = amplitudes[:,1:].T
    pruned = pruned[1:]

    #print(amplitudes)

    G = gainAcc(pruned)

    v = G * np.abs(amplitudes.T[1])/((2*np.pi*pruned))
    maximum = np.max(v)



    """
    opt, cov = cf(gainFunc, pruned[4:-10], (1/maximum)*np.abs(amplitudes.T[0,4:-10])/(2*np.pi*pruned[4:-10]*G[4:-10]))
    print(opt)
    print(F[0])
    print(pruned)
    """
    I = I[:,1:-3]
    F = F[:,1:-3]
    v = v[:-3]
    pruned = pruned[:-3]

    s = np.sort(pruned[4:-1])
    #v = (v-v.mean())/(np.sqrt(v)*v.size)

    #I = (I-np.array([I.mean(axis=1)]).T)/(np.sqrt(i)*I[0].size)

    I = I/np.array([I.max(axis=1)]).T
    v = v/v.max()
    bigPrun.append(pruned)
    bigV.append(v)
    bigF.append(F[0])
    bigI.append(I[Antennas[i]])

    s = np.sort(pruned[4:-1])

    res = I[Antenna]/np.max(I[Antenna]) - v
    print()
    for i in range(4):
        res0 = I[i] - v
        print(stats.wilcoxon(res0))

markers = ["x","+",".","2","^"]
colors = ["#0077BB","#33BBEE","#009988","#EE7733","#CC3311", "#EE3377"]
fig, ax = plt.subplots(2,3,sharey=True, sharex=True)
for i, val in enumerate(bigI):
    j = i 
    if  i > 1:
        j += 1
    x = j % 3
    y = int(np.floor(j/3))

    if i == 0:
        ax[y,x].scatter(bigPrun[i], bigV[i],marker="o", label="Accelerometer",c=colors[-1])
    else:
        ax[y,x].scatter(bigPrun[i], bigV[i],marker="o",c=colors[-1])
    #ax[y,x].set_yscale("log", nonpositive='mask')
    #ax[y,x].set_xscale("log", nonpositive='mask')
    ax[y,x].scatter(bigF[i] ,val, label=f"{np.round(diams[i]/100,1)}mm",c=colors[i],marker=markers[i])
    ax[y,x].grid(c="k",ls="--",alpha=0.25)

ax[0,0].set_ylabel(r"$\frac{I}{I_0}$ [-]")
ax[1,0].set_ylabel(r"$\frac{I}{I_0}$ [-]")

for i in range(3):
    ax[1,i].set_xlabel(r"$f$[Hz]")


ax[0,2].axis("off")
fig.legend(loc="upper right", bbox_to_anchor=(0.43, 0.37, 0.5, 0.5))
#plt.savefig("RefvsAcc.pdf",format="pdf")
plt.show()

fig, ax = plt.subplots(2,3,sharey=True, sharex=True)
for i, val in enumerate(bigI):
    j = i 
    if  i > 1:
        j += 1
    x = j % 3
    y = int(np.floor(j/3))
    ax[y,x].scatter(bigF[i][bigF[i]<2000] ,np.abs((val-bigV[i])/bigV[i])[bigF[i]<2000], label=f"{np.round(diams[i]/100,1)}mm",c=colors[i],marker=markers[i])
    ax[y,x].grid(c="k",ls="--",alpha=0.25)

ax[0,0].set_ylabel(r"$\Delta\frac{I}{I_0}$ [-]")
ax[1,0].set_ylabel(r"$\Delta\frac{I}{I_0}$ [-]")

for i in range(3):
    ax[1,i].set_xlabel(r"$f$[Hz]")


ax[0,2].axis("off")
fig.legend(loc="upper right", bbox_to_anchor=(0.43, 0.37, 0.5, 0.5))
#plt.savefig("ResidualRefvsAcc.pdf",format="pdf")
plt.show()


        