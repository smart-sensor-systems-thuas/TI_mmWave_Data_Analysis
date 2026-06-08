import mmWave_Utils as utils
import numpy as np 
import matplotlib.pyplot as plt
import os
from pathlib import Path
import re
from scipy.stats import linregress
import scipy.stats as stats

def line(x, a, b, da, db):
    y = a*x + b
    ymin = (a - da)*x + b-db
    ymax = (a + da)*x + b+db
    return y, ymin, ymax

rootdir = r'c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\reflector_responsiviteit'

slope = 60.012e12
Fs = 5e6
Frames = 8
Chirps = 128
Rx = 4
samples = 256

for files in os.walk(rootdir):
    for file in files:
        fnames = file

copperData = np.array([])
copperID = np.array([])
conductiveData = np.array([])
conductiveID = np.array([])
data = {}
for i,name in enumerate(fnames):
    if name.split("_")[-1][-4:] != ".bin":
        continue


    size = int(re.findall(r'\d+',name.split("_")[0])[0])
    if size == 8:
        size = 28.55
    elif size == 16:
        size = 39.05
    elif size == 32:
        size = 49.4
    elif size == 48:
        size = 60.015
    elif size == 80:
        size = 70.4

    num = int(re.findall(r'\d+',name.split("_")[2])[0])
    material = name.split("_")[1]    
    y = utils.mmWaveData(f"{rootdir}\\{name}",slope,Fs,Frames,Chirps,Rx,samples)
    if material == "Cond":
        if conductiveID.size == 0:
            conductiveID = np.array([size,num])
            conductiveData = np.array([y])
        else:
            conductiveID = np.vstack((conductiveID, [size,num]))
            conductiveData = np.vstack((conductiveData, [y]))
    else:
        if copperID.size == 0:
            copperID = np.array([size,num])
            copperData = np.array([y])
        else:
            copperID = np.vstack((copperID, [size,num]))
            copperData = np.vstack((copperData, [y]))
print(copperData.shape)
maxCopper = []
maxConductive = []


a, b = copperData[4,0].rangeFFT()
cuX = np.argmax(np.abs(b[0,0]))
print(cuX, a[cuX])
for i in copperData:
    pos, data = i[0].rangeFFT()
    #plt.figure()
    #plt.plot(pos,np.abs(data[0,0]))
    #plt.show()
    tally = []
    data.reshape((4,1024,128))
    for j, val in enumerate(data):
        tally.append(np.mean(np.abs(data[j,:,cuX]),axis=-1))
    maxCopper.append(np.mean(tally))


maxCopper = np.array(maxCopper)

tempID = []
tempData = []
for i,val in enumerate(copperID):
    if True: #val[0] != 70.4:
         tempID.append(val)
         tempData.append(maxCopper[i])

copperID = np.array(tempID)
maxCopper = np.array(tempData)/100

a, b = conductiveData[1,0].rangeFFT()
coX = np.argmax(np.abs(b[0,0]))

for i in conductiveData:
    pos, data = i[0].rangeFFT()
    tally = []
    data.reshape((4,1024,128))
    for j, val in enumerate(data):
        tally.append(np.mean(np.max(np.abs(data[j,:coX]),axis=-1)))
    maxConductive.append(np.mean(tally))

tempID = []
tempData = []
removed = []
for i,val in enumerate(conductiveID):
    if val[0] != 70.4:
         tempID.append(val)
         tempData.append(maxConductive[i])
    else:
         removed.append(maxConductive[i])


conductiveID = np.array(tempID)
maxConductive = np.array(tempData)/100
#maxCopper = maxCopper.reshape(5,5)

cuLine = linregress((3*np.sqrt(3)/8)*(copperID.T[0])**2, maxCopper)
coLine = linregress((3*np.sqrt(3)/8)*(conductiveID.T[0])**2, maxConductive)

print(cuLine)
print("vals")
print(cuLine.rvalue**2,cuLine.pvalue)
print(coLine.rvalue**2,coLine.pvalue)
#maxCopper = maxCopper.reshape(25,1)

x = (3*np.sqrt(3)/8)*np.linspace(0,90,100)**2
cu, cuMin, cuMax = line(x,cuLine.slope,cuLine.intercept, 3*cuLine.stderr, 3*cuLine.intercept_stderr )
co, coMin, coMax = line(x,coLine.slope,coLine.intercept, 3*coLine.stderr, 3*coLine.intercept_stderr )

print(cuLine.slope,3*cuLine.stderr,cuLine.intercept, 3*cuLine.intercept_stderr, cuLine.rvalue**2)
print(coLine.slope,3*coLine.stderr,coLine.intercept, 3*coLine.intercept_stderr, coLine.rvalue**2)

cuDis = f"$y = ({(int(np.round(cuLine.slope,6)*10**6))}\\pm{(np.round(3*cuLine.stderr,6)*10**6)})\\cdot" + "10^{-6}" + f"\\cdot x + {np.round(cuLine.intercept,2)} \\pm {np.round(3*cuLine.intercept_stderr,2)}$"
coDis = f"$y = ({int(np.round(coLine.slope,6)*10**6)}\\pm{np.round(3*coLine.stderr,6)*10**6}) \\cdot" +  "10^{-6}" + f"\\cdot x + {np.round(coLine.intercept,3)} \\pm {np.round(3*coLine.intercept_stderr,3)}$"


fig, ax = plt.subplots(1,2,sharey=True)

for i in ax:
    i.grid(c="k", alpha=0.25, ls="--")
    i.set_xlabel("$A$[mm$^{2}$]")

ax[0].scatter((3*np.sqrt(3)/8)*copperID.T[0]**2, maxCopper,marker="+",color="firebrick",label="Copper Lined")
ax[0].plot(x,cu,c="firebrick", label=cuDis)
ax[0].plot(x,cuMin,c="firebrick",ls="--")
ax[0].plot(x,cuMax,c="firebrick",ls="--")
ax[0].fill_between(x,cuMin, cuMax, color="k", alpha=0.1)

print(removed)
print((3*np.sqrt(3)/8)*np.array([[70.4], [70.4], [70.4], [70.4], [70.4]])**2)
ax[1].scatter((3*np.sqrt(3)/8)*conductiveID.T[0]**2, maxConductive, marker="x", color='k',label="Conductive")
ax[1].plot(x,co,c="k",label=coDis)
ax[1].plot(x,coMin,c="k",ls="--")
ax[1].plot(x,coMax,c="k",ls="--")
ax[1].scatter((3*np.sqrt(3)/8)*np.array([70.4, 70.4, 70.4, 70.4, 70.4])**2, np.array(removed)/100,marker="x", color='gold',label="Removed points")
ax[1].fill_between(x,coMin, coMax, color="k", alpha=0.1)
fig.legend()
ax[0].set_ylabel(r"$\frac{I}{I_0}$[-]")

custom_xlim = (0, 1.01*max(x))
custom_ylim = (np.min(cuMin), 1.01*np.max(cuMax))
plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)

fig.set_figheight(4.7)
fig.set_figwidth(10)
plt.savefig("anderedingen.pdf",format="pdf")
plt.show()
