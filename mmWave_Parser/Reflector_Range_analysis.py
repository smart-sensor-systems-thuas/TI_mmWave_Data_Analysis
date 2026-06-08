import mmWave_Utils as utils
import numpy as np 
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import re
from scipy.stats import linregress

def line(x, a, b, da, db):
    y = a*x + b
    ymin = (a - da)*x + b-db
    ymax = (a + da)*x + b+db
    return y, ymin, ymax



rootdir = r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Conductive_reflector_power"



slope = 60.012e12
Fs = 5e6
Frames = 8
Chirps = 128
Rx = 4
samples = 256

reflectors = []
indexes = []
apertures = []

for i, files in enumerate(os.walk(rootdir)):
    if i == 0:
        for j in files[1]:
            apertures.append(float(j)/100000)
        continue
    measurement = []
    index = []
    for file in files[-1]:
        #print(file,"\n")
        path = f"{files[0]}\\{file}"
        num = int(file.split(".")[0])
        measurement.append(utils.mmWaveData(path=path, slope=slope, sampleRate=Fs, frames=Frames, chirps=Chirps, Rx=Rx, samples=samples))
        index.append(num)
    reflectors.append(measurement)
    indexes.append(index)

df = pd.read_excel(r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Conductive_reflector_power\Afstanden.ods")
ranges = df.to_numpy().T


print(apertures)

ordered = np.copy(ranges)
#for i, val in enumerate(indexes):

newPeak = []
for num in range(5):
    peaks = []
    for i in reflectors[num]:
        a,b = i.rangeFFT()
        peaks.append(np.max(np.mean(np.mean(np.abs(b[0]), axis=1),axis=0)))
    
    oPeak = np.copy(peaks)
    for j,val in enumerate(indexes[num]):
        oPeak[val] = peaks[j]
    newPeak.append(oPeak)

newPeak = np.array(newPeak)#[:,:-1]
apertures  = np.array([apertures]).T
ranges = np.array(ranges)#[:,:-1]

print(ranges.shape)
regLine = linregress(((apertures**2)*ranges**(-4)).flatten(), newPeak.flatten())
print(regLine.rvalue**2)
x = ((apertures**2)*np.array(ranges)**(-4)).flatten()
lin, mini, maxi = line(x,regLine.slope, regLine.intercept, 3*regLine.stderr, 3*regLine.intercept_stderr)

chis = np.sum(((lin-np.array(newPeak).flatten())**2)/lin)

plt.figure()
plt.scatter(x,x)
plt.show()

#np.save(r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Conductive_reflector_power\AdivRConductive",(apertures**2)*np.array(ranges)**(-4))
#np.save(r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Conductive_reflector_power\Pconductive",newPeak)


plt.figure()
plt.scatter(x, np.array(newPeak).flatten())
plt.plot(x,lin)
plt.plot(x,mini)
plt.plot(x, maxi)
plt.xlabel("$\\frac{d^2}{R^4}$[m$^{-2}$]")
plt.show()

plt.figure()
plt.scatter(np.array(ranges)[3], np.array(newPeak)[3])
plt.xlabel("$R$[m]")
plt.show()



"""
cu, cuMin, cuMax = line(apertures[num]*np.array(ranges[num])**(-4),regLine.slope,regLine.intercept, 3*regLine.stderr, 3*regLine.intercept_stderr )

plt.figure()
plt.scatter(ranges[num], oPeak)
plt.plot(ranges[num],cu)
plt.plot(ranges[num],cuMin)
plt.plot(ranges[num],cuMax)
plt.show()
"""  