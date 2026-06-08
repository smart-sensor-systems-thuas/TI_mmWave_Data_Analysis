import mmWave_Utils as utils
import numpy as np 
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import re
from scipy.stats import linregress

rootdir = r'c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Accelerometer_inVivo'
vibDir = r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Trillingsmetingen\7040"
freqFile = f"{vibDir}\\frequencies.npy"
powFile = f"{vibDir}\\Power.npy"

#I = np.load(powFile)
#F = np.load(freqFile)


#f = pd.read_table(r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Accelerometer_inVivo\vast\vibration_103-20260527112603.txt",delimiter=",",header=None).to_numpy()[:,1:] 
#np.save(r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Accelerometer_inVivo\vast\vibration_103", df)
Acdata = np.load(r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Accelerometer_inVivo\vast\vibration_103.npy")[:,1:]

slope = 60.012e12
Fs = 10e6
Frames = 20000
Chirps = 16
Rx = 4
samples = 128

path = r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Betsy\Fastned\10.bin"

antenna = 1

mmData = utils.mmWaveData(path,slope=slope, sampleRate=Fs, frames=Frames, chirps=Chirps, Rx=Rx, samples=samples)
dt = 1/25000

mmData.rangeFFT()
pos = np.argmax(np.abs(mmData.rangeIntensity[0,0]))
arg = mmData.deltaArgument()

print(np.abs(arg).mean())

mmData.dArg[np.abs(arg) > 200*np.abs(arg).mean()] = np.abs(arg).mean()*np.sign(arg[np.abs(arg) > 200*np.abs(arg).mean()])

arg = mmData.interpolate(5e-4,2e-5,16)
print(np.abs(arg).mean())

arg = arg[::10]







print(arg.mean(axis=0))
arg = arg - arg.mean(axis=0)

"""
kern = np.ones(1000)/1000

for i in range(4):
    arg[:,i] = np.convolve(arg[:,i],kern,mode="same")
"""
radFFT = np.fft.fftshift(np.fft.fft(arg[:,antenna]))[arg.shape[0]//2:]



#for i in Acdata[-3]:


i = Acdata[3]

i = i - i.mean()


i[np.abs(i) > 10*np.abs(i).mean()] = np.abs(i).mean()*np.sign(i[np.abs(i) > 10*np.abs(i).mean()])

fig, ax = plt.subplots(1,2)
ax[0].plot(arg[:,0])
ax[1].plot(i)
plt.show()

f = np.arange(0, 12500 ,0.1)
fft = np.fft.fftshift(np.fft.fft(i))[i.size//2:]
fRad = np.arange(0,2000,0.1)

fft = np.abs(fft[1:10000])/(2*np.pi*f[1:10000])
fft[0:2] = 0.01
IRad = np.abs(radFFT[1:10000])
IAcc = fft
#IRad = np.convolve(IRad,np.ones(10)/10,mode="same")


print(IRad.size, IAcc.size)

"""
fig, ax = plt.subplots(1, 2)
ax[0].plot(i)
ax[1].plot(Acdata[4]-Acdata[4].mean())
plt.show()
"""

f1 = np.abs(np.fft.fftshift(np.fft.fft(i)))[125000:]
f1 = (f1/f1.max())[1:]
f2 = np.abs(np.fft.fftshift(np.fft.fft(Acdata[4]-Acdata[4].mean())))[125001:]/f[1:]
print(f2.max())
f2 = f2/f2.max()

"""
fig, ax = plt.subplots(1, 2)
ax[0].plot(f1-f2)
ax[1].plot(f2)#np.abs(np.fft.fftshift(np.fft.fft(Acdata[4]-Acdata[4].mean())))[125000:]/f)
plt.show()
"""

fig, ax = plt.subplots(1, 2)
ax[0].plot(f[1:10000],IAcc)
ax[1].plot(fRad[1:10000], IRad)
plt.show()

plt.figure()
plt.plot(f[1:10000],IAcc-IRad)
plt.show()


IRad = (IRad-IRad.mean())/(np.sqrt(IRad)*IRad.size)
IAcc = (IAcc-IAcc.mean())/(np.sqrt(IAcc)*IAcc.size)


print(np.max(IRad))
print(np.max(IAcc))



corr = np.correlate(IRad, IAcc,mode="same")
plt.figure()
plt.plot(corr)
plt.show()

print(Acdata.size)    
print(Acdata.shape)
