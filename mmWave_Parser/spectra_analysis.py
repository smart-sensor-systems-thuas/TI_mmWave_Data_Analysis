import mmWave_Utils as utils
import numpy as np 
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import re
from scipy.stats import linregress
import scipy.constants as con

def line(x, a, b, da, db):
    y = a*x + b
    ymin = (a - da)*x + b-db
    ymax = (a + da)*x + b+db
    return y, ymin, ymax

"""
def interpolate(arr, Tf, Tc, Nc):
    size = arr.shape[0]
    size *= (Tf)/(Tc*Nc)
    arr = np.concatenate((np.array([arr[0,:]]),arr))
    arr = arr.reshape((arr.shape[0]//Nc, 4, Nc))
    size = int(size)

    frameSize = ((Tf)/(Tc*Nc))
    diff = int((frameSize-1)*Nc)
    for i in np.arange(diff):
        between = np.zeros((arr.shape[0],arr.shape[1],1))
        for j in np.arange(arr.shape[0]):
            idx = i + Nc - 1
            if j == (arr.shape[0] - 1):
                between[j] = np.array([(1/diff)*(arr[j,:,idx] - arr[j-1,:,idx]) + (arr[j,:,idx])]).T
            else:
                between[j] = np.array([(1/diff)*(arr[j+1,:,idx]) + arr[j,:,idx] * (1 - (i/diff))]).T
        
        arr = np.concatenate((arr,between),axis=-1)
    arr = arr.reshape(arr.shape[0]*arr.shape[2], arr.shape[1])
    return arr
"""

def interpolate(arr, Tf, Tc, Nc):
    size = arr.shape[0]
    size *= (Tf)/(Tc*Nc)
    arr = np.concatenate((np.array([arr[0,:]]),arr))
    #print(arr.shape)
    arr = arr.reshape((arr.shape[0]//Nc, 4, Nc))
    size = int(size)
    frameSize = ((Tf)/(Tc*Nc))
    diff = int((frameSize-1)*Nc)
    
    for i in np.arange(diff):
        between = np.zeros((arr.shape[0],arr.shape[1],1))
        for j in np.arange(arr.shape[0]):
            idx = i + Nc - 1
            if j == (arr.shape[0] - 1):
                between[j] = np.array([(1/diff)*(arr[j,:,idx] - arr[j-1,:,idx]) + (arr[j,:,idx])]).T
            else:
                """
                a = np.array([
                    [1, -(2*Tc),        (-2*Tc)**2,        -(2*Tc)**3,         (2*Tc)**4,      -(2*Tc)**5           ],
                    [1, -Tc,            Tc**2,             -Tc**3, Tc**4,      -Tc**5                               ],
                    [1, 0,              0 ,                0           ,       0 ,              0                   ],
                    [1, (Tf-Nc*Tc),     (Tf-Nc*Tc)**2,     (Tf-Nc*Tc)**3,     (Tf-Nc*Tc)**4,     (Tf-Nc*Tc)**5      ],
                    [1, (Tf+(1-Nc)*Tc), (Tf+(1-Nc)*Tc)**2, (Tf+(1-Nc)*Tc)**3, (Tf+(1-Nc)*Tc)**4, (Tf+(1-Nc)*Tc)**5  ],
                    [1, (Tf+(2-Nc)*Tc), (Tf+(2-Nc)*Tc)**2, (Tf+(2-Nc)*Tc)**3, (Tf+(2-Nc)*Tc)**4, (Tf+(2-Nc)*Tc)**5  ]
                ])
                """

                a = np.array([
                    [1, 0],
                    [1, (Tf-Nc*Tc)]
                ])
                """
                b = np.array([
                    arr[j,:,-3],
                    arr[j,:,-2],
                    arr[j,:,-1],
                    arr[j+1,:,0],
                    arr[j+1,:,1],
                    arr[j+1,:,2]
                ])
                """
                
                a = np.array([
                    [1, 0],
                    [1, (Tf-Nc*Tc)]
                ])

                b = np.array([
                    arr[j,:,-1],
                    arr[j+1,:,0]
                ])

                
                coeff = np.linalg.solve(a,b)
                #print(coeff)
                #print(coeff.shape)
                xArr =  np.array([
                    [1],
                    [(Tc*(i+1)/diff)]
                ])
                    
                """
                    [(Tc*(i+1)/diff)**2],
                    [(Tc*(i+1)/diff)**3],
                    [(Tc*(i+1)/diff)**4],
                    [(Tc*(i+1)/diff)**5]
                                 ])
                """
                between[j] = np.matmul(xArr.T, coeff).T
                #between[j] = np.array([(1/diff)*(arr[j+1,:,idx]) + arr[j,:,idx] * (1 - (i/diff))]).T
        
        arr = np.concatenate((arr,between),axis=-1)
    arr = arr.reshape(arr.shape[0]*arr.shape[2], arr.shape[1])
    return arr
rootdir = r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Trillingsmetingen\7040"



slope = 60.012e12
Fs = 5e6
Frames = 20000
Chirps = 16
Rx = 4
samples = 128

vScale = con.c/(4*np.pi*77e9*2e-5)

reflectors = []
frequencies = []
apertures = []
power = []
for i, files in enumerate(os.walk(rootdir)):
    print(files)
    measurement = []
    frequency = []
    for file in files[-1]:
        if file.split(".")[1] =="bin":
            vals = np.zeros((320000,4),dtype=np.complex128)
            
            #print(file,"\n")
            path = f"{files[0]}\\{file}"
            num = int(file.split(".")[0])
            current = utils.mmWaveData(path=path, slope=slope, sampleRate=Fs, frames=Frames, chirps=Chirps, Rx=Rx, samples=samples)
            a,b = current.rangeFFT()
            locs = np.argmax(np.abs(b),axis=-1, keepdims=True)
            for i, val in enumerate(locs):
                for j, r in enumerate(val):
                    vals[i,j] = b[i,j,r[0]]
            
            #print(vals.shape)
            
            #interpolated =  np.diff(interpolated, axis=0)        
            #interpolated = np.concatenate(([interpolated[0]],interpolated))
            #interpolated = vals
        else:
            continue
        """
        if int(1/(num*2e-5)) <= 1000:
            kern  = np.ones(int(1/(num*2e-5)),dtype=np.float64)*(num*2e-5)
        else:
            kern = np.ones(1000)/1000
        
        for i in range(4):
            interpolated[:,i] = np.convolve(interpolated[:,i],kern,mode="same")
        """
        nums = [10, 2187]#[10, 16, 125, 343, 729, 1024, 2187]
        if num in nums:
            
            print(num)
            vals = np.diff(np.angle(vals),axis=0)
            shit = np.copy(vals)
            vals[np.abs(vals) > np.std(np.abs(vals))] = np.abs(vals).mean()*np.sign(vals[np.abs(vals) > np.std(np.abs(vals))])
            interpolated = interpolate(vals,5e-4,2e-5,16)*vScale
            print(interpolated.shape)
                       
            interpolatedShit = interpolate(shit,5e-4,2e-5,16)*vScale


            goodFFT = np.abs(np.fft.fftshift(np.fft.fft(interpolated[:,1]-interpolated[:,1].mean()))[interpolated.shape[0]//2:])
            goodFFT = 10*np.log10(goodFFT/np.max(goodFFT))
            goodFFT[0] = goodFFT[1]
            badFFT = np.abs(np.fft.fftshift(np.fft.fft(interpolatedShit[:,1]-interpolatedShit[:,1].mean()))[interpolatedShit.shape[0]//2:])
            badFFT = 10*np.log10(badFFT/np.max(badFFT))
            badFFT[0] = badFFT[1]
            fig, ax = plt.subplots(2,3)
            for i in ax:
                for j in i:
                    j.grid(c="k",ls="--", alpha=0.25)
            ax[1,0].plot(np.arange(interpolated[:,1].size)/50000,interpolated[:,1],c="rebeccapurple", ls="-.", label="processed")
            ax[0,0].plot(np.arange(interpolatedShit[:,1].size)/50000,interpolatedShit[:,1],c="firebrick", ls="--", label="unprocessed") 
            ax[1,1].plot(np.arange(goodFFT.size)/10,goodFFT,c="rebeccapurple", ls="-.")
            ax[0,1].plot(np.arange(badFFT.size)/10,badFFT,c="firebrick", ls="--")
            ax[1,2].plot(np.arange(goodFFT.size)[num*10-100:num*10+100]/10,goodFFT[num*10-100:num*10+100],c="rebeccapurple", ls="-.")
            ax[0,2].plot(np.arange(badFFT.size)[num*10-100:num*10+100]/10,badFFT[num*10-100:num*10+100],c="firebrick", ls="--")
            ax[1,0].set_xlabel("$T$ [s]")
            ax[1,1].set_xlabel("$f$ [Hz]")
            ax[1,2].set_xlabel("$f$ [Hz]")
            ax[0,0].set_ylabel("$v$ [ms$^{-1}$]")
            ax[1,0].set_ylabel("$v$ [ms$^{-1}$]")
            ax[0,1].set_ylabel("$\\frac{I}{I_p}$ [dB]")
            ax[0,2].set_ylabel("$\\frac{I}{I_p}$ [dB]")
            ax[1,1].set_ylabel("$\\frac{I}{I_p}$ [dB]")
            ax[1,2].set_ylabel("$\\frac{I}{I_p}$ [dB]")
            ax[0,0].set_xlim(0,10)
            ax[1,0].set_xlim(0,10)
            ax[1,1].set_xlim(0,25e3)
            ax[0,1].set_xlim(0,25e3)
            ax[0,2].set_xlim(num-10, num+10)
            ax[1,2].set_xlim(num-10, num+10)
            xticks = [num-10, num-5, num, num+5, num+10]
            ax[0,2].set_xticks(xticks)
            ax[1,2].set_xticks(xticks)
            
            fig.set_figheight(11)
            fig.set_figwidth(17)
            
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.96, top=0.9, wspace=0.25, hspace=0.15)
            fig.legend()#bbox_to_anchor=(1,1,0.5,0.5))
            plt.savefig(f"c:\\Users\\Eigenaar\\Documents\\Afstudeerstage\\Afstuderen lectoraat\\Data\\Trillingsmetingen\\{str(num)}_compressed.jpg",dpi=300,format="jpg")
            plt.show()
        else:
            continue
            
            plt.figure()
            plt.title(file)
            plt.plot(interpolated[:,1])
            #plt.plot(vals[:,1])
            plt.show()
            
            
            plt.figure()
            plt.title(file)
            plt.plot(np.abs(np.fft.fftshift(np.fft.fft(interpolated[:,1]-interpolated[:,1].mean()))[interpolated.shape[0]//2:]))
            plt.show()
            
        P = np.abs(np.fft.fftshift(np.fft.fft(interpolated[:,:]-interpolated[:,:].mean(axis=0)))[interpolated.shape[0]//2:])[num*10,:]
        #print((np.mean(np.concatenate((P[(num-1)*10:(num)*10-3,:], P[(num)*10+3:(num+1)*10,:])),axis=0)), P[num*10,:])
        #P = P[num*10,:]/(np.mean(np.concatenate((P[(num-1)*10+5:(num)*10-3,:], P[(num)*10+3:(num+1)*10+5,:])),axis=0))
        power.append(P)
        frequency.append([num, num, num, num])
    reflectors.append(measurement)
    frequencies.append(frequency)

frequency = np.array(frequency).T
power = np.array(power).T


print(frequency.shape, power.shape)


fig, ax = plt.subplots(2,2)
inds = np.array([[0,1],[2,3]])
for i in range(2):
    for j in range(2):
        ind = inds[j,i]
        ax[j,i].scatter(frequency[0],power[ind])
        # ax[j,i].set_yscale("log")
        # ax[j,i].set_xscale("log")
plt.show()

plt.figure()
plt.scatter(frequency[0],np.mean(power,axis=0))
plt.show()


#np.save(r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Trillingsmetingen\7040\Power", power)
#np.save(r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Trillingsmetingen\7040\frequencies",frequency)

#df = pd.read_excel(r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Conductive_reflector_power\Afstanden.ods")
#ranges = df.to_numpy().T


"""
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
"""