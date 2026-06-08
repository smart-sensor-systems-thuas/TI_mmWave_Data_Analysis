import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from scipy import constants as const
import mmWave_Utils as utils

def CFAR(Array, threshhold, windowsize):
    window = np.ones(windowsize)/windowsize
    Array = Array - (np.real(Array.real.mean()) + np.imag(Array.imag.mean()))
    filter = np.convolve(Array,window,"same")
    """
    print(threshhold)
    plt.figure()
    plt.plot(np.abs(Array))
    plt.plot(np.abs(filter*threshhold))
    plt.show()
    """
    mask = np.abs(Array) > np.abs(filter*threshhold)
    return mask


print("start")
fName = r"c:\\Users\\Eigenaar\\Documents\\Afstudeerstage\\Afstuderen lectoraat\\Data\\initiele_metiningen\\30Hz.bin"

data = Path(r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\Accelerometer_inVivo\vast\11.bin").read_bytes()


#Params

frames = 20000
chirps = 16
Rx = 4
samples = 128
shape = (frames, chirps,Rx, samples)

tF = 5e-4
tC = 20.28e-6

#Load data

y = np.frombuffer(data, dtype=np.int16)
orig = np.copy(y)
y = np.reshape(y,shape)
y_test = np.reshape(y,(frames*chirps,Rx,samples))
y_test = y_test/((2**16)-1)
y = y/((2**16)-1)
con = np.array([])
intermediate = np.zeros(70)


xArr = (const.c/(2*60e12))*np.linspace(0,5e6,(samples)//2)  
dX = 1/((5e6)*const.c/(2*60e12))
idx = int(0.126/dX)

print(con.shape)

plt.figure()
plt.plot(xArr, np.abs(np.fft.fftshift(np.fft.fft(y[0,0,0]))[y[0,0,0].size//2:]),label="$c_1$")
plt.plot(xArr, np.abs(np.fft.fftshift(np.fft.fft(y[10,0,0]))[y[0,0,0].size//2:]),label="$c_{11}$")
plt.plot(xArr, np.abs(np.fft.fftshift(np.fft.fft(y[21,0,0]))[y[0,0,0].size//2:]),label="$c_{22}$")
plt.plot(xArr, np.abs(np.fft.fftshift(np.fft.fft(y[32,0,0]))[y[0,0,0].size//2:]),label="$c_{33}$")
plt.grid(alpha=0.25, c='k',ls="--")
plt.legend()
plt.ylim((0,4.1))
plt.xlim((-0.1,12.5))
plt.xlabel("$r$ [m]")
plt.ylabel("$I$ [-]")
#plt.savefig("chirpRanges")
plt.show()


"""
args = [np.argmax(np.abs(np.fft.fftshift(np.fft.fft(y[0,0,0]))[y[0,0,0].size//2:])), np.argmax(
np.abs(np.fft.fftshift(np.fft.fft(y[10,0,0]))[y[0,0,0].size//2:])), np.argmax(np.abs(np.fft.fftshift(np.fft.fft(y[21,0,0]))[y[0,0,0].size//2:])),
np.argmax(np.abs(np.abs(np.fft.fftshift(np.fft.fft(y[32,0,0]))[y[0,0,0].size//2:])))]

plt.figure()
plt.plot([np.angle(np.fft.fftshift(np.fft.fft(y[0,0,0])))[args[0]], np.angle(np.fft.fftshift(np.fft.fft(y[10,0,0])))[args[1]], np.angle(np.fft.fftshift(np.fft.fft(y[21,0,0])))[args[2]]
np.angle(np.fft.fftshift(np.fft.fft(y[32,0,0])))[args[3]]])
#plt.plot(xArr, np.abs(np.fft.fftshift(np.fft.fft(con[1]))[con[1].size//2:]))
plt.show()
"""

# Extract positional fft 
con = y_test[:,0]

twoDfft = (np.fft.fftshift(np.fft.fft(con)))[con.shape[0]//2:,con.shape[1]//2:]
print(twoDfft.shape)
absArr = np.abs(twoDfft[:,np.argmax(np.abs(np.fft.fftshift(np.fft.fft(y[0,0,0]))[y[0,0,0].size//2:]))])*np.angle(twoDfft[:,np.argmax(np.abs(np.fft.fftshift(np.fft.fft(y[0,0,0]))[y[0,0,0].size//2:]))])
absArr -= np.median(absArr)
plt.figure()
plt.plot(absArr)
plt.title("absArr")
plt.xlabel("$t$ [s]")
plt.ylabel("$I$")
plt.show()

xFFT = np.fft.fftshift(np.fft.fft(absArr))[absArr.size//2:]

tEnd = frames*tF

iFrame = int(tF//tC)
nPrime = iFrame*frames

tPrime = np.linspace(0,tEnd/4,nPrime//2)
xPrime = np.zeros(tPrime.size, dtype=np.complex64)
dI = 0
print(tPrime.size,absArr.size)
print(iFrame, nPrime)

xFFT = twoDfft[:,np.argmax(np.abs(np.fft.fftshift(np.fft.fft(y[0,0,0]))[y[0,0,0].size//2:]))]


plt.figure()
plt.plot(tPrime)
plt.show()

for i,val in enumerate(tPrime):
    
    if i % iFrame < chirps:
        xPrime[i] = xFFT[i-dI]
        I0 = i - dI

    else:
        dI += 1
        if i-dI + 1 < absArr.size:
            xPrime[i] = i % iFrame - chirps + 1 * (xFFT[I0+1] - xFFT[I0]) / 8 + xFFT[I0]
        else:
            xPrime[i] = xFFT[i-dI]
        



plt.figure()
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(xPrime))[xPrime.size//2:]))
plt.xlabel("$t$ [s]")
plt.ylabel("$I$")
plt.title("interp")
plt.show()

#xFFT = xPrime - xPrime.mean()

tArr = np.zeros(xFFT.size)
tArrLine = np.arange(0,frames*5e-4/2,5e-4/chirps)

for i in range(tArr.size):
    if i == 0:
        pass
    
    tArr[i] = tArr[i-1] + tC
    if i%16 == 0:
        tArr[i] = ((i/16)*tF)


fArr = np.arange(0,8/(tF),2/(tF*frames))
#fArr = np.linspace(0,16000,tPrime.size//2)
#print(tArr[::16])
plt.figure()
#plt.scatter(tArr,np.abs(xFFT))
#plt.scatter(,absArr)
plt.scatter(tPrime,xPrime)
#plt.scatter(tArr,np.abs(xFFT)*np.sign(xFFT.real)*np.sign(xFFT.imag))
plt.xlabel("t")
plt.ylabel("x")
plt.show()


"""
cf = CFAR(xFFT,3,100)
print(CFAR(xFFT,3,100))
print(cf)
xFFT = xFFT * (CFAR(xFFT,0,10)==0)
"""
 
phase = np.angle(xFFT)

plt.figure()
plt.plot(np.abs(xFFT))
plt.xlabel("$t$ [s]")
plt.ylabel("$I$")
plt.show()



plt.figure()
plt.plot(fArr, np.abs(np.fft.fftshift(np.fft.fft(np.real(xFFT))))[int(np.abs(xFFT).size//2):])
plt.plot(fArr, np.abs(np.fft.fftshift(np.fft.fft(np.imag(xFFT))))[int(np.abs(xFFT).size//2):])
plt.plot(fArr, np.abs(np.fft.fftshift(np.fft.fft(np.abs(xFFT)-np.mean(np.abs(xFFT)))))[int(np.abs(xFFT).size//2):])
#plt.plot(fArr, np.abs(np.fft.fftshift(np.fft.fft(np.abs(xFFT))))[int(np.abs(xFFT).size//2):])
plt.xlabel("$t$ [s]")
plt.ylabel("$I$")
plt.show()

dPhase = np.zeros(phase.shape, dtype=phase.dtype)


for i,val in enumerate(phase):
    if i != 0:
        dPhase[i] = phase[i] - phase[i-1]
    else:
        dPhase[i] = 0

velocity = (16*const.c/(((5e-4))*4*np.pi * 79 * 10**9))*dPhase
#velCon = velocity* (CFAR(velocity,100,100)==0)
#cf = CFAR(velocity,1000,100).nonzero()
#print(cf)

"""
plt.figure()
plt.plot(CFAR(velocity,100,10))
plt.show()

for index in cf[0]:
    index = int(index)
    velCon[index] = np.mean([velCon[index-1],velCon[index+1]])
"""

print(velocity.shape)
#velocity = velocity * (np.abs(velocity)< 100*np.mean(velocity))
print(velocity.shape)
plt.figure()
plt.plot(phase)
plt.ylabel("$\\phi$[-]")
plt.show()

velCon = np.convolve(velocity,np.ones(100)/100,mode="same")
velCon -= velCon.mean()

plt.figure()
plt.plot(velocity,c='k',label="raw")
plt.plot(velCon,c='r',ls="--",label="convolved")
plt.ylabel("$v$[ms$^{-1}$]")
plt.ylabel("$v$ [ms$^{-1}$]")
plt.xlabel("$n$ [-] ")
plt.xlim(0,10240)
plt.grid(alpha=0.25,c='k',ls="--")
#plt.savefig("velocity")
plt.show()

plt.figure()
plt.plot(velCon,label="convolved")
#plt.plot(velocity)
plt.ylabel("$v$ [ms$^{-1}$]")
plt.xlabel("$n$ [-] ")
plt.xlim(0,10240)
plt.grid(alpha=0.25,c='k',ls="--")
plt.show()

velcon = velocity

velFFT = np.fft.fftshift(np.fft.fft(velCon))[np.fft.fftshift(np.fft.fft(velCon)).size//2:]
shape = velFFT.shape


xArr = np.indices(shape)[0]/(4e-3*velCon.size/128)
print(xArr)  


plt.figure()
plt.plot(xArr, 20*np.log10(np.abs(velFFT/np.max(np.abs(velFFT)))))
plt.xlim(0,np.max(xArr))
plt.grid(alpha=0.25,c='k',ls="--")
plt.xlabel("$f$ [Hz]")
plt.ylabel("$I/I_0$ [dB]")
#plt.savefig("LogFFT1000Hz")
plt.show()


plt.figure()
plt.plot(xArr, (np.abs(velFFT/np.max(np.abs(velFFT)))))
plt.xlabel("$f$ [Hz]")
plt.ylabel("$I/I_0$ [dB]")
plt.show()
print(xArr[np.argmax(np.log10(np.abs(np.fft.fftshift(np.fft.fft(velCon))[np.fft.fftshift(np.fft.fft(velCon)).size//2:]/np.max(np.abs(np.fft.fftshift(np.fft.fft(velCon))[np.fft.fftshift(np.fft.fft(velCon)).size//2:])))))])
#plt.figure()
#plt.imshow(20*np.log10(np.abs(twoDfft.T)))
#plt.show()

