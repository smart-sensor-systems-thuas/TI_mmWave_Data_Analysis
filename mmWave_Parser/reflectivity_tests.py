import mmWave_Utils as utils
import numpy as np 
import matplotlib.pyplot as plt

path = r"c:\Users\Eigenaar\Documents\Afstudeerstage\Afstuderen lectoraat\Data\reflector_responsiviteit\80mm_CuTape_4.bin"
S = 60.012e12 # Hz/s
sampleRate = 5e6 # Hz
frames = 8
chirps = 128
Rx = 4
samples = 256
data = utils.mmWaveData(path=path, slope=S, sampleRate=sampleRate, frames=frames, chirps=chirps, Rx=Rx, samples=256)
x,I = data.rangeFFT()
print(I.shape)

algs = utils.radarAlgorithms(I)
cf = algs.CFAR(1e-4,15,1)

pos = cf[0,0,:].nonzero()
#objects = algs.TrackedObjects(1,1)
plt.figure()
plt.scatter(x[pos],np.abs(I[10,0,:])[pos],marker="x",label="found objects")
plt.plot(x,np.abs(I[10,0,:]),label="intensitiy")
plt.plot(x,np.convolve(np.abs(I[10,0,:]),4*np.ones(10)/10,mode="same"),label="moving average")
plt.plot(x,np.ones(128),label="static threshold")
plt.plot(x,algs.Threshold[10,0,:],label="CA-CFAR")
plt.grid(ls="--",c="k",alpha=0.25)
plt.xlim(0,6.3)
plt.xlabel("$r$[m]")
plt.ylabel("$I$[-]")
plt.legend()
plt.savefig("findingcomp.pdf",format="pdf")
plt.show()


plt.figure()
plt.plot(x, np.abs(I[100,0,:]))
plt.plot(x, np.abs(I[100,1,:]))
plt.plot(x, np.abs(I[100,2,:]))
plt.plot(x, np.abs(I[100,3,:]))
plt.show()

plt.figure()
plt.scatter(0,np.angle(I[100,0,:])[34])
plt.scatter(1,np.angle(I[100,1,:])[34])
plt.scatter(2,np.angle(I[100,2,:])[34])
plt.scatter(3,np.angle(I[100,3,:])[34])
plt.show()