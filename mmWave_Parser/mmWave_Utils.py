import numpy as np
from pathlib import Path
from scipy import constants as const


class mmWaveData:
    def __init__(self,path,slope,sampleRate, frames, chirps, Rx, samples):
        self.shape = (frames, chirps, Rx, samples)
        self.Params = {"slope":slope, "sampleRate":sampleRate}
        self.Bin = Path(path).read_bytes()
        self.Data = np.reshape(np.frombuffer(self.Bin, dtype=np.int16),(frames*chirps,Rx,samples))/(2**16 - 1)
    
    def rangeFFT(self):
        fft = np.fft.fft(self.Data)
        sampleRate = self.Params["sampleRate"]
        slope = self.Params["slope"]
        samples = self.shape[3]
        self.rangeIntensity = np.fft.fftshift(fft,axes=-1)[:,:,samples//2:]
        self.rangeAxis = np.linspace(0, sampleRate//2, samples//2) * (const.c/(2*slope))
        return (self.rangeAxis, self.rangeIntensity)

    def deltaArgument(self):
        pos = np.argmax(np.abs(self.rangeIntensity[0,0,:]))
        arg = np.angle(self.rangeIntensity[:,:,pos])
        self.dArg = np.diff(arg, axis=0)
        return self.dArg

    def interpolate(self, Tf, Tc, Nc):
        size = self.dArg.shape[0]
        size *= (Tf)/(Tc*Nc)
        arr = self.dArg
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
        self.interpolated = arr
        return self.interpolated


class TrackedObject:
    def __init__(self, Receivers, t0):
        """
        Receivers = np.array([Rx1, Rx2, ... , RxN])
        Rxn = [index,val]
        """
        self.timeStamp = np.array([t0])
        self.Receivers = Receivers 

    
    def update(self,timestamp, Receivers, boundary):
        
        print(self.Receivers)
        intermediate = np.zeros(self.Receivers[0].shape)
        hits = 0
        
        for i, val in enumerate(self.Receivers):
        
            print(np.abs(val[0] - Receivers[i][0]))
            if np.abs(val[0] - Receivers[i][0]) < boundary: 
                print(Receivers[i][0])
                print(intermediate)
                intermediate[i] = Receivers[i][0]    
                hits += 1
            else:
                intermediate[i] = [False, False]
        if hits > 0:
            self.Receivers = np.concatenate((self.Receivers,Receivers))
            self.timeStamp = np.append(self.timeStamp, timestamp)
            return 0
        
        else:
            return -1



class radarAlgorithms:
    def __init__(self, rangeIntensity):
        self.rangeIntensity = rangeIntensity
        self.objects = np.array([])
    
    def CFAR(self,Pfa, window, guard):
        kernel = np.ones((guard+window))
        kernel[window:window+guard] = 0
        kernel = np.concatenate((kernel,[0],np.flip(kernel)))/(2*window)
        CF = np.zeros(self.rangeIntensity.shape)
        for i in range(CF.shape[0]):
            for j in range(CF.shape[1]):
                CF[i,j] = np.convolve(np.abs(self.rangeIntensity[i,j]),kernel,mode="same")

        T = 2*window*(Pfa**((-1/(2*window)))-1)
        self.Threshold=T*CF
        self.detections = np.abs(self.rangeIntensity) > T * CF
        return self.detections

    
    
    def AddObjects(self, rx, time,boundary):
        
        print(self.objects.size)
        if self.objects.size == 0:
            self.objects = np.concatenate((self.objects,[TrackedObject(rx,time)]))
            
        else:
            for i in self.objects:
                out = i.update(time, rx, boundary) 
            if out == -1:
                self.objects = np.concatenate((self.objects,[TrackedObject(rx,time)]))
            

    def TrackedObjects(self,boundary,time):

        NR = self.detections.shape[1]
        rx = np.array([])
        for i in range(self.detections.shape[0]):
            indexes = self.detections[i].T.nonzero()
            indexes = list(zip(indexes[0], indexes[1]))
            
            for j,val in enumerate(indexes):
                if val[0] != indexes[j-1][0] or val[1] == 0 or rx.size == 0:
                    if rx.size != 0:
                        self.AddObjects(rx,j, 2)

                    rx = np.array([val[0], self.rangeIntensity[i,val[1],val[0]]])
                else:
                    rx = np.vstack((rx, [val[0], self.rangeIntensity[i, val[1], val[0]]]))
                

        #print(rx)
        return self.objects
        

if __name__ == "__main__":
    print(__name__)