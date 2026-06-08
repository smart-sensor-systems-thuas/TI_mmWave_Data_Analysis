import numpy as np
import time
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

device = torch.device("cuda:0")
print(torch.cuda.is_available())
torch.set_default_device('cuda')
print(torch.get_default_device())

class WaveEqn2D:
    def __init__(self, nx=500, ny=500, c=0.2, h=1, dt=1,
                 use_mur_abc=True, components=[]):
        """Initialize the simulation:

        nx and ny are the dimension of the domain;
        c is the wave speed;
        h and dt are the space and time grid spacings;
        If use_mur_abc is True, the Mur absorbing boundary
        conditions will be used; if False, the Dirichlet
        (reflecting) boundary conditions are used.

        """

        self.nx, self.ny = nx, ny
        self.c = c
        self.h, self.dt = h, dt
        self.use_mur_abc = use_mur_abc
        self.alpha = self.c * self.dt / self.h
        self.alpha2 = self.alpha**2
        self.components = components
        self.u = torch.zeros((3, ny, nx),dtype=torch.float)


    def update(self):
        """Update the simulation by one time tick."""

        # The three planes of u correspond to the time points
        # k+1, k and k-1; i.e. we calculate the next frame
        # of the simulation (k+1) in u[0,...].
        u, nx, ny = self.u, self.nx, self.ny
        u[2] = u[1]     # old k -> new k-1
        u[1] = u[0]     # old k+1 -> new k

        # Calculate the new k+1:
        u[0, 1:ny-1, 1:nx-1]  = self.alpha2 * (
                    u[1, 0:ny-2, 1:nx-1]
                  + u[1, 2:ny,   1:nx-1]
                  + u[1, 1:ny-1, 0:nx-2]
                  + u[1, 1:ny-1, 2:nx]
                  - 4*u[1, 1:ny-1, 1:nx-1]) \
                  + (2 * u[1, 1:ny-1, 1:nx-1]
                  - u[2, 1:ny-1, 1:nx-1])

        if self.use_mur_abc:
            # Mur absorbing boundary conditions.
            kappa = (1 - self.alpha) / (1 + self.alpha) 
            u[0, 0, 1:nx-1] = (u[1, 1, 1:nx-1]
                               - kappa * (
                                     u[0, 1, 1:nx-1]
                                   - u[1, 0, 1:nx-1])
                              )
            u[0, ny-1, 1:nx-1] = (u[1, ny-2, 1:nx-1]
                               + kappa * (
                                   u[1, ny-1, 1:nx-1]
                                 - u[0, ny-2, 1:nx-1])
                              )
            u[0, 1:ny-1, 0] = (u[1, 1:ny-1, 1]
                               - kappa * (
                                   u[0, 1:ny-1, 1]
                               - u[1, 1:ny-1, 0])
                              )
            u[0, 1:ny-1, nx-1] = (u[1, 1:ny-1, nx-2]
                               + kappa * (
                                   u[1, 1:ny-1, nx-1]
                                 - u[0, 1:ny-1, nx-2])
                              )
        if np.size(self.components) > 0:

            for i in self.components:
                i.u = u
                i.calculate()
                #u = i.u
            
                    
                       
class opticalComponent:
    def __init__ (self,posx,posy,ny,**kwargs):
        self.posx = posx
        self.posy = posy
        self.ny = ny
        self.u = None
    def draw(self):        
        pass
    def calculate(self):
        pass

class retroReflector(opticalComponent):
    def __init__(self,posx,posy,ny,**kwargs):
        super().__init__(posx,posy,ny)
        self.aperture = kwargs["aperture"]
        self.u = None
        self.len = None
    def draw(self):
        self.y = torch.arange(-self.aperture//2,self.aperture//2,1,dtype=torch.int)
        self.x = -torch.abs(self.y) + self.posx
        self.y += self.posy + self.ny//2
        self.len = self.y.size()[0]//2
    def calculate(self):
        self.u[0,self.y,self.x] = 0

class hornReflector(opticalComponent):
    def __init__(self,posx,posy,ny,**kwargs):
        super().__init__(posx,posy,ny)
        self.aperture = kwargs["aperture"]
        self.u = None
        self.len = None
    def draw(self):
        self.y = torch.arange(-self.aperture,self.aperture//2,1,dtype=torch.int)
        self.x = -torch.abs(self.y)
        self.y = torch.concatenate((self.y,self.y[-(self.y.size()[0])//3+1:]))
        inbetween = (self.x[-(self.x.size()[0])//3+1:]-self.aperture//2)
        self.x = torch.concatenate((self.x,torch.flip(inbetween,[-1]))) + self.posx
        self.y += self.posy + self.ny//2
        self.len = self.y.size()[0]//4
    def calculate(self):
        self.u[0,self.y,self.x] = 0


A = 100
dt = 0.01
c = 1
T = 20#c/(0.25)
freq = 2 * np.pi*c * dt / T
h = 0.25
nx = 4000
ny = 3000
aperture = 60




#fig, ax = plt.subplots()
#ax.axis("off")
#img = ax.imshow(sim.u[0], vmin=0, vmax=40, cmap='Blues_r')



def update(i):
    """Advance the simulation by one tick."""
    # A regular sinusoidal signal at the centre of the domain.
    if i < int(10*T/dt):
        sim.u[0, ny//2+4, 0] = A * np.sin(i * freq)
        sim.u[0, ny//2-4, 0] = A * np.sin(i * freq)
        sim.u[0, ny//2+12, 0] = A * np.sin(i * freq)
        sim.u[0, ny//2-12, 0] = A * np.sin(i * freq)
    elif i == int(10*T/dt):
        sim.u[0, ny//2+4, 0] = 0
        sim.u[0, ny//2-4, 0] = 0
        sim.u[0, ny//2+12, 0] = 0
        sim.u[0, ny//2-12, 0] = 0
    
    sim.update()

def init():
    """
    Initialization, because we're blitting and need references to the
    animated objects.
    """
    return img,

def animate(i):
    """Draw frame i of the animation."""
    for j in range(40*i,40*i+40):
        update(j)
    img.set_data(toch.abs(sim.u[0]))
    return img,

print("run")

interval, nframes = dt, 450



#ani = animation.FuncAnimation(fig, animate, frames=nframes,
#                              repeat=False,
#                              init_func=init, interval=interval, blit=True)


#ani.save(filename="fasterretroreflectnew.gif", writer="pillow")


#for i in range(nframes):
#    animate(i)

#TMax = int(3*reflector.posx//(const.c*dt))
#print(TMax)




#print(t2-t1)

TMax = 180000
for i in range(1220,2001,20):
    t1 = time.time()
    reflector = retroReflector(int(3000),int(0//h),ny,aperture=int(i))
    reflector.draw()
    sim = WaveEqn2D(nx, ny, dt=dt, use_mur_abc=True,c=c,h=h,components=[reflector])
    I = np.zeros((2,TMax))
    print(f"start Aperture {i/T}")
    for j in range(TMax):
        update(j)
        I[1,j] = np.sum(np.abs(sim.u[0,ny//2-100:ny//2+100,1].cpu().numpy()))
        I[0,j] = j
    with open(f"A{int(i/T)}.npy","wb") as f:
        np.save(f,I)
    f.close()
    t2 = time.time()
    print(t2-t1)
    print(f"end Aperture {i/T}\n")
    

print("done")




