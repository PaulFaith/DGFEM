import matplotlib.pyplot as plt
from DGFEM.checked import *
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#CONSTANTS AND VARIABLES ON THE EXAMPLE WAS GLOBALS1D.
N = 6
#GENERATE SIMPE MESH.
[Nv, VX, K, EToV] = mesh_generator(-1., 1., 80)
#INITIALIZE SOLVER AND CONSTRUCT GRID AND METRIC.
r = jacobi_gauss_lobatto(0, 0, N)
V = vandermonde(N, r)
Dr = differentiation_matrix(N, r, V)
LIFT = surface_integral_dg(N, V)
x = nodes_coordinates(N, EToV, VX)
[rx, J] = geometric_factors(x, Dr)
nx = normals(K)
[EToE, EToF] = connect(EToV)
[vmapM, vmapP, vmapB, mapB,fmask] = build_maps(N, x, EToE, EToF)
Fscale = 1/J[fmask,:]
eps1 = np.concatenate((np.ones(int(K/2)), 2*np.ones(int(K/2))),axis = 0)
mu1 = np.ones(K)
n_p = N+1
epsilon = np.full((n_p, K), eps1)
mu = np.ones((n_p, K))
c=.01
A=1
sig=.05
E = A*np.exp(-((x)/(sig*np.sqrt(2)))**2) 
H = np.zeros((n_p, K))
#E[0,0] = 0
#E[6,79] = 0
#H[0,0] = 0
#H[6,79] = 0

#SOLVE PROBLEM.
#Maxwell section.
finaltime = 20
t = 0
#Runge-Kutta residual storage.
resE = np.zeros((N+1, K))
resH = np.zeros((N+1, K))
#Compute time steps.
xmin = np.amin(np.abs(x[0, :] - x[1, :]))
CFL = 1.0
dt = CFL*xmin
Nsteps = np.ceil(finaltime/dt)
dt = finaltime/Nsteps

nplots = int(Nsteps/7)
fig, axs = plt.subplots(2)
plt.figure(1)

times = np.zeros(int(Nsteps))
EField = np.zeros(int(Nsteps))
EField1 = np.zeros(int(Nsteps))
EField2 = np.zeros(int(Nsteps))
HField1 = np.zeros(int(Nsteps))
HField2 = np.zeros(int(Nsteps))
EField5 = np.zeros(int(Nsteps))
EField6 = np.zeros(int(Nsteps))
EField7 = np.zeros(int(Nsteps))

for tstep in range(int(Nsteps)):
  times[tstep] = t
  EField[tstep] = E[4, 1]
  EField1[tstep] = E[0, 30]
  EField2[tstep] = E[0, 50]
  HField1[tstep] = H[0, 30]
  HField2[tstep] = H[0, 50]
  EField5[tstep] = E[4, 50]
  EField6[tstep] = E[4, 60]
  EField7[tstep] = E[4, 70]
  
  if (tstep % nplots == 0):
    pass
    #axs[int(tstep/nplots)].plot(x, E, '.', ms = 6, color = 'red')
    #axs[int(tstep/nplots)].plot(x, H, '.', ms = 6, color = 'blue')  
  for intrk in range(5):
    [rhsE, rhsH]  = maxwell1d(intrk, tstep, E, H, epsilon, mu, K, Dr, LIFT, rx, nx, vmapP, vmapM, mapB, vmapB, Fscale) 
    resE = rk4("a", intrk)*resE + dt*rhsE
    resH = rk4("a", intrk)*resH + dt*rhsH
    E = E + rk4("b", intrk)*resE
    E[0,0] = 0
    E[6,79] = 0
    H[0,0] = 0
    H[6,79] = 0
    H = H + rk4("b", intrk)*resH
  t = t + dt

print (times)
print (len(E[0, :]))

axs[0].plot(times, EField1, color = 'blue')
axs[0].plot(times, EField2, color = 'green')
axs[1].plot(times, HField1, color = 'blue')
axs[1].plot(times, HField2, color = 'green')


for ax in axs.flat:
    ax.set(xlabel='t', ylabel='u(x,t)')

for ax in axs.flat:
    ax.label_outer()
plt.show()