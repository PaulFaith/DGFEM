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
E = A*np.exp(-((x+.7)/(sig*np.sqrt(2)))**2) 
H = np.zeros((n_p, K))

#SOLVE PROBLEM.
#Maxwell section.
finaltime = 10
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
fig, axs = plt.subplots(8)

for tstep in range(int(Nsteps)):
  if (tstep % nplots == 0):
    axs[int(tstep/nplots)].plot(x, E, '.', ms = 6, color = 'red')
    axs[int(tstep/nplots)].plot(x, H, '.', ms = 6, color = 'blue')  
  for intrk in range(5):
    [rhsE, rhsH]  = maxwell1d(intrk, tstep, E, H, epsilon, mu, K, Dr, LIFT, rx, nx, vmapP, vmapM, mapB, vmapB, Fscale) 
    resE = rk4("a", intrk)*resE + dt*rhsE
    resH = rk4("a", intrk)*resH + dt*rhsH
    E = E + rk4("b", intrk)*resE
    H = H + rk4("b", intrk)*resH
  t = t + dt
for ax in axs.flat:
    ax.set(xlabel='x', ylabel='u(x,t)')

for ax in axs.flat:
    ax.label_outer()
plt.show()