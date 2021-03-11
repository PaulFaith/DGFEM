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
print (rx)
[EToE, EToF] = connect(EToV)
[vmapM, vmapP, vmapB, mapB,fmask] = build_maps(N, x, EToE, EToF)
Fscale = 1/J[fmask,:]
eps1 = np.concatenate((np.ones(int(K/2)), 1*np.ones(int(K/2))),axis = 0)
mu1 = np.ones(K)
n_p = N+1
epsilon = np.full((n_p, K), eps1)
mu = np.ones((n_p, K))
c=.01
A=1
sig=.05
#E =  

E = A*np.exp(-((x+.7)/(sig*np.sqrt(2)))**2)
H = A*np.exp(-((x+.7)/(sig*np.sqrt(2)))**2)
#SOLVE PROBLEM.
#Maxwell section.
finaltime = 2.4
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
print(len(E), len(E[0]))
#fig = plt.figure() #Plot
#ax = plt.axes(projection = '3d')
#d3t = np.zeros((7,80))
#d3t = d3t + t
#my_cmap = plt.get_cmap('autumn')
print(E)
nplots = int(Nsteps/7)
fig, axs = plt.subplots(8)
for tstep in range(int(Nsteps)):
  #if (tstep % 15 == 0):
  if (tstep % nplots == 0):
    axs[int(tstep/nplots)].plot(x, E, '.', ms = 6, color = 'red')
    axs[int(tstep/nplots)].plot(x, A*np.exp(-((x+(.71-t))/(sig*np.sqrt(2)))**2), '.', ms = 6, color = 'blue')  
    #ax.scatter3D(x, E, d3t, alpha = 0.8, cmap = my_cmap, marker ='^')
  for intrk in range(5):
    [rhsE, rhsH]  = maxwell1d(intrk, tstep, E, H, epsilon, mu, K, Dr, LIFT, rx, nx, vmapP, vmapM, mapB, vmapB, Fscale) 
    resE = rk4("a", intrk)*resE + dt*rhsE
    resH = rk4("a", intrk)*resH + dt*rhsH
    E = E + rk4("b", intrk)*resE
    H = H + rk4("b", intrk)*resH
    
    #if (A*np.exp(-((1.0+(.71-t))/(sig*np.sqrt(2)))**2) <= 0.001 and t >= 2):
      #E = E*0.0

    E[0:,79] = 0.0
    H[0:,79] = 0.0
  t = t + dt
  #d3t = d3t + dt
print(E)  
axs[7].plot(x, E, '.', ms = 6, color = 'red')
axs[7].plot(x, A*np.exp(-((x+(.701-t))/(sig*np.sqrt(2)))**2), '.', ms = 6, color = 'blue')
for ax in axs.flat:
    ax.set(xlabel='x', ylabel='u(x,t)')

for ax in axs.flat:
    ax.label_outer()

#ax.set_xlabel('x')
#ax.set_ylabel('u(x,t)')
#ax.set_zlabel('Time')

#PLOT SOLUTION.
plt.show()

