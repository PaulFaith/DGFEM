import matplotlib.pyplot as plt
from DGFEM.checked import *
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
#CONSTANTS AND VARIABLES ON THE EXAMPLE WAS GLOBALS1D.
N = 6
#GENERATE SIMPLE MESH.
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
eps1 = np.concatenate((np.ones(int(K/2)), 1*np.ones(int(K/2))),axis = 0)
for eps in range(int(K//2), int((K//2)+.02*K)):
  eps1[eps] = 10
mu1 = np.ones(K)
n_p = N+1
epsilon = np.full((n_p, K), eps1)
mu = np.ones((n_p, K))
c=.01
A=1
sig=.05
E = A*np.exp(-((x+.7)/(sig*np.sqrt(2)))**2)
H = A*np.exp(-((x+.7)/(sig*np.sqrt(2)))**2)
#SOLVE PROBLEM.
#Maxwell section.
finaltime = 2.5
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

#fig = plt.figure() #Plot
#ax = plt.axes(projection = '3d')
#d3t = np.zeros((7,80))
#d3t = d3t + t
#my_cmap = plt.get_cmap('autumn')

nplots = int(Nsteps/4)
fig, axs = plt.subplots(5)
for tstep in range(int(Nsteps)):
  #if (tstep % 15 == 0):
  if (tstep % nplots == 0):
    axs[int(tstep/nplots)].plot(x, E, '.', ms = 6, color = 'red')
    
    Ext = A*np.exp(-((x+(.7-t))/(sig*np.sqrt(2)))**2) 
    Exr = A*np.exp(-((x+(.7-t))/(sig*np.sqrt(2)))**2)
    
    #axs[int(tstep/nplots)].plot(x, Ext, '.', ms = 6, color = 'blue')
    
    axs[int(tstep/nplots)].plot(x, Exr, '.', ms = 6, color = 'green')
    axs[int(tstep/nplots)].axvline(x=x[0][int(K//2)])
    axs[int(tstep/nplots)].axvline(x=x[0][int((K//2)+.02*K)])  
    #ax.scatter3D(x, E, d3t, alpha = 0.8, cmap = my_cmap, marker ='^')
  for intrk in range(5):
    [rhsE, rhsH]  = maxwell1d(intrk, tstep, E, H, epsilon, mu, K, Dr, LIFT, rx, nx, vmapP, vmapM, mapB, vmapB, Fscale) 
    resE = rk4("a", intrk)*resE + dt*rhsE
    resH = rk4("a", intrk)*resH + dt*rhsH
    E = E + rk4("b", intrk)*resE
    H = H + rk4("b", intrk)*resH
    E[0:,0] = 0.0
    E[0:,79] = 0.0
    H[0:,0] = 0.0
    H[0:,79] = 0.0

    if (round(E[0][int(K//2)]-1, 2) != 0.0 or round(E[0][int((K//2)+.02*K)]-1, 2) != 0.0):
      Ext = A*np.exp(-((x+(.7-t))/(sig*np.sqrt(2)))**2)
      
      print("t= ", round(t,2), "Exi =", round(Ext[0][int(39)], 5), " Eci =", round(E[0][int(39)], 5))
      #print("Exf =", round(Ext[0][int((K//2)+.02*K)-1], 5), " Ecf =", round(E[0][int((K//2)+.02*K)], 5))

      #print("Er", E[0][int(K//2)], "=", Ext[0][int(K//2)], "Ext\n")
      #print("Ext", E[0][int(K//2)])
      #print("Et", E[0][int((K//2)+.02*K)], "\n")
      time.sleep(0)
    
    E[0:,79] = 0.0
    H[0:,79] = 0.0
    #time.sleep(1)
  t = t + dt
  #d3t = d3t + dt
axs[4].plot(x, E, '.', ms = 6, color = 'red')
axs[4].plot(x, Ext, '.', ms = 6, color = 'blue')
axs[4].plot(x, Exr, '.', ms = 6, color = 'green')
axs[4].axvline(x=x[0][int(K//2)])
axs[4].axvline(x=x[0][int((K//2)+.02*K)])
for ax in axs.flat:
    ax.set(xlabel='x', ylabel='E(x,t)')

for ax in axs.flat:
    ax.label_outer()

#ax.set_xlabel('x')
#ax.set_ylabel('u(x,t)')
#ax.set_zlabel('Time')

#PLOT SOLUTION.
plt.show()

