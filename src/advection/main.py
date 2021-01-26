import matplotlib.pyplot as plt
from DGFEM.checked import *
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#CONSTANTS AND VARIABLES ON THE EXAMPLE WAS GLOBALS1D.
N = 8
#GENERATE SIMPE MESH.
[Nv, VX, K, EToV] = mesh_generator(0., 7., 10)
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
u = np.sin(x)
#SOLVE PROBLEM.
#Advec1D section.
finaltime = 1
t = 0
#Runge-Kutta residual storage.
resu = np.zeros((N+1, K))
#Compute time steps.
xmin = np.amin(np.abs(x[0, :] - x[1, :]))
CFL = 0.75
dt = CFL/(2*np.pi)*xmin
dt = .5*dt
Nsteps = np.ceil(finaltime/dt)
dt = finaltime/Nsteps
a = 2*np.pi #Advection speed
fig = plt.figure() #Plot
ax = plt.axes(projection = '3d')
d3t = np.zeros((9,10))
d3t = d3t + t
my_cmap = plt.get_cmap('autumn')
ax.scatter3D(x, u, d3t, alpha = 0.8, c = (x + u + d3t), cmap = my_cmap, marker ='^')

for tstep in range(int(Nsteps)):
  if (tstep % 15 == 0):
    ax.scatter3D(x, u, d3t, alpha = 0.8, c = (x + u + d3t), cmap = my_cmap, marker ='^')
  for intrk in range(5):
    timelocal = t + rk4("c", intrk)*dt
    rhsu = advecrhs1d(u, timelocal, a, K, Dr, LIFT, rx, nx, vmapP, vmapM, Fscale) # NOT DONE...
    resu = rk4("a", intrk)*resu + dt*rhsu
    u = u + rk4("b", intrk)*resu
  t = t+dt
  d3t = d3t + dt

ax.set_xlabel('x')
ax.set_ylabel('u(x,t)')
ax.set_zlabel('Time')

#PLOT SOLUTION.
plt.show()

#EXPORT DATA.
txt = open("variables.txt", "w")
txt.write(f"\nN: {N}\n")
txt.write(f"\nNv: {Nv}\n")
txt.write(f"\nVX: {VX}\n")
txt.write(f"\nK: {K}\n")
txt.write(f"\nEToV: {EToV}\n")
txt.write(f"\nr: {r}\n")
txt.write(f"\nV: {V}\n")
txt.write(f"\nDr: {Dr}\n")
txt.write(f"\nLIFT: {LIFT}\n")
txt.write(f"\nx: {x}\n")
txt.write(f"\nrx: {rx}\n")
txt.write(f"\nJ: {J}\n")
txt.write(f"\nnx: {nx}\n")
txt.write(f"\nEToE: {EToE}\n")
txt.write(f"\nEToF: {EToF}\n")
txt.write(f"\nvmapM: {vmapM}\n")
txt.write(f"\nvmapP: {vmapP}\n")
txt.write(f"\nvmapB: {vmapB}\n")
txt.write(f"\nmapB: {mapB}\n")
txt.write(f"\nfmask: {fmask}\n")
txt.write(f"\nFscale: {Fscale}\n")
txt.write(f"\nu: {u}\n")
txt.write(f"\ndt: {dt}\n")
txt.write(f"\nNsteps: {Nsteps}\n")
txt.write(f"\nxmin: {xmin}\n")
txt.close()
