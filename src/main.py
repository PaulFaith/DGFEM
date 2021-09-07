import matplotlib.pyplot as plt
from DGFEM.checked import *
import matplotlib as mpl
import numpy as np
#CONSTANTS AND VARIABLES ON THE EXAMPLE WAS GLOBALS1D.
N = 4
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

nplots = int(Nsteps/5)
fig, ax = plt.subplots(6)
style = dict(size=10, color='gray')
aux = 0
for tstep in range(int(Nsteps)):
  if (tstep % nplots == 0):
    ax[int(tstep/nplots)].plot(x, np.sin(x-2*np.pi*t), '-r', ms = 1)
    ax[int(tstep/nplots)].plot(x, u, '.b', ms = 3)
    ax[int(tstep/nplots)].text(0 + aux, -1, f't = {t:.2f}')
    aux = aux + 1.2
  for intrk in range(5):
    timelocal = t + rk4("c", intrk)*dt
    rhsu = advecrhs1d(u, timelocal, a, K, Dr, LIFT, rx, nx, vmapP, vmapM, Fscale) # NOT DONE...
    resu = rk4("a", intrk)*resu + dt*rhsu
    u = u + rk4("b", intrk)*resu
  t = t+dt

for ax in ax.flat:
        ax.set(xlabel = f'x, N = {N}, K = {10}', ylabel = 'u(x,t)')

fig.suptitle('DGFEM aproximation of advection equation.')




#PLOT SOLUTION.

plt.show()

