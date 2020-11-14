import json
import argparse
import os.path
import sys
from DGFEM.checked import * 


#CONSTANTS AND VARIABLES ON THE EXAMPLE WAS GLOBALS1D.
N = 8
#GENERATE SIMPE MESH.
[Nv, VX, K, EToV] = mesh_generator(0., 2., 10)
#INITIALIZE SOLVER AND CONSTRUCT GRID AND METRIC.

r = jacobi_gauss_lobatto(0, 0, N)
V = vandermonde(N, r)
Dr = differentiation_matrix(N, r, V)
LIFT = surface_integral_dg(N, V)
x = nodes_coordinates(N, EToV, VX)
[rx, J] = geometrix_factors(x, Dr)
[nx] = normals(K)
[EToE, EToF] = connect(EToV)
[vmapM, vmapP, vmapB, mapB] = buidlmaps(N, x, EToE, EToF)

#INITIAL CONDITIONS.
u = np.sin(x)
#SOLVE PROBLEM.
#Advec1D section.
finaltime = 10
# DONE = "[sol] = advec(u, finaltime)"
t = 0
#Runge-Kutta residual storage.
resu =  np.zeros(N+1, K)
#Compute time steps.
xmin = min(abs(x(1, :) - x(2, :)))
CFL = 0.75
dt = CFL/2*np.pi*xmin
dt = .5*dt
Nsteps = np.ceil(finaltime/dt)
dt = finaltime/Nsteps
#Adcetion speed
a = 2*np.pi
#Outer time step loop

for tstep in range(Nsteps):
  for intrk in range(5):
    timelocal = t + rk4("c", intrk)*dt
    [rhsu] = ADVECRHS1D(u, timelocal, a) # NOT DONE...
    resu = rk4("a", intrk)*resu + dt*rhsu
    u = u + rk4("b", intrk)*resu
  if t == 0 or t == 2.5 or t == 5.0 or t == 7.5 or t == 10:
    plot(x, u) #NOT DONE...
  t = t+dt




  #EXPORT DATA.

#PLOT SOLUTION.

