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
E = np.sin(np.pi*x)*(x<0)
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

fig = plt.figure() #Plot
ax = plt.axes(projection = '3d')
d3t = np.zeros((7,80))
d3t = d3t + t
my_cmap = plt.get_cmap('Reds')
my_cmap1 = plt.get_cmap('Blues')

for tstep in range(int(Nsteps)):
  if (tstep % 10 == 0):
    ax.scatter3D(x, E, d3t, alpha = 0.8, c = (x + E + d3t), cmap = my_cmap, marker ='^')
    ax.scatter3D(x, H, d3t, alpha = 0.8, c = (x + H + d3t), cmap = my_cmap1, marker ='^')
  for intrk in range(5):
    [rhsE, rhsH]  = maxwell1d(intrk, tstep, E, H, epsilon, mu, K, Dr, LIFT, rx, nx, vmapP, vmapM, mapB, vmapB, Fscale)
    resE = rk4("a", intrk)*resE + dt*rhsE
    resH = rk4("a", intrk)*resH + dt*rhsH
    E = E + rk4("b", intrk)*resE
    H = H + rk4("b", intrk)*resH
  t = t + dt
  d3t = d3t + dt

#ax.set_xlabel('x')
#ax.set_ylabel('E(x,t) & H(x,t)')
#ax.set_zlabel('Time')

plt.axis('off')
plt.grid(b=None)

#PLOT SOLUTION.
plt.show()
