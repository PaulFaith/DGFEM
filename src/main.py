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
for finaltime in range(5):
  [sol] = advec(u, finaltime)
  
  #EXPORT DATA.

#PLOT SOLUTION.

