import matplotlib.pyplot as plt
from DGFEM.checked import *
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

Kg = np.array([])
errorEK = np.array([])
fig = plt.figure()

for N in range(1, 6):
  for k_element in range(20, 25, 2):
    #GENERATE SIMPE MESH.
    print(k_element)
    Kg = np.append(Kg, k_element)
    [Nv, VX, K, EToV] = mesh_generator(-2., 2., k_element)
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
    eps1 = np.concatenate((np.ones(int(K/2)), 1.5*np.ones(int(K/2))),axis = 0)
    mu1 = np.ones(K)
    n_p = N+1
    epsilon = np.full((n_p, K), eps1)
    mu = np.ones((n_p, K))
    c=.01
    A=1
    sig=.05
    t = 0
    w = [-2.59, 2.59, -3.69, 3.69]
    E = maxwellexE(1, 1.5, t, x, w[2])
    H = maxwellexH(1, 1.5, t, x, w[2]) 
    #SOLVE PROBLEM.
    #Maxwell section.
    finaltime = 10 
    #Runge-Kutta residual storage.
    resE = np.zeros((N+1, K))
    resH = np.zeros((N+1, K))
    #Compute time steps.
    xmin = np.amin(np.abs(x[0, :] - x[1, :]))
    CFL = .5
    dt = CFL*xmin
    Nsteps = np.ceil(finaltime/dt)
    dt = finaltime/Nsteps
    nplots = int(Nsteps/5)
    errorE = np.array([])
    for tstep in range(int(Nsteps)):
      for intrk in range(5):
        [rhsE, rhsH]  = maxwell1d(intrk, tstep, E, H, epsilon, mu, K, Dr, LIFT, rx, nx, vmapP, vmapM, mapB, vmapB, Fscale) 
        resE = rk4("a", intrk)*resE + dt*rhsE
        resH = rk4("a", intrk)*resH + dt*rhsH
        E = E + rk4("b", intrk)*resE
        H = H + rk4("b", intrk)*resH
      #Calculo del error y la eficiencia del metodo para distintos K y N.
      Eex = maxwellexE(1, 1.5, t, x, w[2])
      if errorE.size == 0 :
        errorE = np.sqrt((np.sum((Eex-E)*(Eex-E)))/len(E)*len(E[0]))
      else :
        errorE = np.append(errorE, np.sqrt((np.sum((Eex-E)*(Eex-E)))/len(E)*len(E[0])))      
      t = t + dt  
    errorEK = np.append(errorEK, np.sum(errorE)/len(errorE))
    errorE = np.delete
    #print(errorEK)
    #print(Kg)
  plt.plot(Kg , errorEK, label = f'N = {N}')
  plt.annotate(f'N = {N}', (Kg[int(len(Kg)//2)],errorEK[int(len(errorEK)//2)]), textcoords="offset points", xytext=(-10,10), ha='center')    
  errorEK = np.array([])
  Kg = np.array([])
legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.show()
  