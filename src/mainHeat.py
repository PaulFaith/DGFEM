import matplotlib.pyplot as plt
from DGFEM.checked import *
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

Kg = np.array([])
errorEK = np.array([])
fig = plt.figure()

for N in range(2, 4):
  for k_element in range(20, 21, 2):
    #GENERATE SIMPE MESH.
    print(k_element)
    Kg = np.append(Kg, k_element)
    [Nv, VX, K, EToV] = mesh_generator(0.0, 2.0*math.pi, k_element)
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
    #Heat section section.
    finaltime = 1.8
    t = 0
    #Runge-Kutta residual storage.
    resu = np.zeros((N+1, K))
    #Compute time steps.
    xmin = np.amin(np.abs(x[0, :] - x[1, :]))
    CFL = .25
    dt = CFL*xmin*xmin
    Nsteps = np.ceil(finaltime/dt)
    dt = finaltime/Nsteps
    nplots = int(Nsteps/5)
    errorE = np.array([])
    fig, axs = plt.subplots(6)
    for tstep in range(int(Nsteps)):
      if (tstep % nplots == 0):
        ux = np.exp(-t)*np.sin(x)
        axs[int(tstep/nplots)].set_ylim(-1,1)
        axs[int(tstep/nplots)].plot(x, u, '.r:', ms = 6, color = 'red')
        axs[int(tstep/nplots)].plot(x, ux, '.b:', ms = 6, color = 'blue')  
      for intrk in range(5):
        timelocal = t + rk4("c", intrk)*dt
        [rhsu]  = HeatCRHS1D(intrk, tstep, E, H, epsilon, mu, K, Dr, LIFT, rx, nx, vmapP, vmapM, mapB, vmapB, Fscale) 
        resu = rk4("a", intrk)*resu + dt#*rhsu

      #Calculo del error y la eficiencia del metodo para distintos K y N.
      ux = np.exp(-t)*np.sin(x)
      if errorE.size == 0 :
        errorE = np.sqrt((np.sum((ux-u)*(ux-u)))/len(u)*len(u[0]))
      else :
        errorE = np.append(errorE, np.sqrt((np.sum((ux-u)*(ux-u)))/len(u)*len(u[0])))
      

      t = t + dt 
    for ax in axs.flat:
        ax.set(xlabel=f'x, N = {N}, K = {k_element}', ylabel='u(x,t)')

    for ax in axs.flat:
        ax.label_outer()


    #PLOT SOLUTION.
    #if errorEK.size == 0 :
      #errorEK = np.sum(errorE)/len(errorE)
    #else : 
    errorEK = np.append(errorEK, np.sum(errorE)/len(errorE))
    errorE = np.delete
    print(errorEK)
    print(Kg)
  #plt.plot(Kg , errorEK, label = f'N = {N}')
  #plt.annotate(f'N = {N}', (Kg[int(len(Kg)//2)],errorEK[int(len(errorEK)//2)]), textcoords="offset points", xytext=(-10,10), ha='center')    
  errorEK = np.array([])
  Kg = np.array([])
plt.show()