#Puropose: Script for building operators, grid, metric and connectivity for 1d solver.

import numpy as np

class Galerkin:
  #CONSTANTS AND VARIABLES ON THE EXAMPLE WAS GLOBALS1D.
  #Constructor.       
  def __init__(NODETOL, Np, Nfp, Nfaces):
    self.NODETOL = NODETOL
    self.Np = Np
    self.Nfp = Nfp
    self.Nfaces = Nfaces
  
  #Compute the basic Legendre Gauss Lobato grid.
  def jacobiGridLGL(self, alpha, beta, N):
    self.alpha = alpha
    self.beta = beta
    self.N = N
    self._x = np.zeros(self.N+1,1)
    if (N == 1):
      self._x(0) = -1.
      self._x(1) = 1.
      return
    [self._xint, self._w] = JacobiGQ(self.alpha + 1, self.beta + 1, self.N - 2)
    self._x = np.append(-1., np.transpose(self._xint), 1.)
  
  def jacobiGQ(self, a, b, N):
    self.a = a
    self.b = b
    self.N = N
    
  #Build reference element matrices.
  def matrixElements(self):
    Pass
  #Create surface integral terms.
  def surfaceIntegral(self):
    Pass
  #Build coordinates of all the nodes.
  def nodeCoordinates(self):
    Pass
  #Calculate geometric factors.
  def geometricFactors(self):
    Pass
  #Compute masks for edge nodes.
  def masksNodes(self):
    Pass
  #Build surface normals and inverse metric at surface.
  def surfaceNormals(self):
    Pass
  #Build connectivity matrix.
  def connectMatrix(self):
    Pass
  #Build connectivity maps.
  def connectMaps(self):
    Pass
