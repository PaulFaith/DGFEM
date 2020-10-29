import numpy as np
import copy
import math

class Mesh:
    #Constructor.       
    def __init__(self, xmin, xmax, K):
        self.xmin = xmin
        self.xmax = xmax
        self.K = K
    
    #Generate node cordinates.
    def coordinates(self):
        return np.linspace(self.xmin, self.xmax, self.K+1, endpoint=True)
    
    #Read element to node connectivity.
    def EToV(self):
        self.EToV = np.zeros((self.K,2))
        for k in range(self.K)
        self.EToV[k,1]=k+1
        self.EToV[k,2]=k+2
        return self.EToV
