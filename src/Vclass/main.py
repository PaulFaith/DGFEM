from galerkin import Galerkin
import matplotlib.pyplot as plt
import math

gal = Galerkin(10, 2, 0.0, 2.0*math.pi)
fig = plt.figure()
for N in range(2, 3):
  gal.N = N
  print(gal.N)  
  for k in range(10, 12):
    gal.K = k
    print(gal.K)
    [u, x] = gal.calculate("H", .8)
    plt.plot(x, u, '.r:', ms = 6, color = 'red')
print(u)
plt.show()

