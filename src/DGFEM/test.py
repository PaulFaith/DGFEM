from checked import jacobi_gauss_lobatto
from scipy import special

r = jacobi_gauss_lobatto(0,0,5)
print(special.roots_jacobi(4,0,0))
