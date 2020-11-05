import numpy as np
import scipy.special
import math

def jacobi_gauss_lobatto(alpha, beta, n_order):
    """
    Compute the order n_order Gauss Lobatto quadrature points, x, associated
    with the Jacobi polynomial.
    
    >>> jacobi_gauss_lobatto(0.0, 0.0, 1)
    array([-1.,  1.])
    >>> jacobi_gauss_lobatto(0,0,3)
    array([-1.       , -0.4472136,  0.4472136,  1.       ])
    >>> jacobi_gauss_lobatto(0,0,4)
    array([-1.        , -0.65465367,  0.        ,  0.65465367,  1.        ])
    
    """
    if n_order==0:
        return np.array([0.0])
    if n_order==1:
        return np.array([-1.0, 1.0])
    if n_order>1:
        x, w = scipy.special.roots_jacobi(n_order-1, alpha+1, beta+1)
        return np.concatenate(([-1.0], x, [1.0]))
    
    raise ValueError('n_order must be positive.')

def mesh_generator(xmin,xmax,k_elem):
    """
    Generate simple equidistant grid with K elements
    >>> [Nv, vx, K, etov] = mesh_generator(0,10,4)
    >>> Nv
    5
    >>> vx_test = ([0.00000000,2.50000000,5.00000000,7.50000000,10.00000000])
    >>> np.allclose(vx,vx_test)
    True
    >>> K
    4
    >>> etov_test = ([[1, 2],[2, 3],[3, 4],[4, 5]])
    >>> np.allclose(etov,etov_test)
    True
    """
    n_v = k_elem+1
    vx = np.zeros(n_v)
    for i in range(n_v):
        vx[i] = (xmax-xmin)*i/(n_v-1)+xmin
    #np.zeros creates a float array. etov should be an integer array
    etov = np.full((k_elem,2),0)
    #etov = np.zeros([K,2])
    for i in range(k_elem):
        etov[i,0] = i+1
        etov[i,1] = i+2

    return [n_v,vx,k_elem,etov]

def vandermonde(n_order, r):
    """
    Initialize Vandermonde matrix
    >>> r = jacobi_gauss_lobatto(0,0,2)
    >>> vandermonde(2,r)
    array([[ 0.70710678, -1.22474487,  1.58113883],
           [ 0.70710678,  0.        , -0.79056942],
           [ 0.70710678,  1.22474487,  1.58113883]])
    """
    vander = np.zeros([len(r), n_order+1])
    
    for j in range(n_order+1):
        vander[:,j] = jacobi_polynomial(r, 0, 0, j)
        
    return vander

def differentiation_matrix(n_order,r,vander):
    """
    Initialize the (r) differentiation matrices
    of the interval evaluated at (r) at order n_order
    V is the 1d Vandermonde matrix
    >>> r = jacobi_gauss_lobatto(0,0,2)
    >>> v = vandermonde(2,r)
    >>> differentiation_matrix(2,r,v)
    array([[-1.5,  2. , -0.5],
           [-0.5,  0. ,  0.5],
           [ 0.5, -2. ,  1.5]])
    >>> r = jacobi_gauss_lobatto(0,0,3)
    >>> v = vandermonde(3,r)
    >>> A1 = differentiation_matrix(3,r,v)
    >>> A2 = ([[-3.00000000e+00,  4.04508497e+00, -1.54508497e+00,  5.00000000e-01], \
               [-8.09016994e-01, -4.05396129e-16,  1.11803399e+00, -3.09016994e-01], \
               [ 3.09016994e-01, -1.11803399e+00,  6.28036983e-16,  8.09016994e-01], \
               [-5.00000000e-01,  1.54508497e+00, -4.04508497e+00,  3.00000000e+00]])
    >>> np.allclose(A1,A2)
    True
    """
    v_r = vandermonde_grad(n_order,r)
    v_inv = np.linalg.inv(vander)
    diff_matrix = np.matmul(v_r,v_inv)
    return diff_matrix

def vandermonde_grad(n_order,r):
    """
    Initialize the gradient of the modal basis (i) at (r)
    at order (n_order)
    >>> r = jacobi_gauss_lobatto(0,0,2)
    >>> vandermonde_grad(2,r)
    array([[ 0.        ,  1.22474487, -4.74341649],
           [ 0.        ,  1.22474487,  0.        ],
           [ 0.        ,  1.22474487,  4.74341649]])
    """
    grad_vander =  np.zeros([len(r),n_order+1])
    for i in range(n_order+1):
        grad_vander[:,i] = jacobi_polynomial_grad(r,0,0,i)
        
    return grad_vander

def jacobi_polynomial_grad(r, alpha, beta, n_order):
    """
    Evaluate the derivative of the Jacobi pol. of type (alpha,beta) > -1
    at points r for order n_order
    >>> r = jacobi_gauss_lobatto(0,0,1)
    >>> jacobi_polynomial_grad(r,0,0,1)
    array([1.22474487, 1.22474487])
    >>> r = jacobi_gauss_lobatto(0,0,3)
    >>> jacobi_polynomial_grad(r,0,0,3)
    array([11.22497216,  0.        ,  0.        , 11.22497216])
    """
    
    der_jacobi_pol = np.zeros([len(r)])

    if n_order == 0:
        return der_jacobi_pol
       
    jacobi_pol = jacobi_polynomial(r,alpha+1,beta+1,n_order-1)

    for i in range(len(r)):
        der_jacobi_pol[i] = math.sqrt(n_order*(n_order+alpha+beta+1))*jacobi_pol[i]

    return der_jacobi_pol

def jacobi_polynomial(r, alpha, beta, n_order):
    """
    Evaluate Jacobi Polynomial
    >>> r = jacobi_gauss_lobatto(0,0,1)
    >>> jacobi_polynomial(r, 0, 0, 1)
    array([-1.22474487,  1.22474487])
    >>> r = jacobi_gauss_lobatto(0,0,2)
    >>> jacobi_polynomial(r, 0, 0, 2)
    array([ 1.58113883, -0.79056942,  1.58113883])
    >>> r = jacobi_gauss_lobatto(0,0,3)
    >>> jacobi_polynomial(r, 0, 0, 3)
    array([-1.87082869,  0.83666003, -0.83666003,  1.87082869])
    >>> r = jacobi_gauss_lobatto(0,0,4)
    >>> jacobi_polynomial(r, 0, 0, 4)
    array([ 2.12132034, -0.90913729,  0.79549513, -0.90913729,  2.12132034])
    """
    jacobi_pol = np.zeros([n_order+1,len(r)]) 
    # Initial values P_0(x) and P_1(x)
    gamma0 = 2**(alpha+beta+1) \
            / (alpha+beta+1) \
            * scipy.special.gamma(alpha+1) \
            * scipy.special.gamma(beta+1) \
            / scipy.special.gamma(alpha+beta+1)

    jacobi_pol[0] = 1.0 / math.sqrt(gamma0)
    if n_order == 0:
    #    return PL.transpose()
        return jacobi_pol[0]

    gamma1 = (alpha+1.) * (beta+1.) / (alpha+beta+3.) * gamma0
    jacobi_pol[1] = ((alpha+beta+2.)*r/2. + (alpha-beta)/2.) / math.sqrt(gamma1)

    if n_order == 1:
    #    return PL.transpose()
        return jacobi_pol[1]
    # Repeat value in recurrence.
    aold = 2. / (2.+alpha+beta) \
        * math.sqrt( (alpha+1.)*(beta+1.) / (alpha+beta+3.))
    
    # Forward recurrence using the symmetry of the recurrence.
    for i in range(n_order-1):
        h1 = 2.*(i+1.) + alpha + beta
        anew = 2. / (h1+2.) \
            * math.sqrt((i+2.)*(i+2.+ alpha+beta)*(i+2.+alpha)*(i+2.+beta) \
                        / (h1+1.)/(h1+3.))
    
        bnew = - (alpha**2 - beta**2) / h1 / (h1+2.)
    
        jacobi_pol[i+2] = 1. / anew * (-aold * jacobi_pol[i] + (r-bnew) * jacobi_pol[i+1])
    
        aold = anew
    
    return jacobi_pol[n_order]