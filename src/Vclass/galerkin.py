import numpy as np
import scipy.special
import math

def jacobi_gauss_lobatto(alpha, beta, n_order):
    """
    OCTAVE CHECKED jacobiGL
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

def jacobi_gauss(alpha, beta, n_order):
    """
    Compute the order n_order Gauss quadrature points, x,
    and weights, w, associated with the Jacobi
    polynomial, of type (alpha,beta) > -1 ( <> -0.5).
    >>> s1 = jacobi_gauss(2,1,0)
    >>> s2 = [-0.2,  2]
    >>> np.allclose(s1,s2)
    True
    >>> s1 = jacobi_gauss(2,1,1)
    >>> s2 = [([-0.54691816,  0.26120387]), ([0.76094757, 0.57238576])]
    >>> np.allclose(s1,s2)
    True
    >>> s1 = jacobi_gauss(2,1,2)
    >>> s2 = [([-0.70882014, -0.13230082,  0.50778763]), ([0.39524241,  0.72312171,  0.21496922])]
    >>> np.allclose(s1,s2)
    True
    """

def mesh_generator(xmin,xmax,k_elem):
    """
    OCVATE CHECKED
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
    OCTAVE CHECKED Vandermonde1D
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
    OCTAVE CHECKED Dmatrix1D
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
    OCTAVE CHECKED
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
    OCTAVE CHECKED, this function is used on vandermonde_grad.
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
    OCTAVE CHECKED JacobiP
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

def surface_integral_dg(n_order,vander):
    """
    OCTAVE CHECKED Lift1D
    Compute surface integral term in DG formulation
    >>> r = jacobi_gauss_lobatto(0,0,2)
    >>> v = vandermonde(2,r)
    >>> surface_integral_dg(2,v)
    array([[ 4.5 ,  1.5 ],
           [-0.75, -0.75],
           [ 1.5 ,  4.5 ]])
    >>> r = jacobi_gauss_lobatto(0,0,3)
    >>> v = vandermonde(3,r)
    >>> surface_integral_dg(3,v)
    array([[ 8.        , -2.        ],
           [-0.89442719,  0.89442719],
           [ 0.89442719, -0.89442719],
           [-2.        ,  8.        ]])
    """
    # n_faces, n_fp and n_p are defined as global variables
    n_faces = 1
    n_fp = 2
    n_p = n_order+1

    emat = np.zeros([n_p,n_faces*n_fp])
    emat[0,0] = 1.0
    emat[n_p-1,1] = 1.0

    v_trans = np.transpose(vander)
    v_i = np.matmul(v_trans,emat)
    lift = np.matmul(vander,v_i)

    return lift

def nodes_coordinates(n_order,etov,vx):
    """
    OCTAVE CHECKED 2 menos
    Part of StartUp1D.m. Defined to be able to define
    methods depedent grid properties
    >>> [Nv,vx,K,etov] = mesh_generator(0,10,4)
    >>> x = nodes_coordinates(4,etov,vx)
    >>> x_test = ([[0.00000000,    2.50000000,    5.00000000,    7.50000000], \
                   [0.43168291,    2.93168291,    5.43168291,    7.93168291], \
                   [1.25000000,    3.75000000,    6.25000000,    8.75000000], \
                   [2.06831709,    4.56831709,    7.06831709,    9.56831709], \
                   [2.50000000,    5.00000000,    7.50000000,   10.00000000]])
    >>> np.allclose(x,x_test)
    True
    """

    jgl = jacobi_gauss_lobatto(0,0,n_order)

    va = etov[:,0]
    vb = etov[:,1]
    vx_va = np.zeros([1,len(va)], dtype = 'g')
    vx_vb = np.zeros([1,len(va)], dtype = 'g')
    for i in range(len(va)):
        vx_va[0,i] = vx[va[i]-1]
        vx_vb[0,i] = vx[vb[i]-1]

    nodes_coord = np.matmul(np.ones([n_order+1,1], dtype = 'g'),vx_va)+0.5*np.matmul((jgl.reshape(n_order+1,1)+1),(vx_vb-vx_va))
    return nodes_coord

def geometric_factors(nodes_coord,diff_matrix):
    """
    OCTAVE CHECKED.
    Compute the metric elements for the local mappings of the 1D elements
    >>> [Nv,vx,K,etov] = mesh_generator(0,10,4)
    >>> x = nodes_coordinates(2,etov,vx)
    >>> r = jacobi_gauss_lobatto(0,0,2)
    >>> V = vandermonde(2,r)
    >>> Dr = differentiation_matrix(2,r,V)
    >>> [rx,J] = geometric_factors(x,Dr)
    >>> rx_test = ([[0.80000,   0.80000,   0.80000,   0.80000], \
                    [0.80000,   0.80000,   0.80000,   0.80000], \
                    [0.80000,   0.80000,   0.80000,   0.80000]])
    >>> J_test =   ([[1.2500,   1.2500,   1.2500,   1.2500], \
                     [1.2500,   1.2500,   1.2500,   1.2500], \
                     [1.2500,   1.2500,   1.2500,   1.2500]])
    >>> np.allclose(rx,rx_test)
    True
    >>> np.allclose(J,J_test)
    True
    """
    xr = np.matmul(diff_matrix,nodes_coord)
    jacobian = xr
    rx = 1/jacobian

    return [rx,jacobian]

def connect(etov):
    """
    OCTAVE CHECKED
    Build global connectivity arrays for 1D grid based on standard
    etov input array from grid generator
    >>> [Nv,vx,K,etov] = mesh_generator(0,10,4)
    >>> [etoe, etof] = connect(etov)
    >>> etoe_test =  ([[1,2], \
                       [1,3], \
                       [2,4], \
                       [3,4]])
    >>> etof_test =  ([[1,1], \
                       [2,1], \
                       [2,1], \
                       [2,2]])
    >>> np.allclose(etoe,etoe_test)
    True
    >>> np.allclose(etof,etof_test)
    True
    >>> [Nv,vx,K,etov] = mesh_generator(-1,22,7)
    >>> [etoe, etof] = connect(etov)
    >>> etoe_test = ([[1,2],\
                      [1,3],\
                      [2,4],\
                      [3,5],\
                      [4,6],\
                      [5,7],\
                      [6,7]])
    >>> etof_test = ([[1,1],\
                      [2,1],\
                      [2,1],\
                      [2,1],\
                      [2,1],\
                      [2,1],\
                      [2,2]])
    >>> np.allclose(etoe,etoe_test)
    True
    >>> np.allclose(etof,etof_test)
    True
    """
    n_faces = 2
    k_elem = np.shape(etov)[0]
    total_faces = n_faces*k_elem
    nv = k_elem+1
    vn = np.arange(0,2)
    sp_ftov = np.zeros([total_faces,nv])

    sk = 0

    for i in range(k_elem):
        for face in range(n_faces):
            sp_ftov[sk][etov[i][vn[face]]-1] = 1
            sk += 1

    sp_ftof = np.matmul(sp_ftov,np.transpose(sp_ftov))-np.identity(total_faces)
    [faces_2,faces_1] = np.where(sp_ftof==1)
    #numpy floor returns floats

    element_1 = np.int64(np.floor(faces_1/n_faces))
    element_2 = np.int64(np.floor(faces_2/n_faces))

    face_1 = np.mod(faces_1,n_faces)
    face_2 = np.mod(faces_2,n_faces)

    ind = np.arange(len(element_1))

    for i in range(len(element_1)):
        ind[i] = np.ravel_multi_index((element_1[i],face_1[i]),dims=(k_elem,n_faces))

    etoe_1 = np.transpose(np.arange(1,k_elem+1).reshape(1,k_elem))
    etoe_2 = np.full([1,n_faces],1)
    etoe = np.matmul(etoe_1,etoe_2)

    etof_1 = np.full([k_elem,1],1)
    etof_2 = np.arange(1,n_faces+1).reshape(1,n_faces)
    etof = np.matmul(etof_1,etof_2)

    for i in range(len(ind)):
        etoe.ravel()[ind[i]] = element_2[i]+1
        etof.ravel()[ind[i]] = face_2[i]+1

    return [etoe, etof]

def normals(k_elem):
    """
    OCTAVE CHECKED
    Compute outward pointing normals at element faces
    >>> normals(4)
    array([[-1., -1., -1., -1.],
           [ 1.,  1.,  1.,  1.]])
    """
    # K is the number of elements, derived from the grid info
    # n_faces and n_fp are defined as global variables
    n_faces = 1
    n_fp = 2
    nx = np.zeros([n_fp*n_faces,k_elem])
    nx[0,:] = -1.0
    nx[1,:] = 1.0
    return nx

def build_maps(n_order,nodes_coord,etoe,etof):
    """
    OCTAVE CHECKED
    Connectivity and boundary tables for nodes given in the K # of elements,
    each with n_order+1 degrees of freedom.
    >>> [Nv,vx,K,etov] = mesh_generator(0,10,4)
    >>> x = nodes_coordinates(4,etov,vx)
    >>> [etoe, etof] = connect(etov)
    >>> [vmap_m,vmap_p,vmap_b,map_b] = build_maps(4,x,etoe,etof)
    >>> vmap_m_test = ([[1,5,6,10,11,15,16,20]])
    >>> np.allclose(vmap_m,vmap_m_test)
    True
    >>> vmap_p_test = ([[1,6,5,11,10,16,15,20]])
    >>> np.allclose(vmap_p,vmap_p_test)
    True
    >>> vmap_b_test = ([[1,20]])
    >>> np.allclose(vmap_b,vmap_b_test)
    True
    >>> map_b_test = ([[1,8]])
    >>> np.allclose(map_b,map_b_test)
    True
    """
    jgl = jacobi_gauss_lobatto(0,0,n_order)
    k_elem = np.size(etoe,0)
    n_p = n_order+1
    n_faces = 2
    n_fp = 1
    #mask defined in globals
    fmask_1 = np.where(np.abs(jgl+1)<1e-10)[0][0]
    fmask_2 = np.where(np.abs(jgl-1)<1e-10)[0][0]
    fmask = [fmask_1,fmask_2]
    node_ids = np.reshape(np.arange(k_elem*n_p),[n_p,k_elem],'F')
    vmap_m = np.full([k_elem,n_fp,n_faces],0)
    vmap_p = np.full([k_elem,n_fp,n_faces],0)

    for k1 in range(k_elem):
        for f1 in range(n_faces):
            vmap_m[k1,:,f1] = node_ids[fmask[f1],k1]

    for k1 in range(k_elem):

        for f1 in range(n_faces):
            k2 = etoe[k1,f1]-1
            f2 = etof[k1,f1]-1

            vid_m = vmap_m[k1,:,f1][0]
            vid_p = vmap_m[k2,:,f2][0]

            x1 = nodes_coord.ravel('F')[vid_m]
            x2 = nodes_coord.ravel('F')[vid_p]

            distance = (x2-x1)**2
            if (distance < 1e-10):
                vmap_p[k1,:,f1] = vid_p

    vmap_m+=1
    vmap_p+=1

    vmap_p = vmap_p.ravel()
    vmap_m = vmap_m.ravel()

    map_b = np.where(vmap_p==vmap_m)[0]
    vmap_b = vmap_m[map_b]

    map_b+=1
    vmap_b

    map_i = 1
    map_o = k_elem*n_faces
    vmap_i = 1
    vmap_0 = k_elem*n_p

    return [vmap_m,vmap_p,vmap_b,map_b,fmask]

def rk4(l, intrk):
  a = np.array([                             0.0, \
                   -567301805773.0/1357537059087.0, \
                  -2404267990393.0/2016746695238.0, \
                  -3550918686646.0/2091501179385.0, \
                   -1275806237668.0/842570457699.0])
  b = np.array([ 1432997174477.0/9575080441755.0, \
                  5161836677717.0/13612068292357.0, \
                   1720146321549.0/2090206949498.0, \
                   3134564353537.0/4481467310338.0, \
                  2277821191437.0/14882151754819.0])
  c = np.array([                             0.0, \
                   1432997174477.0/9575080441755.0, \
                   2526269341429.0/6820363962896.0, \
                   2006345519317.0/3224310063776.0, \
                   2802321613138.0/2924317926251.0])
  if l == "a":
    return a[intrk]
  elif l == "b":
    return b[intrk]
  return c[intrk]

class Galerkin:
    meshi = 0.0
    meshf = 1.0
    def __init__(self, K, N):
        self.K = K
        self.N = N
        self.tf = tf
        [self.Nv, self.VX, self.EToV] = mesh_generator(meshi, meshf*math.pi, self.K)
        self.r = jacobi_gauss_lobatto(0, 0, self.N)
        self.V = vandermonde(self.N, self.r)
        self.Dr = differentiation_matrix(self.N, self.r, self.V)
        self.LIFT = surface_integral_dg(self.N, self.V)
        self.x = nodes_coordinates(self.N, self.EToV, self.VX)
        [self.rx, self.J] = geometric_factors(self.x, self.Dr)
        self.nx = normals(self.K)
        [self.EToE, self.EToF] = connect(self.EToV)
        [self.vmapM, self.vmapP, self.vmapB, self.mapB, self.fmask] = build_maps(self.N, self.x, self.EToE, self.EToF)
        self.Fscale = 1/J[self.fmask,:]
        self.t = 0
    def __string__(self):
        return f"Discrete Galerkin FEM initialized with N = {self.N} and K = {self.K}, lets go mr. Intel."

    def calculate(self, case, u, tf):
        if case == "A":
            pass
        elif case == "Msin":
            pass
        elif case == "Mg":
            pass
        elif case == "Mgglass":
            pass
        elif cases == "H":
            pass
        elif case == "Sch":
            pass
        elif case == "SchNL":
            pass
        else:
            pass

#
#
#
#
#
#
#
#
#
#     u = np.sin(x)
#     finaltime = .8
#
#     #Runge-Kutta residual storage.
#     resu = np.zeros((N+1, K))
#     #Compute time steps.
#     xmin = np.amin(np.abs(x[0, :] - x[1, :]))
#     CFL = .25
#     dt = CFL*xmin*xmin
#     Nsteps = np.ceil(finaltime/dt)
#     dt = finaltime/Nsteps
#     #nplots = int(Nsteps/5)
#     errorE = np.array([])
#     #fig, axs = plt.subplots(6)
#     for tstep in range(int(Nsteps)):
#       #if (tstep % nplots == 0):
#         #ux = np.exp(-t)*np.sin(x)
#         #axs[int(tstep/nplots)].set_ylim(-1,1)
#         #axs[int(tstep/nplots)].plot(x, u, '.r:', ms = 6, color = 'red')
#         #axs[int(tstep/nplots)].plot(x, ux, '.b:', ms = 6, color = 'blue')
#       for intrk in range(5):
#         timelocal = t + rk4("c", intrk)*dt
#         rhsu  = HeatCRHS1D(u, timelocal, k_element, N, Dr, LIFT, rx, nx, vmapP, vmapM, Fscale)
#         resu = rk4("a", intrk)*resu + dt*rhsu
#         u = u + rk4("b", intrk)*resu
#
#       #Calculo del error y la eficiencia del metodo para distintos K y N.
#       ux = np.exp(-t)*np.sin(x)
#       if errorE.size == 0 :
#         errorE = np.sqrt((np.sum((ux-u)*(ux-u)))/len(u)*len(u[0]))
#       else :
#         errorE = np.append(errorE, np.sqrt((np.sum((ux-u)*(ux-u)))/len(u)*len(u[0])))
#
#
#       t = t + dt
#     #for ax in axs.flat:
#         #ax.set(xlabel=f'x, N = {N}, K = {k_element}', ylabel='u(x,t)')
#
#     #for ax in axs.flat:
#         #ax.label_outer()
#
#
#     #PLOT SOLUTION.
#     #if errorEK.size == 0 :
#       #errorEK = np.sum(errorE)/len(errorE)
#     #else :
#     errorEK = np.append(errorEK, np.sum(errorE))#/len(errorE)
#     errorE = np.delete
#     #print(errorEK)
#     #print(Kg)
#   plt.plot(Kg , errorEK, label = f'N = {N}')
#   plt.annotate(f'N = {N}', (Kg[int(len(Kg)//2)],errorEK[int(len(errorEK)//2)]), textcoords="offset points", xytext=(-10,10), ha='center')
#   errorEK = np.array([])
#   Kg = np.array([])
# plt.show()


def advecrhs1d(u, timelocal, a, k_elem, Dr, LIFT, rx, nx, vmap_p, vmap_m, Fscale):
    K=10
    n_faces = 1
    map_O = K*n_faces
    vmap_i = 1
    map_i = 1
    n_fp = 2
    alpha = 1
    du = np.zeros(n_faces*n_fp*k_elem)
    #du reshape
    nxr = np.reshape(nx, len(nx)*len(nx[0]), order='F')
    #nx reshape
    ur = np.reshape(u, len(u)*len(u[0]), order='F')
    du = (ur[vmap_m-1]-ur[vmap_p-1])*(a*nxr)/2

    uin = -np.sin(a*timelocal)
    du[map_i-1] = (ur[vmap_i-1] - uin)*(a*nxr[map_i-1])/2
    du[map_O-1] = 0
    arx = -a*rx
    Dru = np.matmul(Dr,u)
    si = LIFT
    dur = np.reshape(du, (2, int(len(du)/2)), order = 'F')
    Fdu = Fscale*dur
    rhsu = arx*Dru + np.matmul(si,Fdu)
    return rhsu

def maxwell1d(INTRK, tsteps, E, H, epsilon, mu, k_elem, Dr, LIFT, rx, nx, vmap_p, vmap_m, map_b, vmap_b, Fscale):
  n_faces = 1
  n_fp = 2

  Zimp = np.sqrt(mu/epsilon)
  Zimpr = np.reshape(Zimp, len(Zimp)*len(Zimp[0]), order='F')

  dE = np.zeros((n_faces*n_fp*k_elem))
  dH = np.zeros((n_faces*n_fp*k_elem))
  #du reshape
  nxr = np.reshape(nx, len(nx)*len(nx[0]), order='F')
  #nx reshape
  Er = np.reshape(E, len(E)*len(E[0]), order='F')
  Hr = np.reshape(H, len(H)*len(H[0]), order='F')
  dE = Er[vmap_m-1]-Er[vmap_p-1]
  dH = Hr[vmap_m-1]-Hr[vmap_p-1]
  Zimpr = np.reshape(Zimp, len(Zimp)*len(Zimp[0]), order='F')

  Zimpm = np.zeros((n_faces*n_fp*k_elem))
  Zimpm = Zimpr[vmap_m-1]
  Zimpp = np.zeros((n_faces*n_fp*k_elem))
  Zimpp = Zimpr[vmap_p-1]
  Yimpm = np.zeros((n_faces*n_fp*k_elem))
  Yimpm = 1/Zimpm
  Yimpp = np.zeros((n_faces*n_fp*k_elem))
  Yimpp = 1/Zimpp

  Ebc = -Er[vmap_b-1]
  dE[map_b-1] = Er[vmap_b-1] - Ebc
  Hbc = Hr[vmap_b-1]
  dH[map_b-1] = Hr[vmap_b-1] - Hbc

  fluxE = 1/(Zimpm + Zimpp)*(nxr*Zimpp*dH - dE)
  fluxH = 1/(Yimpm + Yimpp)*(nxr*Yimpp*dE - dH)

  fluxEr = np.reshape(fluxE, (2, int(len(fluxE)/2)), order = 'F')
  fluxHr = np.reshape(fluxH, (2, int(len(fluxE)/2)), order = 'F')

  FfluxE = Fscale*fluxEr
  FfluxH = Fscale*fluxHr

  rhsE = (-rx*np.matmul(Dr,H) + np.matmul(LIFT,FfluxE))/epsilon
  rhsH = (-rx*np.matmul(Dr,E) + np.matmul(LIFT,FfluxH))/mu

  return rhsE, rhsH

def maxwellexE(n1, n2, t, x, w):
  xr = np.reshape(x, len(x)*len(x[0]), order='F')
  A = np.zeros(2)
  B = np.zeros(2)
  j = np.complex(0,1)
  A[0] = (n2*np.cos(n2*w))/(n1*np.cos(n1*w))
  A[1] = np.real(np.exp(w*(n1+n2)*j))
  B[0] = np.real(np.exp(-2*n1*w*j))*A[0]
  B[1] = -np.real(np.exp(2*j*n2*w))*A[1]
  Ex1 = np.zeros((len(x)*len(x[0])))
  Ex2 = np.zeros((len(x)*len(x[0])))

  Ex1 = (-A[0]*np.real(np.exp(w*j*n1*xr[:len(xr)//2]))+B[0]*np.real(np.exp(-w*j*n1*xr[:len(xr)//2])))*np.real(np.exp(j*w*t))
  Ex2 = (-A[1]*np.real(np.exp(w*j*n2*xr[len(xr)//2:]))+B[1]*np.real(np.exp(-w*j*n2*xr[len(xr)//2:])))*np.real(np.exp(j*w*t))
  Ex = np.concatenate((Ex1, Ex2), axis = 0)
  Exc = np.reshape(Ex, (len(x), int(len(x[0]))), order='F')

  return Exc

def maxwellexH(n1, n2, t, x, w):
  #No esta completo
  xr = np.reshape(x, len(x)*len(x[0]), order='F')
  A = np.zeros(2)
  B = np.zeros(2)
  j = np.complex(0,1)
  A[0] = (n2*np.cos(n2*w))/(n1*np.cos(n1*w))
  A[1] = np.real(np.exp(j*w*(n1+n2)))
  B[0] = np.real(np.exp(-2*j*n1*w))*A[0]
  B[1] = -np.real(np.exp(2*j*n2*w))*A[1]
  Hx1 = np.zeros((len(x)*len(x[0])))
  Hx2 = np.zeros((len(x)*len(x[0])))

  Hx1 = (A[0]*np.real(np.exp(w*j*n1*xr[:len(xr)//2]))+B[0]*np.real(np.exp(-w*j*n1*xr[:len(xr)//2])))*np.real(np.exp(j*w*t))
  Hx2 = (A[1]*np.real(np.exp(w*j*n2*xr[len(xr)//2:]))+B[1]*np.real(np.exp(-w*j*n2*xr[len(xr)//2:])))*np.real(np.exp(j*w*t))
  Hx = np.concatenate((Hx1, Hx2), axis = 0)
  Hxc = np.reshape(Hx, (len(x), int(len(x[0]))), order='F')

  return Hxc

def HeatCRHS1D(u, timelocal, k_elem, N, Dr, LIFT, rx, nx, vmap_p, vmap_m, Fscale):
    n_faces = 2
    map_O = k_elem*n_faces
    vmap_O = k_elem*(N+1)
    vmap_i = 1
    map_i = 1
    n_fp = 2
    alpha = 1
    du = np.zeros(n_faces*n_fp*k_elem)
    dq = np.zeros(n_faces*n_fp*k_elem)
    #du reshape
    nxr = np.reshape(nx, len(nx)*len(nx[0]), order='F')
    #nx reshape
    ur = np.reshape(u, len(u)*len(u[0]), order='F')
    du = (ur[vmap_m-1]-ur[vmap_p-1])/2
    #print(du, '\n')
    uin = -ur[vmap_i-1]
    uout = -ur[vmap_O-1]
    #print(map_O)
    #print(vmap_O)
    #print(uout, '\n\n', uin, '\n\n')

    du[map_i-1] = (ur[vmap_i-1] - uin)/2
    du[map_O-1] = (ur[vmap_O-1] - uout)/2
    dur = np.reshape(du, (2, int(len(du)/2)), order = 'F')
    du1 =nx*dur
    #print(du)
    #print(du1)
    si = LIFT
    Dru = np.matmul(Dr,u)
    Fdu = Fscale*du1
    q = rx*Dru - np.matmul(si,Fdu)
    qr = np.reshape(q, len(q)*len(q[0]), order='F')
    dq = (qr[vmap_m-1]-qr[vmap_p-1])/2
    qin = qr[vmap_i-1]
    qout = qr[vmap_O-1]
    dq[map_i-1] = (qr[vmap_i-1] - qin)/2
    dq[map_O-1] = (qr[vmap_O-1] - qout)/2
    dqr = np.reshape(dq, (2, int(len(dq)/2)), order = 'F')
    dq1 = nx*dqr
    Drq = np.matmul(Dr,q)
    Fdq = Fscale*dq1
    rhsu = rx*Drq - np.matmul(si,Fdq)
    #input("Press Enter to continue...")
    return rhsu
