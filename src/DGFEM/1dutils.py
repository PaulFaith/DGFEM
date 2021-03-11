import numpy as np
import scipy.special
import math

n_faces = 1
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

    uin = -ur[vmap_i-1]
    uout = -ur[vmap_O-1]

    du[map_i-1] = (ur[vmap_i-1] - uin)/2
    du[map_O-1] = (ur[vmap_O-1] - uout)/2
    dur = np.reshape(du, (2, int(len(du)/2)), order = 'F')
    du1 =nx*dur
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

    
def set_nodes(n_order, vertices):
    """ 
    Sets n_order+1 nodes in equispaced positions using the vertices indicated
    by vx.
    """
    
    vertices_size = vertices.shape[1]
    nodes_x = np.zeros((n_order+1, vertices_size))
    for k in range(vertices_size):
        for i in range(n_order+1):
            nodes_x[i,k] = i * (vertices[1,k] - vertices[0,k]) / n_order + vertices[0,k]
            
    return nodes_x
def node_indices(n_order):
    """
    Generates number of node Indices for order n_order.
    
    >>> node_indices(1)
    array([[1, 0],
           [0, 1]])
        
    >>> node_indices(2)
    array([[2, 0],
           [1, 1],
           [0, 2]])
    """
    n_p = n_order+1
    n_id = np.zeros([n_p, 2])
    for i in range(n_p):
        n_id[i] = [n_order-i, i]     
    return n_id.astype(int)
        

    points = np.zeros(n_order)
    weight = np.zeros(n_order)
    if n_order==0:
        points = -(alpha-beta)/(alpha+beta+2)
        weight = 2
        return [points,weight]
    
    # Form symmetric matrix from recurrence.
    j_matrix = np.zeros([n_order+1,n_order+1])
    h1 = np.zeros(n_order+1)
    aux = np.zeros(n_order)
    
    for i in range(n_order):
        aux[i] = 1+i

    for i in range(n_order+1):
        h1[i] = 2*i+alpha+beta

    j_matrix = np.diag(-0.5*(alpha**2-beta**2)/(h1+2)/h1) \
        + np.diag(2/(h1[0:n_order]+2) \
        * np.sqrt(aux*(aux+alpha+beta)*(aux+alpha) \
        * (aux+beta)/(h1[0:n_order]+1)/(h1[0:n_order]+3)),1)

    eps = np.finfo(np.float).eps

    if (alpha+beta < 10*eps):
        j_matrix[0,0] = 0.0
    
    j_matrix = j_matrix+np.transpose(j_matrix)
    
    [e_val,e_vec] = np.linalg.eig(j_matrix)
    
    points = e_val 
    
    weight = e_vec[0,:]**2*2**(alpha+beta+1)/(alpha+beta+1)*scipy.special.gamma(alpha+1) \
        * scipy.special.gamma(beta+1)/scipy.special.gamma(alpha+beta+1)
    
    return [points,weight]
def filter(n_order,n_c,s,vander):
    """
    Initialize 1D filter matrix of size n_order.
    Order of exponential filter is (even) s with cutoff at n_c;
    >>> r = jacobi_gauss_lobatto(0,0,3)
    >>> v = vandermonde(3,r)
    >>> filter(3,1,1,v)
    array([[ 0.3333842 ,  0.9756328 , -0.14240119, -0.1666158 ],
           [ 0.19512656,  0.66667684,  0.16667684, -0.02848024],
           [-0.02848024,  0.16667684,  0.66667684,  0.19512656],
           [-0.1666158 , -0.14240119,  0.9756328 ,  0.3333842 ]])
    """
    s_even = 2*s
    f_diagonal = np.ones(n_order+1)
    alpha = -np.log(np.finfo(np.float).eps)

    for i in range(n_c,n_order+1):
        f_diagonal[i] = np.exp(-alpha*((i-n_c)/(n_order-n_c))**s_even)

    # F = V*diag(Fdiagonal)*Vinv
    f_i = np.matmul(vander,np.diag(f_diagonal))
    v_inv = np.linalg.inv(vander)
    filter = np.matmul(f_i,v_inv)
    return filter

if __name__ == '__main__':
    import doctest
    doctest.testmod()
