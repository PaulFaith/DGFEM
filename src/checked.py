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
