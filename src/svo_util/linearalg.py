'''
Created on Apr 10, 2013
@author: Stephen O'Hara

Utility functions relating to linear algebra
computations.
'''
import scipy.linalg as LA
import scipy as sp

def nullspace(A, atol=1e-13, rtol=0):
    '''Compute an approximate basis for the nullspace of A.
    The algorithm used by this function is based on the singular value
    decomposition of `A`. This implementation was copied
    from the scipy cookbook: http://www.scipy.org/Cookbook/RankNullspace

    @param A: ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    @param atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    @param rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    @note: If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    @return: ns ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    '''

    A = sp.atleast_2d(A)
    _u, s, vh = LA.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def closestOrthogonal(A):
    '''
    Uses SVD to compute the closest orthogonal
    matrix to input matrix A
    '''
    U,_,Vt = LA.svd(A, full_matrices=False)
    return sp.dot(U,Vt)

def isOrthogonal(A, tol=1e-13):
    '''
    Test whether matrix A is orthogonal, upto
    numerical tolerance. If A is orthogonal then
    sp.dot(A.T,A) will be the identity matrix.
    '''
    (_,p) = A.shape
    Ix = sp.dot( A.T, A) - sp.eye(p)
    return not sp.any(Ix > tol)

def isSameGrassmannPoint(M1,M2, tol=1e-13):
    '''
    Are matrices M1 and M2 two different matrix
    representations of the same point, M, on
    the Grassmann? Assume M in Gr(n,k),
    M1 and M2 are orthogonal with dimensions (nxk)
    True if M1 = sp.dot(M2,X) and sp.dot(X.T,X)==I (kxk)
    '''
    assert isOrthogonal(M1, tol=tol)
    assert isOrthogonal(M2, tol=tol)
    
    Xinv = sp.dot(M1.T,M2)
    #if Xinv is orthogonal, then so is X
    return isOrthogonal(Xinv, tol=tol)

def chordal_dist(M1, M2, already_orthogonal=False):
    '''
    The chordal distance is based on the canonical angles
    between subspaces. This function computes the chordal
    distance between two matrices.
    @param M1: A 2D array (matrix) with rows >= cols.
    @param M2: A 2D array (matrix) with rows >= cols.
    @param already_orthogonal: Specify True if M1 and M2
    are already orthogonal matrices, which will save on
    unnecessary computation. Otherwise, an SVD will be
    used to get an orthogonal representation of each matrix.
    '''
    (r,c) = M1.shape
    assert( r >= c)

    (r,c) = M2.shape
    assert( r >= c)
    
    if already_orthogonal:
        Q1 = M1
        Q2 = M2
    else:
        #get orthonormal bases
        #NOTE: in scipy.linalg, using the thin svd to get the orthonormal bases is MUCH FASTER
        # than using either the LA.orth(A) function or "economy" mode of QR decomposition!
        (Q1,_,_) = LA.svd(M1, full_matrices=False)
        (Q2,_,_) = LA.svd(M2, full_matrices=False)
        
    #canonical angles between subspaces
    X = sp.dot(Q1.T,Q2)
    S = LA.svdvals( X )
    #S = cos(Theta)
    Theta = sp.arccos(S)
    
    #chordal distance is ||sin(Theta)||_2
    return LA.norm( sp.sin(Theta)  )

def grassmann_expmap(tA,p, tol=1e-13):
    '''
    Computes the manifold exp-map of a point, tA, in the tangent space T,
    of grassmann manifold M, with T centered at point p.
    @param tA: The point from TpM to be mapped to the manifold
    @param p: The "pole", or the point where the tangent space is
    incident to the manifold.
    @param tol: Numerical tolerance for assuming a number is zero
    '''   
    U, s, Vt = LA.svd(tA, full_matrices=False)
    s[ s < tol ]= 0   #set extremely small values to zero
    
    cosTheta = sp.cos(s)
    sinTheta = sp.sin(s)
    
    V = Vt.T
    exp_tA = sp.dot(p, sp.dot(V, sp.diag(cosTheta))) + sp.dot(U, sp.diag(sinTheta))
    
    return exp_tA
    

def grassmann_logmap(A,p, tol=1e-13, skip_orthog_check=False):
    '''
    Computes the manifold log-map of (nxk) orthogonal matrix A,
    centered at the point p (i.e. the "pole"), which is also an
    (nxk) orthogonal matrix.
    The log-map takes a point on the manifold and maps it to the
    tangent space which is centered at a given pole.
    The dimension of the tangent space is k(n-k), 
    and points A,p are on Gr(n,k).
    @param A: The orthogonal matrix A, representing a point on
    the grassmann manifold.
    @param p: An orthogonal matrix p, representing a point on
    the grassmann manifold where the tangent space will be formed.
    Also called the "pole".
    @param tol: Numerical tolerance used to set singular values
    to exactly zero when within this tolerance of zero.
    @param skip_orthog_check: Set to True if you can guarantee
    that the inputs are already orthogonal matrices. Otherwise,
    this function will check, and if A and/or p are not orthogonal,
    the closest orthogonal matrix to A (or p) will be used.
    @return: A tuple (log_p(A), ||log_p(A)|| ), representing
    the tangent-space mapping of A, and the distance from the
    mapping of A to the pole in the tangent space.
    '''
    
    #check that A and p are orthogonal, if
    # not, then compute orthogonal representations and
    # send back a warning message.
    if not skip_orthog_check:
        if not isOrthogonal(A):
            print "WARNING: You are calling grassmann_logmap function on non-orthogonal input matrix A"
            print "(This function will compute an orthogonal representation for A using an SVD.)"
            A = closestOrthogonal(A)
        if not isOrthogonal(p):
            print "WARNING: You are calling grassmann_logmap function on non-orthogonal pole p."
            print "(This function will compute an orthogonal representation for p using an SVD.)"
            p = closestOrthogonal(p)
    
    #p_perp is the orthogonal complement to p, = null(p.T)
    p_perp = nullspace(p.T)
    
    #compute p_perp * p_perp.T * A * inv(p.T * A)
    T = sp.dot(p.T,A)
    try:
        Tinv = LA.inv(T)
    except(LA.LinAlgError):
        Tinv = LA.pinv(T)
        
    X = sp.dot( sp.dot( sp.dot(p_perp,p_perp.T), A), Tinv )
    
    u, s, vh = LA.svd(X, full_matrices=False)
    s[ s < tol ]= 0   #set extremely small values to zero
    theta = sp.diag( sp.arctan(s) )
    
    logA = sp.dot(u, sp.dot( theta,vh))    
    normA = sp.trace( sp.dot(logA.T, logA) )
    
    return logA, normA

def grassmann_frobenius_mean(point_set, ctol=1e-5, max_iter=100):
    '''
    Computes the mean of a set of orthogonal matrices (of same size)
    using iterations to find a matrix that minimizes the frobenius
    norm distance to all the points. The code is structured much like
    the Karcher mean, but without the tangent space round-trip mapping,
    and thus may be used to compare whether the karcher mean returns
    better results (perhaps because it better takes into account the
    geometry of the space).
    '''    
    for M in point_set:
        if not isOrthogonal(M):
            raise ValueError("Non-orthogonal point found in input point set.")
    
    N = len(point_set)
    pole = point_set[0]
    i=0
    #step_shrinkage = float(step)/max_iter
    
    print "Iterating to find frobenius mean of %d points..."%N
    while (i<max_iter):      
        accum = sp.zeros_like(pole)
        ds = []
        
        #compute distance from pole to each point
        for M in point_set:
            ds.append( LA.norm(M-pole) )  #frobenius norm
        ds = sp.array(ds)
        
        #normalize distances to find weights
        ws = list( ds / sp.sum(ds) )
            
        #compute new pole as closest orthogonal matrix to weighted sum
        for M,w in zip(point_set,ws):
            accum = accum + w*M

        prev_pole = pole 
        pole = closestOrthogonal(accum)
        
        i += 1
        delta = LA.norm( pole-prev_pole )
        if i % 10 == 0: print "Iter %d: %3.8f"%(i,delta)
        if delta <= ctol:
            print ""
            print "Converged within tolerance after %d steps."%i 
            break
        
    print ""
    return pole    

def grassmann_karcher_mean( point_set, weights=None, step=1, ctol=1e-5, max_iter=100, verbose=True):
    '''
    Compute the karcher mean of a set of points on
    a grassmann manifold.
    @param point_set: A list of orthogonal matrices of same size, representing
    points on a grassmann manifold, nxk
    @param weights: If None, an unweighted mean is returned. Otherwise weights
    is a list of sample length as point_set, used at each iteration.
    @param step: Generally set to 1. Smaller step sizes may converge slower, but
    with more iterations may yield a slightly better answer.
    @param ctol: Convergence tolerance.
    @param max_iter: The maximum number of iterations, at which point the computation
    is stopped, even if the convergence tolerance has not yet been achieved.
    @return: M the karcher mean.
    '''
    step = float(step)
    
    #check that all points in point_set are orthogonal
    for M in point_set:
        if not isOrthogonal(M):
            raise ValueError("Non-orthogonal point found in input point set.")
    
    N = len(point_set)
    
    #initialize pole
    #pole = point_set[0]
    #closest orthogonal matrix to entry-wise mean
    em = sp.zeros_like( point_set[0])
    for M in point_set:
        em += M
    em = (1.0/len(point_set))*em
    pole = closestOrthogonal(em)
        
    i=0
    
    print "Iterating to find Karcher Mean of %d points..."%N
    while (i<max_iter):        
        accum = sp.zeros_like(pole)
        #ds = []
        logMs = []
        #compute tangent distance from each point to pole
        for M in point_set:
            logM, _d = grassmann_logmap(M, pole, skip_orthog_check=True)
            #ds.append(d)
            logMs.append(logM)
        
        #normalize distances to get weights
        ws = [1.0/N]*N if weights is None else list( sp.array(weights, dtype=float) / sp.sum(weights) )
               
        #compute new pole as (weighted) sum in tangent space,
        # mapped back to grassmann space
        for lM,w in zip(logMs,ws):           
            accum = accum + w*step*lM

        prev_pole = pole
        pole = grassmann_expmap(accum, prev_pole)
        i += 1
                   
        _, delta = grassmann_logmap(prev_pole, pole, skip_orthog_check=True)
        if verbose:
            if i % 10 == 0: print "Iter %d: %3.8f"%(i,delta)
        
        #if isSameGrassmannPoint(pole, pole_old, tol=1e-8):
        if delta <= ctol:
            print ""
            print "Converged within tolerance after %d steps."%i 
            break
        
    print ""
    return pole
    
    
    

if __name__ == '__main__':
    pass