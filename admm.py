from numpy import random,reshape,mean,ones,array,sign,arange,delete,log,tensordot,unique,fill_diagonal,shape,std,\
    zeros,where,linspace,load,diag,eye,argsort,argmin,argmax,hstack,trace,e,intersect1d,clip,log
from numpy.linalg import slogdet
from scipy.linalg import eigvalsh,ldl,det,eigh,inv,pinv,cholesky,norm,svd,qr
from scipy.sparse.linalg import eigsh,svds
import time
from scipy.linalg.lapack import dsyevr

import gc

"""
Authors note;

Most of these algorithms have similar structure and have different experimental purposes.
Only well documented function is the ADMM function, which is the one you should use and 
look at if you want to get an idea about the others. 

Also, gc was used because of initial memory/cpu errors encountered. It may be silenced/removed.

Please e-mail selim.aktas@bilkent.edu.tr for anything about the piece of code here.
"""

def ADMM_SDP(A,b,rho,s,max_iter):
    # only solves the sdp
    # no reweighting
    m,n = shape(A)
    
    a = A.flatten()
    v = eye(n).flatten()
    
    y1 = 0
    y2 = 0
    U1 = zeros([n,n])
    U2 = zeros([n,n])
    
    
    M = hstack((reshape(a,[n**2,1]),reshape(v,[n**2,1])))
    m = inv(0.5 * array([[a.dot(a),a.dot(v)],[a.dot(v),v.dot(v)]]) + eye(2))
    
    X1 = zeros([n,n])
    X2 = zeros([n,n])
    X3 = zeros([n,n])
    
    t = 0
    while t <= max_iter:

        t += 1
        # X1 = rho/(1+rho) * (X2 - U1/rho)

        f = X1.flatten() + U1.flatten()/rho + X3.flatten() + U2.flatten()/rho +\
            b * a - y1 * a/rho + v - y2 * v / rho
        
        X2 = f/2 -1/4 * M.dot(m.dot(M.T.dot(f)))
        X2 = reshape(X2,[n,n])
        
        X1 = X2 - U1/rho
        X1 = sign(X1) * (abs(X1)-1/rho).clip(0)

        X3 = X2 - U2/rho
        d,P = eigh(X3)
        i, = where(d>0)
        X3 = P[:,i].dot(diag(d[i])).dot(P[:,i].T)
        # X3 = (P[:,i]*d[i]).dot(P[:,i].T)
        
        y1 = y1 + rho * (tensordot(A,X1) - b)
        y2 = y2 + rho * (trace(X1) - 1)
        U1 = U1 + rho * (X1 - X2)
        U2 = U2 + rho * (X3 - X2)
        if t % 200 == 0:
            print("iteration",t)
            print("res1",tensordot(A,X2) - b)
            print("res2",trace(X2) - 1)
            print("res3",norm(X1-X2))
            print("res4",norm(X3-X2))
            gc.collect()
    d1,p1 = eigsh(X3)
    v1 = p1[:,[-1]]
    admm_set = argsort(abs(v1[:,0]))[-s:]
    return X1,X2,X3,y1,y2,U1,U2,admm_set

def ADMM_RW(A,b,rho,correction,s,max_iter):
    # vanilla admm algorithm
    m,n = shape(A)
    
    L = ones([n,n])
    
    a = A.flatten()
    v = eye(n).flatten()
    
    y1 = 0
    y2 = 0
    U1 = zeros([n,n])
    U2 = zeros([n,n])
    
    
    M = hstack((reshape(a,[n**2,1]),reshape(v,[n**2,1])))
    m = inv(0.5 * array([[a.dot(a),a.dot(v)],[a.dot(v),v.dot(v)]]) + eye(2))
    
    X1 = zeros([n,n])
    X2 = zeros([n,n])
    X3 = zeros([n,n])
    for rw in range(len(correction)):
        t = 0
        while t <= max_iter:
    
            t += 1
    
            f = X1.flatten() + U1.flatten()/rho + X3.flatten() + U2.flatten()/rho +\
                b * a - y1 * a/rho + v - y2 * v / rho
            
            X2 = f/2 -1/4 * M.dot(m.dot(M.T.dot(f)))
            X2 = reshape(X2,[n,n])
            
            X1 = X2 - U1/rho
            X1 = sign(X1) * (abs(X1)-L/rho).clip(0)
    
            X3 = X2 - U2/rho
            d,P = eigh(X3)
            i, = where(d>0)
            X3 = P[:,i].dot(diag(d[i])).dot(P[:,i].T)
            # X3 = (P[:,i]*d[i]).dot(P[:,i].T)
            
            y1 = y1 + rho * (tensordot(A,X1) - b)
            y2 = y2 + rho * (trace(X1) - 1)
            U1 = U1 + rho * (X1 - X2)
            U2 = U2 + rho * (X3 - X2)
            if t % 200 == 0:
                print("iteration",t)
                print("res1",tensordot(A,X2) - b)
                print("res2",trace(X2) - 1)
                print("res3",norm(X1-X2))
                print("res4",norm(X3-X2))
                gc.collect()

        L = 1/(correction[rw] + abs(X3))
        L = L/L.max()
        
        d2,v2 = eigsh(X3,k=3)
        admm_set = argsort(-1 * abs(v2[:,-1]))[:s]
        se,sv = eigsh(A[admm_set,:][:,admm_set],k=3)
        print(se)
        print(b)
        print(d2)
    # d1,p1 = eigsh(X3)
    # v1 = p1[:,[-1]]
    # admm_set = argsort(abs(v1[:,0]))[-s:]
    return X1,X2,X3,y1,y2,U1,U2,admm_set

def ADMM_SRW(A,b,rho,correction,s,max_iter,n1 = 1):
    # admm with diameter given n1
    # reweighting is normalized for better numerical results
    
    m,n = shape(A)
    
    L = ones([n,n])
    
    a = A.flatten()
    v = eye(n).flatten()
    
    y1 = 0
    y2 = 0
    U1 = zeros([n,n])
    U2 = zeros([n,n])
    
    
    M = hstack((reshape(a,[n**2,1]),reshape(v,[n**2,1])))
    m = inv(0.5 * array([[a.dot(a),a.dot(v)],[a.dot(v),v.dot(v)]]) + eye(2))
    
    X1 = zeros([n,n])
    X2 = zeros([n,n])
    X3 = zeros([n,n])
    for rw in range(len(correction)):
        t = 0
        while t <= max_iter:
    
            t += 1
    
            f = X1.flatten() + U1.flatten()/rho + X3.flatten() + U2.flatten()/rho +\
                b * a * n1 - y1 * a/rho + v * n1 - y2 * v / rho
            
            X2 = f/2 -1/4 * M.dot(m.dot(M.T.dot(f)))
            X2 = reshape(X2,[n,n])
            
            X1 = X2 - U1/rho
            X1 = sign(X1) * (abs(X1)-L/rho).clip(0)
    
            X3 = X2 - U2/rho
            d,P = eigh(X3)
            i, = where(d>0)
            X3 = P[:,i].dot(diag(d[i])).dot(P[:,i].T)
            # X3 = (P[:,i]*d[i]).dot(P[:,i].T)
            
            y1 = y1 + rho * (tensordot(A,X1) - b * n1)
            y2 = y2 + rho * (trace(X1) - 1 * n1)
            U1 = U1 + rho * (X1 - X2)
            U2 = U2 + rho * (X3 - X2)
            if t % 200 == 0:
                print("iteration",t)
                print("res1",tensordot(A,X2) - b * n1)
                print("res2",trace(X2) - 1 * n1)
                print("res3",norm(X1-X2))
                print("res4",norm(X3-X2))
                gc.collect()

        RW = abs(X3)
        RW = RW/RW.max()
        L = 1/(correction[rw] + RW)
        
        L = L/L.max()
        
        d2,v2 = eigsh(X3,k=3)
        admm_set = argsort(-1 * abs(v2[:,-1]))[:s]
        se,sv = eigsh(A[admm_set,:][:,admm_set],k=3)
        print(se)
        print(b)
        print(d2)
    # d1,p1 = eigsh(X3)
    # v1 = p1[:,[-1]]
    # admm_set = argsort(abs(v1[:,0]))[-s:]
    return X1,X2,X3,y1,y2,U1,U2,admm_set

def ADMM_base(A,b,rho,correction,sparsity,max_iter,n1 = 1,eps_abs = 1e-3,eps_rel = 1e-3,eps_rw = 1e-3):
    # admm with diameter given n1
    # reweighting is normalized for better numerical results
    # primal-dual stopping criteria is used
    # penalty parameter is adaptively changed depending on
    # primal and dual residual
    
    m,n = shape(A)
    
    L = ones([n,n])
    
    a = A.flatten()
    v = eye(n).flatten()
    
    y1 = 0
    y2 = 0
    U1 = zeros([n,n])
    U2 = zeros([n,n])
    
    r = 0
    s = 0
    mu = 100
    
    eps_pri = n * eps_abs + eps_rel * b
            
    M = hstack((reshape(a,[n**2,1]),reshape(v,[n**2,1])))
    m = inv(0.5 * array([[a.dot(a),a.dot(v)],[a.dot(v),v.dot(v)]]) + eye(2))
    
    X1 = zeros([n,n])
    X2 = zeros([n,n])
    X3 = zeros([n,n])
    for rw in range(len(correction)):
        t = 0
        while t <= max_iter:
    
            t += 1
            
            f = X1.flatten() + U1.flatten()/rho + X3.flatten() + U2.flatten()/rho +\
                b * a * n1 - y1 * a/rho + v * n1 - y2 * v / rho
            
            X2 = f/2 -1/4 * M.dot(m.dot(M.T.dot(f)))
            X2 = reshape(X2,[n,n])
            
            X1_old = X1.copy()
            X1 = X2 - U1/rho
            X1 = sign(X1) * (abs(X1)-L/rho).clip(0)
            
            X3_old = X3.copy()
            X3 = X2 - U2/rho
            d,P = eigh(X3)
            i, = where(d>0)
            # X3 = P[:,i].dot(diag(d[i])).dot(P[:,i].T)
            X3 = (P[:,i]*d[i]).dot(P[:,i].T)
            
            y1 = y1 + rho * (tensordot(A,X1) - b * n1)
            y2 = y2 + rho * (trace(X1) - 1 * n1)
            U1 = U1 + rho * (X1 - X2)
            U2 = U2 + rho * (X3 - X2)
            
            r = norm(X3-X2) ** 2 + norm(X1-X2) ** 2 + (trace(X2) - 1 * n1 ) ** 2 + (tensordot(A,X2) - b * n1 ) ** 2
            s = rho ** 2 * (norm(X1-X1_old) ** 2 + norm(X3-X3_old) ** 2)
            
            if t % 200 == 0:
                print("iteration",t)
                print("primal residual", r)
                print("dual residual", s)
                gc.collect()
                
            eps_dual = 1.4 * n * eps_abs + eps_rel * norm(U1)
            if r < eps_pri ** 2 and s < eps_dual ** 2: 
                break
            elif r > mu * s:
                rho = 2 * rho
            elif s > mu * r:
                rho = rho/2
        
        L_old = L.copy()
        
        RW = abs(X3)
        RW = RW/RW.max()
        L = 1/(correction[rw] + RW)
        
        L = L/L.max()
        
        d2,v2 = eigsh(X3,k=3)
        admm_set = argsort(-1 * abs(v2[:,-1]))[:sparsity]
        se,sv = eigsh(A[admm_set,:][:,admm_set],k=3)
        print(se)
        print(b)
        print(d2)
        
        print("reweigtred")
        print(norm(L-L_old))
        if norm(L-L_old) < eps_rw:
            break
    # d2,v2 = eigsh(X3,k=3)
    # admm_set = argsort(-1 * abs(v2[:,-1]))[:s]
    # se,sv = eigsh(A[admm_set,:][:,admm_set],k=3)
    return X1,X2,X3,y1,y2,U1,U2,admm_set

def ADMM(A,b,rho,correction,sparsity,max_iter,n1 = 1,eps_abs = 1e-5,eps_rel = 1e-5,eps_rw = 1e-5):
    # admm with diameter given n1
    # reweighting is normalized for better numerical results
    # primal-dual stopping criteria is used
    # penalty parameter is adaptively changed depending on
    # primal and dual residual
    """
    Applies PCA Sparsified to covariance matrix A with variance constraint b using
    Alternating Direction Method of Multipliers (ADMM) for solving the subproblem

    Returns the best sparsity pattern computed and corresponding sparse eigenvalue as
    well as primal and dual variables computed at last iteration

    Parameters
    ----------
    A :(n,n) array
        Covariance matrix of the data
    b : float
        Variance constraint
    rho : float
        Initial augmented Lagrangian parameter
    correction : list of floats or array of floats
        Stability parameter to use in each iteration
    sparsity : integer
        Target sparsity level
    max_iter : integer
        Maximum number of ADMM iterations for the subproblem
    n1 : integer, optional
        Trace scale parameter, may improve the result
    eps_abs : float, optional
        Absolute error precision desired for subproblem. Low accuracy may result in early stop
        instead of doing max_iter number of iterations
    eps_rel : float, optional
        Relative error precision desired for subproblem. Low accuracy may result in early stop
        instead of doing max_iter number of iterations
    eps_rw : float, optional
        Absolute error precision desired for master problem. Low accuracy may result in early stop
        instead of doing max_iter number of iterations. Compared against changes in weights.

    Returns
    -------
    X1 : (n, n) array
        Primal variable X1. Responsible for objective.
    X2 : (n, n) array
        Primal variable X2. Responsible for consensus of X variables and constraint feasbility.
    X3 : (n, n) array
        Primal variable X3. Responsible for positive semi-definiteness of the variable.
    y1 : float
        Dual variable corresponding to variance constraint
    y2 : float
        Dual variable corresponding to trace constraint
    U1 : (n, n) array
        Dual variable corresponding to equality of X1 and X2
    U2 : (n, n) array
        Dual variable corresponding to equality of X3 and X2
    best_set : list of size sparsity
        Best sparsity pattern discovered throughout reweighted optimization
    best_val : float
        Best sparse eigenvalue discovered throughout reweighted optimization
    """
    # shape of matrix A
    m,n = shape(A)
    
    # keep track of best support found throughout reweighted iterations
    best_set = []
    best_val = 0
    
    # initial weights for reweighted optimization
    L = ones([n,n])
    
    # flatten version of covariance A and identity matrix I, needed for X2 update
    a = A.flatten()
    v = eye(n).flatten()
    
    # initialize primal & dual residual
    # mu controls the multiplicative factor of which primal and dual residual
    # should be within each other, its actually 10, residuals are kept as squares
    r = 0
    s = 0
    mu = 100
    
    # primal residual threshold
    eps_pri = n * eps_abs + eps_rel * b
    
    # Pre computing matrices required for X2 update
    M = hstack((reshape(a,[n**2,1]),reshape(v,[n**2,1])))
    m = inv(0.5 * array([[a.dot(a),a.dot(v)],[a.dot(v),v.dot(v)]]) + eye(2))
    
    # initialize dual variables
    y1 = 0
    y2 = 0
    U1 = zeros([n,n])
    U2 = zeros([n,n])
    
    # initialize primal variables
    X1 = zeros([n,n])
    X2 = zeros([n,n])
    X3 = zeros([n,n])
    
    # "for" loop for reweighted optimization
    for rw in range(len(correction)):
        # while loop for subproblem, exits after maximum number of iterations or 
        # if primal & dual error criteria is satisfied
        t = 0
        while t <= max_iter:
            t += 1
            
            # X2 update
            # Uses matrix inversion lemma
            f = X1.flatten() + U1.flatten()/rho + X3.flatten() + U2.flatten()/rho +\
                b * a * n1 - y1 * a/rho + v * n1 - y2 * v / rho
            
            X2 = f/2 -1/4 * M.dot(m.dot(M.T.dot(f)))
            X2 = reshape(X2,[n,n])
            
            # X1 update
            # Soft thresholding
            X1_old = X1.copy()
            X1 = X2 - U1/rho
            X1 = sign(X1) * (abs(X1)-L/rho).clip(0)
            
            # X3 Update
            # Projection to positive semi-definite cone
            # i.e discard negative eigenvalues
            X3_old = X3.copy()
            X3 = X2 - U2/rho
            d,P = eigh(X3)
            i, = where(d>0)
            X3 = (P[:,i]*d[i]).dot(P[:,i].T)
            
            # Dual updates
            y1 = y1 + rho * (tensordot(A,X1) - b * n1)
            y2 = y2 + rho * (trace(X1) - 1 * n1)
            U1 = U1 + rho * (X1 - X2)
            U2 = U2 + rho * (X3 - X2)
            
            # primal and dual residuals in squares
            r = norm(X3-X2) ** 2 + norm(X1-X2) ** 2 + (trace(X2) - 1 * n1 ) ** 2 + (tensordot(A,X2) - b * n1 ) ** 2
            s = rho ** 2 * (norm(X1-X1_old) ** 2 + norm(X3-X3_old) ** 2)
            
            # printing some information on subproblem
            if t % 400 == 0:
                print("r",rw,"target sparsity",sparsity)
                print("iteration",t)
                print("primal residual", r)
                print("dual residual", s)
                gc.collect()
            
            # Primal and dual residual stop critertion and 
            # augmented Lagrangian parameter update
            eps_dual = 1.4 * n * eps_abs + eps_rel * norm(U1)
            if r < eps_pri ** 2 and s < eps_dual ** 2: 
                break
            elif r > mu * s:
                rho = 2 * rho
            elif s > mu * r:
                rho = rho/2
        
        # Computing new reweighted optimization weights and 
        # keeping track of amount of change
        # Scaling generally helps the algorithm
        
        L_old = L.copy()
        
        if trace(X3) > 0: 
            RW = abs(X3)
            RW = RW/RW.max()
            L = 1/(correction[rw] + RW)
        elif norm(X2) > 0:
            RW = abs(X2)
            RW = RW/RW.max()
            L = 1/(correction[rw] + RW)
        else:
            RW = abs(X1)
            RW = RW/RW.max()
            L = 1/(correction[rw] + RW)
            
        L = L/L.max()
        
        # Extracting the largest 3 eigenvalues and vectors for the solution
        # Extracting 3 instead of 1 because if the largest eigenvalue is not
        # close 1, distribution of eigenvalues generally give information on 
        # convergence and quality of the solution obtained for sparse PCA
        vals,v2,kmany,isuppz,info = dsyevr(X3,compute_v = 1,range = "I",il = n-2,iu = n,abstol = 1e-12)
        admm_set = argsort(-1 * abs(v2[:,-1]))[:sparsity]
        se,sv,kmany,isuppz,info = dsyevr(A[admm_set,:][:,admm_set],compute_v = 1,range = "I",il = sparsity-2,iu = sparsity,abstol = 1e-12)
        
        # old eigenvector extraction using ARPACK, LAPACK implementation above
        # was more stable
        # d2,v2 = eigsh(X3,k=3)
        # se,sv = eigsh(A[admm_set,:][:,admm_set],k=3)
        
        # update the best sparse PCA found
        if se[2] > best_val:
            best_val = se[2]
            best_set = admm_set

        # printing some information on master problem
        print("max 3 eig of subset",se[:3])
        print("best heuristic",b)
        print("max 3 eig of X",vals[:3])
        
        # stop the reweighted optimization if weights converged
        if norm(L-L_old) < eps_rw:
            break
    return X1,X2,X3,y1,y2,U1,U2,best_set,best_val

def ADMM_mk2(A,b,rho,correction,sparsity,max_iter,n1 = 1,eps_abs = 1e-5,eps_rel = 1e-5,eps_rw = 1e-5):
    # admm with diameter given n1
    # reweighting is normalized for better numerical results
    # primal-dual stopping criteria is used
    # penalty parameter is adaptively changed depending on
    # primal and dual residual
    #
    # This is an experimental version of original ADMM where I would test
    # small changes and their effects on the results, stability/convergence/speed and so on.
    m,n = shape(A)
    
    admm_vals = []
    best_set = []
    best_val = 0
    
    L = ones([n,n])
    
    a = A.flatten()
    v = eye(n).flatten()
    
    y1 = 0
    y2 = 0
    U1 = zeros([n,n])
    U2 = zeros([n,n])
    
    r = 0
    s = 0
    mu = 100
    
    eps_pri = n * eps_abs + eps_rel * b
            
    M = hstack((reshape(a,[n**2,1]),reshape(v,[n**2,1])))
    m = inv(0.5 * array([[a.dot(a),a.dot(v)],[a.dot(v),v.dot(v)]]) + eye(2))
    
    X1 = zeros([n,n])
    X2 = zeros([n,n])
    X3 = zeros([n,n])
    for rw in range(len(correction)):
        t = 0
        while t <= max_iter:
    
            t += 1
            
            f = X1.flatten() + U1.flatten()/rho + X3.flatten() + U2.flatten()/rho +\
                b * a * n1 - y1 * a/rho + v * n1 - y2 * v / rho
            
            X2 = f/2 -1/4 * M.dot(m.dot(M.T.dot(f)))
            X2 = reshape(X2,[n,n])
            
            X1_old = X1.copy()
            X1 = X2 - U1/rho
            X1 = sign(X1) * (abs(X1)-L/rho).clip(0)
            
            X3_old = X3.copy()
            X3 = X2 - U2/rho
            d,P = eigh(X3)
            i, = where(d>0)
            # X3 = P[:,i].dot(diag(d[i])).dot(P[:,i].T)
            X3 = (P[:,i]*d[i]).dot(P[:,i].T)
            
            y1 = y1 + rho * (tensordot(A,X1) - b * n1)
            y2 = y2 + rho * (trace(X1) - 1 * n1)
            U1 = U1 + rho * (X1 - X2)
            U2 = U2 + rho * (X3 - X2)
            
            r = norm(X3-X2) ** 2 + norm(X1-X2) ** 2 + (trace(X2) - 1 * n1 ) ** 2 + (tensordot(A,X2) - b * n1 ) ** 2
            s = rho ** 2 * (norm(X1-X1_old) ** 2 + norm(X3-X3_old) ** 2)
            
            if t % 400 == 0:
                print("r",rw,"target sparsity",sparsity)
                print("iteration",t)
                print("primal residual", r)
                print("dual residual", s)
                gc.collect()
                
            eps_dual = 1.4 * n * eps_abs + eps_rel * norm(U1)
            if r < eps_pri ** 2 and s < eps_dual ** 2: 
                break
            elif r > mu * s:
                rho = 2 * rho
            elif s > mu * r:
                rho = rho/2
        
        L_old = L.copy()
        
        # RW = abs(X3)
        # RW = RW/RW.max()
        # L = 1/(correction[rw] + RW)
        
        if trace(X3) > 0: 
            RW = abs(X3)
            # RW = RW/RW.max()
            RW = RW/RW.max() ** 0.5
            L = 1/(correction[rw] + RW)
        elif norm(X2) > 0:
            RW = abs(X2)
            RW = RW/RW.max()
            L = 1/(correction[rw] + RW)
        else:
            RW = abs(X1)
            RW = RW/RW.max()
            L = 1/(correction[rw] + RW)
            
        L = L/L.max()
        
        vals,v2,kmany,isuppz,info = dsyevr(X3,compute_v = 1,range = "I",il = n-2,iu = n,abstol = 1e-12)
        admm_set = argsort(-1 * abs(v2[:,-1]))[:sparsity]
        se,sv,kmany,isuppz,info = dsyevr(A[admm_set,:][:,admm_set],compute_v = 1,range = "I",il = sparsity-2,iu = sparsity,abstol = 1e-12)
        # d2,v2 = eigsh(X3,k=3)
        # se,sv = eigsh(A[admm_set,:][:,admm_set],k=3)
        
        if se[2] > best_val:
            best_val = se[2]
            best_set = admm_set
        admm_vals += [se[2]]
        print("max 3 eig of subset",se[:3])
        print("best heuristic",b)
        print("max 3 eig of X",vals[:3])
        
        # print("reweigtred")
        # print(norm(L-L_old))
        # surr = log(correction[rw] + abs(X3)).sum()
        # print("surrogate objective",surr)
        
        if norm(L-L_old) < eps_rw:
            break
    # d2,v2 = eigsh(X3,k=3)
    # admm_set = argsort(-1 * abs(v2[:,-1]))[:s]
    # se,sv = eigsh(A[admm_set,:][:,admm_set],k=3)
    return X1,X2,X3,y1,y2,U1,U2,best_set,best_val,admm_vals

def ADMM_fantope(A,b,rho,correction,sparsity,max_iter,n1 = 1,rank = 1,eps_abs = 1e-3,eps_rel = 1e-3,eps_rw = 1e-3):
    # admm with diameter given n1
    # reweighting is normalized for better numerical results
    # primal-dual stopping criteria is used
    # penalty parameter is adaptively changed depending on
    # primal and dual residual
    
    m,n = shape(A)
    
    best_set = []
    best_val = 0
    
    L = ones([n,n])
    
    a = A.flatten()
    v = eye(n).flatten()
    
    y1 = 0
    y2 = 0
    U1 = zeros([n,n])
    U2 = zeros([n,n])
    
    r = 0
    s = 0
    mu = 100
    
    eps_pri = n * eps_abs + eps_rel * b
            
    M = hstack((reshape(a,[n**2,1]),reshape(v,[n**2,1])))
    m = inv(0.5 * array([[a.dot(a),a.dot(v)],[a.dot(v),v.dot(v)]]) + eye(2))
    
    X1 = zeros([n,n])
    X2 = zeros([n,n])
    X3 = zeros([n,n])
    for rw in range(len(correction)):
        t = 0
        while t <= max_iter:
    
            t += 1
            
            f = X1.flatten() + U1.flatten()/rho + X3.flatten() + U2.flatten()/rho +\
                b * a * n1 - y1 * a/rho + v * n1 * rank - y2 * v / rho
            
            X2 = f/2 -1/4 * M.dot(m.dot(M.T.dot(f)))
            X2 = reshape(X2,[n,n])
            
            X1_old = X1.copy()
            X1 = X2 - U1/rho
            X1 = sign(X1) * (abs(X1)-L/rho).clip(0)
            
            X3_old = X3.copy()
            X3 = X2 - U2/rho
            d,P = eigh(X3)
            i1, = where(d>0)
            i2, = where(d<=n1)
            i = intersect1d(i1,i2)
            # X3 = P[:,i].dot(diag(d[i])).dot(P[:,i].T)
            X3 = (P[:,i]*d[i]).dot(P[:,i].T)
            
            y1 = y1 + rho * (tensordot(A,X1) - b * n1)
            y2 = y2 + rho * (trace(X1) - 1 * n1 * rank)
            U1 = U1 + rho * (X1 - X2)
            U2 = U2 + rho * (X3 - X2)
            
            r = norm(X3-X2) ** 2 + norm(X1-X2) ** 2 + (trace(X2) - 1 * n1 * rank ) ** 2 + (tensordot(A,X2) - b * n1 ) ** 2
            s = rho ** 2 * (norm(X1-X1_old) ** 2 + norm(X3-X3_old) ** 2)
            
            if t % 200 == 0:
                print("iteration",t)
                print("primal residual", r)
                print("dual residual", s)
                gc.collect()
                
            eps_dual = 1.4 * n * eps_abs + eps_rel * norm(U1)
            if r < eps_pri ** 2 and s < eps_dual ** 2: 
                break
            elif r > mu * s:
                rho = 2 * rho
            elif s > mu * r:
                rho = rho/2
        
        L_old = L.copy()
        
        RW = abs(X1)
        RW = RW/RW.max()
        L = 1/(correction[rw] + RW)
        
        L = L/L.max()
        
        d2,v2 = eigsh(X3,k=rank)
        admm_set = argsort(-1 * abs(v2[:,-1]))[:sparsity]
        # admm_sets = argsort(-1 * abs(v2),axis = 0)[:sparsity,:]
        # extratcing projections is not trivial
        se,sv = eigsh(A[admm_set,:][:,admm_set],k=3)
        
        if se[-1] > best_val:
            best_val = se[-1]
            best_set = admm_set
        print(se)
        print(b)
        print(d2)
        
        print("reweigtred")
        print(norm(L-L_old))
        if norm(L-L_old) < eps_rw:
            break
    # d2,v2 = eigsh(X3,k=3)
    # admm_set = argsort(-1 * abs(v2[:,-1]))[:s]
    # se,sv = eigsh(A[admm_set,:][:,admm_set],k=3)
    return X1,X2,X3,y1,y2,U1,U2,best_set,best_val

#%%

# m = 100
# n = 100
# s = 40

# B = random.normal(4,10,[m,n])
# B = B - B.mean(axis = 0)
# B = B/B.std(axis = 0)
# A = B.T.dot(B)
# # A = A/norm(A)

# omega = SPCA(B,s)
# em_set,em_val = omega.EM()
# # A = A / norm(A) * n ** 0.5

# d,P = eigh(A)

# correction = [1e-3] * 4
# correction = [1e-3,1e-3,1e-4,1e-4,1e-4,1e-5,1e-5,1e-5]

# max_iter_admm = 600


#%%

# X1,X2,X3,y1,y2,U1,U2,admm_set2 = \
# ADMM(A,em_val,100,correction,s,max_iter_admm,n1 = 1,eps_abs = 1e-4,eps_rel = 1e-4,eps_rw = 1e-3)

# admm_set = argsort(abs(v1[:,0]))[-s:]
# admm_val,admm_vec = omega.eigen_pair(admm_set)
# print("------------------")
# print("em",em_val)
# print("admm",admm_val[0])

#%%

# t0 = time.process_time()
# X1,X2,X3,y1,y2,U1,U2,admm_set = ADMM_mk3(A2,best_val,400,correction,s,max_iter_admm,n1 = 1)
# t1 = time.process_time()
# admm_val1 = omega.eigen_upperbound(admm_set)
# print("admm1",admm_val1)

# print("--------------cgal frobenius---------------------")
# t8 = time.process_time()
# X,y,d2,v2,cgal_set = CGAL_fro(A2,best_val,correction,s,0.005,max_iter_cgal,n2 = -10)
# t9 = time.process_time()
# cgal_val2 = omega.eigen_upperbound(cgal_set)


# print("--------------cgal weighted frobenius---------------------")
# t8 = time.process_time()
# X,y,d2,v2,cgal_set = CGAL_wfro(A2,best_val,correction,s,0.005,max_iter_cgal,n2 = -10)
# t9 = time.process_time()
# cgal_val2 = omega.eigen_upperbound(cgal_set)