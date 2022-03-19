from numpy import random,reshape,mean,ones,array,sign,arange,delete,log,tensordot,unique,fill_diagonal,shape,std,\
    zeros,where,linspace,load,diag,argsort,sqrt,eye,median
from scipy.linalg import eigvalsh,ldl,det,eigh,inv,pinv,cholesky,norm
from scipy.sparse.linalg import eigsh,ArpackNoConvergence
from scipy.linalg.lapack import dsyevr

"""
Authors note;

Most of these algorithms have similar structure and have different experimental purposes.
Only well documented function is the CGAL function, which is the one you should use and 
look at if you want to get an idea about the others. 

Please e-mail selim.aktas@bilkent.edu.tr for anything about the piece of code here.
"""

def CGAL_base(A,best_val,correction,s,beta0,max_iter):
    # cgal mk1
    # Weighted Frobenius norm smoothing
    # base cgal method; do not use it
    m,n = shape(A)
    L = ones([n,n])
    
    X = A/norm(A)
    y = 0
    
    A3 = A/norm(A)
    b = best_val/norm(A)
    for rw in range(len(correction)):
        
        lamda0 = 1
        
        r = tensordot(A3,X) - b
        
        t = 1
        # change = 1
        while t <= max_iter:
            # print("step",t,"change",change)
            t += 1
            eta = 2/t
            lamda = lamda0 * t ** 0.5
            beta = beta0/t ** 0.5
            
            w = X.copy()
    
            i,j = where(X > beta)
            k,l = where(X < -beta)
            w[i,j] = beta
            w[k,l] = -beta
            
            w = w / beta
            
            w = L * w
            
            v = y * A3 + lamda * A3 * r
            
            vals,vec,kmany,isuppz,info = dsyevr(w + v,compute_v = 1,range = "I",il = 1,iu = 1,abstol = 1e-9)
            # val,vec = eigsh(w + v -10 * eye(n),k = 1,tol = 1e-8,which = "SA")

            X = (1 - eta) * X + eta * vec.dot(vec.T)
        
            r = tensordot(A3,X) - b
            
            # bounded travel conditions
            # lipschitz = lamda + 1/beta * operator_norm ** 2
            # sigma = max(lamda,1/2*eta ** 2 * lipschitz * 2/(r ** 2))
            
            sigma = lamda

            y = y + sigma * r
            if t % 1000 == 0:
                print("r",rw,"iteration",t,"constraint diff",r)
                # d2,v2 = eigsh(X,k=3)
                # print("d2\n",d2)
        d2,v2 = eigsh(X,k=3)
        cgal2 = argsort(-1 * abs(v2[:,-1]))[:s]
        se,sv = eigsh(A[cgal2,:][:,cgal2],k=3)
        print(se)
        print(best_val)
        print(d2)
        
        L_old = L.copy()
        
        L = 1/(correction[rw] + abs(X))

        L = L/L.max()
        # operator_norm = L.max()
    return X,y,d2,v2,cgal2


def CGAL_wfro_constant(A,best_val,correction,s,beta0,max_iter,n2 = 0):
    # cgal constant
    # Weighted Frobenius norm smoothing 
    # eigenshifts for kyrlov method
    # constant smoothing
    m,n = shape(A)
    L = ones([n,n])
    
    X = A/norm(A)
    
    y = 0
    
    A3 = A/norm(A)
    b = best_val/norm(A)
    for rw in range(len(correction)):
        
        lamda0 = 1
        # beta0 = 1/n * 0.1
        # beta0 = 0.1
        
        r = tensordot(A3,X) - b

        t = 1
        # change = 1
        while t <= max_iter:
            # print("step",t,"change",change)
            t += 1
            eta = 2/t
            lamda = lamda0 * t ** 0.5
            # lamda = lamda0 * 1
            # beta = beta0/t ** 0.5
            beta = beta0 * 1
            
            w = X.copy()
    
            i,j = where(X > beta)
            k,l = where(X < -beta)
            w[i,j] = beta
            w[k,l] = -beta
            
            w = w / beta
            # w = w/1
            
            w = L * w
            
            v = y * A3 + lamda * A3 * r
            
            vals,vec,kmany,isuppz,info = dsyevr(w + v,compute_v = 1,range = "I",il = 1,iu = 1,abstol = 1e-9)
            # val,vec = eigsh(w + v + n2 * eye(n),k = 1,tol = 1e-8,which = "SA")

            X = (1 - eta) * X + eta * vec.dot(vec.T)
        
            r = tensordot(A3,X) - b
            
            # bounded travel conditions
            # lipschitz = lamda + 1/beta
            # sigma = max(lamda0,1/2*eta ** 2 * lipschitz * 2/(r ** 2))
            sigma = lamda

            y = y + sigma * r
            if t % 1000 == 0:
                print("r",rw,"iteration",t,"constraint diff",r)
                # d2,v2 = eigsh(X,k=3)
                # print("d2\n",d2)
        d2,v2 = eigsh(X,k=3)
        cgal2 = argsort(-1 * abs(v2[:,-1]))[:s]
        se,sv = eigsh(A[cgal2,:][:,cgal2],k=3)
        print(se)
        print(best_val)
        print(d2)
        
        L_old = L.copy()
        
        RW = abs(X)
        RW = RW/RW.max()
        L = 1/(correction[rw] + RW)

        L = L/L.max()
    return X,y,d2,v2,cgal2

def CGAL_fro_constant(A,best_val,correction,s,beta0,max_iter,n2 = 0):
    # cgal constant2
    # Frobenius norm smoothing 
    # constant smoothing
    m,n = shape(A)
    L = ones([n,n])
    
    X = A/norm(A)
    
    y = 0
    
    A3 = A/norm(A)
    b = best_val/norm(A)
    for rw in range(len(correction)):
        
        lamda0 = 1
        
        r = tensordot(A3,X) - b

        t = 1
        # change = 1
        while t <= max_iter:
            # print("step",t,"change",change)
            t += 1
            eta = 2/t
            lamda = lamda0 * t ** 0.5
            # lamda = lamda0 * 1
            # beta = beta0/t ** 0.5
            beta = beta0 * 1
            
            w = L * X
    
            i,j = where(w > beta)
            k,l = where(w < -beta)
            w[i,j] = beta
            w[k,l] = -beta
            
            w = w / beta
            # w = w/1
            
            w = L * w
            
            v = y * A3 + lamda * A3 * r
            
            vals,vec,kmany,isuppz,info = dsyevr(w + v,compute_v = 1,range = "I",il = 1,iu = 1,abstol = 1e-9)
            # val,vec = eigsh(w + v + n2 * eye(n),k = 1,tol = 1e-8,which = "SA")
            
            X = (1 - eta) * X + eta * vec.dot(vec.T)
        
            r = tensordot(A3,X) - b

            # bounded travel conditions
            # lipschitz = lamda + 1/beta
            # sigma = max(lamda0,1/2*eta ** 2 * lipschitz * 2/(r ** 2))
            sigma = lamda

            y = y + sigma * r
            if t % 1000 == 0:
                print("r",rw,"iteration",t,"constraint diff",r)
                # d2,v2 = eigsh(X,k=3)
                # print("d2\n",d2)
        d2,v2 = eigsh(X,k=3)
        cgal2 = argsort(-1 * abs(v2[:,-1]))[:s]
        se,sv = eigsh(A[cgal2,:][:,cgal2],k=3)
        print(se)
        print(best_val)
        print(d2)

        L_old = L.copy()
        
        RW = abs(X)
        RW = RW/RW.max()
        L = 1/(correction[rw] + RW)

        L = L/L.max()
    return X,y,d2,v2,cgal2


def CGAL_wfro(A,best_val,correction,s,beta0,max_iter,n2 = 0,eps_rw = 1e-3):
    # cgal mk2
    # Weighted Frobenius norm smoothing 
    # eigenshifts for kyrlov method
    m,n = shape(A)
    L = ones([n,n])
    
    cgal_set = []
    cgal_val = 0
    
    # X = A/norm(A)
    X = zeros([n,n])
    
    y = 0
    
    A3 = A/norm(A)
    b = best_val/norm(A)
    for rw in range(len(correction)):
        
        lamda0 = 1
        
        r = tensordot(A3,X) - b

        t = 1
        while t <= max_iter:
            # print("step",t,"change",change)
            t += 1
            eta = 2/t
            lamda = lamda0 * t ** 0.5
            beta = beta0/t ** 0.5
            
            w = X.copy()
    
            i,j = where(X > beta)
            k,l = where(X < -beta)
            w[i,j] = beta
            w[k,l] = -beta
            
            w = w / beta
    
            w = L * w
            
            v = y * A3 + lamda * A3 * r
            
            vals,vec,kmany,isuppz,info = dsyevr(w + v,compute_v = 1,range = "I",il = 1,iu = 1,abstol = 1e-9)
            # val,vec = eigsh(w + v + n2 * eye(n),k = 1,tol = 1e-8,which = "SA")

            X = (1 - eta) * X + eta * vec.dot(vec.T)
        
            r = tensordot(A3,X) - b
            
            # bounded travel conditions
            # lipschitz = lamda + 1/beta * operator_norm ** 2
            # sigma = max(lamda,1/2*eta ** 2 * lipschitz * 2/(r ** 2))
            
            sigma = lamda

            y = y + sigma * r
            if t % 1000 == 0:
                print("r",rw,"iteration",t,"constraint diff",r)
                # d2,v2 = eigsh(X,k=3)
                # print("d2\n",d2)
        d2,v2 = eigsh(X,k=3)
        new_set = argsort(-1 * abs(v2[:,-1]))[:s]
        se,sv = eigsh(A[new_set,:][:,new_set],k=3)
        if se[-1] > cgal_val:
            cgal_val = se[-1]
            cgal_set = new_set
        print(se)
        print(best_val)
        print(d2)
        
        L_old = L.copy()
        
        RW = abs(X)
        RW = RW/RW.max()
        L = 1/(correction[rw] + RW)

        L = L/L.max()
        
        print("reweigtred")
        print(norm(L-L_old))
        if norm(L-L_old) < eps_rw:
            break
    return X,y,d2,v2,cgal_set,cgal_val


def CGAL_fro(A,best_val,correction,s,beta0,max_iter,n2 = 0,eps_rw = 1e-3):
    # cgal mk3
    # Frobenius norm smoothing 
    # eigenshifts for kyrlov method
    m,n = shape(A)
    L = ones([n,n])
    
    cgal_set = []
    cgal_val = 0
    
    # X = A/norm(A)
    X = zeros([n,n])
    
    y = 0
    
    A3 = A/norm(A)
    b = best_val/norm(A)
    for rw in range(len(correction)):
        
        lamda0 = 1
        # beta0 = 1/n * 0.1
        # beta0 = 0.1
        
        r = tensordot(A3,X) - b

        t = 1
        while t <= max_iter:
            # print("step",t,"change",change)
            t += 1
            eta = 2/t
            lamda = lamda0 * t ** 0.5
            beta = beta0/t ** 0.5
            
            w = L * X
    
            i,j = where(w > beta)
            k,l = where(w < -beta)
            w[i,j] = beta
            w[k,l] = -beta
            
            w = w / beta
            
            w = L * w
            
            v = y * A3 + lamda * A3 * r
            
            vals,vec,kmany,isuppz,info = dsyevr(w + v,compute_v = 1,range = "I",il = 1,iu = 1,abstol = 1e-9)
            # val,vec = eigsh(w + v + n2 * eye(n),k = 1,tol = 1e-8,which = "SA")

            X = (1 - eta) * X + eta * vec.dot(vec.T)
        
            r = tensordot(A3,X) - b
            
            # bounded travel conditions
            # lipschitz = lamda + 1/beta * operator_norm ** 2
            # sigma = max(lamda,1/2*eta ** 2 * lipschitz * 2/(r ** 2))
            
            sigma = lamda

            y = y + sigma * r
            if t % 1000 == 0:
                print("r",rw,"iteration",t,"constraint diff",r)
                # d2,v2 = eigsh(X,k=3)
                # print("d2\n",d2)
        d2,v2 = eigsh(X,k=3)
        new_set = argsort(-1 * abs(v2[:,-1]))[:s]
        se,sv = eigsh(A[new_set,:][:,new_set],k=3)
        if se[-1] > cgal_val:
            cgal_val = se[-1]
            cgal_set = new_set
        print(se)
        print(best_val)
        print(d2)
        
        L_old = L.copy()
        
        RW = abs(X)
        RW = RW/RW.max()
        L = 1/(correction[rw] + RW)

        L = L/L.max()
        
        print("reweigtred")
        print(norm(L-L_old))
        if norm(L-L_old) < eps_rw:
            break
    return X,y,d2,v2,cgal_set,cgal_val


# Mathematically correct algorithms

def CGAL_fro_dual(A,best_val,correction,s,beta0,max_iter,n2 = 0,eps_rw = 1e-3):
    # cgal mk0
    # Frobenius norm smoothing 
    # eigenshifts for kyrlov method
    # cgal dual update controlled
    m,n = shape(A)
    
    cgal_set = []
    cgal_val = 0
    
    L = ones([n,n])
    
    # X = A/norm(A)
    X = zeros([n,n])
    
    y = 0
    
    A3 = A/norm(A)
    b = best_val/norm(A)
    for rw in range(len(correction)):
        
        lamda0 = 1
        
        r = tensordot(A3,X) - b

        t = 1
        # change = 1
        while t <= max_iter:
            # print("step",t,"change",change)
            t += 1
            eta = 2/t
            lamda = lamda0 * t ** 0.5
            beta = beta0/t ** 0.5
            
            w = L * X
    
            i,j = where(w > beta)
            k,l = where(w < -beta)
            w[i,j] = beta
            w[k,l] = -beta
            
            w = w / beta
            
            w = L * w
            
            lipschitz = lamda0 * (t+1) ** 0.5 + 1/(beta0/(t+1) ** 0.5 * 1)
            
            v = y * A3 + lamda * A3 * r
            
            vals,vec,kmany,isuppz,info = dsyevr(w + v,compute_v = 1,range = "I",il = 1,iu = 1,abstol = 1e-9)
            # val,vec = eigsh(w + v + n2 * eye(n),k = 1,tol = 1e-8,which = "SA")

            X = (1 - eta) * X + eta * vec.dot(vec.T)
        
            r = tensordot(A3,X) - b
            
            # bounded travel conditions
            sigma = min(lamda0 * 1,1/2*eta ** 2 * lipschitz * 2/(r ** 2))
            # sigma = min(lamda0 * 1,1/2*eta ** 2 * lipschitz * 2/(r ** 2))
            
            y = y + sigma * r
            dr = eta * norm(vec.dot(vec.T) - X)
            if t % 500 == 0:
                print("r",rw)
                print("iteration",t)
                print("primal residual", r)
                print("dual residual", dr)
            
            # if abs(r) < 1e-5 and dr < 1e-5: 
            #     break
            
        d2,v2 = eigsh(X,k=3)
        new_set = argsort(-1 * abs(v2[:,-1]))[:s]
        se,sv = eigsh(A[new_set,:][:,new_set],k=3)
        if se[-1] > cgal_val:
            cgal_val = se[-1]
            cgal_set = new_set
        print(se)
        print(best_val)
        print(d2)
        
        L_old = L.copy()
        
        RW = abs(X)
        RW = RW/RW.max()
        L = 1/(correction[rw] + RW)
        
        L = L/L.max()
        
        print("reweigtred")
        print(norm(L-L_old))
        if norm(L-L_old) < eps_rw:
            break
    return X,y,d2,v2,cgal_set,cgal_val

def CGAL_wfro_dual(A,best_val,correction,s,beta0,max_iter,n2 = 0,eps_rw = 1e-3):
    # cgal mk0_s
    # Weighted Frobenius norm smoothing 
    # eigenshifts for kyrlov method
    # cgal dual update controlled
    # 
    # TODO: take lamda0 as input
    """
    Applies PCA Sparsified to covariance matrix A with variance constraint b using
    Conditional Gradient Augmented Lagrangian (CGAL) to solve subproblem

    Returns the best sparsity pattern computed and corresponding sparse eigenvalue as
    well as primal and dual variables computed at last iteration

    Parameters
    ----------
    A :(n,n) array
        Covariance matrix of the data
    best_val : float
        Variance constraint
    correction : list of floats or array of floats
        Stability parameter to use in each iteration
    s : integer
        Target sparsity level
    beta0: float
        Initial smoothing parameter for l1 norm
    max_iter : integer
        Maximum number of CGAL iterations for the subproblem
    n2 : integer, optional
        Eigenshift parameter used for CG-step of the algorithm. It was useful in Krylov methods
        using ARPACK. No longer used since LAPACK version seems to be faster and more stable.
    eps_rw : float, optional
        Absolute error precision desired for master problem. Low accuracy may result in early stop
        instead of doing max_iter number of iterations. Compared against changes in weights.

    Returns
    -------
    X : (n, n) array
        Primal variable X
    y : float
        Dual variable corresponding to variance constraint
    d2 : (3,) array
        Largest 3 eigenvalues of primal variable X.
    v2 : (n, 3) array
        Largest 3 eigenvectors of primal variable X.
    cgal_set : list of size sparsity
        Best sparsity pattern discovered throughout reweighted optimization
    cgal_val : float
        Best sparse eigenvalue discovered throughout reweighted optimization
    """
    # shape of matrix A
    m,n = shape(A)
    
    # keep track of best support found throughout reweighted iterations
    cgal_set = []
    cgal_val = 0
    
    # initial weights for reweighted optimization
    L = ones([n,n])
    
    # Initialize primal variable
    # Different initializations in general have small effect on the result
    # X = A/norm(A)
    X = zeros([n,n])
    
    # initialize dual variable
    y = 0
    
    # Problem scaling. It is necessary for Krylov methods i.e ARPACK
    # LAPACK version seems to be fine under no scaling but CGAL still favors
    # this scaling scheme
    A3 = A/norm(A)
    b = best_val/norm(A)
    
    # for loop for reweighted optimization
    for rw in range(len(correction)):
        
        # initial augmented Lagrangian parameter
        lamda0 = 10
        
        # initial residual for trace constraint
        r = tensordot(A3,X) - b
        
        # while loop for subproblem, exits after maximum number of iterations
        t = 1
        while t <= max_iter:
            # updating the problem parameters at iteration t
            # eta = step size
            # lamda = augmented Lagrangian parameter
            # beta = smoothing parameter
            t += 1
            eta = 2/t
            lamda = lamda0 * t ** 0.5
            beta = beta0/t ** 0.5
            
            # Gradient of the objective function
            w = X.copy()
            
            i,j = where(X > beta)
            k,l = where(X < -beta)
            w[i,j] = beta
            w[k,l] = -beta
            
            w = w / beta
            w = L * w
            
            # gradient of the augmented Lagrangian except objective function 
            v = y * A3 + lamda * A3 * r
            
            # conditional gradient step on primal variable X
            # computes dominant eigenvector of augmented Lagrangian
            vals,vec,kmany,isuppz,info = dsyevr(w + v,compute_v = 1,range = "I",il = 1,iu = 1,abstol = 1e-9)
            
            # CG update on primal variable X
            X = (1 - eta) * X + eta * vec.dot(vec.T)
            
            # Primal & Dual residual update
            r = tensordot(A3,X) - b
            dr = eta * norm(vec.dot(vec.T) - X)
            
            # lipschitz constant of the gradient
            lipschitz = lamda0 * (t+1) ** 0.5 + 1/(beta0/(t+1) ** 0.5 * L.min())
            
            # bounded travel conditions i.e dual step size
            sigma = min(lamda0 * 1,1/2*eta ** 2 * lipschitz * 2/(r ** 2))
            
            # Dual update on y
            y = y + sigma * r
            
            # printing some information on subproblem
            if t % 500 == 0:
                print("r",rw,"target sparsity",s)
                print("iteration",t)
                print("primal residual", r)
                print("dual residual", dr)
        
        # Extracting the largest 3 eigenvalues and vectors for the solution
        # Extracting 3 instead of 1 because if the largest eigenvalue is not
        # close 1, distribution of eigenvalues generally give information on 
        # convergence and quality of the solution obtained for sparse PCA
        
        # d2,v2 = eigsh(X,k=3)
        d2,v2,kmany,isuppz,info = dsyevr(X,compute_v = 1,range = "I",il = n-2,iu = n,abstol = 1e-12)
        new_set = argsort(-1 * abs(v2[:,-1]))[:s]
        # se,sv = eigsh(A[new_set,:][:,new_set],k=3)
        se,sv,kmany,isuppz,info = dsyevr(A[new_set,:][:,new_set],compute_v = 1,range = "I",il = s-2,iu = s,abstol = 1e-12)
        if se[2] > cgal_val:
            cgal_val = se[2]
            cgal_set = new_set

        # Computing new reweighted optimization weights and 
        # keeping track of amount of change
        # Scaling generally helps the algorithm
        
        L_old = L.copy()
        
        RW = abs(X)
        RW = RW/RW.max()
        L = 1/(correction[rw] + RW)
        
        L = L/L.max()
        
        # printing some information on master problem
        print("max 3 eig of subset",se[:3])
        print("best heuristic",best_val)
        print("max 3 eig of X",d2[:3])
        
        # stop the reweighted optimization if weights converged
        if norm(L-L_old) < eps_rw:
            break
    return X,y,d2,v2,cgal_set,cgal_val

def CGAL_wfro_dual_mk2(A,best_val,correction,s,beta0,max_iter,n2 = 0,eps_rw = 1e-3):
    # cgal mk0_s
    # Weighted Frobenius norm smoothing 
    # eigenshifts for kyrlov method
    # cgal dual update controlled
    #
    # This is an experimental version of original CGAL where I would test
    # small changes and their effects on the results, stability/convergence/speed and so on.
    m,n = shape(A)
    
    cgal_vals = []
    cgal_set = []
    cgal_val = 0
    
    L = ones([n,n])
    
    # X = A/norm(A)
    X = zeros([n,n])
    
    y = 0
    
    A3 = A/norm(A)
    b = best_val/norm(A)
    for rw in range(len(correction)):
        
        lamda0 = 12
        
        r = tensordot(A3,X) - b

        t = 1
        # change = 1
        while t <= max_iter:
            # print("step",t,"change",change)
            t += 1
            eta = 2/t
            lamda = lamda0 * t ** 0.5
            beta = beta0/t ** 0.5
            
            w = X.copy()
    
            i,j = where(X > beta)
            k,l = where(X < -beta)
            w[i,j] = beta
            w[k,l] = -beta
            
            w = w / beta
            
            w = L * w
            
            
            lipschitz = lamda0 * (t+1) ** 0.5 + 1/(beta0/(t+1) ** 0.5 * L.min())
            
            v = y * A3 + lamda * A3 * r
            
            vals,vec,kmany,isuppz,info = dsyevr(w + v,compute_v = 1,range = "I",il = 1,iu = 1,abstol = 1e-9)

            X = (1 - eta) * X + eta * vec.dot(vec.T)
            
            r = tensordot(A3,X) - b
            
            # bounded travel conditions
            sigma = min(lamda0 * 1,1/2*eta ** 2 * lipschitz * 2/(r ** 2))
            # sigma = min(lamda0 * 1,1/2*eta ** 2 * lipschitz * 2/(r ** 2))

            y = y + sigma * r
            dr = eta * norm(vec.dot(vec.T) - X)
            if t % 500 == 0:
                print("r",rw,"target sparsity",s)
                print("iteration",t)
                print("primal residual", r)
                print("dual residual", dr)
            
            # if abs(r) < 1e-5 and dr < 1e-5: 
            #     break
            
        d2,v2 = eigsh(X,k=3)
        new_set = argsort(-1 * abs(v2[:,-1]))[:s]
        se,sv = eigsh(A[new_set,:][:,new_set],k=3)
        if se[-1] > cgal_val:
            cgal_val = se[-1]
            cgal_set = new_set
        cgal_vals += [se[-1]]
        print("max 3 eig of subset",se)
        print("best heuristic",best_val)
        print("max 3 eig of X",d2)

        L_old = L.copy()
        
        RW = abs(X)
        RW = RW/RW.max()
        L = 1/(correction[rw] + RW)
        
        L = L/L.max()
        
        # print("reweigtred")
        # print(norm(L-L_old))
        # surr = log(correction[rw] + abs(X)).sum()
        # print("surrogate objective",surr)
        if norm(L-L_old) < eps_rw:
            break
    return X,y,d2,v2,cgal_set,cgal_val,cgal_vals

def CGAL_wfro_dual_fantope(A,best_val,correction,s,beta0,max_iter,n2 = 0,rank = 1,eps_rw = 1e-3):
    # cgal mk0_s
    # Weighted Frobenius norm smoothing 
    # eigenshifts for kyrlov method
    # cgal dual update controlled
    # Fantope instead of spectraplex 
    
    m,n = shape(A)
    
    cgal_set = []
    cgal_val = 0
    
    L = ones([n,n])
    
    # X = A/norm(A)
    X = zeros([n,n])
    
    y = 0
    
    A3 = A/norm(A)
    b = best_val/norm(A)
    for rw in range(len(correction)):
        
        lamda0 = 10
        
        r = tensordot(A3,X) - b

        t = 1
        # change = 1
        while t <= max_iter:
            # print("step",t,"change",change)
            t += 1
            eta = 2/t
            lamda = lamda0 * t ** 0.5
            beta = beta0/t ** 0.5
            
            w = X.copy()
    
            i,j = where(X > beta)
            k,l = where(X < -beta)
            w[i,j] = beta
            w[k,l] = -beta
            
            w = w / beta
            
            w = L * w
            
            
            lipschitz = lamda0 * (t+1) ** 0.5 + 1/(beta0/(t+1) ** 0.5 * L.min())
            
            v = y * A3 + lamda * A3 * r
            
            vals,vec,kmany,isuppz,info = dsyevr(w + v,compute_v = 1,range = "I",il = 1,iu = rank,abstol = 1e-9)
            # val,vec = eigsh(w + v + n2 * eye(n),k = rank,tol = 1e-8,which = "SA")

            X = (1 - eta) * X + eta * vec.dot(vec.T)
            
            r = tensordot(A3,X) - b
            
            # bounded travel conditions
            sigma = min(lamda0 * 10,1/2*eta ** 2 * lipschitz * 2/(r ** 2))
            # sigma = min(lamda0 * 1,1/2*eta ** 2 * lipschitz * 2/(r ** 2))

            y = y + sigma * r
            dr = eta * norm(vec.dot(vec.T) - X)
            if t % 500 == 0:
                print("r",rw)
                print("iteration",t)
                print("primal residual", r)
                print("dual residual", dr)
            
            # if abs(r) < 1e-5 and dr < 1e-5: 
            #     break
            
        d2,v2 = eigsh(X,k=rank + 3)
        new_set = argsort(-1 * abs(v2[:,-1]))[:s]
        se,sv = eigsh(A[new_set,:][:,new_set],k=3)
        if se[-1] > best_val:
            cgal_val = se[-1]
            cgal_set = new_set
        print(se)
        print(best_val)
        print(d2)

        L_old = L.copy()
        
        RW = abs(X)
        RW = RW/RW.max()
        L = 1/(correction[rw] + RW)
        
        L = L/L.max()
        
        print("reweigtred")
        print(norm(L-L_old))
        if norm(L-L_old) < eps_rw:
            break
    return X,y,d2,v2,cgal_set,cgal_val
