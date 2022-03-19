from numpy import random,argsort
from admm import ADMM
from cgal import CGAL_wfro_dual
from scipy.linalg import eigh

# Fix random number generator
random.seed(0)

# Number of observations in data matrix
m = 200

# Number of features in data matrix
n = 50

# Sparsity level
s = 20

# Generate random data of size (m,n)
A = random.normal(0,1,[m,n])

# Emprical scaling of the data 
A = A - A.mean(axis = 0)
A = A/A.std(axis = 0)

# Construct the covariance matrix
A2 = A.T.dot(A)

# Find the largest eigenvector and eigenvalue
lambda1,vector1 = eigh(A2,subset_by_index = [n-1,n-1])

# Truncate the largest eigenvector by sparsity level
support = argsort(abs(vector1[:,0]))[-s:]

# Compute sparse eigen-pair
s_lambda,s_vector = eigh(A2[:,support][support,:],subset_by_index = [s-1,s-1])

# set the variance constraint
varc = s_lambda[0]

#%% CGAL

# Set up stability parameter for CGAL
correction = [1/(4*s)] * 5
# Set maximum number of iterations for CGAL
max_iter_cgal = 2200
# Set initial smoothing parameter for CGAL
beta0 = 0.5/n

# Call the CGAL solver
X,y,eigX3_value,eigX3_vector,cgal_set,cgal_val = \
CGAL_wfro_dual(A2,varc,correction,s,beta0,max_iter_cgal)

#%% ADMM

# Set up stability parameter for ADMM
correction = [1/(2*s)] * 5
# Set maximum number of iterations for ADMM
max_iter_admm = 1000
# Set initial augmented Lagrangian parameter for ADMM
rho = 40

# Call the ADMM solver
X1,X2,X3,y1,y2,U1,U2,admm_set,admm_val = \
ADMM(A2,varc,rho,correction,s,max_iter_admm)

#%%
print("Results")
print("Truncation",varc)
print("CGAL",cgal_val)
print("ADMM",admm_val)