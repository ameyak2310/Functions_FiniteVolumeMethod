#%% Versteeg 4.3
import numpy as np
import cfdScripts as cfd
import anlScripts as anl

#%% Input Variables

# Dimensions and grid
L = 1                           #Length in meter(m)
A = 1E3                           #Area in sq.m (m2)
N = 5                           #Number of node points

# Material Properties
k = 1E-3                         # Thermal conductivity (W/mK)

#Source and sink
q = 0
h = 25

# Boundary Condition
Tamb = 20
T = np.zeros(N)
T[0] = 100;

#%% Solver
x, delta = cfd.oneDgridGen(L,N)             
C        = cfd.oneDcoefMatrix(L,N,k,A,h)      
Su       = cfd.oneDsourceMatrix(L,k,A,N,T,q,h,Tamb) 
sol_nu   = np.linalg.solve(C,Su)            
sol_an   = anl.solAnalytical43(Tamb,T,h,L,x)             
plot     = cfd.visualize(x, sol_nu, sol_an)
 
#%% Results
print("Numerical Solution  = " , sol_nu)
print("Analytical Solution = " , sol_an)
#%% Percentage Error
E = np.zeros(N)
E[:] = np.around(abs((sol_nu[:] - sol_an[:]) / sol_an[:] * 100),2)
print(E)




