#%% Versteeg 4.1
import numpy as np
import cfdScripts as cfd
import anlScripts as anl

#%% Input Variables

# Dimensions and grid
L = 2E-2                        #Length in meter(m)
A = 1                           #Area in sq.m (m2)
N = 5                           #Number of node points

# Material Properties
k = 0.5                         # Thermal conductivity (W/mK)

#Source and Sinks
q = 1000E3
h = 0

# Boundary Condition
Tamb = 0
T = np.zeros(N)
T[0] = 100; T[-1] = 200

#%% Solver
x, delta = cfd.oneDgridGen(L,N)             
C        = cfd.oneDcoefMatrix(L,N,k,A,h)      
Su       = cfd.oneDsourceMatrix(L,k,A,N,T,q,h,Tamb) 
sol_nu   = np.linalg.solve(C,Su)            
sol_an   = anl.solAnalytical42(x,T,L,q,k)             
plot     = cfd.visualize(x, sol_nu, sol_an)
 
#%% Results
print("Numerical Solution  = " , sol_nu)
print("Analytical Solution = " , sol_an)
#%% Percentage Error
E = np.zeros(N)
E[:] = np.around(abs((sol_nu[:] - sol_an[:]) / sol_an[:] * 100),2)



