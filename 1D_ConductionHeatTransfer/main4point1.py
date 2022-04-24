#%% Versteeg 4.1
import numpy as np
import cfdScripts as cfd
import anlScripts as anl

#%% Input Variables

# Dimensions and grid
L = 0.5                        #Length in meter(m)
A = 10E-3                      #Area in sq.m (m2)
N = 5                          #Number of node points

# Material Properties
k = 1000                       # Thermal conductivity (W/mK)

# Sources and Sinks
q = 0
h = 0

# Boundary Condition
Tamb = 0
T = np.zeros(N)
T[0] = 100; T[-1] = 500

#%% Solver
x, delta = cfd.oneDgridGen(L,N)             # Step 01: Grid Generation
C        = cfd.oneDcoefMatrix(L,N,k,A,h)      # Step 02: Coeffcient Matrix
Su       = cfd.oneDsourceMatrix(L,k,A,N,T,q,h,Tamb)  # Step 03: Temperature Matrix
sol_nu   = np.linalg.solve(C,Su)            # Step 04: Numerical Solution
sol_an   = anl.solAnalytical41(x)             # Step 05: Analytical Solution
plot     = cfd.visualize(x, sol_nu, sol_an) # Step 06: Visualize
#%% Results
print("Numerical Solution  = " , sol_nu)
print("Analytical Solution = " , sol_an)
#%%

