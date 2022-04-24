#%% Versteep 5.3
import cfdScripts as cfd
import anlScripts as anl
import numpy as np

# Dimensions and grid
L = 1.0                                        

# Material Properties
gamma = 0.1                          
rho   = 1.0                        
alpha = 1        
#%% Solver Case 1
scheme = "hybrid";N = 5; u = 2.5

# Boundary Condition
phi = np.zeros(N)
phi[0] = 1; phi[-1] = 0

# Solver
x, delta = cfd.oneDgridGen(L,N)
C        = cfd.oneDcoefMatrix(N,rho,u,gamma,delta,scheme,alpha)
Su       = cfd.oneDsourceMatrix(N,rho,u,gamma,delta,phi,scheme)
sol_nu   = np.around(np.linalg.solve(C,Su),3)

sol_an   = anl.solAnalytical52c2(x)
plot     = cfd.visualize(x, sol_nu, sol_an)

print("Solution   Numerical:",sol_nu)
print("Solution Analytical :",sol_an)

#  Percentage Error
E = np.zeros(N)
E[:] = np.around(abs((sol_nu[:] - sol_an[:]) / sol_an[:] * 100),2)
print("Error:", E)

#%%