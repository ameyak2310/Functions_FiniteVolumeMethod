#%% 7.1 Versteeg
import pandas as pd
import cfdScripts as cfd
import anlScripts as anl
import numMethods as num


## Input Variables

# Dimensions and grid
L = 1                                  # Length in meter (m)
A = 10E-3                              # Area in sq.m (m2)

#Material Properties
n = 5                                  # Thermal conductivity (1/K)

# Boundary Condition
Tw = 100                               # Temperature in degree Celcius
Te = 20                                # Temperature in degree Celcius
Ti = 20                                # Temperature in degree Celcius

tan = []; tnu = []; ttd = []; tgj = []
tge = []; tjc = []; tgs = []; tlu = []

Nodes = [5,10,20]#,50,100,200,500,1000,2000,5000,10000]
#%% Calculating computational times for Nodes
for i in Nodes:
    N = i

    x, delta = cfd.oneDgridGen(L,N)                   # Step 01: Grid Generation
    C        = cfd.oneDcoefMatrix(L,N,n)              # Step 02: Coeffcient Matrix
    Su        = cfd.oneDsourceMatrix(L,N,Tw,Te,Ti,n)  # Step 03: Temperature Matrix

    sol_an, t_an = anl.solAnalytical(x,Ti,Tw,n,L);                               Su = cfd.oneDsourceMatrix(L,N,Tw,Te,Ti,n)
    sol_nu, t_nu = num.oneDsolNumPy(C,Su);                                       Su = cfd.oneDsourceMatrix(L,N,Tw,Te,Ti,n)                             
    sol_td, t_td = cfd.oneDtdma(C,Su);                                           Su = cfd.oneDsourceMatrix(L,N,Tw,Te,Ti,n)                               
    sol_ge, t_ge = num.gaussElimantion(C,Su);                                    Su = cfd.oneDsourceMatrix(L,N,Tw,Te,Ti,n)
    sol_gj, t_gj = num.gaussJordan(C,Su);                                        Su = cfd.oneDsourceMatrix(L,N,Tw,Te,Ti,n)
    sol_jc, t_jc = num.jacobi(C, Su, tolerance=1e-5, max_iterations=100);        Su = cfd.oneDsourceMatrix(L,N,Tw,Te,Ti,n)
    sol_gs, t_gs = num.gaussSeidel(C, Su, tolerance=1e-5, max_iterations=100);   Su = cfd.oneDsourceMatrix(L,N,Tw,Te,Ti,n)
    sol_lu, t_lu = num.lu_solve(C,Su)              

    tan.append(t_an)
    tnu.append(t_nu)
    ttd.append(t_td)
    tge.append(t_ge)
    tgj.append(t_gj)
    tjc.append(t_jc)
    tgs.append(t_gs)
    tlu.append(t_lu)
#%% Printing Simulation times
print("__________Simulation time (seconds)__________")
print("Analytical        = ", tan, " x 1E-3 seconds")
print("Numpy Library     = ", tnu, " x 1E-3 seconds")
print("TDMA              = ", ttd, " x 1E-3 seconds")
print("Gauss Elimination = ", tge, " x 1E-3 seconds")
print("Gauss Jordan      = ", tgj, " x 1E-3 seconds")
print("Jacobi Iterative  = ", tjc, " x 1E-3 seconds")
print("Gauss Seidel      = ", tgs, " x 1E-3 seconds")
print("LU Decoposition   = ", tlu, " x 1E-3 seconds\n")

#%% Dumping data into a dataframe

df1D = pd.DataFrame({
                    "Nodes":Nodes,
                    "Anlt":tan,
                    "Nmpy":tnu,
                    "TDMA":ttd,
                    "GEli":tge,
                    "GJor":tgj,
                    "JIte":tjc,
                    "Gsei":tgs,
                    "LUde":tlu })
df1D.to_csv('oneDComparison.csv')
#%%
