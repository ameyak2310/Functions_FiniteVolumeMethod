#%% 7.2 Versteeg#
# Import librabries
import cfdScripts as cfd
import numMethods as num
import numpy as np
import pandas as pd

# Input Variables

# Dimensions and grid
Lx = 0.3   # Length in meter(m) in x direction
Ly = 0.4   # Length in meter(m) in y direction

n  = 1
Nx = 3*n # Number of node points in x direction
Ny = 4*n # Number of node points in y direction

A  = (0.01*0.1)    #10E-3     # Area in sq.m (m2)

# Material Properties
k  = 1000   # Thermal conductivity (W/mK)

# Boundary Condition
TN = 100 # Temperature in degree Celcius
TW = 0 # Temperature in degree Celcius
TE = 0 # Temperature in degree Celcius
TS = 0 # Temperature in degree Celcius

# Sources and Sink
qE = 0
qW = 500E3
qN = 0
qS = 0
#%% Numpy Solution
x,y,deltaX,deltaY = cfd.twoDgridGen(Lx,Ly,Nx,Ny)
def getCSu():
    C  = cfd.twoDcoefMatrix(Nx,Ny,k,A, deltaX,deltaY)
    Su = cfd.twoDSuMatrix(Nx,Ny,k,A,qW,qE,qS,qN,TW,TE,TN,TS, deltaX,deltaY)
    return C,Su
#%% Looping for Nodes
tan = []; tnu = []; ttd = []; tgj = []
tge = []; tjc = []; tgs = []; tlu = []

Nodes = [1,2,]#4,5,6,7,8,9,10,15,20,25,30]
for i in Nodes:
    n = i
    Nx = 3*n # Number of node points in x direction
    Ny = 4*n # Number of node points in y direction
    
    x,y,deltaX,deltaY = cfd.twoDgridGen(Lx,Ly,Nx,Ny)
    C                 = cfd.twoDcoefMatrix(Nx,Ny,k,A, deltaX,deltaY)
    Su                = cfd.twoDSuMatrix(Nx,Ny,k,A,qW,qE,qS,qN,TW,TE,TN,TS, deltaX,deltaY)
    
    sol_nu, t_nu = num.twoDsolNumPy(getCSu()[0],getCSu()[1],Ny,Nx);                                 Su = cfd.twoDSuMatrix(Nx,Ny,k,A,qW,qE,qS,qN,TW,TE,TN,TS, deltaX,deltaY)                           
    # sol_td, t_td = cfd.oneDtdma(getCSu()[0],getCSu()[1]);                                          Su = cfd.twoDSuMatrix(Nx,Ny,k,A,qW,qE,qS,qN,TW,TE,TN,TS, deltaX,deltaY)                        
    # sol_ge, t_ge = num.gaussElimantion(getCSu()[0],getCSu()[1]);                                  Su = cfd.twoDSuMatrix(Nx,Ny,k,A,qW,qE,qS,qN,TW,TE,TN,TS, deltaX,deltaY)
    sol_gj, t_gj = num.gaussJordan(getCSu()[0],getCSu()[1]);                                        Su = cfd.twoDSuMatrix(Nx,Ny,k,A,qW,qE,qS,qN,TW,TE,TN,TS, deltaX,deltaY)
    sol_jc, t_jc = num.jacobi(getCSu()[0], getCSu()[1], tolerance=1e-10, max_iterations=10000);     Su = cfd.twoDSuMatrix(Nx,Ny,k,A,qW,qE,qS,qN,TW,TE,TN,TS, deltaX,deltaY)
    sol_gs, t_gs = num.gaussSeidel(getCSu()[0], getCSu()[1], tolerance=1e-10, max_iterations=10000);Su = cfd.twoDSuMatrix(Nx,Ny,k,A,qW,qE,qS,qN,TW,TE,TN,TS, deltaX,deltaY)
    sol_lu, t_lu = num.lu_solve(getCSu()[0],getCSu()[1])
    
    tnu.append(t_nu)
    # ttd.append(t_td)
    # tge.append(t_ge)
    tgj.append(t_gj)
    tjc.append(t_jc)
    tgs.append(t_gs)
    tlu.append(t_lu)
    
df1D = pd.DataFrame({
                    "Nodes":np.array(Nodes)**2*3*4,
                    "Nmpy":tnu,
                    # "TDMA":ttd,
                    # "GEli":tge,
                    "GJor":tgj,
                    "JIte":tjc,
                    "Gsei":tgs,
                    "LUde":tlu })

df1D.to_csv('twoDComparison.csv')    

print("__________Simulation time (seconds)__________")
print("Numpy Library     = ", tnu, " x 1E-3 seconds")
# print("TDMA              = ", ttd, " x 1E-3 seconds")
# print("Gauss Elimination = ", tge, " x 1E-3 seconds")
print("Gauss Jordan      = ", tgj, " x 1E-3 seconds")
print("Jacobi Iterative  = ", tjc, " x 1E-3 seconds")
print("Gauss Seidel      = ", tgs, " x 1E-3 seconds")
print("LU Decoposition   = ", tlu, " x 1E-3 seconds\n")
#%% Solution and Heat Plots
sol_np, t_np      = num.twoDsolNumPy(getCSu()[0],getCSu()[1],Ny,Nx) 
heatmap           = cfd.twoDvisualize(sol_np,Lx,Ly,TW,TE,TS,TN)