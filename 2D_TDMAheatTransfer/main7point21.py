#%% 7.2 Versteeg#
# Import librabries
import cfdScripts as cfd
import numpy as np
import pandas as pd
import copy

# Dimensions and grid
Lx = 0.3   # Length in meter(m) in x direction
Ly = 0.4   # Length in meter(m) in y direction

n  = 1
Nx = 3*n   # Number of node points in x direction
Ny = 4*n   # Number of node points in y direction

A  = (0.01*0.1)    #10E-3     # Area in sq.m (m2)

# Material Properties
k  = 1000   # Thermal conductivity (W/mK)

# Boundary Condition
TN = 100 # Temperature in degree Celcius
TW = 0   # Temperature in degree Celcius
TE = 0   # Temperature in degree Celcius
TS = 0   # Temperature in degree Celcius

# Sources and Sink
qE = 0
qW = 500E3
qN = 0
qS = 0

# Grid, co-efficient and source matrix generator
x,y,deltaX,deltaY = cfd.twoDgridGen(Lx,Ly,Nx,Ny)
C   = cfd.old_twoDcoefMatrix(Nx,Ny,k,A, deltaX,deltaY)
Su  = cfd.old_twoDSuMatrix(Nx,Ny,k,A,qW,qE,qS,qN,TW,TE,TN,TS, deltaX,deltaY)
#%% 2D TDMA
res = np.ones(len(Su))*100; convergence = 1E-1
maxit = 50; count = 0
sol = np.empty(len(Su))*0

Residuals = []; Iterations = []

while(res.max() > convergence and count < maxit):
    
    sol_old = copy.copy(sol)    
    
    for n in range(Nx):
        print("Sweep = ", n+1, "\n")

        C_clip = []
        for i in range(Ny):
            for j in range(Ny):
                C_clip.append(C[i+ n*Ny][j+ n*Ny])
        C_clip = np.array(C_clip).reshape(Ny,Ny)       
        # print("Coeff Matrix = \n",C,"\n\n", C_clip, "\n")
        

        Su_clip = [];
        for i in range(Ny):
            Su_clip.append(Su[i+ n*Ny])
        # print("Source Matrix = \n",Su,"\n\n", Su_clip, "\n")
            
        
        left=np.zeros(Ny);right=np.zeros(Ny)
        for i in range(Ny):
            if (n == 0):
                left = np.zeros(Ny) 
            else: left[i]  = k*A/deltaY * sol[i+ (n-1)*Ny]    
            
            if (n == Nx-1):
                right = np.zeros(Ny)
            else: right[i] = k*A/deltaY * sol[i+ (n+1)*Ny]
            
        Su_clip = np.array(Su_clip) + left + right 
              
        sol_new = np.around(cfd.oneDtdma(C_clip,Su_clip)[0],2)

        for i in range(Ny):
            sol[i+ n*Ny] = sol_new[i]
            
        # print("Updated Source :\n", np.flip(Su_clip.reshape(Ny,1)), "\n")    
        print("T Matrix = \n", np.rot90(sol.reshape(Nx,Ny)), "\n")
        
        n = n + 1
    
    res = np.abs(sol_old - sol)
    Residuals.append(res.max())
    
    count = count + 1
    Iterations.append(count)
    print("\nEnd of Iteration No. :",count)
    print("\nSolution = \n", np.rot90(sol.reshape(Nx,Ny)), '\n')
    print("Absolute Maximum Residual = \n", np.rot90(res.reshape(Nx,Ny)))
    d = {'Residuals': Residuals, 'Iterations': Iterations}
    df = pd.DataFrame(d)
#%% Plots
hm = cfd.twoDvisualize(np.rot90(sol.reshape(Nx,Ny)),Lx,Ly,TW,TE,TS,TN)
rp = cfd.resPlot(df)
#%%


