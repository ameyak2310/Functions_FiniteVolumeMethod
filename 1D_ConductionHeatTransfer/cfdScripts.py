""" README.

This library contains function for 1D and 2D finite volume methods and its allied
requirments.

Following functions are included in this library

#1 One D Grid generator
#2 One D Co-eff matrox generator
#3 One D Source matrix generator


"""
#%% Importing libraries
import numpy as np
import matplotlib.pyplot as plt


#%% 1D Grid Generation
def oneDgridGen(L,N):
    x = np.ones(N)
    x[0] = (L/N)/2
    for i in range(1,N-1):
        x[i] = x[i-1] + (L/N)
    x[-1] = L -(L/N)/2
    #print("Delta              = ", L/N)
    #print("Distance in meters = ", x)
    return x, (L/N)
#%% 1D Coefficient Coefficient Matrix formation
def oneDcoefMatrix(L,N,k,A,h):
    
    C = np.zeros(N*N).reshape(N,N)
    delta = (L/N)
    aW = k*A / delta
    aE = k*A / delta
    
    sP = np.ones(N) * 0; 
    sP[0] = k*A / delta * 2;   sP[-1] = k*A / delta * 2
    
  
    sI = np.ones(N) * h / (k*A) * delta 
    sI[-1] =  -1 * h / (k*A) * delta 
    
    C = np.zeros(N*N).reshape(N,N)
    for i in range(1,N-1):
        for j in range(1,N-1):
            if (i == j):
                C[i][j] =  aE + aW + sI[i]
                C[i][j-1] = -1 * aE
                C[i][j+1] = -1 * aW                   
    
    C[0,0]   = aE + sP[0]  + sI[0] ; C[0,1]   = -1* aW
    C[-1,-1] = aW + sP[-1] + sI[-1]; C[-1,-2] = -1* aE  
    #print("C = ", C)
    return C
#%% 1D Source matrix formation
def oneDsourceMatrix(L,k,A,N,T,q,h,Tamb):
    Su    = np.zeros(N)
    delta = (L/N)
    Su[:]  = k*A / delta     * T[:]  + q*A*delta + h*delta*Tamb
    Su[0]  = k*A / delta * 2 * T[0]  + q*A*delta + h*delta*Tamb
    Su[-1] = k*A / delta * 2 * T[-1] + q*A*delta + h*delta*Tamb
    #print("Su = ", Su)
    return Su
#%% Visualization
def visualize(x, sol_nu, sol_an):
    plt.scatter(x, sol_nu, color = 'k', label='Numerical')
    plt.plot   (x, sol_an, color = 'r', label='Analytical')
    plt.xlabel('Distance (meters)'); plt.ylabel('Temperature (deg C)')
    #plt.xlim(0, 0.5); plt.ylim(100, 500); plt.legend(loc='best'); plt.show()