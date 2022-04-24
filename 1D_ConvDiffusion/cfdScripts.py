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
def oneDcoefMatrix(N,rho,u,gamma,delta,scheme,alpha):
    
    C = np.zeros(N*N).reshape(N,N)
    F = rho * u
    D = gamma / delta 
    
    if (scheme == "CDS"):
        for i in range(1,N-1):
            for j in range(1,N-1):
                if (i == j):
                    C[i][j] =  (D - F/2) + (D + F/2)
                    C[i][j-1] = -1* (D + F/2)
                    C[i][j+1] = -1* (D - F/2)                  
        
        C[0,0]   = (D - F/2) - (-1* (2*D + F)); C[0,1]   = -1*(D - F/2) 
        C[-1,-1] = (D + F/2) - (-1* (2*D - F)); C[-1,-2] = -1*(D + F/2)  
        
    elif (scheme == "UDS"):
        for i in range(1,N-1):
            for j in range(1,N-1):
                if (i == j):
                    C[i][j] =  (D + F) + (D)
                    C[i][j-1] = -1* (D + F)
                    C[i][j+1] = -1* (D)                  
        
        C[0,0]   = D + (2*D + F) ; C[0,1]   = -1*(D) 
        C[-1,-1] = (D + F) + 2*D ; C[-1,-2] = -1*(D + F) 

    elif (scheme == "hybrid"):
        aW = max(F, (D+F/2), 0)
        aE = max(-1*F, (D-F/2),0)
        for i in range(1,N-1):
            for j in range(1,N-1):
                if (i == j):
                    C[i][j] =  aW + aE + 0
                    C[i][j-1] = -1* aW
                    C[i][j+1] = -1* aE                  
        
        C[0,0]   = (2*D + F) ; C[0,1]   = 0
        C[-1,-1] = (2*D + F) ; C[-1,-2] = -1*F

    elif (scheme == "QUICK"):
        
        aWW = -1/8*alpha*F
        aW  = D + (6/8*alpha*F) + (1/8*alpha*F)     + (3/8*(1-alpha)*F)
        aE  = D - (3/8*alpha*F) - (6/8*(1-alpha)*F) - (1/8*(1-alpha)*F)
        aEE = -1/8*(1-alpha)*F
        
        for i in range(2,N-2):
            for j in range(2,N-2):
                if (i == j):
                    C[i][j] =  aWW + aW + aE + aEE
                    C[i][j-2] = -1* aWW
                    C[i][j-1] = -1* aW
                    C[i][j+1] = -1* aE  
                    C[i][j+2] = aEE
        
        C[0,0],  C[0,1],  C[0,2]            = ((D+1/3*D-3/8*F) + (8/3*D+2/8*F+F)),(-1*((D+1/3*D-3/8*F))),(1/8*(1-alpha)*F)
        C[1,0],  C[1,1],  C[1,2],  C[1,3]   = (-1*(D+7/8*F+1/8*F)),((D+7/8*F+1/8*F)+(D-3/8*F)- 1/4*F),(-1*(D-3/8*F)),(0)
        C[-2,-4],C[-2,-3],C[-2,-2],C[-2,-1] = (-1*aWW),(-1*aW),(aWW + aW + aE + aEE),(-1*aE)
        C[-1,-3],C[-1,-2],C[-1,-1]          = (-1*aWW),(-1*(D+(1/3*D)+(6/8*F))), (aWW + (D+(1/3*D)+(6/8*F)) + (8/3*D-F))
                
    
    #print("C = ", C)
    return C
#%% 1D Source matrix formation
def oneDsourceMatrix(N,rho,u,gamma,delta, phi, scheme):
    Su     = np.zeros(N)
    F = rho * u
    D = gamma / delta
    
    if (scheme == "CDS"):
        Su[0]  = (2*D + F) * phi[0]
        Su[-1] = (2*D - F) * phi[-1]
        
        
    elif (scheme == "UDS"):
        Su[0]  = (2*D + F) * phi[0]
        Su[-1] = (2*D) * phi[-1]
        
    elif (scheme == "hybrid"):
        Su[0]  = (2*D + F) * phi[0]
        Su[-1] = (2*D) * phi[-1]
        
    elif (scheme == "QUICK"):
        Su[0]  = (8/3*D + 2/8*F + F)*phi[0]
        Su[1]  = -1* 1/4*F * phi[0]
        
    #print("Su = ", Su)
    return Su
#%% Visualization
def visualize(x, sol_nu, sol_an):
    plt.scatter(x, sol_nu, color = 'k', label='Numerical')
    plt.plot   (x, sol_an  , color = 'r', label='Analytical')
    plt.xlabel('Distance (meters)'); plt.ylabel('Temperature (deg C)')
    #plt.xlim(0, 0.5); plt.ylim(100, 500); plt.legend(loc='best'); plt.show()