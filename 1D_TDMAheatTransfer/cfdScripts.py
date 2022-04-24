""" README.

This library contains function for 1D and 2D finite volume methods and its allied
requirments.

Following functions are included in this library

#1 One D Grid generator
#2 One D Co-eff matrox generator
#3 One D Source matrix generator
#4 One D TDMA
#5 Two D Grid generator
#6 Two D Co-eff matrox generator
#7 Two D Source matrix generator
#8 Two D Visulization - Heat map generator
#9 Residual plotter

"""
#%%

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import time
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
def oneDcoefMatrix(L,N,n):
    delta = (L/N)
    C = np.zeros(N*N).reshape(N, N)
    for i in range(0,N-1):
        for j in range(0,N-1):
            if (i == j): 
                C[i][j]   = (1 / delta) + (1 / delta) + (n**2 * delta) # = ap
                C[i][j+1] =  -1 / delta                                # = ae
                C[i+1][j] =  -1 / delta                                # = aw
    C[0][0]     = (1 / delta) + (1 * n**2 * delta + 2/delta)           # = ae + ap
    C[-1][-1]   = (1 / delta) + (1 * n**2 * delta)                     # = ae + Sp
    #print("C = ", C)
    return C
#%% 1D Source matrix formation
def oneDsourceMatrix(L,N,Tw,Te,Ti,n):
    delta = (L/N)
    T     = np.zeros(N) + (n**2 * delta * Ti)
    T[0]  = (n**2 * delta * Ti) + 2 * Tw/ delta
    T[-1] = (n**2 * delta * Ti) 
    #print("T = ", T)
    return T
#%% 1D TDMA
def oneDtdma(C,T):
    t0 = time.time()
    N = len(T)    
    # Generates Dj, alpha, beta
    D = np.empty(N)*0
    a = np.empty(N)*0
    b = np.empty(N)*0
    for i in range(N):
        for j in range(N):
            if i == j:
                try: 
                    D[i]     = C[i][j] 
                except: pass
                try: 
                    a[i] = -1 * C[i][j+1]
                except: pass
                try:
                    b[i]  = -1 * C[i][j-1]
                except: pass
    
    # Generates Aj
    A = np.empty(N) * 0
    for i in range(N):
        try:
            A[i] = a[i] / (D[i] - b[i] * A[i-1])
        except: pass

    # Generates CDash
    CDash = np.empty(N) * 0
    for i in range(N):
        try:
            CDash[i] = (b[i] * CDash[i-1] + T[i]) / (D[i] - b[i] * A[i-1])
        except: pass

    # Generates solution matrix phi
    phi = np.empty(N) * 0
    for i in reversed(range(N)):
        try:
            phi[i] = A[i] * phi[i+1] + CDash[i]
        except:
            phi[i] = CDash[i]
    
    tf = time.time()
    return np.around(phi,3), round((tf-t0)*1E3,5)
#%% 2D Grid Generation
def twoDgridGen(Lx,Ly,Nx,Ny):
    x = np.linspace((Lx/Nx/2),Lx-(Lx/Nx/2),Nx)
    y = np.linspace((Ly/Ny/2),Ly-(Ly/Ny/2),Ny)
    deltaX = Lx/Nx
    deltaY = Ly/Ny
    return x,y,deltaX,deltaY
#%% 2D Coefficient Matrix formation
def twoDcoefMatrix(Nx,Ny,k,A, deltaX,deltaY):
    aW = k*A/deltaX #* 0 +1*-1
    aE = k*A/deltaX #* 0 +1*-1
    aN = k*A/deltaY #* 0 +1*-1
    aS = k*A/deltaY #* 0 +1*-1
    C = np.zeros(Nx*Ny*Nx*Ny).reshape(Nx*Ny,Nx*Ny)
    for i in range(0,Nx*Ny):
            for j in range(0,Nx*Ny):
                if (i == j):
                    C[i][j]  = aW + aE + aN + aS
                    # march forward
                    try:
                        C[i][j+1] = -1*aE
                        try: C[i][j+Nx] = -1*aS
                        except: pass
                    except: pass
    
                    # march backward
                    try:
                        C[i+1][j] = -1*aW       # = aw
                        try:
                            C[i+Nx][j] = -1*aN  # = an
                        except: pass
                    except: pass
    # West 0,3,6,9
    for i in range(0,Nx*Ny,Nx):
        for j in range(0,Ny*Nx,Nx):
            if (i == j):
                # print(i,j)
                C[i][j]   = aN + aS + aE
                C[i][j-1] = 0               
    # East 2,5,8,11
    for i in range(Nx-1,Nx*Ny-1,Nx):
        for j in range(Nx-1,Ny*Nx-1,Nx):
            if (i == j):
                # print(i,j)
                C[i][j]  = aN + aS + aW
                C[i][j+1] = 0
    # North 0,1,2
    for i in range(0,Nx):
        for j in range(0,Nx):
            if (i == j):
                # print(i,j)
                C[i][j]  = aE + aS + aW + 2*k*A/deltaX
    # South
    for i in range(-1,-1*(Nx+1),-1):
        for j in range(-1,-1*(Nx+1),-1):
            if (i == j):
                # print(i,j)
                C[i,j] = aW + aE + aN 
    # Corners
    C[0,0]         = aS + aW + 2*k*A/deltaX
    C[Nx-1,Nx-1]   = aS + aE + 2*k*A/deltaX
    C[-1,-1]       = aN + aW
    C[-1*Nx,-1*Nx] = aN + aE
    return C
#%% # Coefficient Coefficient Matrix formation   
def old_twoDcoefMatrix(Nx,Ny,k,A, deltaX,deltaY):
    C = np.zeros(Nx*Ny*Nx*Ny).reshape(Nx*Ny,Nx*Ny)
    #Central Nodes
    for i in range(0,Nx*Ny):
            for j in range(0,Ny*Nx):
                if (i == j):
                    C[i][j]  = 2*k*A/deltaX + 2*k*A/deltaY                
                    # march forward
                    try: 
                        C[i][j+1] = -1*k*A/deltaX
                        try: C[i][j+Ny] = -1*k*A/deltaY
                        except: pass
                    except: pass

                    # march backward
                    try:
                        if (j-1>0): C[i][j-1] = -1*k*A/deltaX       # = aw
                        try:
                            if (j-Ny>0): C[i][j-Ny] = -1*k*A/deltaY  # = an
                        except: pass
                    except: pass

    # West
    for i in range(1,Nx):
        for j in range(0,Ny*Nx):
            if (i == j):
                C[i][:]    =  0
                C[i][j]    =  3*k*A/deltaY                         
                C[i][j+1]  = -1*k*A/deltaX # North
                C[i][j+Ny] = -1*k*A/deltaY # 
                C[i][j-1]  = -1*k*A/deltaX # South

    #East
    for i in range(1,Nx):
        for j in range(0,Ny*Nx):
            if (i == j):
                C[i+(Nx*Ny-Ny)][:]   = 0
                C[i+(Nx*Ny-Ny)][j+(Nx*Ny-Ny)]    =  3*k*A/deltaY         
                C[i+(Nx*Ny-Ny)][j+(Nx*Ny-Ny)+1]  = -1*k*A/deltaX   # = an
                C[i+(Nx*Ny-Ny)][j+(Nx*Ny-Ny)-1]  = -1*k*A/deltaX   # = as
                C[i+(Nx*Ny-Ny)][j+(Nx*Ny-Ny)-Ny] = -1*k*A/deltaY   # = ae


    # North
    for i in range(2*Ny-1,Ny*Nx-Ny,Ny):
        for j in range(0,Ny*Nx):
            if (i == j):
                C[i][:]    = 0
                C[i][j]    =  3*k*A/deltaX + 2*k*A/deltaY          
                C[i][j-1]  = -1*k*A/deltaX  # = as
                C[i][j+Ny] = -1*k*A/deltaY  # = ae
                C[i][j-Ny] = -1*k*A/deltaY  # = aw


    # South
    for i in range(2*Ny-1,Ny*Nx-Ny,Ny):
        for j in range(0,Ny*Nx):
            if (i == j):
                C[i-Nx][:]       = 0
                C[i-Nx][j-Nx]    =  3*k*A/deltaX         
                C[i-Nx][j-Nx+1]  = -1*k*A/deltaX   # = an
                C[i-Nx][j-Nx+Ny] = -1*k*A/deltaY  # = aw
                C[i-Nx][j-Nx-Ny] = -1*k*A/deltaY  # = ae

    # Corners
    C[0][:]   =  0
    C[0][0]   =  k*A/deltaX + k*A/deltaY   # aSE
    C[0][1]   = -1*k*A/deltaX              # an
    C[0][Ny]  = -1*k*A/deltaX              # aw

    C[Ny-1][:]         = 0
    C[Ny-1][Ny-1]      = 2*k*A/deltaX + 2*k*A/deltaY   # aNE 
    C[Ny-1][Ny-1-1]    = -1*k*A/deltaY                  # aw
    C[Ny-1][Ny-1+Ny]   = -1*k*A/deltaY                  # as

    C[-1*Ny][:]        = 0
    C[-1*Ny][-1*Ny]    = k*A/deltaX + k*A/deltaY   # aSW
    C[-1*Ny][-1*Ny+1]  = -1*k*A/deltaX              # ae
    C[-1*Ny][-1*Ny-Ny] = -1*k*A/deltaY              # an

    C[-1][:]           = 0
    C[-1][-1]          = 2*k*A/deltaX + 2*k*A/deltaY   # aSE
    C[-1][-1-1]        = -1*k*A/deltaX                  # aw
    C[-1][-1-Ny]       = -1*k*A/deltaY                  # an
    return C
#%% 2D Source matrix generator
def twoDSuMatrix(Nx,Ny,k,A,qW,qE,qS,qN,TW,TE,TN,TS, deltaX,deltaY):
    Su = np.zeros(Nx*Ny)   
    
    # West 0,3,6,9
    for i in range(0,Nx*Ny,Nx):
        for j in range(0,Ny*Nx,Nx):
            if (i == j):
                # print(i,j)
                Su[i]   = Su[i] + qW*A + 2*k*A/deltaX*TW                
    # East 2,5,8,11
    for i in range(Nx-1,Nx*Ny-1,Nx):
        for j in range(Nx-1,Ny*Nx-1,Nx):
            if (i == j):
                # print(i,j)
                Su[i]   = Su[i] + qE*A + 2*k*A/deltaX*TE
    # North 0,1,2
    for i in range(0,Nx):
        for j in range(0,Nx):
            if (i == j):
                # print(i,j)
                Su[i]   = Su[i] + qN*A + 2*k*A/deltaX*TN
    # South -1,-2,-3
    for i in range(-1,-1*(Nx+1),-1):
        for j in range(-1,-1*(Nx+1),-1):
            if (i == j):
                # print(i,j)
                Su[i]   = Su[i] + qS*A+ 2*k*A/deltaX*TS
    # Corners
    
    
    Su[0]     = qN*A + qW*A+ 2*k*A/deltaX*(TW+TN)
    Su[Nx-1]  = qN*A + qE*A+ 2*k*A/deltaX*(TE+TN)
    Su[-1]    = qS*A + qE*A+ 2*k*A/deltaX*(TS+TW)
    Su[-1*Nx] = qS*A + qW*A+ 2*k*A/deltaX*(TS+TE)
    
    return Su
#%% 2D Source matrix generator      
def old_twoDSuMatrix(Nx,Ny,k,A,qW,qE,qS,qN,TW,TE,TN,TS, deltaX,deltaY):
    Su        = np.zeros(Nx*Ny)

    Su                     = np.zeros(Nx*Ny)
    Su[1      :Ny-1      ] = qW*A
    Su[2*Ny-1:Ny*Nx-Ny:Ny] = 2*k*A/deltaY*TN
    Su[Nx*Ny-Ny+1:Nx*Ny-1] = 0
    Su[0:Nx*Ny:Ny]         = 0

    Su[0]                  = qW*A
    Su[Ny-1]               = 2*k*A/deltaY*TN + qW*A
    Su[-1*Ny]              = 0
    Su[-1]                 = 2*k*A/deltaY*TN   
    return Su
#%% 2D Heat Mapper
def twoDvisualize(sol_np,Lx,Ly,Tw,Te,Ts,Tn):
    # plt.figure(figsize=(6, 4), dpi = 600)
    # hm = plt.imshow(sol_np, extent=(0, Lx, 0, Ly), 
    #            interpolation='none', cmap="jet",#"coolwarm",
    #            vmin=min(Tw,Te,Ts,Tn), vmax=max(Tw,Te,Ts,Tn)+200)
    # plt.colorbar()
    #plt.show()
    
    import seaborn as sns
    plt.figure(figsize=(10,10), dpi = 300)
    heatmap = sns.heatmap((sol_np), linewidth = 1 , cmap="jet", annot = True, fmt=".2f")
    plt.show() 
    return heatmap
#%% Residual Plotter
def resPlot(df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6), dpi = 300)  
    sns.axes_style("darkgrid")
    # sns.set(style='darkgrid')
    plot = sns.lineplot(x='Iterations', y='Residuals', data=df,linewidth=2.5).set(title='Residual Decay')
    sns.set(rc={"figure.figsize":(6, 4)})
    # plot.set(xscale="log", yscale="log")
    return plot
#%%    