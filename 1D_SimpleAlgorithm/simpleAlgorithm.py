"""

"""
#%%
import numpy as np

#%% Velocity Corrector
def getvCorr(A, Ai, rho,n,L,pStag,uStar,pStar):    
    Su = np.empty(n)*0
    Fe = np.empty(n)*0
    Fw = np.empty(n)*0
    aw = np.empty(n)*0
    ae = np.empty(n)*0
    ap = np.empty(n)*0
    dp = np.empty(n)*0
    
    for i in range(n): 
        if i == 0:
            Fe[i] = rho * A[i+1] * (uStar[i] + uStar[i+1]) * 0.5
            Fw[i] = rho * A[i] * uStar[i] * (0.5*(A[i]+A[i+1])) / A[i]

            aw[i] = 0
            ae[i] = 0
            ap[i] = Fe[i] + Fw[i] * 0.5 * ((0.5*(A[i]+A[i+1])) / A[i])**2
            Su[i] = (pStag - pStar[i+1]) * (A[i]+A[i+1]) * 0.5 + Fw[i] * (0.5*(A[i]+A[i+1])) / Ai * uStar[i]
            dp[i] = (A[i] + A[i+1]) * 0.5 / ap[i]
        elif (i>0) and (i<n-1):
            Fe[i] = rho * A[i+1] * (uStar[i] + uStar[i+1]) * 0.5 
            Fw[i] = rho * A[i]   * (uStar[i-1] + uStar[i]) * 0.5

            aw[i] = max(Fw[i],0)
            ae[i] = max(0,-1*Fe[i])
            ap[i] = ae[i] + aw[i] + (Fe[i]-Fw[i]) 
            Su[i] = (pStar[i]-pStar[i+1]) * (A[i] + A[i+1]) * 0.5
            dp[i] = (A[i] + A[i+1]) * 0.5 / ap[i]
        elif i == n-1:
            Fe[i] = rho * A[i+1] * uStar[i] * 0.5 *(A[i]+A[i+1]) / A[i+1]
            Fw[i] = rho * A[i]   * (uStar[i-1] + uStar[i])  * 0.5

            aw[i] = max(Fw[i],0)
            ae[i] = max(0,-1*Fe[i])
            ap[i] = ae[i] + aw[i] + (Fe[i]-Fw[i])
            Su[i] = (pStar[i] - pStar[i+1]) * (A[i] + A[i+1]) * 0.5
            dp[i] = (A[i] + A[i+1]) * 0.5 / ap[i]

    uC = np.empty(n*n).reshape(n,n)*0

    for i in range(n):
        for j in range(n):
            if i == j:
                uC[i][j] = ap[i]
                try:
                    uC[i][j+1] = -1*ae[i]
                except: pass

                try:
                    uC[i][j-1] = -1*aw[i]
                except: pass

    uCalc = np.linalg.solve(uC,Su)
    return uCalc, dp, uC, Su

#%% Pressure Corrector
def getpCorr(n,rho,A,uCalc,dp):
    aw = np.empty(n+1)*0
    ae = np.empty(n+1)*0
    ap = np.empty(n+1)*0
    bDash = np.empty(n+1)*0

    for i in range(n+1):
        if i == 0:
            aw[i]    = 0
            ae[i]    = 0
            ap[i]    = 1
            bDash[i] = 0
        elif (i>0) and (i<n):   
            aw[i]    = rho * 0.5 * (A[i-1] + A[i]) * dp[i-1]
            ae[i]    = rho * 0.5 * (A[i] + A[i+1]) * dp[i]
            FwStar   = rho * 0.5 * (A[i-1] + A[i]) * uCalc[i-1]
            FeStar   = rho * 0.5 * (A[i] + A[i+1]) * uCalc[i]
            ap[i]    = aw[i] + ae[i]
            bDash[i] = FwStar - FeStar
        elif i == n:
            aw[i]    = 0
            ae[i]    = 0
            ap[i]    = 1
            bDash[i] = 0

    pC = np.empty((n+1)*(n+1)).reshape(n+1,n+1)*0

    for i in range(n+1):
        for j in range(n+1):
            if i == j:
                pC[i][j] = ap[i]
                try:
                    pC[i][j+1] = -1*ae[i]
                except: pass

                try:
                    pC[i][j-1] = -1*aw[i]
                except: pass
    pDash = np.linalg.solve(pC,bDash)
    return pDash

#%% Next Guess
def getnextGuess(n, u,uStar, dp, pDash, pStar, alpha):
    uCalc = np.empty(n)*0
    for i in range(n):
        uCalc[i] = u[i] + dp[i] * (pDash[i]-pDash[i+1])
    
    pCalc  = pStar + pDash
    
    uNext = (1-alpha) * uStar + alpha * uCalc
    pNext = (1-alpha) * pStar + alpha * pCalc
    return uNext, pNext, uCalc, pCalc 

#%% Checking for continuity
def checkContinuity(n,uCalc ,A):
    mCalc = np.empty(n)*0
    for i in range(n):
        mCalc[i] = uCalc[i] * 0.5* (A[i]+A[i+1])
    return mCalc

#%% Residual calulator
def getresidual(uC,uNext,uStar,Su):
    res = np.matmul(uC,uStar) - Su
    return res

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