#%% Ver steeg 6.2
import numpy as np
import pandas as pd
import simpleAlgorithm as sa

# Input variables
rho      = 1 # kg/m3
L        = 2 # m
n        = 4 # nodes
deltaX   = L/n # m
Ai       = 0.5; Ao = 0.1 # m2
pStag    = 10 # Pa
alpha    = 0.8
convergence = 1E-5

# Initial Guess
mdotStar = 1.0 # Kg/s
A        = np.linspace(0,L,n+1)*((Ao-Ai)/L) + Ai # Area
uStar    = mdotStar / (np.linspace((L/n/2),L-(L/n/2),n)*(A[-1]-A[0])/L + A[0]) # Velocity
pStar    = np.linspace(pStag,0,n+1)
res      = np.ones(n) * 1E2

#%% Looping SIMPLE 
count = 1; Residuals = []; Iterations = []
while (abs(sum(res)) > convergence):

    # print("Iteration no. :",count," Velocity Guess = ", uStar)
    # print("Iteration no. :",count," Pressure Guess = ", pStar)
      
    uCalc, dp, uC, Su    = sa.getvCorr(A,Ai,rho,n,L,pStag,uStar,pStar)
    # print("uCalc = ", uCalc)
    # print("dp    = ", dp)
    # print("uC    = ", uC)
    # print("Su    = ", Su)   
    
    pDash        = sa.getpCorr(n,rho,A,uCalc,dp)
    # print("pdash = ", pDash)
    
    uNext, pNext,uCalc, pCalc = sa.getnextGuess(n, uCalc,uStar, dp, pDash, pStar, alpha)
    # print("pCalc = ", pCalc)
    # print("uCalc = ", uCalc)
    # print("pNext = ", pNext)
    # print("uNext = ", uNext)    
    
    mCalc        = sa.checkContinuity(n,uNext,A)
    # print("mCalc = ", mCalc)
    
    res          = sa.getresidual(uC,uNext,uStar,Su)
    
    # print("Results of iteration ", count)
    # print("Iteration no. :",count," Residual = ", res)
    # print("Velocity Field (m/s)  = ", uNext)
    # print("Pressure Field (Pa)   = ",  pNext)
    print("Iteration no.:",count," Mass flow Rate (kg/s) = ", np.mean(mCalc))
    # print("************************ End of iteration", count, " ************************ ")

    uStar = uNext
    pStar = pNext
    count = count + 1
    
    Residuals.append(res.max())
    Iterations.append(count-1)
    if count > 100: break

d = {'Residuals': Residuals, 'Iterations': Iterations}
df = pd.DataFrame(d)
#%% Residual Plotter
rp = sa.resPlot(df)