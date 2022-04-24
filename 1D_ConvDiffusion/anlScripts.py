# Importing libraries
import numpy as np


#%% Analytical Solution Versteeg 4.1
def solAnalytical41(x):
    sol = 800*x+100
    #sol = ((Te - Tw)/L + q/k/2*(L - x)) * x +  Tw
    #sol = Ti + (Tw - Ti)/(np.cosh(n*L)) *(np.cosh(n*(L - x)))
    return np.around(sol,3)

#%% Analytical Solution Versteeg 4.2
def solAnalytical42(x,T,L,q,k):
    sol = ((T[-1] - T[0])/L + q/k/2*(L - x)) * x +  T[0]
    return np.around(sol,3)

#%% Analytical Solution Versteeg 4.3
def solAnalytical43(Tamb,T,h,L,x):
    sol = Tamb + ((T[0] - Tamb) * (np.cosh(h**0.5*(L - x))) / (np.cosh(h**0.5*L)))
    return np.around(sol,3)

#%% Analytical Solution Versteeg 5.1 Case 1
def solAnalytical51c1(x):
    sol = (2.7183 - np.exp(x)) / 1.7183
    return np.around(sol,3)

#%% Analytical Solution Versteeg 5.1 Case 2
def solAnalytical51c2(x):
    sol = 1 + (1-np.exp(25*x))/(7.2E10)
    return np.around(sol,3)

#%% Analytical Solution Versteeg 5.2 Case 2
def solAnalytical52c2(x):
    sol = 1 + (1-np.exp(25*x))/(7.2E10)
    return np.around(sol,3)