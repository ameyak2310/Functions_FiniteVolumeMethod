# Importing libraries
import numpy as np
import time

#%% Analytical Solution Versteeg 7.1
def solAnalytical(x,Ti,Tw,n,L):
    t0 = time.time()
    #sol_an = 800*x+100
    #sol_an = ((Te - Tw)/L + q/k/2*(L - x)) * x +  Tw
    sol = Ti + (Tw - Ti)/(np.cosh(n*L)) *(np.cosh(n*(L - x)))
    tf = time.time()
    return np.around(sol,3), round((tf-t0)*1E3,5)

#%%