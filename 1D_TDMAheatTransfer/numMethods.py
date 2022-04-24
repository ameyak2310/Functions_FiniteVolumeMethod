""" README.

This library contains function based on differnet numerical solution techniques 
for solving system of equations.

[C][phi] = [Su] 

Co-efficient matrix C and Source matrix Su is to be fed into for 
each function, and the function return the Solution [phi] and the 
time taken for computation (t) in seconds (time 1E-3)

Following functions are included in this library

#1 Gauss Elimination (Direct)
#2 LU decompositon (Direct)
#3 Numpy based One D solution (Direct-LU Decomp. based)
#4 Numpy based Two D solution (Direct-LU Decomp. based)
#5 Gauss Jordan (Direct)
#6 Jacobi Iterative (Iterative)
#7 Gauss Seidel (Iterative)

"""
#%% Importing libraries
import numpy as np
import time

#%% GAUSS ELIMINATION
def gaussElimantion(C,T):
    N = len(T)
    t0 = time.time()
    sol = np.zeros(N)
    # Check for diagonal Dominance
    # Find diagonal coefficients
    dia = np.diag(np.abs(C)) 

    # Find row sum without diagonal
    off = np.sum(np.abs(C), axis=1) - dia 

    if np.all(dia > off):
        # Forward Elimation
        for i in range(N-1):
            for j in range(i+1,N):
                fctr = C[j,i] / C[i,i]
                for k in range(i, N):
                    C[j,k] = C[j,k] - fctr * C[i,k]
                T[j] = T[j] - fctr * T[i]

         # Back-substitution
        sol[N-1] = T[N-1] / C[N-1,N-1]
        for i in range(N-2,-1,-1):
            Sum = T[i]
            for j in range(i+1,N):
                Sum = Sum - C[i,j] * sol[j]
            sol[i] = Sum/C[i,i]

    else:
        print('NOT diagonally dominant')
        print('System of equation cannot be solved using Gauss Elimination Method !') 
    tf = time.time()
    return np.around(sol,3) , round((tf-t0)*1E3,5)

#%% JACOBI ITERATIVE
def jacobi(A, b, tolerance=1e-10, max_iterations=10000):
    t0 = time.time()
    x = np.zeros_like(b, dtype=np.double)  
    T = A - np.diag(np.diagonal(A))  
    for k in range(max_iterations):     
        x_old  = x.copy()
        x[:] = (b - np.dot(T, x)) / np.diagonal(A)
        delta = np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) 
        if delta < tolerance:
            break   
    tf = time.time()
    return np.around(x,3), round((tf-t0)*1E3,5)

#%% GAUSS SEIDEL
def gaussSeidel(A, b, tolerance=1e-10, max_iterations=10000):
    t0 = time.time()
    x = np.zeros_like(b, dtype=np.double)
    for k in range(max_iterations):
        x_old  = x.copy()
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x_old[(i+1):])) / A[i ,i]           
        delta = np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf)
        if delta < tolerance:
            break         
    tf = time.time()
    return np.around(x,3), round((tf-t0)*1E3,5)

#%% LU DECOMPOSITION
def lu(A):
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n, dtype=np.double)
    for i in range(n):
        factor = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factor
        U[i+1:] -= factor[:, np.newaxis] * U[i]        
    return L, U

def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros_like(b, dtype=np.double);
    y[0] = b[0] / L[0, 0]
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i,:i], y[:i])) / L[i,i]        
    return y

def back_substitution(U, y):
    n = U.shape[0]
    x = np.zeros_like(y, dtype=np.double);
    x[-1] = y[-1] / U[-1, -1]
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - np.dot(U[i,i:], x[i:])) / U[i,i]        
    return x

def lu_solve(A, b):
    t0 = time.time()
    L, U = lu(A)
    y = forward_substitution(L, b)
    sol = back_substitution(U, y)
    tf = time.time()
    return np.around(sol,3), round((tf-t0)*1E3,5)

#%% GAUSS JORDAN
def gaussJordan(A, b):
    t0 = time.time()
    temp_mat = np.c_[A, b]
    
    #Get the number of rows
    n = temp_mat.shape[0]
    
    #Loop over rows
    for i in range(n):
        p = np.abs(temp_mat[i:, i]).argmax()
        p += i 
        if p != i:
            temp_mat[[p, i]] = temp_mat[[i, p]]
            

        temp_mat = temp_mat / np.diagonal(temp_mat)[:, np.newaxis]
        factor = temp_mat[:i, i] 
        temp_mat[:i] -= factor[:, np.newaxis] * temp_mat[i]
            
        factor = temp_mat[i+1:, i] 
        temp_mat[i+1:] -= factor[:, np.newaxis] * temp_mat[i]
    
    tf = time.time()
    if temp_mat[:,n:].shape[1] == 1:
        return np.around(temp_mat[:,n:].flatten(),3), round((tf-t0)*1E3,5)
    else:
        return np.around(temp_mat[:,n:],3), round((tf-t0)*1E3,5)
#%% ONE D NUMPY LIBRARY SOLUTION
def oneDsolNumPy(C,T):
    t0 = time.time()
    sol = np.linalg.solve(C,T) 
    tf = time.time()
    return np.around(sol,3), round((tf-t0)*1E3,5)

#%% TWO D NUMPY LIBRARY SOLUTION
def twoDsolNumPy(C,Su,Ny,Nx):
    t0 = time.time()
    sol = np.linalg.solve(C,Su).reshape(Ny,Nx)
    tf = time.time()
    return np.around(sol,3), round((tf-t0)*1E3,5)
#%%