import numpy as np
from adal import adal_solver
from irwa import irwa_solver

A_ineq = np.array([[-1,0],
                   [0,-1]])
b_ineq = np.array([1,4])

A_eq = np.array([[0,0]])
b_eq = np.array([0])


H = np.array([[1,-1],[-1,1]])
g = np.array([0,0])

x_irwa, k = irwa_solver(H,g,A_eq,b_eq,A_ineq,b_ineq)
x_adal, k_adal = adal_solver(H,g,A_eq,b_eq,A_ineq,b_ineq)

print(x_irwa)
print(x_adal)
