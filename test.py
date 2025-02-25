import numpy as np
from adal import adal_solver
from irwa import irwa_solver
from functions import exact_penalty_func


n = 100  # number of variables
m1 = 20  # number of equality constraints
m2 = 20  # number of inequality constraints

# equality constraints 
A_eq = np.random.rand(m1, n)
b_eq = np.zeros(m2)

# inequality constraints
A_ineq = np.random.rand(m2, n)
b_ineq = np.random.rand(m2)

# define the phi(x) with H and g
k = 5
P = np.random.rand(n, 5)
H = np.dot(P, P.T)/2 + 0.1 * np.eye(n)
g = np.random.rand(n)


x_irwa, k_irwa = irwa_solver(H, g, A_eq, b_eq, A_ineq, b_ineq)
x_adal, k_adal = adal_solver(H, g, A_eq, b_eq, A_ineq, b_ineq)

val_irwa = exact_penalty_func(H, g, x_irwa, A_eq, b_eq, A_ineq, b_ineq)
val_adal = exact_penalty_func(H, g, x_adal, A_eq, b_eq, A_ineq, b_ineq)

print(f"IRWA function value with penalty: {val_irwa}.")
print(f"ADAL function value with penalty: {val_adal}.")
