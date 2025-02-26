import numpy as np
from adal import adal_solver
from irwa import irwa_solver
from functions import exact_penalty_func, quadratic_form
import osqp
from scipy.sparse import csc_matrix
import time


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


start_irwa = time.time()
x_irwa, k_irwa = irwa_solver(H, g, A_eq, b_eq, A_ineq, b_ineq)
end_irwa = time.time()
running_time_irwa = end_irwa-start_irwa

start_adal = time.time()
x_adal, k_adal = adal_solver(H, g, A_eq, b_eq, A_ineq, b_ineq)
end_adal = time.time()
running_time_adal = end_adal-start_adal


# Compare with osqp, l <= Ax <= u

m = m1+m2
u = np.zeros(m)
l = np.zeros(m)
A = csc_matrix(np.vstack([A_eq, A_ineq]))
b = np.hstack([b_eq, b_ineq])
P = csc_matrix(H)
q = g

# equality constraints: l = u = -b
for i in range(m1):
    l[i] = -b[i]
    u[i] = -b[i]
# inequality constraints, l = -inf, u = -b
for i in range(m1,m):
    l[i] = -np.inf
    u[i] = -b[i]
    
start_osqp = time.time()
prob = osqp.OSQP()
prob.setup(P=P, q=q, A=A, l=l, u=u)
result_osqp = prob.solve()
end_osqp = time.time()
running_time_osqp = end_osqp-start_osqp


val_irwa_penalty = exact_penalty_func(H, g, x_irwa, A_eq, b_eq, A_ineq, b_ineq)
val_adal_penalty = exact_penalty_func(H, g, x_adal, A_eq, b_eq, A_ineq, b_ineq)
val_osqp_penalty = exact_penalty_func(H, g, result_osqp.x, A_eq, b_eq, A_ineq, b_ineq)
val_irwa_pri = quadratic_form(H, g, x_irwa)
val_adal_pri = quadratic_form(H, g, x_adal)
val_osqp_pri = quadratic_form(H, g, result_osqp.x)

print("------------------------------------------------------------")
print(f"IRWA function value with penalty: {val_irwa_penalty}")
print(f"ADAL function value with penalty: {val_adal_penalty}")
print(f"OSQP function value with penalty: {val_osqp_penalty}")
print("------------------------------------------------------------")
print(f"IRWA function value without penalty: {val_irwa_pri}")
print(f"ADAL function value without penalty: {val_adal_pri}")
print(f"OSQP function value without penalty: {val_osqp_pri}")
print("------------------------------------------------------------")
print(f"IRWA running time: {running_time_irwa}")
print(f"ADAL running time: {running_time_adal}")
print(f"OSQP running time: {running_time_osqp}")