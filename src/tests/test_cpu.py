import numpy as np
from ..functions import *
from ..algorithms.irwa import irwa_solver
from ..algorithms.adal import adal_solver
import time


scale = 1
n = 1000*scale  # number of variables
m1 = 300*scale  # number of equality constraints
m2 = 300*scale  # number of inequality constraints

# equality constraints 
A_eq = np.random.rand(m1, n)
b_eq = np.zeros(m1)

# inequality constraints
A_ineq = np.random.rand(m2, n)
b_ineq = np.random.rand(m2)

# Add infeasible constraints of equality: Ax+b=0, Ax-b=0
m1_infeasible = 0
m1 += 2*m1_infeasible
A_eq_infeasible = np.random.rand(m1_infeasible, n)
A_eq = np.vstack([A_eq, A_eq_infeasible, A_eq_infeasible])
b_eq_infeasible = np.random.rand(m1_infeasible)
b_eq = np.hstack([b_eq, b_eq_infeasible, -b_eq_infeasible])

# Add infeasible constraints of inequality: Ax+1<=0, -Ax<=0
m2_infeasible = 0
m2 += 2*m2_infeasible
A_ineq_infeasible = np.random.rand(m2_infeasible, n)
A_ineq = np.vstack([A_ineq, A_ineq_infeasible, -A_ineq_infeasible])
b_ones_infeasible = np.ones(m2_infeasible)
b_zeros_infeasible = np.zeros(m2_infeasible)
b_ineq = np.hstack([b_ineq, b_ones_infeasible, b_zeros_infeasible])


# define the phi(x) with H and g
P = np.random.rand(n, 5)
H = np.dot(P, P.T) + 0.1 * np.eye(n)
g = np.random.rand(n)


# start IRWA
start_irwa = time.monotonic()
x_irwa, k_irwa, n_cg_steps_irwa, time_cg_irwa = irwa_solver(H, g, A_eq, b_eq, A_ineq, b_ineq)
end_irwa = time.monotonic()
running_time_irwa = end_irwa-start_irwa

# start ADAL
start_adal = time.monotonic()
x_adal, k_adal, n_cg_steps_adal, time_cg_adal = adal_solver(H, g, A_eq, b_eq, A_ineq, b_ineq)
end_adal = time.monotonic()
running_time_adal = end_adal-start_adal


# compute function value with/without penalty
val_irwa_penalty = penalized_quadratic_objective(H, g, A_eq, b_eq, A_ineq, b_ineq, x_irwa)
val_adal_penalty = penalized_quadratic_objective(H, g, A_eq, b_eq, A_ineq, b_ineq, x_adal)
val_irwa_pri = quadratic_objective(H, g, x_irwa)
val_adal_pri = quadratic_objective(H, g, x_adal)
penalty_irwa = val_irwa_penalty - val_irwa_pri
penalty_adal = val_adal_penalty - val_adal_pri

# show the comparison
print("------------------------------------------------------------")
print(f"Number of variables: {n}")
print(f"Number of constraints: {m1} + {m2} = {m1+m2}")
print("------------------------------------------------------------")
print(f"IRWA function value with penalty: {val_irwa_penalty:.6f}")
print(f"ADAL function value with penalty: {val_adal_penalty:.6f}")
print("------------------------------------------------------------")
print(f"IRWA function value without penalty: {val_irwa_pri:.6f}")
print(f"ADAL function value without penalty: {val_adal_pri:.6f}")
print("------------------------------------------------------------")
print(f"IRWA penalty: {penalty_irwa:.3e}")
print(f"ADAL penalty: {penalty_adal:.3e}")
print("------------------------------------------------------------")
print(f"IRWA running time: {running_time_irwa:.3f}s, iteration: {k_irwa}")
print(f"ADAL running time: {running_time_adal:.3f}s, iteration: {k_adal}")
print("------------------------------------------------------------")
print(f"IRWA CG steps: {n_cg_steps_irwa}, CG total computation time: {time_cg_irwa:.4f}s")
print(f"ADAL CG steps: {n_cg_steps_adal}, CG total computation time: {time_cg_adal:.4f}s")
print("------------------------------------------------------------")
