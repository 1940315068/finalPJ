import torch
from ..functions import *
from ..algorithms.irwa import irwa_solver
from ..algorithms.adal import adal_solver
import time


# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)
print(f"Using device: {device}")


scale = 5
n = 1000*scale  # number of variables
m1 = 300*scale  # number of equality constraints
m2 = 300*scale  # number of inequality constraints

# equality constraints 
A_eq = torch.rand(m1, n)
b_eq = torch.zeros(m1)

# inequality constraints
A_ineq = torch.rand(m2, n)
b_ineq = torch.rand(m2)

# Add infeasible constraints of equality: Ax+b=0, Ax-b=0
m1_infeasible = 0
m1 += 2*m1_infeasible
A_eq_infeasible = torch.rand(m1_infeasible, n)
A_eq = torch.vstack([A_eq, A_eq_infeasible, A_eq_infeasible])
b_eq_infeasible = torch.rand(m1_infeasible)
b_eq = torch.hstack([b_eq, b_eq_infeasible, -b_eq_infeasible])

# Add infeasible constraints of inequality: Ax+1<=0, -Ax<=0
m2_infeasible = 0
m2 += 2*m2_infeasible
A_ineq_infeasible = torch.rand(m2_infeasible, n)
A_ineq = torch.vstack([A_ineq, A_ineq_infeasible, -A_ineq_infeasible])
b_ones_infeasible = torch.ones(m2_infeasible)
b_zeros_infeasible = torch.zeros(m2_infeasible)
b_ineq = torch.hstack([b_ineq, b_ones_infeasible, b_zeros_infeasible])


# define the phi(x) with H and g
P = torch.rand(n, 5)
H = torch.matmul(P, P.T) + 0.1 * torch.eye(n)
g = torch.rand(n)

# start IRWA
start_irwa = time.time()
x_irwa, k_irwa, n_cg_steps_irwa, time_cg_irwa = irwa_solver(H, g, A_eq, b_eq, A_ineq, b_ineq)
end_irwa = time.time()
running_time_irwa = end_irwa-start_irwa

# start ADAL
start_adal = time.time()
x_adal, k_adal, n_cg_steps_adal, time_cg_adal = adal_solver(H, g, A_eq, b_eq, A_ineq, b_ineq)
end_adal = time.time()
running_time_adal = end_adal-start_adal


# compute function value with/without penalty
val_irwa_penalty = penalized_quadratic_objective(H, g, A_eq, b_eq, A_ineq, b_ineq, x_irwa)
val_irwa_pri = quadratic_objective(H, g, x_irwa)

val_adal_penalty = penalized_quadratic_objective(H, g, A_eq, b_eq, A_ineq, b_ineq, x_adal)
val_adal_pri = quadratic_objective(H, g, x_adal)


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
print(f"IRWA running time: {running_time_irwa:.3f}s, iteration: {k_irwa}")
print(f"ADAL running time: {running_time_adal:.3f}s, iteration: {k_adal}")
print("------------------------------------------------------------")
print(f"IRWA CG steps: {n_cg_steps_irwa}, CG total computation time: {time_cg_irwa:.4f}s")
print(f"ADAL CG steps: {n_cg_steps_adal}, CG total computation time: {time_cg_adal:.4f}s")
print("------------------------------------------------------------")
