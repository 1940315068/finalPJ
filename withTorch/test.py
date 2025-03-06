import torch
from irwa import irwa_solver
from adal import adal_solver
import time
from functions import *


# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)
print(f"Using device: {device}")


n = 4000  # number of variables
m1 = 400  # number of equality constraints
m2 = 400  # number of inequality constraints

# equality constraints 
A_eq = torch.rand(m1, n)
b_eq = torch.zeros(m1)

# inequality constraints
A_ineq = torch.rand(m2, n)
b_ineq = torch.rand(m2)

# define the phi(x) with H and g
P = torch.rand(n, 5)
H = torch.matmul(P, P.T) + 0.1 * torch.eye(n)
g = torch.rand(n)

# start IRWA
start_irwa = time.time()
x_irwa, k_irwa = irwa_solver(H, g, A_eq, b_eq, A_ineq, b_ineq)
end_irwa = time.time()
running_time_irwa = end_irwa-start_irwa

# start ADAL
start_adal = time.time()
x_adal, k_adal = adal_solver(H, g, A_eq, b_eq, A_ineq, b_ineq)
end_adal = time.time()
running_time_adal = end_adal-start_adal

# compute function value with/without penalty
val_irwa_penalty = exact_penalty_func(H, g, x_irwa, A_eq, b_eq, A_ineq, b_ineq)
val_irwa_pri = quadratic_form(H, g, x_irwa)

val_adal_penalty = exact_penalty_func(H, g, x_adal, A_eq, b_eq, A_ineq, b_ineq)
val_adal_pri = quadratic_form(H, g, x_adal)

# show the comparison
print("------------------------------------------------------------")
print(f"IRWA function value with penalty: {val_irwa_penalty}")
print(f"ADAL function value with penalty: {val_adal_penalty}")
print("------------------------------------------------------------")
print(f"IRWA function value without penalty: {val_irwa_pri}")
print(f"ADAL function value without penalty: {val_adal_pri}")
print("------------------------------------------------------------")
print(f"IRWA running time: {running_time_irwa}")
print(f"ADAL running time: {running_time_adal}")

