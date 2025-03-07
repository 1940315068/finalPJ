import torch
from irwa import irwa_solver
from adal import adal_solver
import time
from functions import *
import osqp
from scipy.sparse import csc_matrix
import numpy as np


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


# Compare with osqp, l <= Ax <= u
m = m1+m2
u = np.zeros(m)
l = np.zeros(m)
A = csc_matrix(torch.vstack([A_eq, A_ineq]).cpu().numpy())
b = torch.hstack([b_eq, b_ineq]).cpu().numpy()
P = csc_matrix(H.cpu().numpy())
q = g.cpu().numpy()

# equality constraints: l = u = -b
for i in range(m1):
    l[i] = -b[i]
    u[i] = -b[i]
# inequality constraints, l = -inf, u = -b
for i in range(m1,m):
    l[i] = -np.inf
    u[i] = -b[i]

# start OSQP
start_osqp = time.time()
prob = osqp.OSQP()
prob.setup(P=P, q=q, A=A, l=l, u=u)
result_osqp = prob.solve()
end_osqp = time.time()
running_time_osqp = end_osqp-start_osqp
x_osqp = torch.from_numpy(result_osqp.x).to(device)  # Convert NumPy array to Torch tensor and move it to the device


# compute function value with/without penalty
val_irwa_penalty = exact_penalty_func(H, g, x_irwa, A_eq, b_eq, A_ineq, b_ineq)
val_irwa_pri = quadratic_form(H, g, x_irwa)

val_adal_penalty = exact_penalty_func(H, g, x_adal, A_eq, b_eq, A_ineq, b_ineq)
val_adal_pri = quadratic_form(H, g, x_adal)

val_osqp_penalty = exact_penalty_func(H, g, x_osqp, A_eq, b_eq, A_ineq, b_ineq)
val_osqp_pri = quadratic_form(H, g, x_osqp)


# show the comparison
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

