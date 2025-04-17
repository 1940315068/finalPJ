import torch
from ..functions import *
from ..algorithms.irwa import irwa_solver
from ..algorithms.adal import adal_solver
import time
from .data_gen import *


# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)
print(f"Using device: {device}")


scale = 5
n = 1000*scale  # number of variables
m1 = 300*scale  # number of equality constraints
m2 = 300*scale  # number of inequality constraints

# data = generate_portfolio_data(n_assets=n, n_factors=n//100, target_return=0.1, numpy_output=False, include_shorting=True)
# data = generate_bounded_least_squares_data(n=n, m=m1, numpy_output=False)
# data = generate_svm_data(n=n, m=50, numpy_output=False)
data = generate_random_data(n=n,m1=m1,m2=m2, numpy_output=False)

H = data['torch']['H'].to(device)
g = data['torch']['g'].to(device)
A_eq = data['torch']['A_eq'].to(device)
b_eq = data['torch']['b_eq'].to(device)
A_ineq = data['torch']['A_ineq'].to(device)
b_ineq = data['torch']['b_ineq'].to(device)

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
val_irwa_pri = quadratic_objective(H, g, x_irwa)

val_adal_penalty = penalized_quadratic_objective(H, g, A_eq, b_eq, A_ineq, b_ineq, x_adal)
val_adal_pri = quadratic_objective(H, g, x_adal)


# show the comparison
print("------------------------------------------------------------")
print(f"Number of variables: {g.shape[0]}")
print(f"Number of constraints: {b_eq.shape[0]} + {b_ineq.shape[0]} = {b_eq.shape[0] + b_ineq.shape[0]}")
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
