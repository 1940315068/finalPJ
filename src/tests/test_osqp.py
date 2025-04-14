import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import osqp
import time
from scipy.sparse import csc_matrix
from ..algorithms import irwa, adal
from ..functions import penalized_quadratic_objective, quadratic_objective
from .data_gen import generate_random_data


def solve_with_osqp(H, g, A_eq, b_eq, A_ineq, b_ineq, algebra='builtin'):
    """Solve using OSQP solver."""
    P = csc_matrix(H)
    A = csc_matrix(np.vstack([A_eq, A_ineq]))
    l = np.hstack([-b_eq, -np.inf * np.ones(len(b_ineq))])
    u = np.hstack([-b_eq, -b_ineq])
    start = time.monotonic()
    prob_osqp = osqp.OSQP(algebra=algebra)
    prob_osqp.setup(P=P, q=g, A=A, l=l, u=u, verbose=False)
    result_osqp = prob_osqp.solve()
    running_time = time.monotonic() - start
    x = result_osqp.x
    iters = result_osqp.info.iter
    print_solver_results(f"OSQP-{algebra}", x, iters, running_time, H, g, A_eq, b_eq, A_ineq, b_ineq)
    return x, iters, running_time, result_osqp.info.run_time, result_osqp.info.setup_time, result_osqp.info.solve_time


def solve_with_irwa(H, g, A_eq, b_eq, A_ineq, b_ineq):
    """Solve using IRWA algorithm."""
    start = time.monotonic()
    x, iters, _, _ = irwa.irwa_solver(H, g, A_eq, b_eq, A_ineq, b_ineq, verbose=False)
    running_time = time.monotonic() - start
    print_solver_results("IRWA-GPU", x, iters, running_time, H, g, A_eq, b_eq, A_ineq, b_ineq)
    return x, iters, running_time


def solve_with_adal(H, g, A_eq, b_eq, A_ineq, b_ineq):
    """Solve using ADAL algorithm."""
    start = time.monotonic()
    x, iters, _, _ = adal.adal_solver(H, g, A_eq, b_eq, A_ineq, b_ineq, verbose=False)
    running_time = time.monotonic() - start
    print_solver_results("ADAL-GPU", x, iters, running_time, H, g, A_eq, b_eq, A_ineq, b_ineq)
    return x, iters, running_time


def print_solver_results(solver_name, x, iters, running_time, H, g, A_eq, b_eq, A_ineq, b_ineq):
    """Print formatted results for a solver including objective values and constraints.
    
    Args:
        solver_name (str): Name of the solver
        x (np.array): Solution vector
        iters (int): Number of iterations
        running_time (float): Execution time in seconds
        H, g, A_eq, b_eq, A_ineq, b_ineq: Problem parameters
    """
    val_penalty = penalized_quadratic_objective(H, g, A_eq, b_eq, A_ineq, b_ineq, x)
    val_primal = quadratic_objective(H, g, x)
    penalty = val_penalty - val_primal
    
    print(f"{solver_name:<12} {val_penalty:>14.6f} {val_primal:>14.6f} "
          f"{penalty:>12.3e} {iters:>8} {running_time:>10.4f}")


if __name__ == "__main__":
    # Problem parameters
    scale = 5
    n = 1000 * scale  # number of variables
    m1 = 300 * scale  # number of equality constraints
    m2 = 300 * scale  # number of inequality constraints

    # Generate problem data
    torch_float_dtype = torch.float64
    data = generate_random_data(n=n, m1=m1, m2=m2, numpy_output=True, torch_output=True, torch_float_dtype=torch_float_dtype)
    
    # Extract problem matrices
    H, g = data['numpy']['H'], data['numpy']['g']
    A_eq, b_eq = data['numpy']['A_eq'], data['numpy']['b_eq']
    A_ineq, b_ineq = data['numpy']['A_ineq'], data['numpy']['b_ineq']

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch_float_dtype)
    torch.set_default_device(device)

    # GPU (PyTorch) data
    H_torch = data['torch']['H'].to(device)
    g_torch = data['torch']['g'].to(device)
    A_eq_torch = data['torch']['A_eq'].to(device)
    b_eq_torch = data['torch']['b_eq'].to(device)
    A_ineq_torch = data['torch']['A_ineq'].to(device)
    b_ineq_torch = data['torch']['b_ineq'].to(device)

    """Run and compare all available solvers."""
    print(f"{'Solver':<12} {'Val(penalty)':>14} {'Val(primal)':>14} {'Penalty':>12} {'Iters':>8} {'Time(s)':>10}")
    print("-"*80)
    solve_with_irwa(H_torch, g_torch, A_eq_torch, b_eq_torch, A_ineq_torch, b_ineq_torch)
    solve_with_adal(H_torch, g_torch, A_eq_torch, b_eq_torch, A_ineq_torch, b_ineq_torch)
    _, _, t1, run_time_1, setup_time_1, solve_time_1 = solve_with_osqp(H, g, A_eq, b_eq, A_ineq, b_ineq, algebra="cuda")
    _, _, t2, run_time_2, setup_time_2, solve_time_2 = solve_with_osqp(H, g, A_eq, b_eq, A_ineq, b_ineq, algebra="mkl")
    print("-"*80)
    print()
    print("-"*80)
    print(f"{'Total time':<15} {'Run time':>14} {'Set-up time':>14} {'Solve time':>14}")
    print(f"{t1:<15.4f} {run_time_1:>14.4f} {setup_time_1:>14.4f} {solve_time_1:>14.4f}")
    print(f"{t2:<15.4f} {run_time_2:>14.4f} {setup_time_2:>14.4f} {solve_time_2:>14.4f}")
    print("-"*80)
    print()
