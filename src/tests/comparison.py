import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import osqp
import cvxpy as cp
from cvxopt import matrix, solvers
from scipy.optimize import minimize
from scipy.sparse import csc_matrix
import time
from ..algorithms import irwa, adal
from ..functions import penalized_quadratic_objective, quadratic_objective
from .data_gen import generate_optimization_data


def solve_with_irwa(H, g, A_eq, b_eq, A_ineq, b_ineq):
    """Solve using IRWA algorithm."""
    start = time.time()
    x, iters, _, _ = irwa.irwa_solver(H, g, A_eq, b_eq, A_ineq, b_ineq, verbose=False)
    running_time = time.time() - start
    print_solver_results("IRWA", x, iters, running_time, H, g, A_eq, b_eq, A_ineq, b_ineq)
    return x, iters, running_time


def solve_with_adal(H, g, A_eq, b_eq, A_ineq, b_ineq):
    """Solve using ADAL algorithm."""
    start = time.time()
    x, iters, _, _ = adal.adal_solver(H, g, A_eq, b_eq, A_ineq, b_ineq, verbose=False)
    running_time = time.time() - start
    print_solver_results("ADAL", x, iters, running_time, H, g, A_eq, b_eq, A_ineq, b_ineq)
    return x, iters, running_time


def solve_with_osqp(H, g, A_eq, b_eq, A_ineq, b_ineq):
    """Solve using OSQP solver."""
    P = csc_matrix(H)
    A = csc_matrix(np.vstack([A_eq, A_ineq]))
    l = np.hstack([-b_eq, -np.inf * np.ones(len(b_ineq))])
    u = np.hstack([-b_eq, -b_ineq])
    start = time.time()
    prob_osqp = osqp.OSQP()
    prob_osqp.setup(P=P, q=g, A=A, l=l, u=u, verbose=False)
    result_osqp = prob_osqp.solve()
    running_time = time.time() - start
    x = result_osqp.x
    iters = result_osqp.info.iter
    print_solver_results("OSQP", x, iters, running_time, H, g, A_eq, b_eq, A_ineq, b_ineq)
    return x, iters, running_time


def solve_with_cvxpy(H, g, A_eq, b_eq, A_ineq, b_ineq):
    """Solve using CVXPY interface."""
    n = H.shape[0]
    x_cvxpy = cp.Variable(n)
    obj_cvxpy = cp.Minimize(0.5 * cp.quad_form(x_cvxpy, H) + g.T @ x_cvxpy)
    constraints_cvxpy = [A_eq @ x_cvxpy == -b_eq, A_ineq @ x_cvxpy <= -b_ineq]
    start = time.time()
    prob_cvxpy = cp.Problem(obj_cvxpy, constraints_cvxpy)
    prob_cvxpy.solve() 
    running_time = time.time() - start
    x = x_cvxpy.value
    iters = prob_cvxpy.solver_stats.num_iters
    print_solver_results("CVXPY", x, iters, running_time, H, g, A_eq, b_eq, A_ineq, b_ineq)
    return x, iters, running_time


def solve_with_cvxopt(H, g, A_eq, b_eq, A_ineq, b_ineq):
    """Solve using CVXOPT solver."""
    solvers.options['show_progress'] = False
    P = matrix(H)
    q = matrix(g)
    G = matrix(A_ineq)
    h = matrix(-b_ineq)
    A = matrix(A_eq)
    b = matrix(-b_eq)
    start_time = time.time()
    sol = solvers.qp(P, q, G, h, A, b)
    running_time = time.time() - start_time
    x = np.array(sol['x']).flatten()
    iters = sol['iterations'] if 'iterations' in sol else 0
    print_solver_results("CVXOPT", x, iters, running_time, H, g, A_eq, b_eq, A_ineq, b_ineq)
    return x, iters, running_time


def solve_with_scipy(H, g, A_eq, b_eq, A_ineq, b_ineq):
    """Solve using SciPy's SLSQP solver."""
    n = H.shape[0]
    x0 = np.zeros(n)
    constraints_scipy = [
        {'type': 'eq', 'fun': lambda x: A_eq @ x + b_eq},
        {'type': 'ineq', 'fun': lambda x: -A_ineq @ x - b_ineq}
    ]
    start = time.time()
    result_scipy = minimize(lambda x: quadratic_objective(H,g,x), x0, constraints=constraints_scipy)
    running_time = time.time() - start
    x = result_scipy.x
    iters = result_scipy.nit
    print_solver_results("SciPy", x, iters, running_time, H, g, A_eq, b_eq, A_ineq, b_ineq)
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
    val_penalty = penalized_quadratic_objective(H, g, x, A_eq, b_eq, A_ineq, b_ineq)
    val_primal = quadratic_objective(H, g, x)
    penalty = val_penalty - val_primal
    
    # Calculate constraint violations
    eq_violation = np.linalg.norm(A_eq @ x + b_eq)
    ineq_violation = np.linalg.norm(np.maximum(A_ineq @ x + b_ineq, 0))
    
    print(f"{solver_name:<8} {val_penalty:14.6f} {val_primal:14.6f} "
          f"{penalty:12.3e} {eq_violation:12.3e} {ineq_violation:12.3e} "
          f"{iters:8} {running_time:10.4f}")

def compare_solvers(H, g, A_eq, b_eq, A_ineq, b_ineq):
    """Run and compare all available solvers."""
    print("\nSOLVER COMPARISON")
    print(f"{'Solver':<8} {'Val(penalty)':>14} {'Val(primal)':>14} {'Penalty':>12} "
          f"{'Eq Viol':>12} {'Ineq Viol':>12} {'Iters':>8} {'Time(s)':>10}")
    print("-"*100)
    
    # List of all solver functions
    solvers = [
        solve_with_irwa,
        solve_with_adal,
        solve_with_osqp,
        solve_with_cvxpy,
        solve_with_cvxopt,
        solve_with_scipy
    ]
    
    for solver in solvers:
        solver(H, g, A_eq, b_eq, A_ineq, b_ineq)


if __name__ == "__main__":
    # Problem parameters
    scale = 1
    n = 1000 * scale  # number of variables
    m1 = 300 * scale  # number of equality constraints
    m2 = 300 * scale  # number of inequality constraints

    # Generate problem data
    data = generate_optimization_data(n=n, m1=m1, m2=m2, numpy_output=True, torch_output=False)
    
    # Extract problem matrices
    H, g = data['numpy']['H'], data['numpy']['g']
    A_eq, b_eq = data['numpy']['A_eq'], data['numpy']['b_eq']
    A_ineq, b_ineq = data['numpy']['A_ineq'], data['numpy']['b_ineq']

    # Run comparison
    compare_solvers(H, g, A_eq, b_eq, A_ineq, b_ineq)