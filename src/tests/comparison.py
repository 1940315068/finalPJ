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
    start = time.time()
    x, iters, _, _ = irwa.irwa_solver(H, g, A_eq, b_eq, A_ineq, b_ineq, verbose=False)
    end = time.time()
    running_time = end - start
    return x, iters, running_time
    

def solve_with_adal(H, g, A_eq, b_eq, A_ineq, b_ineq):
    start = time.time()
    x, iters, _, _ = adal.adal_solver(H, g, A_eq, b_eq, A_ineq, b_ineq, verbose=False)
    end = time.time()
    running_time = end - start
    return x, iters, running_time


def solve_with_osqp(H, g, A_eq, b_eq, A_ineq, b_ineq):
    P = csc_matrix(H)
    A = csc_matrix(np.vstack([A_eq, A_ineq]))
    l = np.hstack([-b_eq, -np.inf * np.ones(len(b_ineq))])
    u = np.hstack([-b_eq, -b_ineq])
    start = time.time()
    prob_osqp = osqp.OSQP()
    prob_osqp.setup(P=P, q=g, A=A, l=l, u=u, verbose=False)
    result_osqp = prob_osqp.solve()
    end = time.time()
    running_time = end - start
    x = result_osqp.x
    iters = result_osqp.info.iter
    return x, iters, running_time


def solve_with_cvxpy(H, g, A_eq, b_eq, A_ineq, b_ineq):
    n = H.shape[0]
    x_cvxpy = cp.Variable(n)
    obj_cvxpy = cp.Minimize(0.5 * cp.quad_form(x_cvxpy, H) + g.T @ x_cvxpy)
    constraints_cvxpy = [A_eq @ x_cvxpy == -b_eq, A_ineq @ x_cvxpy <= -b_ineq]
    start = time.time()
    prob_cvxpy = cp.Problem(obj_cvxpy, constraints_cvxpy)
    prob_cvxpy.solve() 
    end = time.time()
    running_time = end - start
    x = x_cvxpy.value
    iters = prob_cvxpy.solver_stats.num_iters
    return x, iters, running_time


def solve_with_cvxopt(H, g, A_eq, b_eq, A_ineq, b_ineq):
    solvers.options['show_progress'] = False
    P = matrix(H)
    q = matrix(g)
    G = matrix(A_ineq)
    h = matrix(-b_ineq)
    A = matrix(A_eq)
    b = matrix(-b_eq)
    start_time = time.time()
    sol = solvers.qp(P, q, G, h, A, b)
    end_time = time.time()
    x = np.array(sol['x']).flatten()
    running_time = end_time - start_time
    iters = sol['iterations'] if 'iterations' in sol else 0
    return x, iters, running_time


def solve_with_scipy(H, g, A_eq, b_eq, A_ineq, b_ineq):
    n = H.shape[0]
    x0 = np.zeros(n)
    constraints_scipy = [
            {'type': 'eq', 'fun': lambda x: A_eq @ x + b_eq},
            {'type': 'ineq', 'fun': lambda x: -A_ineq @ x - b_ineq}
        ]
    start = time.time()
    result_scipy = minimize(lambda x: quadratic_objective(H,g,x), x0, method='SLSQP', constraints=constraints_scipy)
    end = time.time()
    running_time = end - start
    x = result_scipy.x
    iters = result_scipy.nit
    return x, iters, running_time


if __name__ == "__main__":
    scale = 1
    n = 1000*scale  # number of variables
    m1 = 300*scale  # number of equality constraints
    m2 = 300*scale  # number of inequality constraints

    # Generate problem data
    data = generate_optimization_data(n=n, m1=m1, m2=m2, numpy_output=True, torch_output=False)
            
    # CPU (NumPy) data
    H, g = data['numpy']['H'], data['numpy']['g']
    A_eq, b_eq = data['numpy']['A_eq'], data['numpy']['b_eq']
    A_ineq, b_ineq = data['numpy']['A_ineq'], data['numpy']['b_ineq']


    x_irwa, iters_irwa, time_irwa = solve_with_irwa(H, g, A_eq, b_eq, A_ineq, b_ineq)
    x_adal, iters_adal, time_adal = solve_with_adal(H, g, A_eq, b_eq, A_ineq, b_ineq)
    x_osqp, iters_osqp, time_osqp = solve_with_osqp(H, g, A_eq, b_eq, A_ineq, b_ineq)
    x_cvxpy, iters_cvxpy, time_cvxpy = solve_with_cvxpy(H, g, A_eq, b_eq, A_ineq, b_ineq)
    x_cvxopt, iters_cvxopt, time_cvxopt = solve_with_cvxopt(H, g, A_eq, b_eq, A_ineq, b_ineq)
    x_scipy, iters_scipy, time_scipy = solve_with_scipy(H, g, A_eq, b_eq, A_ineq, b_ineq)

    # compute function value with/without penalty
    val_penalty_irwa = penalized_quadratic_objective(H, g, x_irwa, A_eq, b_eq, A_ineq, b_ineq)
    val_penalty_adal = penalized_quadratic_objective(H, g, x_adal, A_eq, b_eq, A_ineq, b_ineq)
    val_penalty_osqp = penalized_quadratic_objective(H, g, x_osqp, A_eq, b_eq, A_ineq, b_ineq)
    val_penalty_cvxpy = penalized_quadratic_objective(H, g, x_cvxpy, A_eq, b_eq, A_ineq, b_ineq)
    val_penalty_cvxopt = penalized_quadratic_objective(H, g, x_cvxopt, A_eq, b_eq, A_ineq, b_ineq)
    val_penalty_scipy = penalized_quadratic_objective(H, g, x_scipy, A_eq, b_eq, A_ineq, b_ineq)

    val_primal_irwa = quadratic_objective(H, g, x_irwa)
    val_primal_adal = quadratic_objective(H, g, x_adal)
    val_primal_osqp = quadratic_objective(H, g, x_osqp)
    val_primal_cvxpy = quadratic_objective(H, g, x_cvxpy)
    val_primal_cvxopt = quadratic_objective(H, g, x_cvxopt)
    val_primal_scipy = quadratic_objective(H, g, x_scipy)

    penalty_irwa = val_penalty_irwa - val_primal_irwa
    penalty_adal = val_penalty_adal - val_primal_adal
    penalty_osqp = val_penalty_osqp - val_primal_osqp
    penalty_cvxpy = val_penalty_cvxpy - val_primal_cvxpy
    penalty_cvxopt = val_penalty_cvxopt - val_primal_cvxopt
    penalty_scipy = val_penalty_scipy - val_primal_scipy


    # show the comparison
    print("------------------------------------------------------------")
    # Prepare all data in lists
    solvers = ["IRWA", "ADAL", "OSQP", "CVXPY", "CVXOPT", "SciPy"]
    val_penalties = [val_penalty_irwa, val_penalty_adal, val_penalty_osqp,
                     val_penalty_cvxpy,  val_penalty_cvxopt, val_penalty_scipy]
    val_primaries = [val_primal_irwa, val_primal_adal, val_primal_osqp,
                     val_primal_cvxpy, val_primal_cvxopt, val_primal_scipy]
    penalties = [penalty_irwa, penalty_adal, penalty_osqp,
                 penalty_cvxpy, penalty_cvxopt, penalty_scipy]
    iterations = [iters_irwa, iters_adal, iters_osqp, iters_cvxpy, iters_cvxopt, iters_scipy]
    times = [time_irwa, time_adal, time_osqp, time_cvxpy, time_cvxopt, time_scipy]

    # Print header
    print("\nSOLVER COMPARISON")
    print(f"{'Solver':<8} {'Val(penalty)':>14} {'Val(primal)':>14} {'Penalty':>12} {'Iters':>8} {'Time(s)':>10}")
    print("-"*70)

    # Print each solver's data
    for i in range(len(solvers)):
        print(f"{solvers[i]:<8} {val_penalties[i]:14.6f} {val_primaries[i]:14.6f} "
            f"{penalties[i]:12.3e} {iterations[i]:8} {times[i]:10.4f}")

    # Find best solver
    # best_idx = min(range(len(val_penalties)), itersey=lambda i: val_penalties[i])
    # print(f"\nBest solver: {solvers[best_idx]} (value: {val_penalties[best_idx]:.6f})")