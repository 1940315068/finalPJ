import torch
from ..functions import *
from ..cg import cg
import time


def irwa_solver(H, g, A_eq, b_eq, A_ineq, b_ineq, x0=None, max_iter=1000):
    print("========================================")
    print("----------------- IRWA -----------------")

    # Get the length from input
    n = g.size(0)  # number of variables
    m1 = b_eq.size(0)  # number of equality constraints
    m2 = b_ineq.size(0)  # number of inequality constraints
    m = m1 + m2  # total number of constraints

    # Hyper parameters:
    epsilon0 = 20000 * torch.ones(m)  # relaxation vector, positive
    eta = 0.8  # scaling parameter, in (0,1)
    gamma = 0.15  # scaling parameter, > 0
    M = 100  # scaling parameter, > 0
    sigma = 1e-6  # termination tolerance
    sigma_prime = 1e-6  # termination tolerance

    # Set x0 as zeros if not provided
    if x0 is None:
        x0 = torch.zeros(n)

    # Initialization
    x = x0.clone()
    epsilon = epsilon0
    A = torch.vstack([A_eq, A_ineq])
    b = torch.hstack([b_eq, b_ineq])
    n_cg_steps = 0  # number of cg steps in total
    time_cg = 0

    # Iteration
    for k in range(1, max_iter+1):
        # Step 1. Solve the reweighted subproblem for x^(k+1)
        w = compute_w(x, epsilon, A, b, m1, m2)
        v = compute_v(x, A, b, m1, m2)

        # Use conjugate gradient to solve the linear system

        # Define a function to compute (H+ATWA)p
        def matvec(p):
            return (H @ p + (A.T @ (w * (A @ p))))
        
        rhs = - (g + A.T @ (w * v))
        cg_start_time = time.time()
        maxiter = max(10, n//10)
        x_next, cg_steps = cg(matvec, rhs, maxiter=maxiter, x0=x, rtol=1e-1) 
        cg_end_time = time.time()
        time_cg += (cg_end_time - cg_start_time)
        # print(f"CG steps: {cg_steps:03d},  max_w = {max(w):.2e},  min_w = {min(w):.2e},  condition number of ATWA: {np.linalg.cond(coeff_matrix):.2e}")
        n_cg_steps += cg_steps

        # Step 2. Set the new relaxation vector epsilon^(k+1)
        q = torch.matmul(A, x_next - x)
        # set r, r[i] = A[i]x+b[i] for i in I1, r[i] = max(A[i]x+b[i], 0) for i in I2
        z = torch.hstack([-9999*torch.ones(m1), torch.zeros(m2)])
        Ax_plus_b = torch.matmul(A, x) + b
        r = torch.max(Ax_plus_b, z)

        if torch.all(torch.abs(q) <= M * (r**2 + epsilon**2)**(0.5 + gamma)):
            epsilon_next = epsilon * eta
            # Keep the original epsilon value for the satisfied constraints
            cond1 = (torch.arange(len(Ax_plus_b)) < m1) & torch.isclose(Ax_plus_b, torch.tensor(0.0), atol=1e-6)  # i < m1 and (Ax+b)[i] == 0
            cond2 = (torch.arange(len(Ax_plus_b)) >= m1) & (Ax_plus_b <= 0)  # i >= m1 and (Ax+b)[i] <= 0
            indices = torch.where(cond1 | cond2)[0]
            epsilon_next[indices] = epsilon[indices]

        # Step 3. Check stopping criteria
        norm_dx = torch.norm(x_next - x)
        norm_eps = torch.norm(epsilon)
        if norm_dx <= sigma:  # and norm_eps <= sigma_prime:
            print(f"Iteration ends at {k} times.")
            # Show the current function value with/without penalty
            val = quadratic_objective(H, g, x)
            val_penalty = penalized_quadratic_objective(H, g, x, A_eq, b_eq, A_ineq, b_ineq)
            print(f"Function value without penalty : {val}")
            print(f"Function value with penalty :    {val_penalty}")
            # Show the current norm of dx and epsilon
            print(f"Current norm of dx: {norm_dx}")
            print(f"Current norm of epsilon: {norm_eps}")
            print("========================================")
            return x, k, n_cg_steps, time_cg

        x = x_next
        epsilon = epsilon_next
        # Output the function value every 100 iterations
        if k % 100 == 0:
            print(f"Iteration {k}:")
            # Show the current function value with/without penalty
            val = quadratic_objective(H, g, x)
            val_penalty = penalized_quadratic_objective(H, g, x, A_eq, b_eq, A_ineq, b_ineq)
            print(f"Function value without penalty : {val}")
            print(f"Function value with penalty :    {val_penalty}")
            # Show the current norm of dx and epsilon
            print(f"Current norm of dx: {norm_dx}")
            print(f"Current norm of epsilon: {norm_eps}")
            print()

    print(f"No converge in {k} iterations.")
    print("========================================")
    return x, k, n_cg_steps, time_cg


def compute_w(x, epsilon, A, b, m1: int, m2: int):
    m = b.size(0)  # total number of constraints
    assert m == m1 + m2, f"Error: m (total constraints) should be equal to m1 + m2. Got m={m}, m1={m1}, m2={m2}"\
    # Compute A @ x + b for all constraints
    Ax_plus_b = A @ x + b
    # Equality constraints (first m1 constraints)
    eq_part = torch.abs(Ax_plus_b[:m1])**2 + epsilon[:m1]**2
    # Inequality constraints (remaining m2 constraints)
    ineq_part = torch.max(Ax_plus_b[m1:], torch.tensor(0.0))**2 + epsilon[m1:]**2
    # Combine the results as the diagonal elements
    w = torch.concatenate([eq_part, ineq_part])**(-0.5)
    return w


def compute_v(x_tilde, A, b, m1: int, m2: int):
    m = b.size(0)  # total number of constraints
    assert m == m1 + m2, f"Error: m (total constraints) should be equal to m1 + m2. Got m={m}, m1={m1}, m2={m2}"
    # Compute A @ x_tilde + b for all constraints
    Ax_plus_b = torch.matmul(A, x_tilde) + b
    # Inequality constraints (first m1 constraints)
    ineq_part = b[:m1]
    # Equality constraints (remaining m2 constraints)
    eq_part = b[m1:] - torch.min(Ax_plus_b[m1:], torch.tensor(0.0))
    # Combine the results
    v = torch.cat([ineq_part, eq_part])
    return v