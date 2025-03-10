import torch
from functions import *
from cg_torch import cg_torch


def irwa_solver(H, g, A_eq, b_eq, A_ineq, b_ineq, x0=None, max_iter=1000):
    print("========================================")
    print("----------------- IRWA -----------------")

    # Get the length from input
    n = g.size(0)  # number of variables
    m1 = b_eq.size(0)  # number of equality constraints
    m2 = b_ineq.size(0)  # number of inequality constraints
    m = m1 + m2  # total number of constraints

    # Hyper parameters:
    epsilon0 = 2000 * torch.ones(m)  # relaxation vector, positive
    eta = 0.8  # scaling parameter, in (0,1)
    gamma = 0.15  # scaling parameter, > 0
    M = 10  # scaling parameter, > 0
    sigma = 1e-7  # termination tolerance
    sigma_prime = 1e-7  # termination tolerance

    # Set x0 as zeros if not provided
    if x0 is None:
        x0 = torch.zeros(n)

    # Initialization
    x = x0
    epsilon = epsilon0
    A = torch.vstack([A_eq, A_ineq])
    b = torch.hstack([b_eq, b_ineq])

    # Iteration
    for k in range(max_iter):
        # Step 1. Solve the reweighted subproblem for x^(k+1)
        W = compute_W(x, epsilon, A, b, m1, m2)
        v = compute_v(x, A, b, m1, m2)

        # Use conjugate gradient to solve the linear system
        coeff_matrix = H + torch.matmul(A.T, torch.matmul(W, A))
        rhs = -(g + torch.matmul(A.T, torch.matmul(W, v)))
        x_next = cg_torch(coeff_matrix, rhs, x0=x, max_iter=n*2)

        # Step 2. Set the new relaxation vector epsilon^(k+1)
        q = torch.matmul(A, x_next - x)
        # set r, r[i] = A[i]x+b[i] for i in I1, r[i] = max(A[i]x+b[i], 0) for i in I2
        z = torch.hstack([-9999*torch.ones(m1), torch.zeros(m2)])
        Ax_plus_b = torch.matmul(A, x) + b
        r = torch.max(Ax_plus_b, z)

        if torch.all(torch.abs(q) <= M * (r**2 + epsilon**2)**(0.5 + gamma)):
            epsilon_next = epsilon * eta

        # Step 3. Check stopping criteria
        norm_dx = torch.norm(x_next - x)
        norm_eps = torch.norm(epsilon)
        if norm_dx <= sigma and norm_eps <= sigma_prime:
            print(f"Iteration ends at {k} times.")
            # Show the current function value with/without penalty
            val = quadratic_form(H, g, x)
            val_penalty = exact_penalty_func(H, g, x, A_eq, b_eq, A_ineq, b_ineq)
            print(f"Function value without penalty : {val}")
            print(f"Function value with penalty :    {val_penalty}")
            # Show the current norm of dx and epsilon
            print(f"Current norm of dx: {norm_dx}")
            print(f"Current norm of epsilon: {norm_eps}")
            print("========================================")
            return x, k

        x = x_next
        epsilon = epsilon_next
        # Output the function value every 100 iterations
        if k % 100 == 0:
            print(f"Iteration {k}:")
            # Show the current function value with/without penalty
            val = quadratic_form(H, g, x)
            val_penalty = exact_penalty_func(H, g, x, A_eq, b_eq, A_ineq, b_ineq)
            print(f"Function value without penalty : {val}")
            print(f"Function value with penalty :    {val_penalty}")
            # Show the current norm of dx and epsilon
            print(f"Current norm of dx: {norm_dx}")
            print(f"Current norm of epsilon: {norm_eps}")
            print()

    print(f"No converge in {k} iterations.")
    print("========================================")
    return x, k


def compute_W(x, epsilon, A, b, m1: int, m2: int):
    m = b.size(0)  # total number of constraints
    assert m == m1 + m2, f"Error: m (total constraints) should be equal to m1 + m2. Got m={m}, m1={m1}, m2={m2}"

    # diagonal_elements = torch.zeros(m)
    # # Inequality constraints
    # for i in range(m1):
    #     diagonal_elements[i] = (torch.abs(torch.matmul(A[i], x) + b[i])**2 + epsilon[i]**2)**(-0.5)
    # # Equality constraints
    # for i in range(m1, m):
    #     diagonal_elements[i] = (torch.max(torch.matmul(A[i], x) + b[i], torch.tensor(0.0))**2 + epsilon[i]**2)**(-0.5)
    
    # Compute A x + b for all constraints
    Ax_plus_b = torch.matmul(A, x) + b
    # Inequality constraints (first m1 constraints)
    ineq_part = torch.abs(Ax_plus_b[:m1])**2 + epsilon[:m1]**2
    # Equality constraints (remaining m2 constraints)
    eq_part = torch.max(Ax_plus_b[m1:], torch.tensor(0.0))**2 + epsilon[m1:]**2
    # Combine the results
    diagonal_elements = torch.cat([ineq_part, eq_part])**(-0.5)
    # Generate the diagonal matrix W
    W = torch.diag(diagonal_elements)
    return W


def compute_v(x_tilde, A, b, m1: int, m2: int):
    m = b.size(0)  # total number of constraints
    assert m == m1 + m2, f"Error: m (total constraints) should be equal to m1 + m2. Got m={m}, m1={m1}, m2={m2}"
    # v = torch.zeros(m)
    # # Inequality constraints
    # for i in range(m1):
    #     v[i] = b[i]
    # # Equality constraints
    # for i in range(m1, m):
    #     v[i] = b[i] - torch.min(torch.matmul(A[i], x_tilde) + b[i], torch.tensor(0.0))

    # Compute A @ x_tilde + b for all constraints
    Ax_plus_b = torch.matmul(A, x_tilde) + b
    # Inequality constraints (first m1 constraints)
    ineq_part = b[:m1]
    # Equality constraints (remaining m2 constraints)
    eq_part = b[m1:] - torch.min(Ax_plus_b[m1:], torch.tensor(0.0))
    # Combine the results
    v = torch.cat([ineq_part, eq_part])
    return v