import numpy as np
from functions import *
from cg import cg


def adal_solver(H, g, A_eq, b_eq, A_ineq, b_ineq, x0=None, max_iter=1000):
    print("========================================")
    print("----------------- ADAL -----------------")

    # Get the length from input
    n = len(g)  # number of variables
    m1 = len(b_eq)  # number of inequality constraints
    m2 = len(b_ineq)  # number of equality constraints
    m = m1+m2  # total number of constraints

    # Hyper parameters:
    u0 = np.zeros(m)  # relaxation vector, non-negative  
    mu = 0.01  # penalty parameter, > 0 
    sigma = 1e-7  # termination tolerance 
    sigma_prime = 1e-7  # termination tolerance 

    # Set x0 as zeros if not provided 
    if x0 is None:
        x0 = np.zeros(n)

    # Initialization
    x = x0
    u = u0
    A = np.vstack([A_eq, A_ineq])
    b = np.hstack([b_eq, b_ineq])
    k = 0
    # Check condition number
    mat = H + mu * (np.dot(A.T, A))
    print(f"Condition number (2-norm) in solving x_adal: {np.linalg.cond(mat)}")
    print(f"Condition number (1-norm) in solving x_adal: {np.linalg.cond(mat, 1)}")
    print(f"Condition number (inf-norm) in solving x_adal: {np.linalg.cond(mat, np.inf)}")

    for k in range(max_iter):
        # Step 1: Solve the augmented Lagrangian subproblem for x^(k+1) and p^(k+1)
        
        # Solve for p^(k+1), set the values of p[i] explicitly
        p_next = np.zeros(m)
        s = np.dot(A, x) + b + 1/mu*u
        # Equality constraints
        for i in range(m1):
            p_next[i] = np.sign(s[i]) * max(np.abs(s[i]) - 1/mu, 0)
        # Inequality constraints
        for i in range(m1, m):
            p_next[i] = max(s[i] - 1/mu, 0) - max(-s[i], 0)
        
        # Solve for x^(k+1), use conjugate gradient to solve the linear system
        coeff_matrix = H + mu * (np.dot(A.T, A))
        rhs = - (g + np.dot(A.T, u) + mu * np.dot(A.T, b - p_next))
        x_next = cg(coeff_matrix, rhs, maxiter=n*2, x0=x) 
        
        # Step 2: Set the new multiplier u^(k+1)
        residual = np.dot(A, x_next) + b - p_next
        u_next = u + mu * residual

        # Step 3: Check the stopping criterion
        norm_dx = np.linalg.norm(x_next - x)
        norm_residual = np.max(np.abs(residual))
        if norm_dx <= sigma and norm_residual <= sigma_prime:
            print(f"Iteration ends at {k} times.")
            # show the current function value with/without penalty
            val = quadratic_form(H, g, x)
            val_penalty = exact_penalty_func(H,g,x,A_eq,b_eq,A_ineq,b_ineq)
            print(f"Function value without penalty : {val}")
            print(f"Function value with penalty :    {val_penalty}")
            # show the current norm of residual and dx
            print(f"Current norm* of residual Ax + b - p = {norm_residual}")
            print(f"Current norm of dx: {norm_dx}")
            print("========================================")
            return x, k

        x = x_next
        # p = p_next
        u = u_next

        # Output the function value every 100 iterations
        if k % 100 == 0:
            print(f"Iteration {k}:")
            # show the current function value with/without penalty
            val = quadratic_form(H, g, x)
            val_penalty = exact_penalty_func(H,g,x,A_eq,b_eq,A_ineq,b_ineq)
            print(f"Function value without penalty : {val}")
            print(f"Function value with penalty :    {val_penalty}")
            # show the current norm of dx and residual
            print(f"Current norm of dx: {norm_dx}")
            print(f"Current norm* of residual Ax + b - p : {norm_residual}")
            print()

    print(f"No converge in {k} iterations.")
    print("========================================")
    return x, k


