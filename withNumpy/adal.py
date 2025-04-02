import numpy as np
from functions import *
from cg import cg
import time


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
    mu = min(0.1, 1/n)  # penalty parameter, > 0 
    sigma = 1e-6  # termination tolerance 
    sigma_prime = 1e-6  # termination tolerance 

    # Set x0 as zeros if not provided 
    if x0 is None:
        x0 = np.zeros(n)

    # Initialization
    x = x0
    u = u0
    A = np.vstack([A_eq, A_ineq])
    b = np.hstack([b_eq, b_ineq])
    k = 0
    n_cg_steps = 0  # number of cg steps in total
    time_cg = 0
    
    # Define a function to compute (H+mu*ATA)p
    def matvec(p):
        return (H @ p + mu * (A.T @ (A @ p)))
    
    for k in range(max_iter):
        # Step 1: Solve the augmented Lagrangian subproblem for x^(k+1) and p^(k+1)
        
        # Solve for p^(k+1), set the values of p[i] explicitly
        # Compute s = A @ x + b + 1/mu * u
        s = A @ x + b + 1/mu * u
        # Equality constraints (first m1 constraints)
        s_eq = s[:m1]
        p_eq = np.sign(s_eq) * np.maximum(np.abs(s_eq) - 1/mu, 0)
        # Inequality constraints (remaining m2 constraints)
        s_ineq = s[m1:]
        p_ineq = np.maximum(s_ineq - 1/mu, 0) - np.maximum(-s_ineq, 0)
        # Combine the results
        p_next = np.concatenate([p_eq, p_ineq])
        
        # Solve for x^(k+1), use conjugate gradient to solve the linear system
        rhs = - (g + np.matmul(A.T, u) + mu * np.matmul(A.T, b - p_next))
        cg_start_time = time.time()
        maxiter = max(10, n//10)
        x_next, cg_steps = cg(matvec, rhs, maxiter=maxiter, x0=x, rtol=1e-1) 
        # x_next = inv_matrix @ rhs; cg_steps = 0  # directly scompute the solution with the inverse matrix
        cg_end_time = time.time()
        time_cg += (cg_end_time - cg_start_time)
        n_cg_steps += cg_steps
        
        # Step 2: Set the new multiplier u^(k+1)
        residual = A @ x_next + b - p_next
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
            return x, k, n_cg_steps, time_cg

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
        
        # Check primal infeasibility by computing the support function of C at du
        if k % 20 == 0:
            y = mu * residual  # y=du
            ATy = np.matmul(A.T, y)
            norm_ATy = np.linalg.norm(ATy)
            if norm_ATy < 1e-6:
                support_func_eq = -np.sum(b[:m1] * y[:m1])
                support_func_ineq = np.sum(np.where(y[m1:] >= 0, -b[m1:] * y[m1:], 9999))  # 9999 as +infty
                if support_func_eq < 0:
                    # print(f"Iteration ends at {k} times: Primal infeasible of equality constraints!")
                    # return x,k
                    raise ValueError(f"Iteration ends at {k} times: Primal infeasible!")
                elif support_func_ineq < 0:
                    # print(f"Iteration ends at {k} times: Primal infeasible of inequality constraints!")
                    # return x,k
                    raise ValueError(f"Iteration ends at {k} times: Primal infeasible of inequality constraints!")
                    
            

    print(f"No converge in {k} iterations.")
    print("========================================")
    return x, k, n_cg_steps, time_cg


