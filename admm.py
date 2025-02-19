import numpy as np
from scipy.sparse.linalg import cg  # Conjugate Gradient solver
from scipy.linalg import norm


def admm_solver(H, g, A_eq, b_eq, A_ineq, b_ineq, x0=None, max_iter=1000):
    print("========================================")
    print("----------------- ADMM -----------------")
    # Get the length from imput
    n = len(g)  # number of variables
    m1 = len(b_eq)  # number of inequality constraints
    m2 = len(b_ineq)  # number of equality constraints
    m = m1+m2  # total number of constraints

    # Hyper parameters:
    u0 = np.ones(m)  # relaxation vector, positive  
    mu = 10  # penalty parameter, > 0 
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
    p = np.dot(A, x) + b

    for k in range(max_iter):
        # Step 1: Solve the augmented Lagrangian subproblem for x^(k+1) and p^(k+1)
        
        # Use conjugate gradient to solve the linear system
        coeff_matrix = H + mu * (np.dot(A.T, A))
        rhs = - (g + np.dot(A.T, u) + mu * np.dot(A.T, b - p))
        x_new, info = cg(coeff_matrix, rhs, maxiter=100, x0=x)  # Conjugate Gradient
        if info != 0:
            raise RuntimeError("CG not converge!")

        # Then, solve for p^(k+1)
        tmp = np.dot(A, x_new) + b
        p_new = (1 / (2 + mu)) * (u + mu * tmp)
        
        # Apply the constraints to p
        for i in range(m1, m):
            p1 = max(0, (1 / (2 + mu)) * (u[i] + mu * tmp[i]))
            f1 = max(p1, 0) ** 2 - u[i] * p1 + mu / 2 * (tmp[i] ** 2 + p1 ** 2 - 2 * tmp[i] * p1)
            p2 = min(0, (1 / mu) * (u[i] + mu * tmp[i]))
            f2 = max(p2, 0) ** 2 - u[i] * p2 + mu / 2 * (tmp[i] ** 2 + p2 ** 2 - 2 * tmp[i] * p2)
            if f1 > f2:
                p_new[i] = p2
            else:
                p_new[i] = p1
        
        # Step 2: Set the new multiplier u^(k+1)
        residual = np.dot(A, x_new) + b - p_new
        u_new = u + mu * residual

        # Step 3: Check the stopping criterion
        if norm(x_new - x) <= sigma and np.max(np.abs(residual)) <= sigma_prime:
            print(f"Iteration ends at {k} times.")
            val = quadratic_form(H,g,x)
            print(f"Function value = {val}")
            print("========================================")
            return x, k

        x = x_new
        p = p_new
        u = u_new

        # Output the function value every 100 iterations
        if k % 100 == 0:
            # Calculate the function value here 
            func_value = quadratic_form(H, g, x) 
            # func_value_penalty = obj_func_penalty(x, g, H, A, b, l, m)
            print(f"Iteration {k}")
            print(f"Function value = {func_value}")
            # print(f"Function value with penalty = {func_value_penalty}")
            print(f"Current residual Ax + b - p = {norm(residual)}")
            print()

    print(f"No converge! Iteration ends at {k} times. ")
    print("========================================")
    return x, k


# Compute the quadratic form: g^T x + 1/2 x^T H x
def quadratic_form(H, g, x) -> float:
    gTx = np.dot(g.T, x)
    xTHx = np.dot(x.T, np.dot(H, x))
    return gTx + 0.5 * xTHx

