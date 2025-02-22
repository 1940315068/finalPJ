import numpy as np
from scipy.sparse.linalg import cg  # Conjugate Gradient solver
from scipy.linalg import norm


def adal_solver(H, g, A_eq, b_eq, A_ineq, b_ineq, x0=None, max_iter=1000):
    print("========================================")
    print("----------------- ADAL -----------------")
    # Get the length from imput
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
    # p = np.dot(A, x) + b

    for k in range(max_iter):
        # Step 1: Solve the augmented Lagrangian subproblem for x^(k+1) and p^(k+1)
        
        # Solve for p^(k+1)
        p_next = np.zeros(m)
        s = np.dot(A, x) + b + 1/mu*u
        # Equality constraints
        for i in range(m1):
            p_next[i] = np.sign(s[i]) * max(np.abs(s[i]) - 1/mu, 0)
        # Inequality constraints
        for i in range(m1, m):
            p_next[i] = max(s[i] - 1/mu, 0) - max(-s[i], 0)
        
        # Solve for x^(k+1)
        coeff_matrix = H + mu * (np.dot(A.T, A))
        rhs = - (g + np.dot(A.T, u) + mu * np.dot(A.T, b - p_next))
        x_next, info = cg(coeff_matrix, rhs, maxiter=100, x0=x)  # Conjugate Gradient
        if info != 0:
            raise RuntimeError("CG not converge!")
        
        # Step 2: Set the new multiplier u^(k+1)
        residual = np.dot(A, x_next) + b - p_next
        u_next = u + mu * residual

        # Step 3: Check the stopping criterion
        norm_dx = norm(x_next - x)
        norm_residual = np.max(np.abs(residual))
        if norm_dx <= sigma and norm_residual <= sigma_prime:
            print(f"Iteration ends at {k} times.")
            val = quadratic_form(H,g,x)
            print(f"Function value = {val}")
            print(f"Current norm* of residual Ax + b - p = {norm_residual}")
            print(f"Current norm of dx: {norm_dx}")
            print("========================================")
            return x, k

        x = x_next
        # p = p_next
        u = u_next

        # Output the function value every 100 iterations
        if k % 100 == 0:
            # Calculate the function value here 
            func_value = quadratic_form(H, g, x) 
            # func_value_penalty = obj_func_penalty(x, g, H, A, b, l, m)
            print(f"Iteration {k}")
            print(f"Function value = {func_value}")
            # print(f"Function value with penalty = {func_value_penalty}")
            print(f"Current norm* of residual Ax + b - p = {norm_residual}")
            print(f"Current norm of dx: {norm_dx}")
            print()

    print(f"No converge! Iteration ends at {k} times. ")
    print("========================================")
    return x, k


# Compute the quadratic form: g^T x + 1/2 x^T H x
def quadratic_form(H, g, x) -> float:
    gTx = np.dot(g.T, x)
    xTHx = np.dot(x.T, np.dot(H, x))
    return gTx + 0.5 * xTHx

