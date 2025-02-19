import numpy as np
from scipy.sparse.linalg import cg


def irwa_solver(H, g, A_eq, b_eq, A_ineq, b_ineq, x0=None, max_iter=1000):

    print("========================================")
    print("----------------- IRWA -----------------")

    # Get the length from imput
    n = len(g)  # number of variables
    m1 = len(b_eq)  # number of equality constraints
    m2 = len(b_ineq)  # number of inequality constraints
    m = m1+m2  # total number of constraints

    # Hyper parameters:
    epsilon0 = 200*np.ones(m)  # relaxation vector, positive  
    eta = 0.7  # scaling parameter, in (0,1) 
    gamma = 0.15  # scaling parameter, > 0 
    M = 10  # scaling parameter, > 0 
    sigma = 1e-7  # termination tolerance 
    sigma_prime = 1e-7  # termination tolerance 

    # Set x0 as zeros if not provided 
    if x0 is None:
        x0 = np.zeros(n)

    # Initialization
    x = x0
    epsilon = epsilon0
    A = np.vstack([A_eq, A_ineq])
    b = np.hstack([b_eq, b_ineq])

    # Iteration
    for k in range(max_iter):
        # Step 1. Solve the reweighted subproblem for x^(k+1) 
        W = compute_W(x, epsilon, A, b, m1, m2)
        v = compute_v(x, A, b, m1, m2)

        # Use conjugate gradient to solve the linear system
        coeff_matrix = H + A.T @ W @ A
        rhs = - (g + A.T @ W @ v)
        x_next, info = cg(coeff_matrix, rhs, maxiter=100, x0=x)
        if info != 0:
            raise RuntimeError("CG not converge!")
        
        # Step 2. Set the new relaxation vector epsilon^(k+1)
        q = np.zeros(m)
        r = np.zeros(m)
        for i in range(m1):
            q[i] = np.dot(A[i], x_next - x)
            r[i] = np.dot(A[i], x) + b[i]
        for i in range(m1, m):
            q[i] = np.dot(A[i], x_next - x)
            r[i] = max(np.dot(A[i], x) + b[i], 0)
        
        if np.all(np.abs(q) <= M*(r**2 + epsilon**2)**(0.5+gamma)):
            epsilon_next = epsilon * eta

        # Step 3. Check stopping criteria
        if np.linalg.norm(x_next-x) <= sigma and np.linalg.norm(epsilon) <= sigma_prime:
            print(f"Iteration ends at {k} times.")
            val = quadratic_form(H,g,x)
            print(f"Function value = {val}")
            print("========================================")
            return x, k
        
        x = x_next
        epsilon = epsilon_next
        # Output the function value every 100 iterations
        if k % 100 == 0:
            # Calculate the function value here 
            func_value = quadratic_form(H, g, x) 
            # func_value_penalty = obj_func_penalty(x, g, H, A, b, l, m)
            print(f"Iteration {k}")
            print(f"Function value = {func_value}")
            # print(f"Function value with penalty = {func_value_penalty}")
            print()
        
    print(f"No converge! Iteration ends at {k} times. ")
    print("========================================")
    return x, k

# Compute the quadratic form: g^T x + 1/2 x^T H x
def quadratic_form(H, g, x) -> float:
    gTx = np.dot(g.T, x)
    xTHx = np.dot(x.T, np.dot(H, x))
    return gTx + 0.5 * xTHx


def compute_W(x, epsilon, A, b, m1:int, m2:int):
    m= len(b)  # total number of constraints
    assert m == m1 + m2, f"Error: m (total constraints) should be equal to m1 + m2. Got m={m}, m1={m1}, m2={m2}"
    diagonal_elements = np.zeros(m)
    # Inequality constraints
    for i in range(m1):
        diagonal_elements[i] = (np.abs(np.dot(A[i], x) + b[i])**2 + epsilon[i]**2)**(-1/2) 
    # Equality constraints
    for i in range(m1, m):
        diagonal_elements[i] = (max(np.dot(A[i], x) + b[i], 0)**2 + epsilon[i]**2)**(-1/2)   
    # Generate the diagonal matrix W
    W = np.diag(diagonal_elements)
    return W


def compute_v(x_tilde, A, b, m1:int, m2:int):
    m = len(b)  # total number of constraints
    assert m == m1 + m2, f"Error: m (total constraints) should be equal to l + m. Got m={m}, m1={m1}, m2={m2}"
    v = np.zeros(m)
    # Inequality constraints
    for i in range(m1):
        v[i] = b[i]
    # Equality constraints
    for i in range(m1, m):
        v[i] = b[i] - min(np.dot(A[i], x_tilde) + b[i], 0)
    return v