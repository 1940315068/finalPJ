import torch
from functions import quadratic_form, exact_penalty_func
from cg_torch import cg_torch

def adal_solver(H, g, A_eq, b_eq, A_ineq, b_ineq, x0=None, max_iter=1000):
    """
    ADAL solver for constrained optimization on GPU.

    Parameters:
        H (torch.Tensor): Symmetric positive definite matrix (n x n).
        g (torch.Tensor): Vector (n).
        A_eq (torch.Tensor): Equality constraint matrix (m1 x n).
        b_eq (torch.Tensor): Equality constraint vector (m1).
        A_ineq (torch.Tensor): Inequality constraint matrix (m2 x n).
        b_ineq (torch.Tensor): Inequality constraint vector (m2).
        x0 (torch.Tensor, optional): Initial guess for the solution. Defaults to zero vector.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        device (torch.device, optional): Device to run the computation on (e.g., "cuda" or "cpu").

    Returns:
        x (torch.Tensor): Solution to the optimization problem.
        k (int): Number of iterations performed.
    """

    print("========================================")
    print("----------------- ADAL -----------------")

    # Get the size from input
    n = g.size(0)  # number of variables
    m1 = b_eq.size(0)  # number of equality constraints
    m2 = b_ineq.size(0)  # number of inequality constraints
    m = m1 + m2  # total number of constraints

    # Hyper parameters:
    u0 = torch.zeros(m)  # relaxation vector, non-negative
    mu = 0.001  # penalty parameter, > 0
    sigma = 1e-6  # termination tolerance
    sigma_prime = 1e-6  # termination tolerance

    # Set x0 as zeros if not provided
    if x0 is None:
        x0 = torch.zeros(n)

    # Initialization
    x = x0.clone()
    u = u0
    A = torch.vstack([A_eq, A_ineq])
    b = torch.hstack([b_eq, b_ineq])
    k = 0

    for k in range(max_iter):
        # Step 1: Solve the augmented Lagrangian subproblem for x^(k+1) and p^(k+1)

        # Solve for p^(k+1)
        # Compute s = A @ x + b + 1 / mu * u
        s = torch.matmul(A, x) + b + 1 / mu * u
        # Equality constraints (first m1 constraints)
        s_eq = s[:m1]
        p_eq = torch.sign(s_eq) * torch.max(torch.abs(s_eq) - 1 / mu, torch.tensor(0.0))
        # Inequality constraints (remaining m2 constraints)
        s_ineq = s[m1:]
        p_ineq = torch.max(s_ineq - 1 / mu, torch.tensor(0.0)) - torch.max(-s_ineq, torch.tensor(0.0))
        # Combine the results
        p_next = torch.cat([p_eq, p_ineq])

        # Solve for x^(k+1)
        coeff_matrix = H + mu * (torch.matmul(A.T, A))
        rhs = -(g + torch.matmul(A.T, u) + mu * torch.matmul(A.T, b - p_next))
        x_next = cg_torch(coeff_matrix, rhs, x0=x, max_iter=n*2)  # Conjugate Gradient, on GPU

        # Step 2: Set the new multiplier u^(k+1)
        residual = torch.matmul(A, x_next) + b - p_next
        u_next = u + mu * residual

        # Step 3: Check the stopping criterion
        norm_dx = torch.norm(x_next - x).item()
        norm_residual = torch.max(torch.abs(residual)).item()
        if norm_dx <= sigma and norm_residual <= sigma_prime:
            print(f"Iteration ends at {k} times.")
            # show the current function value with/without penalty
            val = quadratic_form(H, g, x)
            val_penalty = exact_penalty_func(H,g,x,A_eq,b_eq,A_ineq,b_ineq)
            print(f"Function value without penalty : {val}")
            print(f"Function value with penalty :    {val_penalty}")
            # show the current norm of dx and residual
            print(f"Current norm of dx: {norm_dx}")
            print(f"Current norm* of residual Ax + b - p : {norm_residual}")
            print("========================================")
            return x, k

        x = x_next
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
            ATy = torch.matmul(A.T, y)
            norm_ATy = torch.norm(ATy)
            if norm_ATy < 1e-6:
                support_func_eq = -torch.sum(b[:m1] * y[:m1])
                support_func_ineq = torch.sum(torch.where(y[m1:] >= 0, -b[m1:] * y[m1:], 9999))  # 9999 as +infty
                if support_func_eq < 0:
                    # print(f"Iteration ends at {k} times: Primal infeasible of equality constraints!")
                    # return x,k
                    raise ValueError(f"Iteration ends at {k} times: Primal infeasible!")
                elif support_func_ineq < 0:
                    print(f"Iteration ends at {k} times: Primal infeasible of inequality constraints!")
                    return x,k
                    # raise ValueError(f"Iteration ends at {k} times: Primal infeasible of inequality constraints!")
                    
        

    print(f"No converge! Iteration ends at {k} times. ")
    print("========================================")
    return x, k