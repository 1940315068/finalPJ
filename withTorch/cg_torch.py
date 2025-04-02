import torch

def cg_torch(matvec, b, x0=None, maxiter=None, rtol=1e-6, atol=1e-12):
    """
    Conjugate Gradient Method for solving the linear system Ax = b using PyTorch.

    Parameters:
        A (torch.Tensor): Symmetric positive definite matrix (n x n).
        b (torch.Tensor): Right-hand side vector (n).
        x0 (torch.Tensor, optional): Initial guess for the solution. Defaults to zero vector.
        maxiter (int, optional): Maximum number of iterations. Defaults to n.
        rtol (float, optional): Relative tolerance for convergence (to the initial residual). Defaults to 1e-6.
        atol (float, optional): Absolute tolerance for convergence. Defaults to 1e-12.

    Returns:
        x (torch.Tensor): Solution to the linear system.

    Raises:
        ValueError: If A is not symmetric.
    """

    # Ensure A is symmetric
    # if not torch.allclose(A, A.T):
    #     raise ValueError("Matrix A must be symmetric.")

    n = b.size(0)
    if maxiter is None:
        maxiter = n

    # Initialize solution
    if x0 is None:
        x = torch.zeros(n)
    else:
        x = x0.clone()

    # Initial residual
    r = b - matvec(x)
    p = r.clone()
    rs_old = torch.dot(r, r)
    rs_init = rs_old

    for k in range(maxiter):
        Ap = matvec(p)
        alpha = rs_old / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = torch.dot(r, r)
        if rs_new < rtol * rs_init or rs_new < atol:
            # print(f"Conjugate Gradient converged after {i+1} iterations.")
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, k+1


# Example
if __name__ == "__main__":
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(device)

    # Construct a symmetric positive definite matrix A and vector b
    n = 100
    P = torch.rand(n, 5)  # rank 5 matrix
    # A = torch.matmul(P, P.T) + 0.1 * torch.eye(n)  # Ensure A is positive definite
    b = torch.rand(n) 

    def matvec(p):
        """Compute A @ p where A = P @ P.T + 0.1*I, without forming A explicitly"""
        return P @ (P.T @ p) + 0.1 * p  # Equivalent to A @ p

    # Solve using Conjugate Gradient
    x, cg_steps = cg_torch(matvec, b, maxiter=1000)

    # Verify the solution
    Ax = matvec(x)
    residual_norm = torch.norm(Ax - b).item()
    print("Residual norm:", residual_norm)
    print("CG steps:", cg_steps)
