import numpy as np


def cg(matvec, b, x0=None, maxiter=None, rtol=1e-6, atol=1e-12):
    """
    Conjugate Gradient Method for solving Ax = b, where A is implicitly defined by matvec(p) = A @ p.

    Parameters:
        matvec (callable): Function that computes A @ p for any vector p.
        b (np.ndarray): Right-hand side vector (n).
        x0 (np.ndarray, optional): Initial guess for the solution. Defaults to zero vector.
        maxiter (int, optional): Maximum number of iterations. Defaults to n.
        rtol (float, optional): Relative tolerance for convergence (to the initial residual). Defaults to 1e-6.
        atol (float, optional): Absolute tolerance for convergence. Defaults to 1e-12.

    Returns:
        x (np.ndarray): Solution to the linear system.
        k (int): Number of iterations.
    """
    n = b.size
    if maxiter is None:
        maxiter = n

    # Initialize solution
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # Initial residual
    r = b - matvec(x)  # Compute A @ x using matvec
    p = r.copy()  # Initial search direction
    rs_old = np.dot(r, r)  # Dot product of residual
    rs_init = rs_old
    
    for k in range(maxiter):
        Ap = matvec(p)  # Compute A @ p using matvec (e.g., B @ (C @ (D @ p)))
        alpha = rs_old / np.dot(p, Ap)  # Step size
        x = x + alpha * p  # Update solution
        r = r - alpha * Ap  # Update residual

        rs_new = np.dot(r, r)  # New residual dot product
        if rs_new < rtol * rs_init or rs_new < atol:  # Check convergence
            break

        p = r + (rs_new / rs_old) * p  # Update search direction
        rs_old = rs_new  # Update residual dot product

    return x, k+1


# Example
if __name__ == "__main__":
    # Construct a symmetric positive definite matrix A and vector b
    n = 100
    P = np.random.randn(n, 5)  # Rank 5 matrix
    # A = np.matmul(P, P.T) + 0.1 * np.eye(n)  # Ensure A is positive definite
    b = np.random.randn(n)

    def matvec(p):
        """Compute A @ p where A = P @ P.T + 0.1*I, without forming A explicitly"""
        return P @ (P.T @ p) + 0.1 * p

    # Solve using Conjugate Gradient
    x, cg_steps = cg(matvec, b, maxiter=1000)

    # Verify the solution
    Ax = matvec(x)
    residual_norm = np.linalg.norm(Ax - b)
    print("Residual norm:", residual_norm)
    print("CG steps:", cg_steps)
    