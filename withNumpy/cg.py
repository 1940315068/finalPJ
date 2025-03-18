import numpy as np


def cg(A, b, x0=None, maxiter=None, rtol=1e-10):
    """
    Conjugate Gradient Method for solving the linear system Ax = b using NumPy.

    Parameters:
        A (np.ndarray): Symmetric positive definite matrix (n x n).
        b (np.ndarray): Right-hand side vector (n).
        x0 (np.ndarray, optional): Initial guess for the solution. Defaults to zero vector.
        max_iter (int, optional): Maximum number of iterations. Defaults to n.
        rtol (float, optional): Relative tolerance for convergence. Defaults to 1e-10.

    Returns:
        x (np.ndarray): Solution to the linear system.

    Raises:
        ValueError: If A is not symmetric.
    """

    # Ensure A is symmetric
    A = 0.5 * (A + A.T)

    n = b.size
    if maxiter is None:
        maxiter = n

    # Initialize solution
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # Initial residual
    r = b - np.dot(A, x)
    p = r.copy()  # Initial search direction

    rs_old = np.dot(r, r)  # Dot product of residual

    for i in range(maxiter):
        Ap = np.dot(A, p)  # Matrix-vector product
        alpha = rs_old / np.dot(p, Ap)  # Step size
        x = x + alpha * p  # Update solution
        r = r - alpha * Ap  # Update residual

        rs_new = np.dot(r, r)  # New residual dot product
        if np.linalg.norm(r) < rtol * np.linalg.norm(b):  # Check convergence
            # print(f"Conjugate Gradient converged after {i+1} iterations.")
            break

        p = r + (rs_new / rs_old) * p  # Update search direction
        rs_old = rs_new  # Update residual dot product

    return x


# Example
if __name__ == "__main__":
    # Construct a symmetric positive definite matrix A and vector b
    n = 100
    P = np.random.randn(n, 5)  # Rank 5 matrix
    A = np.dot(P, P.T) + 0.1 * np.eye(n)  # Ensure A is positive definite
    b = np.random.randn(n)

    # Solve using Conjugate Gradient
    x = cg(A, b, maxiter=1000)

    # Verify the solution
    residual_norm = np.linalg.norm(np.dot(A, x) - b)
    print("Residual norm:", residual_norm)