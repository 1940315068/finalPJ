import numpy as np
import torch
from typing import Union, Callable, Optional, Tuple


def cg(
    matvec: Callable[[Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]],
    b: Union[np.ndarray, torch.Tensor],
    x0: Optional[Union[np.ndarray, torch.Tensor]] = None,
    maxiter: Optional[int] = None,
    rtol: float = 1e-6,
    atol: float = 1e-12
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """
    Conjugate Gradient Method for solving Ax = b, where A is implicitly defined by matvec(p) = A @ p.
    Supports both NumPy arrays and PyTorch tensors.

    Parameters:
        matvec (callable): Function that computes A @ p for any vector p.
        b (np.ndarray or torch.Tensor): Right-hand side vector (n).
        x0 (np.ndarray or torch.Tensor, optional): Initial guess for the solution. Defaults to zero vector.
        maxiter (int, optional): Maximum number of iterations. Defaults to n.
        rtol (float, optional): Relative tolerance for convergence (to the initial residual). Defaults to 1e-6.
        atol (float, optional): Absolute tolerance for convergence. Defaults to 1e-12.

    Returns:
        x (same type as input): Solution to the linear system.
        k (int): Number of iterations.

    Raises:
        TypeError: If input types are neither NumPy arrays nor PyTorch tensors, or input types are mixed.
    """
    # Check input type validity
    is_numpy = isinstance(b, np.ndarray)
    is_torch = isinstance(b, torch.Tensor)
    
    if not (is_numpy or is_torch):
        raise TypeError(f"Input 'b' must be either a NumPy array or PyTorch tensor. Got type:{type(b)}.")
    
    # Check for type consistency between b and x0
    if x0 is not None:
        x0_is_numpy = isinstance(x0, np.ndarray)
        x0_is_torch = isinstance(x0, torch.Tensor)
        if (is_numpy and x0_is_torch) or (is_torch and x0_is_numpy):
            raise TypeError(f"Mixed input types: b is {type(b)}, x0 is {type(x0)}. Use either both torch or both numpy.")
    
    # Select appropriate operations based on input type
    if is_torch:
        # PyTorch operations
        dot = torch.dot
        zeros_like = torch.zeros_like
        clone_or_copy = lambda x: x.clone()
    else:
        # NumPy operations
        dot = np.dot
        zeros_like = np.zeros_like
        clone_or_copy = np.copy
    
    # Get problem size
    n = b.size(0) if is_torch else b.size
    
    # Set default maxiter if not provided
    if maxiter is None:
        maxiter = n

    # Initialize solution
    if x0 is None:
        x = zeros_like(b)
    else:
        x = clone_or_copy(x0)

    # Initial residual
    r = b - matvec(x)
    p = clone_or_copy(r)
    rs_old = dot(r, r)
    rs_init = rs_old
    
    # Main CG iteration loop
    for k in range(1, maxiter+1):
        # Matrix-vector product
        Ap = matvec(p)
        
        # Compute step size
        alpha = rs_old / dot(p, Ap)
        
        # Update solution and residual
        x = x + alpha * p
        r = r - alpha * Ap

        # Check convergence
        rs_new = dot(r, r)
        if rs_new < rtol * rs_init or rs_new < atol:
            break

        # Update search direction
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, k


# Example
if __name__ == "__main__":
    """ Numpy part """
    print("========================================")
    print("-----------------Numpy------------------")
    
    # Construct a symmetric positive definite matrix A and vector b
    n = 100
    P = np.random.randn(n, 5)  # Rank 5 matrix
    b = np.random.randn(n)
    def matvec(p:np.ndarray):
        """Compute A @ p where A = P @ P.T + 0.1*I, without forming A explicitly"""
        return P @ (P.T @ p) + 0.1 * p
    
    # Solve using Conjugate Gradient
    x, cg_steps = cg(matvec, b, maxiter=1000)

    # Verify the solution
    Ax = matvec(x)
    residual_norm = np.linalg.norm(Ax - b)
    print(f"Residual norm: {residual_norm:.3e}.  CG steps: {cg_steps}")
    print("========================================")
    print()
    
    
    """ Torch part """
    print("========================================")
    print("-----------------Torch------------------")
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(device)
    
    # Construct the matrix and vector with numpy data
    P_torch = torch.from_numpy(P).clone()
    b_torch = torch.from_numpy(b).clone()
    def matvec_torch(p:torch.Tensor):
        """Compute A @ p where A = P @ P.T + 0.1*I, without forming A explicitly"""
        return P_torch @ (P_torch.T @ p) + 0.1 * p
    
    # Solve using Conjugate Gradient
    x_torch, cg_steps_torch = cg(matvec_torch, b_torch, maxiter=1000)

    # Verify the solution
    Ax_torch = matvec_torch(x_torch)
    residual_norm_torch = torch.norm(Ax_torch - b_torch).item()
    print(f"Residual norm: {residual_norm_torch:.3e}.  CG steps: {cg_steps_torch}")
    print("========================================")
    print()
