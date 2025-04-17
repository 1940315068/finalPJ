import torch
import numpy as np
import time
from typing import Union, Optional, Tuple
from ..functions import quadratic_objective, penalized_quadratic_objective
from ..cg import cg


def adal_solver(
    H: Union[np.ndarray, torch.Tensor],
    g: Union[np.ndarray, torch.Tensor],
    A_eq: Union[np.ndarray, torch.Tensor],
    b_eq: Union[np.ndarray, torch.Tensor],
    A_ineq: Union[np.ndarray, torch.Tensor],
    b_ineq: Union[np.ndarray, torch.Tensor],
    x0: Optional[Union[np.ndarray, torch.Tensor]] = None,
    max_iter: int = 1000,
    verbose: bool = True
) -> Tuple[Union[np.ndarray, torch.Tensor], int, int, float]:
    """
    Alternating Direction Augmented Lagrangian (ADAL) solver for constrained quadratic programming.

    Solves:
        minimize    (1/2) * x^T H x + g^T x
        subject to  A_eq x + b_eq == 0
                    A_ineq x + b_ineq <= 0

    Parameters:
        H (np.ndarray or torch.Tensor): Quadratic coefficient matrix.
        g (np.ndarray or torch.Tensor): Linear coefficient vector.
        A_eq (np.ndarray or torch.Tensor): Equality constraint matrix.
        b_eq (np.ndarray or torch.Tensor): Equality constraint vector.
        A_ineq (np.ndarray or torch.Tensor): Inequality constraint matrix.
        b_ineq (np.ndarray or torch.Tensor): Inequality constraint vector.
        x0 (np.ndarray or torch.Tensor, optional): Initial guess for the solution. Defaults to zero vector.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        x (np.ndarray or torch.Tensor). Final solution vector.
        k (int): Number of iterations performed.
        n_cg_steps (int): Total number of CG steps used.
        time_cg (float): Total time spent in CG.
        
    Raises:
        TypeError: If input types are neither NumPy arrays nor PyTorch tensors, or input types are mixed.
    """
    if verbose:
        print("========================================")
        print("----------------- ADAL -----------------")

    # Type and consistency check
    is_numpy = isinstance(g, np.ndarray)
    is_torch = isinstance(g, torch.Tensor)
    expected_type = type(g)
    if not (is_numpy or is_torch):
        raise TypeError(f"Unsupported input type: {expected_type.__name__}. Must be NumPy or PyTorch.")

    all_inputs = [H, g, A_eq, b_eq, A_ineq, b_ineq] + ([x0] if x0 is not None else [])
    if not all(isinstance(item, expected_type) for item in all_inputs):
        raise TypeError(f"Mixed input types. All inputs must be of type {expected_type.__name__}.")

    backend = get_backend(g)

    # Problem dimensions
    n = g.shape[0]        # number of variables
    m1 = b_eq.shape[0]    # number of equality constraints
    m2 = b_ineq.shape[0]  # number of inequality constraints
    m = m1 + m2           # total number of constraints

    # Hyperparameters
    u = backend['zeros'](m)
    mu = min(0.1, 1.0 / n)
    sigma = 1e-6
    sigma_prime = 1e-6

    # Initialization
    x = backend['copy'](x0) if x0 is not None else backend['zeros'](n)
    A = backend['vstack']([A_eq, A_ineq])
    b = backend['hstack']([b_eq, b_ineq])
    n_cg_steps = 0
    time_cg = 0
    rho = 1.0
    rho_max = 1.0
    rho_scale = 1.1

    def matvec(p):
        return H @ p + rho * (A.T @ (mu * (A @ p)))

    for k in range(1, max_iter + 1):
        # Step 1. Solve subproblem for p^(k+1)
        s = A @ x + b + (1.0 / mu) * u
        s_eq = s[:m1]
        s_ineq = s[m1:]

        p_eq = backend['sign'](s_eq) * backend['max'](backend['abs'](s_eq) - 1.0 / mu, backend['tensor'](0.0))
        p_ineq = backend['max'](s_ineq - 1.0 / mu, backend['tensor'](0.0)) - backend['max'](-s_ineq, backend['tensor'](0.0))
        p_next = backend['cat']([p_eq, p_ineq])
        
        # Step 2. Solve subproblem for x^(k+1), use conjugate gradient to solve the linear system
        rhs = -(g + rho * (A.T @ u) + rho * mu * (A.T @ (b - p_next)))
        cg_start_time = time.monotonic()
        maxiter = max(50, n // 250)  # max iteration for cg
        rtol_cg = 0.15
        x_next, cg_steps = cg(matvec, rhs, maxiter=maxiter, x0=x, rtol=rtol_cg)
        cg_end_time = time.monotonic()
        time_cg += (cg_end_time - cg_start_time)
        n_cg_steps += cg_steps

        # Step 3. Update multipliers u^(k+1)
        residual = A @ x_next + b - p_next
        u_next = u + mu * residual
        
        rho = min(rho_max, rho*rho_scale)
        
        # Step 4. Check the stopping criterion
        norm_dx = backend['norm'](x_next - x)
        norm_residual = backend['max_val'](backend['abs'](residual))

        if norm_dx <= sigma and norm_residual <= sigma_prime:
            if verbose:
                print(f"Iteration ends at {k}.")
                val = quadratic_objective(H, g, x)
                val_penalty = penalized_quadratic_objective(H, g, A_eq, b_eq, A_ineq, b_ineq, x)
                print(f"{'Objective (no penalty):':<24} {val:.6f}")
                print(f"{'Objective (penalized):':<24} {val_penalty:.6f}")
                print(f"||dx|| = {norm_dx:.2e}, max(|residual|) = {norm_residual:.2e}")
                print("========================================")
            return x, k, n_cg_steps, time_cg

        x = x_next
        u = u_next

        if verbose and k % 100 == 0:
            print(f"Iteration {k}:")
            val = quadratic_objective(H, g, x)
            val_penalty = penalized_quadratic_objective(H, g, A_eq, b_eq, A_ineq, b_ineq, x)
            print(f"{'Objective (no penalty):':<24} {val:.6f}")
            print(f"{'Objective (penalized):':<24} {val_penalty:.6f}")
            print(f"||dx|| = {norm_dx:.2e}, max(|residual|) = {norm_residual:.2e}\n")

    if verbose:
        print(f"No convergence in {k} iterations.")
        print("========================================")
    return x, k, n_cg_steps, time_cg


def get_backend(x):
    if isinstance(x, torch.Tensor):
        return {
            'dot': torch.dot,
            'abs': torch.abs,
            'max': torch.max,
            'sign': torch.sign,
            'zeros': torch.zeros,
            'ones': torch.ones,
            'norm': torch.norm,
            'cat': torch.cat,
            'vstack': torch.vstack,
            'hstack': torch.hstack,
            'copy': lambda x: x.clone(),
            'tensor': torch.tensor,
            'max_val': lambda x: torch.max(x),
            'backend': 'torch'
        }
    elif isinstance(x, np.ndarray):
        return {
            'dot': np.dot,
            'abs': np.abs,
            'max': np.maximum,
            'sign': np.sign,
            'zeros': np.zeros,
            'ones': np.ones,
            'norm': np.linalg.norm,
            'cat': np.concatenate,
            'vstack': np.vstack,
            'hstack': np.hstack,
            'copy': np.copy,
            'tensor': lambda x: x,
            'max_val': np.max,
            'backend': 'numpy'
        }
    else:
        raise TypeError("Input must be a NumPy array or PyTorch tensor.")
