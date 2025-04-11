import torch
import numpy as np
import time
from typing import Union, Optional, Tuple
from ..functions import quadratic_objective, penalized_quadratic_objective
from ..cg import cg


def irwa_solver(
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
    Iteratively Reweighted Algorithm (IRWA) solver for constrained quadratic programming.

    Solves the optimization problem:
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
        print("----------------- IRWA -----------------")
    
    # Type and consistency check
    is_numpy = isinstance(g, np.ndarray)
    is_torch = isinstance(g, torch.Tensor)
    expected_type = type(g)
    if not (is_numpy or is_torch):
        raise TypeError(f"Unsupported input type: {expected_type.__name__}. Must be NumPy or PyTorch.")

    all_inputs = [H, g, A_eq, b_eq, A_ineq, b_ineq] + ([x0] if x0 is not None else [])
    if not all(isinstance(item, expected_type) for item in all_inputs):
        raise TypeError(f"Mixed input types. All inputs must be of type {expected_type.__name__}.")

    backend_ops = get_backend(g)
    
    # Problem dimensions
    n = g.shape[0]        # number of variables
    m1 = b_eq.shape[0]    # number of equality constraints
    m2 = b_ineq.shape[0]  # number of inequality constraints
    m = m1 + m2           # total number of constraints

    # Hyper parameters:
    epsilon0 = 20000 * backend_ops['ones'](m)
    eta = 0.8
    gamma = 0.15
    M = 100
    sigma = 1e-6
    sigma_prime = 1e-4

    # Initialization
    x = backend_ops['copy'](x0) if x0 is not None else backend_ops['zeros'](n)
    epsilon = epsilon0
    A = backend_ops['vstack']([A_eq, A_ineq])
    b = backend_ops['hstack']([b_eq, b_ineq])
    n_cg_steps = 0
    time_cg = 0

    for k in range(1, max_iter+1):
        # Step 1. Solve the reweighted subproblem for x^(k+1), use conjugate gradient to solve the linear system 
        w = compute_w(x, epsilon, A, b, m1, m2, backend_ops)
        v = compute_v(x, A, b, m1, m2, backend_ops)

        def matvec(p):
            return H @ p + (A.T @ (w * (A @ p)))

        rhs = -(g + A.T @ (w * v))
        cg_start_time = time.monotonic()
        maxiter = max(30, n // 100)  # max iteration for cg
        rtol_cg = 0.15
        x_next, cg_steps = cg(matvec, rhs, maxiter=maxiter, x0=x, rtol=rtol_cg)
        cg_end_time = time.monotonic()
        time_cg += (cg_end_time - cg_start_time)
        n_cg_steps += cg_steps

        # Step 2. Set the new relaxation vector epsilon^(k+1)
        q = A @ (x_next - x)
        Ax_plus_b = A @ x + b
        z = backend_ops['hstack']([-9999 * backend_ops['ones'](m1), backend_ops['zeros'](m2)])
        r = backend_ops['max'](Ax_plus_b, z)

        if backend_ops['all'](backend_ops['abs'](q) <= M * (r**2 + epsilon**2)**(0.5 + gamma)):
            epsilon_next = epsilon * eta
            # Keep the original epsilon value for the satisfied constraints
            cond1 = (backend_ops['arange'](m) < m1) & backend_ops['isclose'](Ax_plus_b, backend_ops['tensor'](0.0), atol=1e-6)
            cond2 = (backend_ops['arange'](m) >= m1) & (Ax_plus_b <= 0)
            indices = backend_ops['where'](cond1 | cond2)[0]
            epsilon_next[indices] = epsilon[indices]

        # Step 3. Check the stopping criterion
        norm_dx = backend_ops['norm'](x_next - x)
        min_eps = backend_ops['min_value'](epsilon)
        if norm_dx <= sigma and min_eps <= sigma_prime:
            if verbose:
                print(f"Iteration ends at {k}.")
                val = quadratic_objective(H, g, x)
                val_penalty = penalized_quadratic_objective(H, g, A_eq, b_eq, A_ineq, b_ineq, x)
                print(f"{'Objective (no penalty):':<24} {val:.6f}")
                print(f"{'Objective (penalized):':<24} {val_penalty:.6f}")
                print(f"||dx|| = {norm_dx:.2e}, min(epsilon) = {min_eps:.2e}")
                print("========================================")
            return x, k, n_cg_steps, time_cg

        x = x_next
        epsilon = epsilon_next

        if verbose and k % 100 == 0:
            print(f"Iteration {k}:")
            val = quadratic_objective(H, g, x)
            val_penalty = penalized_quadratic_objective(H, g, A_eq, b_eq, A_ineq, b_ineq, x)
            print(f"{'Objective (no penalty):':<24} {val:.6f}")
            print(f"{'Objective (penalized):':<24} {val_penalty:.6f}")
            print(f"||dx|| = {norm_dx:.2e}, min(epsilon) = {min_eps:.2e}\n")

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
            'min': torch.min,
            'cat': torch.cat,
            'zeros': torch.zeros,
            'ones': torch.ones,
            'norm': torch.norm,
            'arange': torch.arange,
            'isclose': torch.isclose,
            'vstack': torch.vstack,
            'hstack': torch.hstack,
            'copy': lambda x: x.clone(),
            'tensor': torch.tensor,
            'all': torch.all,
            'any': torch.any,
            'sqrt': torch.sqrt,
            'where': torch.where,
            'max_value': lambda x: torch.max(x),
            'min_value': lambda x: torch.min(x),
            'backend': 'torch'
        }
    elif isinstance(x, np.ndarray):
        return {
            'dot': np.dot,
            'abs': np.abs,
            'max': np.maximum,
            'min': np.minimum,
            'cat': np.concatenate,
            'zeros': np.zeros,
            'ones': np.ones,
            'norm': np.linalg.norm,
            'arange': np.arange,
            'isclose': np.isclose,
            'vstack': np.vstack,
            'hstack': np.hstack,
            'copy': np.copy,
            'tensor': lambda x: x,
            'all': np.all,
            'any': np.any,
            'sqrt': np.sqrt,
            'where': np.where,
            'max_value': np.max,
            'min_value': np.min,
            'backend': 'numpy'
        }
    else:
        raise TypeError("Input must be a NumPy array or PyTorch tensor.")

def compute_w(x, epsilon, A, b, m1, m2, backend_ops):
    m = b.shape[0]
    Ax_plus_b = A @ x + b
    eq_part = backend_ops['abs'](Ax_plus_b[:m1])**2 + epsilon[:m1]**2
    ineq_part = backend_ops['max'](Ax_plus_b[m1:], backend_ops['tensor'](0.0))**2 + epsilon[m1:]**2
    w = backend_ops['cat']([eq_part, ineq_part])**(-0.5)
    return w

def compute_v(x, A, b, m1, m2, backend_ops):
    Ax_plus_b = A @ x + b
    eq_part = b[:m1]
    ineq_part = b[m1:] - backend_ops['min'](Ax_plus_b[m1:], backend_ops['tensor'](0.0))
    v = backend_ops['cat']([eq_part, ineq_part])
    return v
