import numpy as np
import torch
from typing import Union


def quadratic_objective(
    H: Union[torch.Tensor, np.ndarray],
    g: Union[torch.Tensor, np.ndarray],
    x: Union[torch.Tensor, np.ndarray]
) -> float:
    """Compute g^T x + 0.5 * x^T H x. Inputs must be all torch or all numpy."""
    # Check input type validity
    is_numpy = isinstance(x, np.ndarray)
    is_torch = isinstance(x, torch.Tensor)
    if not (is_numpy or is_torch):
        raise TypeError("Input 'x' must be either a NumPy array or PyTorch tensor")
    
    # Check for type consistency
    input_types = {type(H), type(g), type(x)}
    if len(input_types) > 1:
        raise TypeError(f"Mixed input types: {input_types}. Use either all torch or all numpy.")
    
    # Compute based on input type
    if is_torch:
        gTx = torch.dot(g, x)
        xTHx = x @ H @ x  # PyTorch handles 1D/2D automatically
    else:
        gTx = np.dot(g, x)
        xTHx = x.T @ H @ x  # NumPy needs explicit transpose
    
    return float(gTx + 0.5 * xTHx)

def penalized_quadratic_objective(
    H: Union[torch.Tensor, np.ndarray],
    g: Union[torch.Tensor, np.ndarray],
    x: Union[torch.Tensor, np.ndarray],
    A_eq: Union[torch.Tensor, np.ndarray],
    b_eq: Union[torch.Tensor, np.ndarray],
    A_ineq: Union[torch.Tensor, np.ndarray],
    b_ineq: Union[torch.Tensor, np.ndarray]
) -> float:
    """Compute objective with L1 penalties. Inputs must be all torch or all numpy."""
    # Check input type validity
    is_numpy = isinstance(x, np.ndarray)
    is_torch = isinstance(x, torch.Tensor)
    if not (is_numpy or is_torch):
        raise TypeError("Input 'x' must be either a NumPy array or PyTorch tensor")
    
    # Check for type consistency
    tensors = [H, g, x, A_eq, b_eq, A_ineq, b_ineq]
    input_types = {type(t) for t in tensors}
    if len(input_types) > 1:
        raise TypeError(f"Mixed input types: {input_types}. Use either all torch or all numpy.")

    # Select appropriate operations based on input type
    if is_torch:
        dot = torch.dot
        matmul = torch.matmul
        abs_sum = lambda a: a.abs().sum()
        max_zero_sum = lambda a: a.clamp(min=0).sum()
    else:
        dot = np.dot
        matmul = np.matmul
        abs_sum = lambda a: np.abs(a).sum()
        max_zero_sum = lambda a: np.maximum(a, 0).sum()
    

    # Core computation
    gTx = dot(g, x)
    xTHx = dot(x, matmul(H, x))
    penalty_eq = abs_sum(matmul(A_eq, x) + b_eq)
    penalty_ineq = max_zero_sum(matmul(A_ineq, x) + b_ineq)

    return float(gTx + 0.5 * xTHx + penalty_eq + penalty_ineq)