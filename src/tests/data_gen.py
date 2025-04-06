import numpy as np
import torch


def generate_optimization_data(n=1000, m1=300, m2=300, 
                             m1_infeasible=0, m2_infeasible=0,
                             numpy_output=True, torch_output=True, seed=None):
    """
    Generate optimization problem data in NumPy and/or PyTorch formats.
    
    The problem is of the form:
    min (1/2)x^T H x + g^T x
    s.t. A_eq x = b_eq
         A_ineq x <= b_ineq
    
    Parameters:
    -----------
    n : int
        Number of variables (default=1000)
    m1 : int
        Number of equality constraints (default=300)
    m2 : int
        Number of inequality constraints (default=300)
    m1_infeasible : int
        Number of infeasible equality constraints to add (default=0)
    m2_infeasible : int
        Number of infeasible inequality constraints to add (default=0)
    numpy_output : bool
        Whether to return NumPy arrays (default=True)
    torch_output : bool
        Whether to return PyTorch tensors (default=True)
    seed : int or None
        Random seed for reproducibility (default=None)
    
    Returns:
    --------
    dict
        Dictionary containing the problem data in requested formats with keys:
        - 'numpy': Contains NumPy arrays (if numpy_output=True)
        - 'torch': Contains PyTorch tensors (if torch_output=True)
        Each contains the following elements:
        - 'H': Quadratic term matrix (n x n)
        - 'g': Linear term vector (n,)
        - 'A_eq': Equality constraint matrix (m1 x n)
        - 'b_eq': Equality constraint vector (m1,)
        - 'A_ineq': Inequality constraint matrix (m2 x n)
        - 'b_ineq': Inequality constraint vector (m2,)
        - 'n': Number of variables
        - 'm1': Final number of equality constraints (including infeasible ones)
        - 'm2': Final number of inequality constraints (including infeasible ones)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Initialize output dictionary
    output = {}
    
    # Generate equality constraints
    A_eq = np.random.rand(m1, n)
    b_eq = np.zeros(m1)
    
    # Generate inequality constraints
    A_ineq = np.random.rand(m2, n)
    b_ineq = np.random.rand(m2)
    
    # Add infeasible equality constraints: Ax+b=0, Ax-b=0
    if m1_infeasible > 0:
        A_eq_infeasible = np.random.rand(m1_infeasible, n)
        A_eq = np.vstack([A_eq, A_eq_infeasible, A_eq_infeasible])
        b_eq_infeasible = np.random.rand(m1_infeasible)
        b_eq = np.hstack([b_eq, b_eq_infeasible, -b_eq_infeasible])
        m1 += 2 * m1_infeasible
    
    # Add infeasible inequality constraints: Ax+1<=0, -Ax<=0
    if m2_infeasible > 0:
        A_ineq_infeasible = np.random.rand(m2_infeasible, n)
        A_ineq = np.vstack([A_ineq, A_ineq_infeasible, -A_ineq_infeasible])
        b_ones_infeasible = np.ones(m2_infeasible)
        b_zeros_infeasible = np.zeros(m2_infeasible)
        b_ineq = np.hstack([b_ineq, b_ones_infeasible, b_zeros_infeasible])
        m2 += 2 * m2_infeasible
    
    # Generate quadratic objective: (1/2)x^T H x + g^T x
    P = np.random.rand(n, 5)  # Low-rank component
    H = np.dot(P, P.T) + 0.1 * np.eye(n)  # Ensure positive definiteness
    g = np.random.rand(n)
    
    # Prepare outputs
    if numpy_output:
        output['numpy'] = {
            'H': H,
            'g': g,
            'A_eq': A_eq,
            'b_eq': b_eq,
            'A_ineq': A_ineq,
            'b_ineq': b_ineq,
            'n': n,
            'm1': m1,
            'm2': m2
        }
    
    if torch_output:
        output['torch'] = {
            'H': torch.from_numpy(H).to(torch.float64),
            'g': torch.from_numpy(g).to(torch.float64),
            'A_eq': torch.from_numpy(A_eq).to(torch.float64),
            'b_eq': torch.from_numpy(b_eq).to(torch.float64),
            'A_ineq': torch.from_numpy(A_ineq).to(torch.float64),
            'b_ineq': torch.from_numpy(b_ineq).to(torch.float64),
            'n': n,
            'm1': m1,
            'm2': m2
        }
    
    return output
