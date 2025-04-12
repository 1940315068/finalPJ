import numpy as np
import torch


def generate_random_data(n=1000, m1=300, m2=300, 
                             numpy_output=True, torch_output=True, seed=None, 
                             torch_float_dtype = torch.float64):
    """
    Generate optimization problem data in NumPy and/or PyTorch formats.
    
    The problem is of the form:
    min (1/2)x^T H x + g^T x
    s.t. A_eq x + b_eq = 0
         A_ineq x + b_ineq <= 0
    
    Parameters:
    -----------
    n (int): Number of variables (default=1000)
    m1 (int): Number of equality constraints (default=300)
    m2 (int): Number of inequality constraints (default=300)
    numpy_output (bool): Whether to return NumPy arrays (default=True)
    torch_output (bool): Whether to return PyTorch tensors (default=True)
    seed (int, optional): Random seed for reproducibility (default=None)
    torch_float_dtype (dtype): Dtype of torch tensors. (default=float32)
    
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
    """
    torch.set_default_dtype(torch_float_dtype)
    
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
            'b_ineq': b_ineq
        }
    
    if torch_output:
        output['torch'] = {
            'H': torch.from_numpy(H).clone().to(torch_float_dtype),
            'g': torch.from_numpy(g).clone().to(torch_float_dtype),
            'A_eq': torch.from_numpy(A_eq).clone().to(torch_float_dtype),
            'b_eq': torch.from_numpy(b_eq).clone().to(torch_float_dtype),
            'A_ineq': torch.from_numpy(A_ineq).clone().to(torch_float_dtype),
            'b_ineq': torch.from_numpy(b_ineq).clone().to(torch_float_dtype)
        }
    
    return output


def generate_infeasible_data(n=1000, m1=300, m2=300, 
                             m1_infeasible=10, m2_infeasible=10,
                             numpy_output=True, torch_output=True, seed=None, 
                             torch_float_dtype = torch.float64):
    """
    Generate infeasible optimization problem data in NumPy and/or PyTorch formats.
    
    The problem is of the form:
    min (1/2)x^T H x + g^T x
    s.t. A_eq x + b_eq = 0
         A_ineq x + b_ineq <= 0
    
    Parameters:
    -----------
    n (int): Number of variables (default=1000)
    m1 (int): Number of equality constraints (default=300)
    m2 (int): Number of inequality constraints (default=300)
    m1_infeasible (int): Number of infeasible equality constraints to add (default=10)
    m2_infeasible (int): Number of infeasible inequality constraints to add (default=10)
    numpy_output (bool): Whether to return NumPy arrays (default=True)
    torch_output (bool): Whether to return PyTorch tensors (default=True)
    seed (int, optional): Random seed for reproducibility (default=None)
    torch_float_dtype (dtype): Dtype of torch tensors. (default=float32)
    
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
    """
    torch.set_default_dtype(torch_float_dtype)
    
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
            'b_ineq': b_ineq
        }
    
    if torch_output:
        output['torch'] = {
            'H': torch.from_numpy(H).clone().to(torch_float_dtype),
            'g': torch.from_numpy(g).clone().to(torch_float_dtype),
            'A_eq': torch.from_numpy(A_eq).clone().to(torch_float_dtype),
            'b_eq': torch.from_numpy(b_eq).clone().to(torch_float_dtype),
            'A_ineq': torch.from_numpy(A_ineq).clone().to(torch_float_dtype),
            'b_ineq': torch.from_numpy(b_ineq).clone().to(torch_float_dtype)
        }
    
    return output



import numpy as np
import torch

def generate_portfolio_data(
    n_assets=1000,
    n_factors=10,
    target_return=0.1,
    include_shorting=False,
    numpy_output=True,
    torch_output=True,
    seed=None,
    torch_float_dtype=torch.float64
):
    """
    Generate quadratic programming data for Markowitz minimum-risk portfolio optimization:
        min (1/2)w^T Σ w
        s.t. 1^T w = 1
             μ^T w >= R_target
             w >= 0 (if no shorting)
    
    Parameters:
    -----------
    n_assets (int): Number of assets (default=1000)
    n_factors (int): Number of factors for covariance matrix (default=10)
    target_return (float): Required portfolio return (default=0.1)
    include_shorting (bool): Allow negative weights (default=False)
    numpy_output (bool): Return NumPy arrays (default=True)
    torch_output (bool): Return PyTorch tensors (default=True)
    seed (int, optional): Random seed for reproducibility (default=None)
    torch_float_dtype (dtype). PyTorch tensor dtype (default=torch.float64)
    
    Returns:
    --------
    dict
        Dictionary containing the problem data in requested formats with keys:
        - 'numpy': Contains NumPy arrays (if numpy_output=True)
        - 'torch': Contains PyTorch tensors (if torch_output=True)
        Each contains the following elements:
        - 'H': Quadratic term matrix (n x n)
        - 'g': Linear term vector (n,)
        - 'A_eq': Equality constraint matrix (m1 x n)  = [-mu; 1]
        - 'b_eq': Equality constraint vector (m1,)  = - [target_return; 1]
        - 'A_ineq': Inequality constraint matrix (m2 x n)  = -I
        - 'b_ineq': Inequality constraint vector (m2,)  = 0
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # ==================== Generate Factor Model Data ====================
    # Factor loading matrix (n_assets × n_factors)
    B = np.random.randn(n_assets, n_factors) * 0.1
    
    # Factor covariance matrix (n_factors × n_factors)
    F = np.random.randn(n_factors, n_factors)
    Sigma_f = F @ F.T / n_factors + np.eye(n_factors) * 0.1
    
    # Idiosyncratic risk (diagonal matrix)
    idio_vols = np.random.uniform(0.15, 0.35, n_assets)
    D = np.diag(idio_vols**2)
    
    # Total covariance matrix (n_assets × n_assets)
    Sigma = B @ Sigma_f @ B.T + D
    
    # Expected returns vector
    mu = np.random.uniform(-0.05, 0.15, n_assets)
    
    # ==================== Construct QP Problem ====================
    # Objective function: min (1/2)w^T Σ w
    H = Sigma
    g = np.zeros(n_assets)
    
    # Equality constraints: 1^T w =  1
    A_eq = np.ones(n_assets).reshape(1, n_assets)  # (1, n_assets)
    b_eq = -np.array([1.0])  # (1,)
    
    # Inequality constraints: -μ^T w + R <=0; -w <= 0 (if no shorting)
    A_ineq = -mu.reshape(1, n_assets)  # (1, n_assets)
    b_ineq = np.array([target_return])  # (1,)
    if not include_shorting:
        A_ineq = np.vstack([A_ineq, -np.eye(n_assets)])    # (1 + n_assets, n_assets)
        b_ineq = np.hstack([b_ineq, np.zeros(n_assets)])  # (1 + n_assets,)
    
    # ==================== Prepare Outputs ====================
    output = {}
    
    if numpy_output:
        output['numpy'] = {
            'H': H,
            'g': g,
            'A_eq': A_eq,
            'b_eq': b_eq,
            'A_ineq': A_ineq,
            'b_ineq': b_ineq
        }
    
    if torch_output:
        output['torch'] = {
            'H': torch.from_numpy(H).to(torch_float_dtype),
            'g': torch.from_numpy(g).to(torch_float_dtype),
            'A_eq': torch.from_numpy(A_eq).to(torch_float_dtype),
            'b_eq': torch.from_numpy(b_eq).to(torch_float_dtype),
            'A_ineq': torch.from_numpy(A_ineq).to(torch_float_dtype),
            'b_ineq': torch.from_numpy(b_ineq).to(torch_float_dtype)
        }
    
    return output
