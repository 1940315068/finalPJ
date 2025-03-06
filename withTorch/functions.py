import torch


# Compute the quadratic form: g^T x + 1/2 x^T H x
def quadratic_form(H, g, x) -> float:
    gTx = torch.dot(g, x).item()
    xTHx = torch.dot(x, torch.matmul(H, x)).item()
    return gTx + 0.5 * xTHx

# return g^T x + 1/2 x^T H x + penalty_eq + penalty_ineq
def exact_penalty_func(H, g, x, A_eq, b_eq, A_ineq, b_ineq) -> float:
    gTx = torch.dot(g, x).item()
    xTHx = torch.dot(x, torch.matmul(H, x)).item()
    penalty_eq = torch.sum(torch.abs(torch.matmul(A_eq, x) + b_eq)).item()
    penalty_ineq = torch.sum(torch.maximum(torch.matmul(A_ineq, x) + b_ineq, torch.zeros_like(b_ineq))).item()
    return gTx + 0.5 * xTHx + penalty_eq + penalty_ineq