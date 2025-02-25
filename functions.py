import numpy as np


# Compute the quadratic form: g^T x + 1/2 x^T H x
def quadratic_form(H, g, x) -> float:
    gTx = np.dot(g.T, x)
    xTHx = np.dot(x.T, np.dot(H, x))
    return gTx + 0.5 * xTHx

# return g^T x + 1/2 x^T H x + penalty_eq + penalty_ineq
def exact_penalty_func(H, g, x, A_eq, b_eq, A_ineq, b_ineq) -> float:
    gTx = np.dot(g.T, x)
    xTHx = np.dot(x.T, np.dot(H, x))
    penalty_eq = np.sum(np.abs(A_eq @ x + b_eq))
    penalty_ineq = np.sum(np.maximum(A_ineq @ x + b_ineq, 0))
    return gTx + 0.5 * xTHx + penalty_eq + penalty_ineq