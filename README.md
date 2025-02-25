# Introduction to QP problem

Quadratic Programming (QP) is a type of optimization problem where the objective function is quadratic, and the constraints are linear. It can be formulated as follows:
$$
\begin{align*}
\min_{x} \quad & \varphi(x), \quad \text{where} \quad \varphi(x) := g^T x + \frac{1}{2} x^T H x \\
\textit{s.t.} \quad & Ax + b \in \mathcal{C}
\end{align*}
$$
where $ x \in \R^n$ is the decision variable, $ g \in \R^{n} $ is a vector, $ H \in \R^{n\times n} $ is a symmetric matrix, $ A \in \R^{m \times n} $ is a matrix, $b \in \R^m$ is a vector, $ \mathcal{C} \subseteq \R^m $ is a nonempty, closed set. 

In simple terms, the constraint $ A x + b \in \mathcal{C}$ includes both the equality constraints $ (a_i^Tx+b_i = 0) $ and inequality constraints $ (a_i^Tx+b_i \leq 0) $. Hence the problem can also be written as follows: 
$$
\begin{align*}
\min_{x} \quad & g^T x + \frac{1}{2} x^T H x \\
\textit{s.t.} \quad 
& a_i^T x + b_i = 0, \ \forall i \in \mathcal{I}_{1} := \{1, 2, \cdots, m_1\}\\
& a_i^T x + b_i \leq 0, \ \forall i \in \mathcal{I}_{2} := \{m_1+1, \cdots, m\}
\end{align*}
$$
where for any $i\in\mathcal{I} := \mathcal{I}_1 \cap \mathcal{I}_2$, $ a_i^T \in \R^{n} $ is the $i$-th row of $A$ and $b_i \in \R$ is the $i$-th element of $b$. 


# Exact penalty subproblem 

The central focus of the following algorithms is the numerical solution of exact penalty subproblems, which we define to be any problem of the form:
$$
\begin{align*}
\min_{x\in\mathcal{X}} \quad J(x), 
\quad \text{where} \quad 
J(x) :=  g^T x + \frac{1}{2} x^T H x + \sum_{i\in\mathcal{I}_1} \lvert a_i^T x + b_i \rvert + \sum_{i\in\mathcal{I}_2} \max\{a_i^T x + b_i, 0\}
\end{align*}
$$