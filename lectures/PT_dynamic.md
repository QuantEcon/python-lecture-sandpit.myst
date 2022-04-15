---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Section IV

+++

## 1 Formulation
Now, let's consider the repeated versions of the previous models. 

- The repeated version allows a given individual's consumption and effort to be variable over time and for the distribution of agent characteristics in the population to change over time. 
- Now we assume a discount factor less than 1 to bound the objective. Then the social problem is to maximize the discounted sum of social surpluses subject to each agent receiving a given ex ante expected discounted utility. 
- Under infinite periods, this is exactly a dynamic programming setting. Theoretically we can guarantee that the dynamic programming problem has a unique solution and thus functional fixed point iteration algorithm works (Please refer to section V for details).
- Thus, we can rewrite the problem into a dynamic programming one, with most parts are the same, except that we now have a fourth dimension, $\mathbf{W'}$, and that we also have the value function in the objective itself.

The repeated problem can be formulated as:

$$
\begin{align*}
\max_{\Pi^{w}} & \ s(w) = \sum_{\mathbf{A} \times \mathbf{Q} \times \mathbf{C}  \times \mathbf{W'}} \{(q-c) + \beta s^*(w')\}\Pi^{w}(a,q,c, w') \\
\mbox{subject to} \ 
\mbox{C5:} & \ w = \sum_{\mathbf{A}\times\mathbf{Q}\times\mathbf{C}\times\mathbf{W'}}\{U(a,c)+\beta w'\}\Pi^{w}(a,q,c,w') \\
\mbox{C6:} & \ \sum_{\mathbf{C}\times\mathbf{W'}} \Pi^{w}(\bar{a}, \bar{q}, c, w')=P(\bar{q} \mid \bar{a}) \sum_{\mathbf{Q} \times \mathbf{C} \times \mathbf{W'}} \Pi^{w}(\bar{a}, q, c, w') \\
\mbox{C7:} & \ \sum_{\mathbf{A} \times \mathbf{Q} \times \mathbf{C} \times \mathbf{W'}} \Pi^{w}(a, q, c, w')=1, \Pi^{w}(a, q, c, w') \geqq 0 \\
\mbox{C8:} & \ \sum_{\mathbf{Q} \times \mathbf{C} \times \mathbf{W'}} \{U(a,c)+\beta w'\}\Pi^{w}(a,q,c,w') \geqq \sum_{\mathbf{Q} \times \mathbf{C}  \times \mathbf{W'}} \{U[\hat{a}, c] + \beta w' \} \frac{P(q \mid \hat{a})}{P(q \mid a)} \Pi^{w}(a, q, c, w')
\end{align*} 
$$

Let's begin with some imports!

```{code-cell} ipython3
import numpy as np
import cvxpy as cp
import cylp

from time import time
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
```

## 2 Solve the repeated problem

+++

### 2.1 The first version

+++

For the convenience of reuse, we firstly define a function "solve_static_problem" that solves the static problem and a function "solve_repeated_problem" that solves the repeated problem given $\mathbf{W'}$ and $s(w')$.

We treated the three-dimensional decision variables as a list of two-dimensional matrix CVXPY variables and the four-dimensional one as a matrix of matrix CVXPY variables, like what we have done before.

By calling "solve_repeated_problem" at each iteration, we then define a function "solve_multi_period_economy" that solves the multi-period economy with either finite periods or infinite periods.

```{code-cell} ipython3
# Define the function that solves the static problem
def solve_static_problem(W=None, u=None, A=None, Q=None, C=None, P=None, problem_type=None):
    '''
    Function: Solve the static problem
    
    Parameters
    ----------
    W: 1-D array
        The expected utility.
    u: function
        The utility function in terms of actions and consumptions.
    A: 1-D array
        The finite set of possible actions.
    Q: 1-D array
        The finite set of possible outputs.
    C: 1-D array
        The finite set of possible consumptions.
    P: 2-D array
        The probability matrix of outputs given an action.
    problem_type: str, "full information" or "unobserved-actions"
        The problem type, i.e. the full information problem or the unobserved-action problem.
        
    Returns
    -------
    s_W: 1-D array
        The optimal values of surplus for each w in w_vec.
    Pi: 4-D array
        The probility of (a, q, c) given w.
    '''
    
    # Define parameter
    n_A, n_Q, n_C = len(A), len(Q), len(C)
    A_ind, Q_ind, C_ind = range(n_A), range(n_Q), range(n_C)
    
    Phi = np.array([[q-c for c in C] for q in Q])
    U = np.array([[u(a, c) for c in C] for a in A])
    
    w = cp.Parameter()
        
    # Define variable Pi_x
    Pi_list = list(np.zeros(n_A))
    
    for a_ind in A_ind:
        Pi_list[a_ind] = cp.Variable((n_Q, n_C))

    # Define objective function
    obj_expr = cp.sum([cp.sum(cp.multiply(Pi_list[a_ind], Phi)) for a_ind in A_ind])
    obj = cp.Maximize(obj_expr)
    
    # Define constraints
    C1 = [cp.sum([cp.sum([cp.sum(cp.multiply(Pi_list[a_ind][q_ind, :], U[a_ind, :])) for a_ind in A_ind]) 
                  for q_ind in Q_ind]) == w]
    C2 = [(cp.sum(Pi_list[a_ind], axis = 1)[q_ind] == P[a_ind, q_ind]*cp.sum(Pi_list[a_ind])) 
          for a_ind in A_ind for q_ind in Q_ind]
    C3 = [cp.sum([cp.sum(Pi_list[a_ind]) for a_ind in A_ind]) == 1] + [(Pi_list[a_ind] >= 0) for a_ind in A_ind]
        
    problem_type = problem_type.lower()
    if problem_type == "full information":
        constraints = C1 + C2 + C3
    else:
        C4 = [(cp.sum([cp.sum(cp.multiply(Pi_list[a_ind][q_ind, :],U[a_ind, :])) for q_ind in Q_ind]) >= 
               cp.sum([cp.sum(cp.multiply(Pi_list[a_ind][q_ind, :],U[a_ind_hat, :]))*P[a_ind_hat, q_ind]/P[a_ind, q_ind] 
                       for q_ind in Q_ind])) 
              for a_ind in A_ind for a_ind_hat in A_ind]
        constraints = C1 + C2 + C3 + C4

    # Create the problem
    problem = cp.Problem(obj, constraints)
    
    # Initialize output variables
    s_W = np.zeros(len(W))
    Pi = np.zeros((len(W), len(A), len(Q), len(C)))
    
    # Solve the problem
    for i in range(len(W)):
        w.value = W[i]
        problem.solve(solver=cp.CBC)
        s_W[i] = obj_expr.value
        for a_ind in A_ind:
            Pi[i, a_ind, :, :] = Pi_list[a_ind].value
    
    return s_W, Pi
```

```{code-cell} ipython3
# Define the function that solves the dynamic problem at one iteration
def solve_repeated_problem(W=None, u=None, A=None, Q=None, C=None, W_prime=None, s_W_prime=None, P=None,
                           problem_type=None, β=0.8):
    '''
    Function: Solve the dynamic problem at one iteration
    
    Parameters
    ----------
    W: 1-D array
        The expected utility.
    u: function
        The utility function in terms of actions and consumptions.
    A: 1-D array
        The finite set of possible actions.
    Q: 1-D array
        The finite set of possible outputs.
    C: 1-D array
        The finite set of possible consumptions.
    W_prime: 1-D array
        The finite set of possible w_prime.
    s_W_prime: 1-D array
        The finite set of optimal values of surplus of w_prime.
    P: 2-D array
        The probability matrix of outputs given an action.
    problem_type: str, "full information" or "unobserved-actions"
        The problem type, i.e. the full information problem or the unobserved-action problem.
    β: float, optional
        The discouted factor. The value is 0.8 by default.
        
    Returns
    -------
    s_W: 1-D array
        The optimal values of surplus for each w in w_vec.
    Pi: 5-D array
        The probility of (a, q, c, w_prime) given w.
    '''
    
    # Define parameters
    n_A, n_Q, n_C, n_W_prime = len(A), len(Q), len(C), len(W_prime)
    A_ind, Q_ind, C_ind, W_prime_ind = range(n_A), range(n_Q), range(n_C), range(n_W_prime)
    
    U = np.array([[u(a, c) for c in C] for a in A])
    Phi = np.array([[[q-c+β*s_w_prime for s_w_prime in s_W_prime] for c in C] for q in Q])
    U_disc = np.array([[[u(a, c)+β*w_prime for w_prime in W_prime] for c in C] for a in A])
    
    w = cp.Parameter()
        
    # Define variables
    Pi_list = list(np.zeros(n_A))
    for a_ind in A_ind:
        Pi_list[a_ind] = list(np.zeros(n_Q))
        for q_ind in Q_ind:
            Pi_list[a_ind][q_ind] = cp.Variable((n_C, n_W_prime))

    # Define the objective function
    obj_expr = cp.sum([cp.sum(cp.multiply(Phi[q_ind, :, :], Pi_list[a_ind][q_ind])) for a_ind in A_ind for q_ind in Q_ind])
    obj = cp.Maximize(obj_expr)
    
    # Define constraints
    C5 = [cp.sum([cp.sum(cp.multiply(U_disc[a_ind, :, :], Pi_list[a_ind][q_ind])) 
                  for a_ind in A_ind for q_ind in Q_ind]) == w]
    C6 = [(cp.sum(Pi_list[a_ind][q_ind]) == 
           P[a_ind, q_ind]*cp.sum([cp.sum(Pi_list[a_ind][q_ind_1]) for q_ind_1 in Q_ind])) 
          for a_ind in A_ind for q_ind in Q_ind]
    C7 = [cp.sum([cp.sum(Pi_list[a_ind][q_ind]) for a_ind in A_ind for q_ind in Q_ind]) == 1]
    C7 = C7 + [(Pi_list[a_ind][q_ind] >= 0) for a_ind in A_ind for q_ind in Q_ind]
   
    problem_type = problem_type.lower()
    if problem_type == "full information":
        constraints = C5 + C6 + C7
    else:
        C8 = [(cp.sum([cp.sum(cp.multiply(U_disc[a_ind, :, :], Pi_list[a_ind][q_ind])) for q_ind in Q_ind]) >= 
               cp.sum([cp.sum(cp.multiply(U_disc[a_ind_hat, :, :], Pi_list[a_ind][q_ind]))*P[a_ind_hat, q_ind]/P[a_ind, q_ind] 
                       for q_ind in Q_ind]))
              for a_ind in A_ind for a_ind_hat in A_ind] 
        constraints = C5 + C6 + C7 + C8

    # Create the problem
    problem = cp.Problem(obj, constraints)
    
    # Initialize output variables
    s_W = np.zeros(len(W))
    Pi = np.zeros((len(W), len(A), len(Q), len(C), len(W_prime))) 
    
    # Solve the problem
    for i in range(len(W)):
        w.value = W[i]
        problem.solve(solver = cp.CBC)
        s_W[i] = obj_expr.value
        for a_ind in A_ind:
            for q_ind in Q_ind:
                Pi[i, a_ind, q_ind, :, :] = Pi_list[a_ind][q_ind].value
    
    return s_W, Pi
```

```{code-cell} ipython3
# Define the function that solve the infinite-period or finite-period economy
def solve_multi_period_economy(u=None, A=None, Q=None, C=None, P=None, problem_type=None, T=None, β=0.8, N=100, 
                               s_W_0=None, tol=1e-8):
    '''
    Function: Solve the multi-period problem, either infinite-period or finite-period
    
    Parameters
    ----------
    u: function
        The utility function in terms of actions and consumptions.
    A: 1-D array
        The finite set of possible actions.
    Q: 1-D array
        The finite set of possible outputs.
    C: 1-D array
        The finite set of possible consumptions.
    P: 2-D array
        The probability matrix of outputs given an action.
    problem_type: str, "full information" or "unobserved-actions"
        The problem type, i.e. the full information problem or the unobserved-action problem.
    T: int, optional
        The number of periods. If T is None, the algorithm solves the infinite-period economy. If T is some
        integer, the algorithm solves the T-period economy. By default, T is None.
    β: float, optional
        The discouted factor in (0,1). The value is 0.8 by default.
    N: int, optional
        The length of discretized parameter space W.
    s_W_0: 1-D array, optional
        The initial guess for s_W with a length of N.
    tol: float, optional
        The precision of convergence.
        
    Returns
    -------
    s_W: 1-D array
        The optimal values of convergent surplus for each w in w_vec.
    Pi: 5-D array
        The convergent probility of (a, q, c, w') given w for an infinite-period economy. Or, the convergent 
        probility (a, q, c, w_{T-1}) given w_T for a finite-period economy.
    '''
    
    if β >= 1 or β <= 0:
        raise ValueError('β should lie in [0,1]')
        
    if T is None:
        # Discretize the parameter space W
        problem_type = problem_type.lower()
        if problem_type == "full information":
            w_l = u(A.max(), C.min())/(1-β)
            w_u = u(A.min(), C.max())/(1-β)
        else:
            w_l = u(A.min(), C.min())/(1-β)
            w_u = u(A.min(), C.max())/(1-β)
        W = np.linspace(w_l, w_u, N)

        # Assign initial value for s_W
        if s_W_0 is not None:
            s_W_prime = s_W_0
        else:
            s_W_prime = np.zeros(N)

        # Iterate
        optimal = False
        iteration = 1
        while not optimal:
            print('Iteration %i in process'%iteration)
            start_time = time()
            s_W, Pi = solve_repeated_problem(W=W, u=u, A=A, Q=Q, C=C, W_prime=W, s_W_prime=s_W_prime, P=P,
                                                     problem_type=problem_type, β=β)
            end_time = time()
            print('Iteration %i finished in:'%iteration, round(end_time-start_time, 2), 's')
            print('---------')
            if np.max(np.abs(s_W-s_W_prime)) <= tol:
                optimal = True
            else:
                s_W_prime = s_W
                
            iteration += 1
    
    if T is not None:
        # Discretize the parameter space W
        W_mat = np.zeros((T, N))
        
        problem_type = problem_type.lower()
        if problem_type == "full information":
            w_l = u(A.max(), C.min())
            w_u = u(A.min(), C.max())
        else:
            w_l = u(A.min(), C.min())
            w_u = u(A.min(), C.max())
        W_mat = np.cumsum(np.logspace(0, T-1, T, base=β).reshape(T, 1)*np.linspace(w_l, w_u, N).reshape(1, N), 
                          axis=0)
        
        # Solve the 1-period economy
        print('Solving the 1-period economy')
        print('-------')
        s_W, Pi = solve_static_problem(W=W_mat[0, :], u=u, A=A, Q=Q, C=C, P=P, problem_type=problem_type)
        
        if T != 1:
            for t in range(2, T+1):
                print('Solving the %i-period economy'%t)
                print('-------')
                s_W_prime = np.copy(s_W)
                s_W, Pi = solve_repeated_problem(W=W_mat[t-1,:], u=u, A=A, Q=Q, C=C, W_prime=W_mat[t-2,:], 
                                                 s_W_prime=s_W_prime, P=P, problem_type=problem_type, β=β)
    return s_W, Pi
```

Now, we have already had methods to solve all kinds of problems. Here we give some specific examples.

Let's begin with defining parameters we need.

```{code-cell} ipython3
# Define the function u[a,c]
def u(a, c):
    return c**0.5/0.5 + (1-a)**0.5/0.5

# Define the parameters
A = np.array([0, 0.2, 0.4, 0.6])
Q = np.array([1,2])
C = np.linspace(0, 2.25, 81)
P = np.array([[0.9, 0.1], 
              [0.6, 0.4], 
              [0.4, 0.6], 
              [0.25, 0.75]])
β = 0.8
N = 50
```

```{code-cell} ipython3
# Solve the finite-period economy
%time s_W_T, Pi_T = solve_multi_period_economy(u, A, Q, C, P, "unobserved-actions", T=3, N=N)
```

```{code-cell} ipython3
T = 3
w_l = u(A.min(), C.min())
w_u = u(A.min(), C.max())
W_mat = np.cumsum(np.logspace(0, T-1, T, base=β).reshape(T, 1)*np.linspace(w_l, w_u, N).reshape(1, N), axis=0)
W = W_mat[2, :]

plt.figure(figsize=(6.5, 6.5))
plt.plot(W, s_W_T, "k-.")
plt.text(8, 3, "3-Period Unobserved Action", size=12)
plt.title("Figure\n Optimized surplus function", y=-0.2)
plt.show()
```

For the infinite-period economy, we use the solution to the static problem as the initial value for the algorithm. The algorithm converges at 40-th iteration.

So far, the algorithm is not fast enough. So, we relax our parameters as N=50 and tol=1e-5.

You may skip this parts because it will take a long time to finish. We provide an improved algorithm in the following section.

```{code-cell} ipython3
# Solve the static unobserved-actions problem
w_l = u(A.min(), C.min())/(1-β)
w_u = u(A.min(), C.max())/(1-β)
W = np.linspace(w_l, w_u, N)
%time s_W_0, Pi_0 = solve_static_problem(W*(1-β), u, A, Q, C, P, "unobserved-actions")
```

```{code-cell} ipython3
# Solve the infinite-period unobserved-actions economy
%time s_W, Pi = solve_multi_period_economy(u, A, Q, C, P, "unobserved-actions", N=N, s_W_0=s_W_0/(1-β), tol=1e-5)
```

```{code-cell} ipython3
plt.figure(figsize=(6.5, 6.5))
plt.plot(W, s_W, "k-.")
plt.xlim([5.0, 25.0])
plt.ylim([-7.5, 10.0])
plt.xlabel("w")
plt.ylabel("s(w)")
plt.title("Figure\n Optimized surplus function", y=-0.2)
plt.text(15, 6.5, "Infinity Unobserved Action", size=12)
plt.show()
```

### 2.2 The second version

+++

Actually, each iteration of this algorithm takes about 12 seconds and the whole process takes about ten minutes. It seems not too bad. But, here we set N as 50 and the tolerance as 1e-5. N is not large enough to draw perfect pictures. And tolerance as 1e-5 is also not ideal. If set N as 100 and tolerance as 1e-8, each iteration would take about 65 seconds and the process needs 81 iterations to convergence. As a whole, the algorithm needs one and a half hours to finish. 

As inspired by the Section VI in the paper, we develop a new function "solve_repeated_problem_v2" which implements an algorithm that divides one period into two sub-periods with two sub-linear-programming problems. This new function significantly shortens the running time.

Here, we separate the utility function $U[a, c] = 2\sqrt{c} + 2\sqrt{1-a}$ into two independent parts. So in the new function, we provide no argument "u".

For each period, $a, q, c, w'$ are random variables but $w$ are given. We seek to solve the optimal probability $\Pi^w(a, q, c, w') = Pr(a, q, c, w'|w)$ and optimal surplus $s(w)$.

Now, instead of directly solving the original problem, we solve two sub-problems. At each period, suppose the decision is made by two steps:

1. The output is realized but the consumption is still undecided.
2. The consumption is decided.

In the first step, $a, q, w^m$ are random variables but $w$ are given. We seek to find the optimal probability of $(a, q, w^m)$ that maximize the surplus function $s(w)$, where $w^m$ is the expected utility of the second step. 

In the second step, $c, w'$ are random variables but $w^m$ are given. We seek to find the optimal probability of $(c, w')$ that maximize the surplus function $s^m(w^m)$.

The problem of the first step is:
$$
\begin{align*}
\max_{\Pi^{w}} & \ s(w) = \sum_{\mathbf{A} \times \mathbf{Q} \times \mathbf{W^m}} \{q + s^m(w^m)\} \Pi^w(a, q, w^m) \\
\mbox{subject to} \ 
\mbox{C5:} & \ w = \sum_{\mathbf{A}\times\mathbf{Q}\times\mathbf{W^m}}\{2 \sqrt{1-a} +w^m\} \Pi^w(a, q, w^m) \\
\mbox{C6:} & \ \sum_{\mathbf{W^m}} \Pi^w(\bar{a}, \bar{q}, w^m)=P(\bar{q} \mid \bar{a}) \sum_{\mathbf{Q} \times \mathbf{W^m}} \Pi^w(\bar{a}, q, w^m) \\
\mbox{C7:} & \ \sum_{\mathbf{A} \times \mathbf{Q} \times \mathbf{W^m}} \Pi^w(a, q, w^m)=1, \Pi^w(a, q, w^m) \geqq 0 \\
\mbox{C8:} & \ \sum_{\mathbf{Q} \times \mathbf{W^m}} \{2 \sqrt{1-a}+w^m\} \Pi^w(a, q, w^m) \geqq \\
& \ \sum_{\mathbf{Q} \times \mathbf{W^m}} \{2 \sqrt{1-\hat{a}} + w^m \} \frac{P(q \mid \hat{a})}{P(q \mid a)} \Pi^w(a, q, w^m)
\end{align*}
$$

The problem of the second step is:

$$
\begin{align*}
\max_{\Pi^{w^m}} & \ s^m(w^m) = \sum_{\mathbf{C}  \times \mathbf{W'}} \{\beta s^*(w') - c\}\Pi^{w^m}(c, w') \\
\mbox{subject to} \ 
\mbox{C5:} & \ w^m = \sum_{\mathbf{C} \times \mathbf{W'}} \{ 2\sqrt{c}+\beta w'\}\Pi^{w^m}(c, w') \\
\mbox{C7:} & \ \sum_{\mathbf{C} \times \mathbf{W'}} \Pi^{w^m}(c, w')=1, \Pi^{w^m}(c, w') \ge 0 \\
\end{align*}
$$

We need to solve the problem of step 2 firstly in order to get the value of $s^m(w^m)$. After solving this problem, we also get the optimal probability $\Pi^{w^m}(c, w') = Pr(c, w'|w^m)$. 

Then, we solve the problem of step 1 and get $s(w)$ and $\Pi^{w}(a, q, w^m) = Pr(a, q, w^m|w)$.

```{code-cell} ipython3
# Define the function that solves the dynamic problem at one iteration
def solve_repeated_problem_2(W=None, W_m=None, A=None, Q=None, C=None, W_prime=None, s_W_prime=None, P=None,
                             problem_type=None, β=0.8):
    '''
    Function: Solve the dynamic problem at one iteration
    
    Parameters
    ----------
    W: 1-D array
        The expected utility.
    A: 1-D array
        The finite set of possible actions.
    Q: 1-D array
        The finite set of possible outputs.
    C: 1-D array
        The finite set of possible consumptions.
    W_prime: 1-D array
        The finite set of possible w_prime.
    s_W_prime: 1-D array
        The finite set of optimal values of surplus of w_prime.
    P: 2-D array
        The probability matrix of outputs given an action.
    problem_type: str, "full information" or "unobserved-actions"
        The problem type, i.e. the full information problem or the unobserved-action problem.
    β: float, optional
        The discouted factor. The value is 0.8 by default.
        
    Returns
    -------
    s_W: 1-D array
        The optimal values of surplus for each w in w_vec.
    Pi_W_s1: 4-D array
        The probility of (a, q, w_m) given w.
    Pi_W_m_s2: 3-D array
        The probility of (c, w_prime) given w_m.
    '''
    
    n_A, n_Q, n_C, n_W, n_W_m, n_W_prime = len(A), len(Q), len(C), len(W), len(W_m), len(W_prime)
    A_ind, Q_ind, C_ind, W_ind, W_m_ind, W_prime_ind = range(n_A), range(n_Q), range(n_C), range(n_W), range(n_W_m), range(n_W_prime)
    
    # Problem of step 2
    
    # Define parameters
    Phi_s2 = np.array([[β*s_w_prime - c for s_w_prime in s_W_prime] for c in C])
    U_disc_s2 = np.array([[2*c**0.5+β*w_prime for w_prime in W_prime] for c in C])
    
    w_m_para = cp.Parameter()
    
    
    # Define variables
    Pi_w_m = cp.Variable((n_C, n_W_prime))

    # Define the objective function
    obj_expr_s2 = cp.sum(cp.multiply(Phi_s2, Pi_w_m))
    obj_s2 = cp.Maximize(obj_expr_s2)
    
    # Define constraints
    C5_s2 = [cp.sum(cp.multiply(U_disc_s2, Pi_w_m)) == w_m_para]
    C7_s2 = [cp.sum(Pi_w_m) == 1] + [Pi_w_m >= 0]
    
    # Create the problem of step 2
    problem_s2 = cp.Problem(obj_s2, C5_s2 + C7_s2)
    
    # Solve the probelm of step 2
    s_W_m = np.zeros(n_W_m)
    Pi_W_m_s2 = np.zeros((n_W_m, n_C, n_W_prime))
    for w_m, w_m_ind in zip(W_m, W_m_ind):
        w_m_para.value = w_m
        problem_s2.solve(solver = cp.CBC)
        s_W_m[w_m_ind] = obj_expr_s2.value
        Pi_W_m_s2[w_m_ind, :, :] = Pi_w_m.value
    
    # Problem of step 1
    
    # Define parameters
    Phi_s1 = np.array([[(q+s_w_m) for s_w_m in s_W_m] for q in Q])
    U_disc_s1 = np.array([[[2*(1-a)**0.5+w_m for w_m in W_m] for q in Q] for a in A])
    U_disc_hat_s1 = np.array([[[[(2*(1-A[a_hat_ind])**0.5+W_m[w_m_ind])*P[a_hat_ind, q_ind]/P[a_ind, q_ind] 
                                 for w_m_ind in W_m_ind] for q_ind in Q_ind] for a_ind in A_ind] 
                              for a_hat_ind in A_ind])
    
    w_para = cp.Parameter()
    
    # Define variables
    Pi_w_list = list(np.zeros(n_A))
    for a_ind in A_ind:
        Pi_w_list[a_ind] = cp.Variable((n_Q, n_W_m))
    
    # Define the objective function
    obj_expr_s1 = cp.sum([cp.sum(cp.multiply(Phi_s1, Pi_w_list[a_ind])) for a_ind in A_ind])
    obj_s1 = cp.Maximize(obj_expr_s1)
                                       
    # Define constraints
    C5_s1 = [cp.sum([cp.sum(cp.multiply(U_disc_s1[a_ind, :, :], Pi_w_list[a_ind])) for a_ind in A_ind]) == w_para]
    C6_s1 = [(cp.sum(Pi_w_list[a_ind][q_ind, :]) == P[a_ind, q_ind] * cp.sum(Pi_w_list[a_ind])) 
             for q_ind in Q_ind for a_ind in A_ind]
    C7_s1 = [cp.sum([cp.sum(Pi_w_list[a_ind]) for a_ind in A_ind]) == 1]
    C7_s1 = C7_s1 + [(Pi_w_list[a_ind] >= 0) for a_ind in A_ind]
    
    problem_type = problem_type.lower()
    if problem_type == "full information":
        constraints_s1 = C5_s1 + C6_s1 + C7_s1
    else:
        C8_s1 = [(cp.sum(cp.multiply(U_disc_s1[a_ind, :, :], Pi_w_list[a_ind])) >= 
                 cp.sum(cp.multiply(U_disc_hat_s1[a_hat_ind, a_ind, :, :], Pi_w_list[a_ind]))) 
                 for a_ind in A_ind for a_hat_ind in A_ind] 
        constraints_s1 = C5_s1 + C6_s1 + C7_s1 + C8_s1
    
    # Create the problem of step 1
    problem_s1 = cp.Problem(obj_s1, constraints_s1)
    
    # Solve the problem of step 1
    s_W = np.zeros(n_W)
    Pi_W_s1 = np.zeros((n_W, n_A, n_Q, n_W_m))
    for w, w_ind in zip(W, W_ind):
        w_para.value = w
        problem_s1.solve(solver = cp.CBC)
        s_W[w_ind] = obj_expr_s1.value
        for a_ind in A_ind:
            Pi_W_s1[w_ind, a_ind, :, :] = Pi_w_list[a_ind].value                
    return s_W, Pi_W_s1, Pi_W_m_s2
```

Let's test this new function and compare the results with results from the previous function.

```{code-cell} ipython3
# Define the function u[a,c]
def u(a, c):
    return c**0.5/0.5 + (1-a)**0.5/0.5

# Define the parameters
A = np.array([0, 0.2, 0.4, 0.6])
Q = np.array([1,2])
C = np.linspace(0, 2.25, 81)
P = np.array([[0.9, 0.1], 
              [0.6, 0.4], 
              [0.4, 0.6], 
              [0.25, 0.75]])
β = 0.8
N = 100
N_m = 100

W_l = u(A.min(), C.min())/(1-β)
W_u = u(A.min(), C.max())/(1-β)
W = np.linspace(W_l, W_u, N)

W_m_l = β*W_l + 2*C.min()**0.5
W_m_u = β*W_u + 2*C.max()**0.5
W_m = np.linspace(W_m_l, W_m_u, N_m)

%time s_W_0, Pi_0 = solve_static_problem(W*(1-β), u, A, Q, C, P, "unobserved-actions")
```

```{code-cell} ipython3
# The new function
start_time = time()
s_W_2, _, _ = solve_repeated_problem_2(W=W, W_m=W_m, A=A, Q=Q, C=C, W_prime=W, s_W_prime=s_W_0, P=P, 
                                       problem_type="unobserved-actions", β=0.8)
end_time = time()
print("The process finished in:", end_time-start_time, "s")
```

```{code-cell} ipython3
# The previous function
start_time = time()
s_W_1, _ = solve_repeated_problem(W=W, u=u, A=A, Q=Q, C=C, W_prime=W, s_W_prime=s_W_0, P=P, 
                                     problem_type="unobserved-actions", β=0.8)
end_time = time()
print("The process finished in:", end_time-start_time, "s")
```

```{code-cell} ipython3
# Compare results
plt.plot(np.abs(s_W_2 - s_W_1))
plt.show()
print("max of error:", np.abs(s_W_2 - s_W_1).max())
print("min of error:", np.abs(s_W_2 - s_W_1).min())
print("mean of error:", np.abs(s_W_2 - s_W_1).mean())
print("std of error:", np.abs(s_W_2 - s_W_1).std())
print("largest error point: W[%i]"%np.abs(s_W_2 - s_W_1).argmax())
```

The results show that this new algorithm is an approximation to the previous one. The more exciting thing is this new algorithm is much faster than the previous one.

However, there does exist difference between two results. The reason is the choice of the intermediate parameter space $W_m$. 

$W_m = [2\sqrt{c_{min}} + w'_{min}, \ 2\sqrt{c_{max}} + w'_{max}]$, which is suppsoed to be a continuous inteval including uncountable points. Here, we again discretized $W_m$ into 100 points for the numerical purpose. Discrete $W_m$ could only be an approximation to the previous problem.

However, taking $W[6]$ as an example, which has the largest error, we can show that the error goes smaller as $N_m$ goes larger.

```{code-cell} ipython3
s_W6_1, _ = solve_repeated_problem(W=[W[6]], u=u, A=A, Q=Q, C=C, W_prime=W, s_W_prime=s_W_0, P=P, 
                                   problem_type="unobserved-actions", β=0.8)

N_m_space = [100, 200, 300, 400, 500]
err = np.ones(len(N_m_space))
for i in range(len(N_m_space)):
    W_m = np.linspace(W_m_l, W_m_u, N_m_space[i])
    s_W6_2, _, _ = solve_repeated_problem_2(W=[W[6]], W_m=W_m, A=A, Q=Q, C=C, W_prime=W, s_W_prime=s_W_0, P=P, 
                                            problem_type="unobserved-actions", β=0.8)
    err[i] = abs(s_W6_2[0] - s_W6_1[0])

plt.plot(err)
plt.show()
print("Error:", err)
```

Now, let's solve the multi-period economy using the function "solve_repeated_problem_2". The new function "solve_multi_period_economy_2" is almost same as the previous one.

```{code-cell} ipython3
# Define the function that solve the infinite-period or finite-period economy
def solve_multi_period_economy_2(A=None, Q=None, C=None, P=None, problem_type=None, T=None, β=0.8, N=100, N_m=100, 
                                 s_W_0=None, tol=1e-8):
    '''
    Function: Solve the multi-period problem, either infinite-period or finite-period
    
    Parameters
    ----------
    A: 1-D array
        The finite set of possible actions.
    Q: 1-D array
        The finite set of possible outputs.
    C: 1-D array
        The finite set of possible consumptions.
    P: 2-D array
        The probability matrix of outputs given an action.
    problem_type: str, "full information" or "unobserved-actions"
        The problem type, i.e. the full information problem or the unobserved-action problem.
    T: int, optional
        The number of periods. If T is None, the algorithm solves the infinite-period economy. If T is some
        integer, the algorithm solves the T-period economy. By default, T is None.
    β: float, optional
        The discouted factor in (0,1). The value is 0.8 by default.
    N: int, optional
        The length of discretized parameter space W.
    N_m: int, optional
        The length of discretized parameter space W_m.
    s_W_0: 1-D array, optional
        The initial guess for s_W with a length of N.
    tol: float, optional
        The precision of convergence.
        
    Returns
    -------
    s_W: 1-D array
        The optimal values of convergent surplus for each w in w_vec.
    Pi_W_s1: 4-D array
        The probility of (a, q, w_m) given w.
    Pi_W_m_s2: 3-D array
        The probility of (c, w_prime) given w_m.
    '''
    
    if β >= 1 or β <= 0:
        raise ValueError('β should lie in [0,1]')
        
    # Define the function u[a,c]
    def u(a, c):
        return c**0.5/0.5 + (1-a)**0.5/0.5
        
    if T is None:
        # Discretize the parameter space W and W_m
        problem_type = problem_type.lower()
        if problem_type == "full information":
            w_l = u(A.max(), C.min())/(1-β)
            w_u = u(A.min(), C.max())/(1-β)
        else:
            w_l = u(A.min(), C.min())/(1-β)
            w_u = u(A.min(), C.max())/(1-β)
        W = np.linspace(w_l, w_u, N)
        
        W_m_l = β*W_l + 2*C.min()**0.5
        W_m_u = β*W_u + 2*C.max()**0.5
        W_m = np.linspace(W_m_l, W_m_u, N_m)

        # Assign initial value for s_W
        if s_W_0 is not None:
            s_W_prime = s_W_0
        else:
            s_W_prime = np.zeros(N)

        # Iterate
        optimal = False
        iteration = 1
        while not optimal:
            print('Iteration %i in process'%iteration)
            start_time = time()
            s_W, Pi_W_s1, Pi_W_m_s2 = solve_repeated_problem_2(W=W, W_m=W_m, A=A, Q=Q, C=C, W_prime=W, 
                                                               s_W_prime=s_W_prime, P=P, 
                                                               problem_type=problem_type, β=β)
            end_time = time()
            print('Iteration %i finished in:'%iteration, round(end_time-start_time, 2), 's')
            print('---------')
            
            if np.max(np.abs(s_W-s_W_prime)) <= tol:
                optimal = True
            else:
                s_W_prime = s_W
                
            iteration += 1
    
    if T is not None:
        # Discretize the parameter space W
        W_mat = np.zeros((T, N))
        
        problem_type = problem_type.lower()
        if problem_type == "full information":
            w_l = u(A.max(), C.min())
            w_u = u(A.min(), C.max())
        else:
            w_l = u(A.min(), C.min())
            w_u = u(A.min(), C.max())
        W_mat = np.cumsum(np.logspace(0, T-1, T, base=β).reshape(T, 1)*np.linspace(w_l, w_u, N).reshape(1, N), 
                          axis=0)
        
        # Solve the 1-period economy
        print('Solving the 1-period economy')
        print('-------')
        s_W, Pi = solve_static_problem(W=W_mat[0, :], u=u, A=A, Q=Q, C=C, P=P, problem_type=problem_type)
        
        if T != 1:
            for t in range(2, T+1):
                print('Solving the %i-period economy'%t)
                print('-------')
                s_W_prime = np.copy(s_W)
                W_m_l = β*W_mat[t-2,:].min() + 2*C.min()**0.5
                W_m_u = β*W_mat[t-2,:].max() + 2*C.max()**0.5
                W_m = np.linspace(W_m_l, W_m_u, N_m)
                s_W, Pi_W_s1, Pi_W_m_s2 = solve_repeated_problem_2(W=W_mat[t-1,:], W_m=W_m, A=A, Q=Q, C=C, 
                                                                 W_prime=W_mat[t-2,:], s_W_prime=s_W_prime, P=P, 
                                                                 problem_type=problem_type, β=β)
    return s_W, Pi_W_s1, Pi_W_m_s2
```

Let's present the numerical examples using new algorithm.

```{code-cell} ipython3
# Define the parameters
A = np.array([0, 0.2, 0.4, 0.6])
Q = np.array([1,2])
C = np.linspace(0, 2.25, 81)
P = np.array([[0.9, 0.1], 
              [0.6, 0.4], 
              [0.4, 0.6], 
              [0.25, 0.75]])
β = 0.8
N = 100
N_m = 100
```

```{code-cell} ipython3
# Solve the finite-period economy
%time s_W_T, Pi_W_s1_T, Pi_W_m_s2_T = solve_multi_period_economy_2(A, Q, C, P, "unobserved-actions", T=3, N=N, N_m=N_m)
```

```{code-cell} ipython3
T = 3
w_l = u(A.min(), C.min())
w_u = u(A.min(), C.max())
W_mat = np.cumsum(np.logspace(0, T-1, T, base=β).reshape(T, 1)*np.linspace(w_l, w_u, N).reshape(1, N), axis=0)
W = W_mat[2, :]

plt.figure(figsize=(6.5, 6.5))
plt.plot(W, s_W_T, "k-.")
plt.text(8, 3, "3-Period Unobserved Action", size=12)
plt.title("Figure\n Optimized surplus function", y=-0.2)
plt.show()
```

```{code-cell} ipython3
# Solve the static unobserved-actions problem
w_l = u(A.min(), C.min())/(1-β)
w_u = u(A.min(), C.max())/(1-β)
W = np.linspace(w_l, w_u, N)
%time s_W_0, Pi_0 = solve_static_problem(W*(1-β), u, A, Q, C, P, "unobserved-actions")
```

```{code-cell} ipython3
# Solve the infinite-period unobserved-actions economy
%time
s_W, Pi_W_s1, Pi_W_m_s2 = solve_multi_period_economy_2(A, Q, C, P, "unobserved-actions", N=N, N_m=N_m, 
                                                       s_W_0=s_W_0/(1-β), tol=1e-8)
```

```{code-cell} ipython3
plt.figure(figsize=(6.5, 6.5))
plt.plot(W, s_W, "k-.")
plt.xlim([5.0, 25.0])
plt.ylim([-7.5, 10.0])
plt.xlabel("w")
plt.ylabel("s(w)")
plt.title("Figure\n Optimized surplus function", y=-0.2)
plt.text(15, 6.5, "Infinity Unobserved Action", size=12)
plt.show()
```

Let's plot the figures to illustrate the solutions. Before that, we need to convert $\Pi^{w}(a, q, w^m)$ and $\Pi^{w^m}(c, w')$ into $\Pi^{w}(a, q, c, w')$.

The optimal probability $\Pi^{w}(a, q, c, w')$ can be obtained by the following equations:

$$
\begin{array}\
& & \Pi^w(a, q, c, w') \\
&=& Pr(a, q, c, w'|w) \\
&=& \sum_{W^m} Pr(a, q, c, w', w^m|w) \\
&=& \sum_{W^m} Pr(a, q, w^m|w) Pr(c, w'|a, q, w^m, w) \\
&=& \sum_{W^m} Pr(a, q, w^m|w) Pr(c, w'|w^m) \\
&=& \sum_{W^m} \Pi^{w}(a, q, w^m) \Pi^{w^m}(c, w') \\
\end{array}
$$

```{code-cell} ipython3
n_A, n_Q, n_C, n_W, n_W_prime = 4, 2, 81, N, N
A_ind, Q_ind, C_ind, W_ind, W_prime_ind = range(n_A), range(n_Q), range(n_C), range(n_W), range(n_W_prime)

Pi = np.array([[[[[Pi_W_s1[w_ind, a_ind, q_ind, :]@Pi_W_m_s2[:, c_ind, w_prime_ind] 
                   for w_prime_ind in W_prime_ind] for c_ind in C_ind] for q_ind in Q_ind] 
                for a_ind in A_ind] for w_ind in W_ind])  
```

#### Figure 5

```{code-cell} ipython3
# Solve the static full information
W_full = np.linspace(5, 25, N)
%time s_W_1, Pi_1 = solve_static_problem(W_full*(1-β), u, A, Q, C, P, "full information")
```

```{code-cell} ipython3
plt.figure(figsize=(6.5, 6.5))
plt.plot(W, s_W, "k-.")
plt.plot(W, s_W_0/(1-β), "yellow")
plt.plot(W_full, s_W_1/(1-β), "red")
plt.xlim([5.0, 25.0])
plt.ylim([-7.5, 10.0])
plt.hlines(0, 5.0, 25.0, linestyle="dashed")
plt.xlabel("w")
plt.ylabel("s(w)")
plt.title("Figure 5\n Optimized surplus function", y=-0.2)
plt.text(5.4, -2.0, "Full Information (top)", size=12)
plt.text(5.4, -3.0, "T = infinity Unobserved Action (middle)", size=12)
plt.text(5.4, -4.0, "T = 1 Unobserved Action", size=12)
plt.show()
```

#### Figure 6

```{code-cell} ipython3
# Calculate expected efforts
# T=1 Unobserved Action
X, Y, Z, N = list(range(len(A))), list(range(len(Q))), list(range(len(C))), list(range(len(W)))
Ea_1 = np.array([np.sum([A[x]*Pi_0[i,x,:,:] for x in X]) for i in N])

# T=infinity unobserved Action
Ea_inf = np.array([np.sum([A[x]*Pi[i,x,:,:,:] for x in X]) for i in N])

# Plot expected efforts
plt.figure(figsize=(6.5, 6.5))
plt.plot(W, Ea_1)
plt.plot(W, Ea_inf)
plt.xlabel("w")
plt.ylabel("E{a(w)}")
plt.xlim([5.0, 25.0])
plt.ylim([0.0, 0.8])
plt.title("Figure 6\n Actions", y=-0.2)
plt.text(14, 0.60, "T = infinity Unobserved Action (top)", size=10)
plt.text(14, 0.55, "T = 1 Unobserved Action (bottom)", size=10)
plt.show()
```

#### Figure 7

```{code-cell} ipython3
def ex_con(Pi, A, Q, C, W, type="infinity"):
    X, Y, Z, N = list(range(len(A))), list(range(len(Q))), list(range(len(C))), list(range(len(W)))
    Ec = np.zeros((len(N),len(X),len(Y)))
    for i in N:
        for x in X:
            for y in Y:
                if type == "infinity":
                    total_prob = np.sum(Pi[i,x,y,:,:])
                    if total_prob <= 1e-9:
                        Ec[i,x,y] = float("-inf")
                    else:
                        Ec[i,x,y] = np.sum([np.sum(C[z]*Pi[i,x,y,z,:]) for z in Z])/total_prob
                elif type == "one":
                    total_prob = np.sum(Pi[i,x,y,:])
                    if total_prob <= 1e-9:
                        Ec[i,x,y] = float("-inf")
                    else:
                        Ec[i,x,y] = np.sum([C[z]*Pi[i,x,y,:] for z in Z])/total_prob                   
    return Ec
```

```{code-cell} ipython3
Ec_inf = ex_con(Pi, A, Q, C, W)

# %matplotlib inline
# %matplotlib notebook
# Plot expected consumption
plt.figure(figsize=(10.5, 10.5))
for x in X:
    for y in Y:
        plt.plot(W, Ec_inf[:,x,y])
plt.xlabel("w")
plt.ylabel("E(c) given a, q, w")
plt.xlim([5.0, 25.0])
plt.ylim([0.0, 2.25])
plt.title("Figure 7\n Unobserved Action Consumption", y=-0.3)
plt.annotate("a=.4, q=2", xy=(13.5, 0.5), xytext=(10.5, 0.7), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2, q=2", xy=(20.0, 1.3), xytext=(15.5, 1.65), arrowprops={"arrowstyle":"-"})
plt.annotate("a=0, q=(1,2)", xy=(24, 2.15), xytext=(15.0, 2.15), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2, q=2", xy=(10.1, 0.01), xytext=(7.5, 0.03), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.4, q=2", xy=(10.5, 0.10), xytext=(7.5, 0.15), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.6, q=2", xy=(11.5, 0.25), xytext=(8.5, 0.30), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.6, q=1", xy=(12.5, 0.05), xytext=(14.5, 0.10), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.4, q=1", xy=(15.0, 0.35), xytext=(18, 0.2), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2, q=1", xy=(20.0, 1.1), xytext=(21.5, 0.75), arrowprops={"arrowstyle":"-"})
plt.annotate("", xy=(10.0, 0), xytext=(11.5, -0.1), arrowprops={"arrowstyle":"-"})
plt.annotate("a=0, q=(1,2)\na={.2,.4}, q=1", fontsize=15, xy=(10.5, 0), xytext=(5.5, -0.3))
plt.annotate(r"$\{$",fontsize=35, xy=(10.5, 0), xytext=(4.5, -0.3))
plt.annotate(r"$\}$",fontsize=35, xy=(10.5, 0), xytext=(9.5, -0.3))
plt.show()
```

#### Figure 8

```{code-cell} ipython3
def ex_ut(Pi, A, Q, C, W):
    X, Y, Z, N = list(range(len(A))), list(range(len(Q))), list(range(len(C))), list(range(len(W)))
    Ew = np.zeros((len(N),len(X),len(Y)))
    for i in N:
        for x in X:
            for y in Y:
                total_prob = np.sum(Pi[i,x,y,:,:])
                if total_prob <= 1e-9:
                    Ew[i,x,y] = float("-inf")
                else:
                    Ew[i,x,y] = np.sum([np.sum(W[w]*Pi[i,x,y,:,w]) for w in N])/total_prob
    return Ew
```

```{code-cell} ipython3
Ew_inf = ex_ut(Pi, A, Q, C, W)

# %matplotlib inline
# %matplotlib notebook
# Plot expected consumption
plt.figure(figsize=(7.5, 7.5))
marker = [["o","v"],[">","<"],["x","1"],["2","3"]]
for x in X:
    for y in Y:
        plt.plot(W, Ew_inf[:,x,y],marker=marker[x][y])
plt.plot(W,W,"k-.")
plt.xlabel("w")
plt.ylabel("E(w') given a, q, w")
plt.xlim([10.0, 25.0])
plt.ylim([10.0, 25.0])
plt.title("Figure 8\n Future Utility", y=-0.2)
plt.annotate("a=.4, q=2", xy=(14.0, 15.0), xytext=(10.5, 17.0), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2, q=2", xy=(19.5, 20.0), xytext=(15.0, 23.0), arrowprops={"arrowstyle":"-"})
plt.annotate("a=0, q=(1,2)", xy=(24.5, 24.5), xytext=(18.0, 24.5), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2, q=2", xy=(10.0, 10.7), xytext=(7.5, 10.7), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.4, q=2", xy=(10.3, 11.2), xytext=(7.5, 11.2), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.6, q=2", xy=(11.5, 12.2), xytext=(10.1, 14.0), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.6, q=1", xy=(11.7, 10.7), xytext=(13.5, 10.7), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.4, q=1", xy=(15.0, 14.0), xytext=(16.5, 12.5), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2, q=1", xy=(20.0, 19.5), xytext=(21.0, 18.0), arrowprops={"arrowstyle":"-"})
plt.annotate("", xy=(10.1, 10.1), xytext=(12.0, 9.2), arrowprops={"arrowstyle":"-"})
plt.annotate("a=0, q=(1,2)\na={.2,.4}, q=1", xy=(10.1, 10.1), xytext=(9.5, 8.5))
plt.annotate(r"$\{$",fontsize=25, xy=(10.1, 10.1), xytext=(8.5, 8.5))
plt.annotate(r"$\}$",fontsize=25, xy=(10.1, 10.1), xytext=(12.5, 8.5))
plt.show()
```

For figure 9 to 12, Phelan and Townsend used $\beta = 0.95$. So, we firstly run the function to get the solution to the problem with $\beta = 0.95$.

However, this case took many steps to converge. With the improved algorithm, it didn't converge after 1000 iterations!

After 1000 iterations, we had a memory leak problem and Python was forced to restart.

The algorithm needs more further improvements to handle this problem.

We just presented the codes and running processing here. However, we recommend not to run this cell.

We use the case with $\beta = 0.8$ to show figures anyway.

```{code-cell} ipython3
# Figure 9-12
# Define the function U[a,c]
# def u(a,c):
#     return c**0.5/0.5 + (1-a)**0.5/0.5

# A = np.array([0,0.2,0.4,0.6])
# Q = np.array([1,2])
# C = np.linspace(0,2.25,81)
# P = np.array([[0.9,0.1],[0.6,0.4],[0.4,0.6],[0.25,0.75]])
# beta = 0.95
# N = 50
# w_l = u(A.min(), C.min())/(1-beta)
# w_u = u(A.min(), C.max())/(1-beta)
# W = np.linspace(w_l, w_u, N)
# s_W_0_new, Pi_0_new = solve_static_problem(W*(1-beta), u, A, Q, C, P, "unobserved-actions")
# s_W_new, Pi_W_s1_new, Pi_W_m_s2_new = solve_multi_period_economy_2(A, Q, C, P, "unobserved-actions", β=beta, N=N, 
#                                                                    s_W_0=s_W_0_new/(1-beta), tol=1e10-5)
```

```{code-cell} ipython3
# n_A, n_Q, n_C, n_W, n_W_prime = 4, 2, 81, N, N
# A_ind, Q_ind, C_ind, W_ind, W_prime_ind = range(n_A), range(n_Q), range(n_C), range(n_W), range(n_W_prime)

# Pi_new = np.array([[[[[Pi_W_s1_new[w_ind, a_ind, q_ind, :]@Pi_W_m_s2_new[:, c_ind, w_prime_ind] 
#                        for w_prime_ind in W_prime_ind] for c_ind in C_ind] for q_ind in Q_ind] 
#                     for a_ind in A_ind] for w_ind in W_ind])  
```

```{code-cell} ipython3
s_W_new = s_W
Pi_new = Pi
```

```{code-cell} ipython3
# Ec_beta = ex_con(Pi_new, A, Q, C, W)
Ew_beta = ex_ut(Pi_new, A, Q, C, W)
```

```{code-cell} ipython3
def simulation(W, C, s_W, T, Pi, Ew, seed=12345):
    # initial w such that s(w)=0
    w_index = np.argwhere(np.abs(s_W) == np.min(np.abs(s_W)))[0][0]
    w0 = W[w_index]
    date = np.arange(T)
    
    # set seed for random number
    np.random.seed(seed)
    randn = np.random.rand(T,8)
    
    w_index1, w_index2 = w_index, w_index
    w_series = w0*np.ones(T+1)
    c_series = np.zeros(T)
    Pi_c = list(np.zeros(T))
    Pi_w = list(np.zeros(T))
    
    for i in range(T):
        
        w_index_temp1 = w_index1
        
        Pi_temp_a = Pi[w_index_temp1, :, :, :, :].sum(axis=1).sum(axis=1).sum(axis=1)
        Pi_temp_a_cum = np.cumsum(Pi_temp_a/np.sum(Pi_temp_a))
        a_index = np.sum(randn[i,0] >= Pi_temp_a_cum)
        Pi_temp_q = Pi[w_index_temp1, a_index, :, :, :].sum(axis=1).sum(axis=1)
        Pi_temp_q_cum = np.cumsum(Pi_temp_q/np.sum(Pi_temp_q))
        q_index = np.sum(randn[i,1] >= Pi_temp_q_cum)
        
        Pi_temp_w = Pi[w_index_temp1, a_index, q_index, :, :].sum(axis=0)
        Pi_temp_w_cum = np.cumsum(Pi_temp_w/np.sum(Pi_temp_w))
        w_index1 = np.sum(randn[i,2] >= Pi_temp_w_cum)
        
        # simulation for consumption as well as its distribution
        Pi_c[i] = Pi[w_index_temp1, a_index, q_index, :, w_index1]
        Pi_c[i] /= np.sum(Pi_c[i])
        Pi_temp_c_cum = np.cumsum(Pi_c[i])
        c_index = np.sum(randn[i,3] >= Pi_temp_c_cum)
        c_series[i] = C[c_index]
        
        # simulation for expected utility
        w_series[i+1] = Ew[w_index_temp1, a_index, q_index]
        
        # simulation for distribution over future utility
        Pi_temp_a = Pi[w_index2, :, :, :, :].sum(axis=1).sum(axis=1).sum(axis=1)
        Pi_temp_a_cum = np.cumsum(Pi_temp_a/np.sum(Pi_temp_a))
        a_index = np.sum(randn[i,4] >= Pi_temp_a_cum)
        Pi_temp_q = Pi[w_index2, a_index, :, :, :].sum(axis=1).sum(axis=1)
        Pi_temp_q_cum = np.cumsum(Pi_temp_q/np.sum(Pi_temp_q))
        q_index = np.sum(randn[i,5] >= Pi_temp_q_cum)
        Pi_temp_c = Pi[w_index2, a_index, q_index, :, :].sum(axis=1)
        Pi_temp_c_cum = np.cumsum(Pi_temp_c/np.sum(Pi_temp_c))
        c_index = np.sum(randn[i,6] >= Pi_temp_c_cum)
        Pi_w[i] = Pi[w_index2, a_index, q_index, c_index, :]
        Pi_w[i] /= np.sum(Pi_w[i])
        w_index2 = np.sum(randn[i,7] >= np.cumsum(Pi_w[i]))
    
    return c_series, w_series, Pi_w, Pi_c
```

```{code-cell} ipython3
c_series = np.zeros((80,4))
w_series = np.zeros((81,4))
for i in range(4):
    c_series[:,i], w_series[:,i], _, _ = simulation(W, C, s_W_new, 80, Pi_new, Ew_beta, seed=(12345+i))
```

#### Figure 9

```{code-cell} ipython3
# Plot consumption simulation
date_c = np.arange(80)+1
plt.figure(figsize=(6.5, 6.5))
plt.plot(date_c, c_series[:,0])
plt.plot(date_c, c_series[:,1])
plt.plot(date_c, c_series[:,2])
plt.plot(date_c, c_series[:,3])
plt.xlabel("date")
plt.ylabel("consumption")
plt.xlim([0, 80])
plt.ylim([0.00, 2.25])
plt.title("Figure 9\n Individual Consumptions", y=-0.2)
plt.show()
```

#### Figure 10

```{code-cell} ipython3
# Plot expected utility simulation
date_w = np.arange(81)
plt.figure(figsize=(6.5, 6.5))
plt.plot(date_w, w_series[:,0])
plt.plot(date_w, w_series[:,1])
plt.plot(date_w, w_series[:,2])
plt.plot(date_w, w_series[:,3])
plt.xlabel("date")
plt.ylabel("expected utility")
plt.title("Figure 10\n Individual Utilities", y=-0.2)
plt.show()
```

```{code-cell} ipython3
%time _, _, Pi_w, Pi_c = simulation(W, C, s_W_new, 80, Pi_new, Ew_beta)
```

#### Figure 11

```{code-cell} ipython3
# Plotting distribution for consumption
%matplotlib inline

date_mat_c = np.reshape(np.arange(80)+1,(80,1))*np.ones((1,len(C)))
c_mat = np.ones((80,1))@np.reshape(C,(1,len(C)))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.title("Figure 11 \n Consumptions over time",y=-0.3)
plt.xlabel('date')
plt.ylabel('consumption')
ax.set_zlabel('percentage')

surf = ax.plot_surface(date_mat_c,c_mat,np.array(Pi_c),cmap='viridis')
plt.show()
```

#### Figure 12

```{code-cell} ipython3
# Plotting distribution for future utilities
%matplotlib inline
date_mat_w = np.reshape(np.arange(80)+1,(80,1))*np.ones((1,len(W)))
W_mat = np.ones((80,1))@np.reshape(W,(1,len(W)))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.title("Figure 12 \n Utilities over time",y=-0.3)
plt.xlabel('date')
plt.ylabel('w')
ax.set_zlabel('percentage')

surf = ax.plot_surface(date_mat_w,W_mat,np.array(Pi_w),cmap='viridis')
plt.show()
```
