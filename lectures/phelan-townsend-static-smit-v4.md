---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
# Importing the required libraries
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from pulp import *

import cvxopt
import cylp

from quantecon.optimize import linprog_simplex

import prettytable as pt
import matplotlib.pyplot as plt
import time
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

from warnings import filterwarnings
```

# 1 The Full-Information Economy

## 1.1 Setting And Formulation

In this section, we consider the social planning problem for an economy with a continuum of identical agents each having an identical but independent production technology taking in an agent's own labour and producing the single consumption good as a function of this labour and an independent shock. 
    
**Setting:**

 - Action $a$ from a finite set of possible actions $\mathbf{A} \subset \mathbf{R}_{+}$ 
 - Output $q \in \text{finite }\mathbf{Q} \subset \mathbf{R}_{+}$  determined by an exogenous probability $\mathbf{P}(q|a)>0$ for all $q$
 - Consumption $c \in \text{finite } \mathbf{C} \subset \mathbf{R}_{+}$ with mixed outcomes allowed
 - Utility $U(a,c): \mathbf{A}\times\mathbf{C} \to \mathbf{R}_{+}$ with assumptions:
   - $U(a,c)$ is strictly concave over changes in $c$ holding $a$ constant
   - Higher actions induce higher expected utilities
 - Ex ante utility level:
   - Lowest ex ante utility level with certain highest labor assignment and lowest consumption receipt, $\underline{w}$
   - Highest ex ante utility level with certain lowest labor assignment and highest consumption receipt, $\overline{w}$
   - Arbitrary utility level in between, $w = \alpha\underline{w}+(1-\alpha)\overline{w}\in\mathbf{W}=[\underline{w},\overline{w}]$, $\alpha\in[0,1]$
 - $d_{0}(w)$ denotes the fraction of agent whose required utility is $w$
 - $\Pi^{w}(a,q,c)$: choice variable, the probability for an agent required to receive $w$ of taking action $a$, having output $q$ occur in his own production technology and receiving consumption amount $c$
    
**Formulation:**
- Contract: Such function $\Pi^{w}$ for given $w\in\mathbf{W}$ satisfying certain constraints
- Allocation: a collection of contracts for each $w$ in the support $d_{0}(w)$ for a given distribution $d_{0}$
- **Full Information Problem (FIP)**: 
$$
\begin{aligned}
 & \max_{\Pi^w(a,q,c)} s(w)=\sum_{\mathbf{A} \times \mathbf{Q} \times \mathbf{C}}(q-c)\Pi^{w}(a, q, c) \\
s.t. \text{C1:} & w = \sum_{\mathbf{A}\times\mathbf{Q}\times\mathbf{C}}U(a,c)\Pi^{w}(a,q,c) \\
&\text{(discounted expected utility)} \\ 
\text{C2:} & \sum_{\mathbf{C}} \Pi^{w}(\bar{a}, \bar{q}, c)=P(\bar{q} \mid \bar{a}) \sum_{\mathbf{Q} \times \mathbf{C}} \Pi^{w}(\bar{a}, q, c), \forall (\bar{a},\bar{q})\in\mathbf{A}\times\mathbf{Q} \\
& \text{(coincide conditional probability with nature)} \\
\text{C3:} & \sum_{\mathbf{A} \times \mathbf{Q} \times \mathbf{C}} \Pi^{w}(a, q, c)=1 \\
& \Pi^{w}(a, q, c) \geqq 0, \forall (a, q, c) \in \mathbf{A} \times \mathbf{Q} \times \mathbf{C} \\
& \text{(probability measure)} \\
\end{aligned}
$$
    
**Solution:**
- Total social surplus for initial distribution $d_{0}$: $S^{*}(d_{0})\equiv \sum_{\mathbf{W}}s^{*}(w)d_{0}(w) \geqq 0$ for feasibility
- **Pareto Optima**: An initial distribution of utilities $d_{0}$ and its associated surplus maximizing plans $\{\Pi^{w*}\}^{w\in\mathbf{W}}$ represent a Pareto optimum if the support of $d_{0}$ lies within the non-increasing portion of s*(w).
</font>
    
**Parameterization:**
- $\mathbf{A} = \{0\}$
- $\mathbf{C} = \{0,1,4,5\}$
- $\mathbf{Q} = \{\underline{q},\overline{q}\}$
- $\mathbf{P}(\underline{q} | a=0) = \mathbf{P}(\overline{q} | a=0) = \frac{1}{2}$
- $U(a,c) = c^{0.5}$
- $w = 1.5$
</font>

    
**Primal Problem:**
    
Let $\Pi = \left( \begin{matrix} \Pi^{w}(0,\underline{q},0) & \Pi^{w}(0,\underline{q},1) & \Pi^{w}(0,\underline{q},4) & \Pi^{w}(0,\underline{q},5) \\ \Pi^{w}(0,\overline{q},0) & \Pi^{w}(0,\overline{q},1) & \Pi^{w}(0,\overline{q},4) & \Pi^{w}(0,\overline{q},5) \end{matrix} \right) $,
$\Phi = \left( \begin{matrix} \underline{q}-0 & \underline{q}-1 & \underline{q}-4 & \underline{q}-5 \\ \overline{q}-0 & \overline{q}-1 & \overline{q}-4 & \overline{q}-5 \end{matrix} \right) $,
$u = (0^{0.5},1^{0.5},4^{0.5},5^{0.5})'$.

Then FIP can be written as:
$$
\begin{aligned}
& \max_{\Pi_{xy} \ge 0} \ \sum_{xy} \Pi_{xy} \Phi_{xy} \\
s.t. \text{C1:} & \ \sum_{x=1}^N \sum_{y=1}^M \Pi_{xy} \cdot u_y = w = 1.5\\
\text{C2:} & \ \sum_{y=1}^M \Pi_{1 y} = P(\underline{q}|a=0) \sum_{xy}\Pi_{xy} \\
& \ \sum_{y=1}^M \Pi_{2 y} = P(\overline{q}|a=0) \sum_{xy}\Pi_{xy} \\
\text{C3:} & \  \sum_{xy}\Pi_{xy} = 1 \\
\end{aligned}
$$
where, $N=2,M=4$.

This is equivalent to:
$$
\begin{aligned}
& \max_{\Pi_{xy} \ge 0} \ \sum_{xy} \Pi_{xy} \Phi_{xy} \\
s.t. \text{C1:} & \ \sum_{x=1}^N \sum_{y=1}^M \Pi_{xy} \cdot u_y = w = 1.5\\
\text{C2:} & \ \sum_{y=1}^M \Pi_{1 y} = P(\underline{q}|a=0) \cdot 1 = \frac{1}{2} \\
& \ \sum_{y=1}^M \Pi_{2 y} = P(\overline{q}|a=0) \cdot 1 = \frac{1}{2}\\
\text{C3:} & \  \sum_{xy}\Pi_{xy} = 1 \\
\end{aligned}
$$
Call this formulation as the **"elementary version"**.

We can rewrite this in the compact matrix form as follow:
$$
\begin{aligned}
& \max_{\Pi_{xy} \ge 0} \ Tr (\Pi' \Phi)\\
s.t. \text{C1:} & \ \mathbf{1}_2' \cdot \Pi \cdot u = w \\
\text{C2:} & \ \Pi \cdot \mathbf{1}_4 = p \\
\text{C3:} & \ \mathbf{1}_2' \cdot \Pi \cdot \mathbf{1}_4 = 1 \\
\end{aligned}
$$
where, $p=(P(\underline{q}|a=0),P(\overline{q}|a=0))=(\frac{1}{2},\frac{1}{2})$.

Using vectorizing and Kronecker product, FIP can be formulated as:
$$
\begin{aligned}
\max_{z \ge 0} & \ vec(\Phi)' z\\
s.t. \text{C1:} & \ (u' \otimes \mathbf{1}_2') \cdot z = w \\
\text{C2:} & \ (\mathbf{1}_4' \otimes \mathbf{I}_2) \cdot z = p \\
\text{C3:} & \ (\mathbf{1}_4' \otimes \mathbf{1}_2) \cdot z = 1 \\
\end{aligned}
$$
where, $z = vec(\Pi)$.

As notated in "optimalAssignment_v4.ipynb", FIP can be written as:
$$
\max_{z \ge 0} \ vec(\Phi)^\prime z \\
s.t. 
Az = 
\left( \begin{array}\
u' \otimes \mathbf{1}_2' \\
\mathbf{1}_4' \otimes \mathbf{I}_2 \\
\mathbf{1}_4' \otimes \mathbf{1}_2'
\end{array}\right)
z
= 
\left( \begin{array}\
w \\
p \\
1
\end{array}\right)
=b,\\
$$
Call this formulation as the **"compact version"**.


As derivation above shows, we can either implement the "elemnentary version" by CVXPY and PuLP or the "compact version" by CVXPY and Scipy.linprog. All these computational implementations are showed in the "optimalAssignment_v4.ipynb". In section 1.2, we will use Scipy.linprog to implement the "compact version" and CVXPY and PuLP to implement "elementary version".

**Dual Problem:**
    
By mathematical derivation (see "DSS_Chpt5.ipynb" writen by Jiahui, the proof is similiar to those in "DSS_Chpt5.ipynb"), it can be easily proved the dual problem of FIP is:
$$
\min_{\nu_1, \mu, \nu_2} w\nu_1 + \sum_{x=1}^N p_x \mu_x + \nu_2\\
s.t. u_y \nu_1 + \mu_x + \nu_2 \ge \Phi_{xy}, \forall x,y \\
\nu_1, \mu, \nu_2 \ unrestricted
$$
where, N = 2, $\nu_1$ is the dual variable corresponding to C1, $\mu$ is the dual variable corresponding to C2 and $\nu_2$ is the dual variable corresponding to C3. Notice, here, $\mu$ is a vector with two entries while $\nu_1$ and $\nu_2$ are both scalars.
    
Call this the **"elementary version"** of the dual problem.
    
Based on the "compact version" of the primal problem, we can directly derive the "compact version" of the dual problem since the "compact version" of the primal problem is a standard form of LP.

The **"compact version"** of the dual problem is:
$$
\min_{\lambda} b' \lambda\\
s.t. A' \lambda \ge vec(\Phi)\\
\lambda \ unrestricted
$$
where, $\lambda$ is the dual variable vector with $\lambda_1 = \nu_1$, $\left( \begin{array}\ \lambda_2 \\ \lambda_3 \end{array} \right)= \mu$, $\lambda_4 = \nu_2$.

Furthermore, by rewriting the "elementary version" of the dual problem, we can derive the same "compact version" of the dual problem as shown above.(See "optimalAssignment_v4.ipynb" written by Jiahui, the mathematical statement is similiar to that in this file.)

Again, we can use CVXPY and Scipy.linprog to solve the "compact version" of the dual problem and CVXPY and PuLP to the "elementary version" of the dual problem. Those are also shown in "optimalAssignment_v4.ipynb". In section 1.3, we will use Scipy.linprog and CVXPY to implement the "compact version" and CVXPY and PuLP to implement "elementary version".

```{code-cell} ipython3
# Define parameters
N = 2
M = 4
ql = 1   # an arbitrary value for ql
qu = 10  # an arbitrary value for qu
Phi = np.array([[ql-0, ql-1, ql-4, ql-5], [qu-0, qu-1, qu-4, qu-5]])
u = np.reshape(np.array([0**0.5, 1**0.5, 4**0.5, 5**0.5]), (4, 1))
w = 1.5
p = np.reshape(np.array([1/2, 1/2]), (2, 1))

# Vectorlize matrix Phi
Phi_vec = Phi.reshape(N*M, 1, order = 'F')

# Construct matrix A by Kronecker product
A1 = np.kron(u.T,np.ones((1, N)))
A2 = np.kron(np.ones((1, M)), np.identity(N))
A3 = np.kron(np.ones((1, M)), np.ones((1, N)))
A = np.vstack([A1, A2, A3])

# Construct vector b
b = np.vstack([w, p, 1])

bounds = [(0, None)] * N*M

X = list(range(N))
Y = list(range(M))
```

## 1.2 Solve The Primal Problem

### 1.2.1 SciPy


All three methods from `scipy.linprog` give different solutions to the prorgamming problem.

```{code-cell} ipython3
# Different methods available in scipy
method = ['revised simplex', 'simplex', 'interior-point']
n = len(method)

soln_tb_scipy = pt.PrettyTable()
soln_tb_scipy.field_names = ["Method", "Optimal Value", 
                             "# of Iterations", "Success", "Solutions", "Time(s)"]

print("Summary of Solutions under Different Methods with SCIPY")
scipy_times = [] # stores time taken for each execution
for i in range(n):
    # Solve the FIP
    command = "res = linprog(-Phi_vec, A_eq=A, b_eq=b, bounds=bounds, method = '" + method[i] + "')"
    in_time = time.time()
    exec(command)
    out_time = time.time()
    # Add the results to the table
    scipy_times.append(round(out_time-in_time, 3))
    soln_tb_scipy.add_row(["", "", "", "", "", ""])
    soln_tb_scipy.add_row([method[i], round(-res.fun, 2), res.nit, res.success, np.round(res.x, 2),
                           scipy_times[-1]])

print(soln_tb_scipy)
```

### 1.2.2 CVXPY

```{code-cell} ipython3
# Define variable Pi
Pi = cp.Variable((N, M))

# Define objective function
obj_expr = cp.sum(cp.multiply(Pi, Phi))
obj = cp.Maximize(obj_expr)

# Define constraints
constraints = [cp.sum(Pi@u) == w, cp.sum(Pi, axis=1) == np.reshape(p, 2), cp.sum(Pi) == 1, Pi >= 0]

# Create the model
cvxpy_primal = cp.Problem(obj,constraints)
```

```{code-cell} ipython3
# Different solvers available in cvxpy
solver = ["cp.ECOS", "cp.OSQP", "cp.SCS"]

n = len(solver)

soln_tb_cv = pt.PrettyTable()
soln_tb_cv.field_names = ['Solver', 'DCP', 'Status', 'Optimal Value', 'Solution to Primal', 'Solution to Dual',
                          'Time(s)']

cvxpy_times = [] # stores time taken for each execution
print("Summary of Solutions under Different Solvers with CVXPY")
for i in range(n):
    command = "cvxpy_primal.solve(solver=" + solver[i] + ")"
    in_time = time.time()
    exec(command)
    out_time = time.time()
    cvxpy_times.append(round(out_time-in_time, 3))
    soln_tb_cv.add_row(['', '', '', '', '', '', ''])
    soln_tb_cv.add_row([solver[i][3:], cvxpy_primal.is_dcp(), cvxpy_primal.status, 
                     round(cvxpy_primal.value, 2), tuple(np.round(np.hstack(Pi.value), 2)), 
                     tuple(np.round(np.hstack([constraints[i].dual_value for i in range(3)]),2)),
                       cvxpy_times[-1]])

print(soln_tb_cv)
```

### 1.2.3 PuLP

```{code-cell} ipython3
# Create the model to contain the problem data
pulp_primal = LpProblem("pulp_primal", LpMaximize)

# Define variable Pi
Pi_index = [(x,y) for x in X for y in Y]
Pi = LpVariable.dicts("Pi", Pi_index, 0, None)

# Add objective function
pulp_primal += lpSum([Pi[x,y]*Phi[x,y] for x in X for y in Y]), "Objective Function"

# Add constraints
pulp_primal += lpSum([Pi[x,y]*u[y] for y in Y for x in X]) == w, "C1"
for x in X:
    pulp_primal += lpSum([Pi[x,y] for y in Y]) == p[x,0], "C2_%i"%x
pulp_primal += lpSum([Pi[x,y] for y in Y for x in X]) == 1, "C3"

# Solve the primal problem
in_time = time.time()
pulp_primal.solve()
out_time = time.time()

pulp_times = [round(out_time-in_time, 3)] # stores time taken for execution

# Print results
print("status:", LpStatus[pulp_primal.status])
print("Time taken(s):", pulp_times[0])
print("fun:", round(value(pulp_primal.objective), 3))
print("Pi:")
for x in X:
      print([round(Pi[x,y].varValue,3) for y in Y])
```

### 1.2.4 QuantEcon

We will use quantecon's `linprog_simplex` to solve the same problem.

```{code-cell} ipython3
in_time = time.time()
result_qe = linprog_simplex(Phi_vec.flatten(), A_eq=A, b_eq=b.flatten())
out_time = time.time()

qe_times = [round(out_time-in_time, 3)] # stores time taken for execution
# Print results
print("success:", result_qe.success)
print("Time taken(s):", qe_times[0])
print("Optimal value:", round(result_qe.fun, 2))
print("solution:", [round(val, 2) for val in result_qe.x])
```

### 1.2.5 Time Comparison

```{code-cell} ipython3
time_tb = pt.PrettyTable()
time_tb.field_names = ['Library', 'Best Time(seconds)']
time_comparison = [('scipy', scipy_times), ('cvxpy', cvxpy_times), ('pulp', pulp_times), ('quantecon', qe_times)]

print('Table showing best time taken by each library to solve the primal problem')
for lib_name, time_vals in time_comparison:
    time_tb.add_row(['', ''])
    time_tb.add_row([lib_name, min(time_vals)])

print(time_tb)
```

## 1.3 Solve The Dual Problem

### 1.3.1 `scipy.linprog`

```{code-cell} ipython3
# Solve the dual problem
# Please note the bounds of \lambda since it is unrestricted and scipy by default uses [0, infinity) as bounds
in_time = time.time()
res_d = linprog(b, A_ub = -A.transpose(), b_ub = -Phi_vec, bounds=[(None, None)]*b.shape[0])
out_time = time.time()
scipy_time = round(out_time - in_time, 3)
print("success:", res_d.success)
print("fun:", round(res_d.fun, 2))
print("nu_1:", round(res_d.x[0], 2))
print("mu:", np.round([res_d.x[i] for i in [1, 2]], 2))
print("nu_2:", round(res_d.x[3], 2))
print("Time taken(s):", scipy_time)
```

### 1.3.2 CVXPY

First, let implement the "compact version".

```{code-cell} ipython3
# Define variable _lambda
_lambda = cp.Variable((4,1))

# Define objective function
obj_expr_d = b.transpose() @ _lambda
obj_d = cp.Minimize(obj_expr_d)

# Define constraints
constraints_d = [A.transpose() @ _lambda >= Phi_vec]

# Create the model
cvxpy_dual = cp.Problem(obj_d, constraints_d)

# Solve the dual problem
cvxpy_dual.solve()

#print the result
print("status:", cvxpy_dual.status)
print("fun:", np.round(obj_expr_d.value,2))
print("nu_1:", np.round(_lambda.value[0],2))
print("mu:", np.round([_lambda.value[1]],2), np.round([_lambda.value[2]],2))
print("nu_2:", np.round(_lambda.value[3],2))
```

This result is the same as the primal problems we implement above, meaning the "compact version" of the dual problem is theoretically correct!!

Next, let's implement the "elementary version" of the dual problem.

```{code-cell} ipython3
# Define variable
nu_1 = cp.Variable()
mu = cp.Variable((2, 1))
nu_2 = cp.Variable()

# Define objective function
obj_expr_d = w * nu_1 + p.transpose() @ mu + nu_2
obj_d = cp.Minimize(obj_expr_d)

# Define constraints
constraints_d = [(u[y] * nu_1 + mu[x] + nu_2 >= Phi[x, y]) for x in X for y in Y]

# Create the model
cvxpy_dual = cp.Problem(obj_d,constraints_d)

in_time = time.time()
# Solve the dual problem
cvxpy_dual.solve()
out_time = time.time()

cvxpy_time = round(out_time - in_time, 3)
# print the result
print("status:", cvxpy_dual.status)
print("fun:", np.round(obj_expr_d.value, 2))
print("nu_1:", np.round(nu_1.value, 2))
print("mu:", np.round(mu.value, 2))
print("nu_2:", np.round(nu_1.value, 2))
print("Time taken(s):", cvxpy_time)
```

### 1.3.3 PuLP

```{code-cell} ipython3
# Create the model to contain the problem data
pulp_dual = LpProblem("pulp_dual", LpMinimize)

# Define variable u and v
nu_1 = LpVariable("nu_1", None, None)
mu = LpVariable.dicts("mu", [0,1], None, None)
nu_2 = LpVariable("nu_2", None, None)

# Add objective function
pulp_dual += w * nu_1 + p[0] * mu[0] + p[1] * mu[1] + nu_2,"Objective Function"

# Add constraints
for x in X:
    for y in Y:
        pulp_dual += u[y,0] * nu_1 + mu[x] + nu_2 >= Phi[x,y],"Constraint%i_%i"%(x,y)

in_time = time.time()
# Solve the dual problem
pulp_dual.solve()
out_time = time.time()
pulp_time = round(out_time - in_time, 3)

# Print results
print("status:", LpStatus[pulp_primal.status])
print("fun:", round(value(pulp_primal.objective), 2))
print("nu_1:", round(nu_1.varValue, 2))
print("mu:", [round(mu[x].varValue, 3) for x in range(2)])
print("nu_2:", round(nu_2.varValue, 2))
print("Time taken(s):", pulp_time)
```

### 1.3.4 Time Comparison

```{code-cell} ipython3
time_tb = pt.PrettyTable()
time_tb.field_names = ['Library', 'Time Taken(seconds)']
time_comparison = [('scipy', scipy_time), ('cvxpy', cvxpy_time), ('pulp', pulp_time)]

print('Table showing the time taken by each library to solve the dual problem')
for lib_name, time_val in time_comparison:
    time_tb.add_row(['', ''])
    time_tb.add_row([lib_name, time_val])

print(time_tb)
```

# 2 The Static Unobserved-Action Economy

## 2.1 Setting And Formulation
    
**Setting:**
 - An agent's action is unobservable by everyone other than the agent himself
 - Additional constraints requiring that obeying the action recommendation is always weakly preferred
    - The lowest possible ex ante expected utility for the unobserved action economy is that of receiving the lowest consumption and the lowest labour amount
 - Additional constriants: for all assigned and possible alternative actions $(a,\hat{a})\in \mathbf{A}\times\mathbf{A}$
    - $\sum_{\mathbf{Q} \times \mathbf{C}} U[a, c]\left\{\Pi^{w}(c \mid q, a) P(q \mid a)\right\} \geqq \sum_{\mathbf{Q} \times \mathbf{C}} U[\hat{a}, c]\left\{\Pi^{w}(c \mid q, a) P(q \mid \hat{a})\right\}$
    - where $\Pi^{w}(c \mid q, a)$ is the conditional probability implied by $\Pi^{w}(a, q, c)$
    - $\Pi^{w}(c \mid q, a) P(q \mid a)$ is the probability of a given $(q, c)$ combination given that action $a$ is recommended and that this action $a$ is taken
    - $\Pi^{w}(c \mid q, a) P(q \mid \hat{a})$ is the probability of a given $(q, c)$ combination given that action $a$ is announced and deviation action $\hat{a}$ is taken instead
    - $\Pi^{w}(q, c \mid a) = \Pi^{w}(c \mid q, a) P(q \mid a) \implies \sum_{\mathbf{Q} \times \mathbf{C}} U[a, c] \Pi^{w}(q, c \mid a) \geqq \sum_{\mathbf{Q} \times \mathbf{C}} U[\hat{a}, c] \frac{\mathbf{P}(q \mid \hat{a})}{\mathbf{P}(q \mid a)} \Pi^{w}(q, c \mid a) \\ \qquad\qquad\qquad\qquad\qquad\quad\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\implies \sum_{\mathbf{Q} \times \mathbf{C}} U[a, c] \Pi^{w}(a, q, c) \geqq \sum_{\mathbf{Q} \times \mathbf{C}} U[\hat{a}, c] \frac{P(q \mid \hat{a})}{P(q \mid a)} \Pi^{w}(a, q, c) \qquad \text{ (C4)}$ 
    - The ration $\frac{P(q \mid \hat{a})}{P(q \mid a)}$ gives how many more times likely it is that output $q$ will occur given deviation action $\hat{a}$ as opposed to recommended action $a$, and thus updates the joint probability of observing recommended action $a$, output $q$, and consumption $c$.
    
**Formulation:**
$$
\begin{aligned}
& \max_{\Pi^{w}} \sum_{A \times Q \times C} (q-c)\Pi^{w}(a,q,c) \\
s.t. 
\text{C1:} & \ w = \sum_{\mathbf{A}\times\mathbf{Q}\times\mathbf{C}}U(a,c)\Pi^{w}(a,q,c) \\
\text{C2:} & \ \sum_{\mathbf{C}} \Pi^{w}(\bar{a}, \bar{q}, c)=P(\bar{q} \mid \bar{a}) \sum_{\mathbf{Q} \times \mathbf{C}} \Pi^{w}(\bar{a}, q, c) \\
\text{C3:} & \ \sum_{\mathbf{A} \times \mathbf{Q} \times \mathbf{C}} \Pi^{w}(a, q, c)=1, \Pi^{w}(a, q, c) \geqq 0 \\
\text{C4:} & \ \sum_{\mathbf{Q} \times \mathbf{C}} U[a, c] \Pi^{w}(a, q, c) \geqq \sum_{\mathbf{Q} \times \mathbf{C}} U[\hat{a}, c] \frac{P(q \mid \hat{a})}{P(q \mid a)} \Pi^{w}(a, q, c)
\end{aligned}
$$
    
**Parameterization:**
 - $U[a,c] = [(c^{0.5}/0.5+(1-a)^{0.5}/0.5]$
 - $\mathbf{A} = \{0, 0.2, 0.4, 0.6\}$
 - $\mathbf{Q} = \{1, 2\}$
 - $\mathbf{C} = \{c = 0.028125n: 0\leq n\leq80,n\in\mathbb{N}\}$: 81 equally spaced points between 0 and 2.25
 - The technology relating action to the probability of each output:
    
|a|P(q=1)|P(q=2)|
|:-:|:---:|:---:|
|0|0.9|0.1|
|0.2|0.6|0.4|
|0.4|0.4|0.6|
|0.6|0.25|0.75|
    
**Primal Problem:**
    
Firstly, define a three-dimension matrix $\Pi^w$ with a size of $(l \times m \times n) = (4 \times 2 \times 81)$, in which each dimension represents $A$, $Q$ or $C$ respectively. For example, $\Pi_{3,2,5}^w = \Pi^w(a=0.4,q=2,c=0.140625)$. In the same way, we define a two-dimension matrix $\Phi$ with entries $\Phi_{yz} = q_y - c_z$, a two-dimension matrix $U$ with entries $U_{x,z} = U[a_x,c_z] = [(c_z^{0.5}/0.5+(1-a_x)^{0.5}/0.5]$ and a two-dimension matrix $P$ with entries $P_{xy} = P(q=q_y|a=a_x)$.

Then the problem can be formulated as follow:
$$
\begin{aligned}
& \max_{\Pi_{xyz}^w \ge 0} \ \sum_{x=1}^4\sum_{y=1}^2\sum_{z=1}^{81} \Pi^w_{xyz} \Phi_{yz} \\
s.t. \text{C1:} & \ \sum_{x=1}^4\sum_{y=1}^2\sum_{z=1}^{81} U_{xz} \Pi^w_{xyz} = w\\
\text{C2:} & \ \sum_{z=1}^{81} \Pi^w_{xyz} = P_{xy} \sum_{y=1}^2 \sum_{z=1}^{81} \Pi^w_{xyz}, \forall x,y \\
\text{C3:} & \ \sum_{x=1}^4\sum_{y=1}^2\sum_{z=1}^{81} \Pi^w_{xyz} = 1 \\
\text{C4:} & \ \sum_{y=1}^2\sum_{z=1}^{81} U_{xz}\Pi^w_{x,y,z} \ge \sum_{y=1}^2\sum_{z=1}^{81} U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \Pi^w_{x,y,z}, \forall x, x^*
\end{aligned}
$$

Call this formulation as the "elementay version" of the primal problem.

It is worthy to emphasis that this formulation contains a three-dimension matrix $\Pi$. As what we have done before, we can expand matrix $\Pi$ by one axis and then convert it to a two-dimension matrix. In this way, we obtain a new formulation of this problem(see section 3.1). Furthermore, we can reuse "vectorization" to convert two-dimension version of matrix $\Pi$ to a vector and then get a third formulation of the problem. However, this third version is much more sophisticated than we can imagine.

Here, we directly implement the "elementary version" of the primal problem by CVXPY. Variables in CVXPY have at most two axises, so that it is impossible to create a three-dimension matrix of cp.Variable. The way we do this is to create four two-dimension matrices of cp.Variable.

    
**Dual Problem:**
    
As for the dual problem, although formulating three formulations' dual problems is all possible, here we only formulate the "elementary version" of the dual problem. Like statements in "DSS_Chpt5.ipynb" written by Jiahui, we firstly rewrite the primal problem in the standard form of LP. Notice here, we need to add **slack variables** $s_{xx^*}$ to convert the C4 into equation constrains. For simplicity, we write $\Pi$ instead of $\Pi^w$. The standard form of primal problem is:
    
$$
\begin{aligned}
& \min_{\Pi_{xyz}} \ -\sum_x \sum_y \sum_z \Pi_{xyz} \Phi_{yz} \\
s.t. \text{C1:} & \ \sum_x \sum_y \sum_z  U_{xz} \Pi_{xyz} = w\\
\text{C2:} & \ \sum_z \Pi_{xyz} - P_{xy} \sum_y \sum_z \Pi_{xyz} = 0, \forall x,y \\
\text{C3:} & \ \sum_x \sum_y \sum_z \Pi_{xyz} = 1 \\
\text{C4:} & \ \sum_y \sum_z U_{xz}\Pi_{xyz} = \sum_y \sum_z U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \Pi_{x,y,z} + s_{xx^*}, \forall x, x^* \\
& \ s_{xx^*} \ge 0, \forall x, x^* \\
& \ \Pi_{xyz} \ge 0, \forall x, y, z
\end{aligned}
$$
    
Denote $\alpha(scalar), \beta(l\times m = 4\times2), \gamma(scalar), \delta(l\times l =4\times4), \lambda(l\times l =4\times4)$ and $\mu(l\times m \times n=4\times 2 \times 81)$ are dual variables corresponding to C1, C2, C3, C4, $\ s_{xx^*} \ge 0$ and $\Pi_{xyz} \ge 0$, respectively.

Then the Langrangian function is:
$$
\begin{align}
& L(\Pi,s,\alpha,\beta,\gamma,\delta,\lambda,\mu) \\
= & -\sum_x \sum_y \sum_z \Pi_{xyz} \Phi_{yz} + (\sum_x \sum_y \sum_z U_{xz} \Pi_{xyz} - w)\alpha \\
&+ \sum_x \sum_y (\sum_z \Pi_{xyz} - P_{xy} \sum_y \sum_z \Pi_{xyz}) \beta_{xy} + (\sum_x \sum_y \sum_z \Pi_{xyz} -1)\gamma \\
&+ \sum_x \sum_{x^*} (\sum_y \sum_z U_{xz}\Pi_{xyz} - \sum_y \sum_z U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \Pi_{x,y,z} - s_{xx^*}) \delta_{xx^*}\\
&-  \sum_x \sum_{x^*} \lambda_{xx^*} s_{xx^*} - \sum_x \sum_y \sum_z \mu_{xyz} \Pi_{xyz} \\
= & -\sum_x \sum_y \sum_z \Pi_{xyz} \Phi_{yz} + \sum_x \sum_y \sum_z U_{xz} \Pi_{xyz}\alpha - w\alpha \\
&+ \sum_x \sum_y \sum_z \Pi_{xyz} \beta_{xy} - \sum_x \sum_y (P_{xy} \sum_y \sum_z \Pi_{xyz}) \beta_{xy} + \sum_x \sum_y \sum_z \Pi_{xyz} \gamma - \gamma \\
&+ \sum_x \sum_{x^*} (\sum_y \sum_z U_{xz}\Pi_{xyz})\delta_{xx^*} - \sum_x \sum_{x^*} (\sum_y \sum_z U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \Pi_{x,y,z}) \delta_{xx^*}  - \sum_x \sum_{x^*} s_{xx^*} \delta_{xx^*}\\
&-  \sum_x \sum_{x^*} \lambda_{xx^*} s_{xx^*} - \sum_x \sum_y \sum_z \mu_{xyz} \Pi_{xyz}
\end{align}
$$

Let's simplify several terms of $L$:
$$
\begin{align}
& \sum_x \sum_y (P_{xy} \sum_y \sum_z \Pi_{xyz}) \beta_{xy} \\
= & \sum_x \sum_y (\beta_{xy} P_{xy}) \cdot (\sum_y \sum_z \Pi_{xyz}) \\
= & \sum_x (\sum_y \sum_z \Pi_{xyz}) \sum_y (\beta_{xy} P_{xy}) \\
= & \sum_x (\sum_y \sum_z \Pi_{xyz} \sum_y \beta_{xy} P_{xy}) \\
= & \sum_x \sum_y \sum_z \Pi_{xyz} \cdot (\sum_y \beta_{xy} P_{xy}) \\
& \\
& \sum_x \sum_{x^*} (\sum_y \sum_z U_{xz}\Pi_{xyz})\delta_{xx^*} \\
= & \sum_x (\sum_y \sum_z \sum_{x^*} U_{xz}\Pi_{xyz} \delta_{xx^*}) \\
= & \sum_x (\sum_y \sum_z \Pi_{xyz} \cdot \sum_{x^*} U_{xz} \delta_{xx^*}) \\
= & \sum_x \sum_y \sum_z \Pi_{xyz} \cdot (\sum_{x^*} U_{xz} \delta_{xx^*}) \\
& \\
& \sum_x \sum_{x^*} (\sum_y \sum_z U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \Pi_{x,y,z}) \delta_{xx^*} \\
= & \sum_x (\sum_y \sum_z \sum_{x^*} U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \Pi_{x,y,z} \delta_{xx^*}) \\
= & \sum_x (\sum_y \sum_z \Pi_{x,y,z} \cdot (\sum_{x^*} U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \delta_{xx^*})) \\
= & \sum_x \sum_y \sum_z \Pi_{x,y,z} \cdot (\sum_{x^*} U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \delta_{xx^*}) \\
\end{align}
$$

Substuting these three euqations into $L$:
$$
\begin{align}
L = & \sum_x \sum_y \sum_z \Pi_{xyz} (-\Phi_{yz} + \alpha U_{xz} + \beta_{xy} - \sum_y \beta_{xy} P_{xy} + \gamma + \sum_{x^*} U_{xz} \delta_{xx^*} - \sum_{x^*} U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \delta_{xx^*} - \mu_{xyz})\\
&+ \sum_x \sum_{x^*} s_{xx^*} (-\delta_{xx^*} - \lambda_{xx^*})\\
&- \alpha w - \gamma \\
\end{align}
$$

Let $C = -\Phi_{yz} + \alpha U_{xz} + \beta_{xy} - \sum_y \beta_{xy} P_{xy} + \gamma + \sum_{x^*} U_{xz} \delta_{xx^*} - \sum_{x^*} U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \delta_{xx^*} - \mu_{xyz}$, then the dual function is:
$$
g(\alpha,\beta,\gamma,\delta) 
= \inf_{\Pi,s}L 
= \begin{cases}
- \alpha w - \gamma, & \text{if } C = 0 \text{ and } -\delta_{xx^*}-\lambda_{xx^*} = 0 \\
- \infty, & \text{otherwise} 
\end{cases}
$$

The dual problem is:
$$
\min_{\alpha,\beta,\gamma,\delta} \alpha w + \gamma\\
s.t. -\Phi_{yz} + \alpha U_{xz} + \beta_{xy} - \sum_y \beta_{xy} P_{xy} + \gamma + \sum_{x^*} U_{xz} \delta_{xx^*} - \sum_{x^*} U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \delta_{xx^*} \ge 0, \forall x,y,z \\
\delta_{xx^*} \le 0, \forall x,x^*
$$

We use CVXPY to implement this dual problem.

```{code-cell} ipython3
# Define function U[a,c]
def u(a, c):
    return c**0.5/0.5 + (1-a)**0.5/0.5

# Define parameters
A = np.array([0, 0.2, 0.4, 0.6])
Q = np.array([1, 2])
C = np.linspace(0, 2.25, 81)

l = len(A)
m = len(Q)
n = len(C)

X = list(range(l))
Y = list(range(m))
Z = list(range(n))

Phi = np.array([[Q[y]-C[z] for z in Z] for y in Y])
U = np.array([[u(A[x],C[z]) for z in Z] for x in X])
P = np.array([[0.9, 0.1], [0.6, 0.4], [0.4, 0.6], [0.25, 0.75]])

w = cp.Parameter()
```

## 2.2 Solve The Primal Problem

```{code-cell} ipython3
# Define variable Pi_x
for x in X:
    exec("Pi_%i = cp.Variable((m , n))"%x)

# Define objective function
obj_expr = cp.sum([cp.sum(cp.multiply(eval("Pi_%i"%x), Phi)) for x in X])
obj = cp.Maximize(obj_expr)

# Define constraints
C1 = [cp.sum([cp.sum([cp.sum(cp.multiply(eval("Pi_%i"%x)[y,:], U[x,:])) for x in X]) for y in Y]) == w]
C2 = [(cp.sum(eval("Pi_%i"%x),axis = 1)[y] == P[x,y]*cp.sum(eval("Pi_%i"%x))) for x in X for y in Y]
C3 = [cp.sum([cp.sum(eval("Pi_%i"%x)) for x in X]) == 1] + [(eval("Pi_%i"%x) >= 0) for x in X]
C4 = [(cp.sum([cp.sum(cp.multiply(eval("Pi_%i"%x)[y,:],U[x,:])) for y in Y]) >= 
     cp.sum([cp.sum(cp.multiply(eval("Pi_%i"%x)[y,:],U[x_star,:]))*P[x_star,y]/P[x,y] for y in Y])) for x in X for x_star in X]

# Create the model
cvxpy_primal = cp.Problem(obj,C1+C2+C3+C4)
```

```{code-cell} ipython3
# Set parameter value of w
w.value = 3

# Solve the primal problem
in_time = time.time()
cvxpy_primal.solve(solver = cp.ECOS)
out_time = time.time()

cvxpy_time = round(out_time - in_time, 3)
# Print results
print("status:", cvxpy_primal.status)
print("fun:", round(obj_expr.value, 8))
print("Time taken(s):", cvxpy_time)
```

## 2.3 Solve The Dual Problem

```{code-cell} ipython3
# Define dual variables
alpha = cp.Variable()
beta = cp.Variable((4, 2))
gamma = cp.Variable()
delta = cp.Variable((4, 4))

# Define objective function
obj_expr_d = alpha * w + gamma
obj_d = cp.Minimize(obj_expr_d)

# Define constraints
constraints_d = [(-Phi[y,z] + alpha * U[x,z] + beta[x,y] - cp.sum([beta[x,y]*P[x,y] for y in Y]) + gamma 
               + cp.sum([U[x,z]*delta[x,x_star] for x_star in X]) 
               - cp.sum([U[x_star,z]*P[x_star,y]/P[x,y]*delta[x,x_star] for x_star in X])
               >= 0) for x in X for y in Y for z in Z] + [delta[x,x_star] <= 0 for x in X for x_star in X]

# Create the model
cvxpy_dual = cp.Problem(obj_d, constraints_d)
```

```{code-cell} ipython3
# Set parameter w
w.value = 3

# Solve the dual problem
in_time = time.time()
cvxpy_dual.solve(solver = cp.CBC)
out_time = time.time()

cvxpy_time = round(out_time - in_time, 3)

# Print results
print("status:", cvxpy_dual.status)
print("fun:", round(obj_expr_d.value, 8))
print("Time taken(s):", cvxpy_time)
```

## 2.4 Figures

In section 2.2, we solve the primal problem setting $w = 3$. In this section, for the purpose of plotting, we shall give $w$ dynamic values and compute corresponding results based on the codes for the primal problem in section 2.2.
    
**Notice:** When plotting figure 3 and 4, only cp.CBC solver can derive the same figures as presented in Phelan & Townsend, 1991.

```{code-cell} ipython3
num_points = 100

# Set dynamic values for w
w_values = np.linspace(1,5,num_points)

# Create models of Full Information and Unobserved Action
cvxpy_full = cp.Problem(obj,C1+C2+C3)
cvxpy_unobs = cp.Problem(obj,C1+C2+C3+C4)

# Initialize variables
sw_full = np.ones(num_points)
sw_unobs = np.ones(num_points)
Ea_full = np.ones(num_points) * float("-inf")
Ea_unobs = np.ones(num_points) * float("-inf")
Ec_full = np.ones((num_points,l,m)) * float("-inf")
Ec_unobs = np.ones((num_points,l,m)) * float("-inf")

# Compute results corresponding to different values of w
for i in range(num_points):
    w.value = w_values[i]
    
    # Solve the Full Information model and save results into vatiables
    cvxpy_full.solve(solver = cp.CBC)  
    sw_full[i] = obj_expr.value             
    if cvxpy_full.status == "optimal":
        Ea_full[i] = np.sum([A[x]*eval("Pi_%i"%x).value[y,z] for x in X for y in Y for z in Z])
        Ec_full[i,:,:] = [[(np.sum([C[z]*eval("Pi_%i"%x).value[y,z] for z in Z])/np.sum([eval("Pi_%i"%x).value[y,z] for z in Z]))
                           for y in Y] for x in X]
    
    
    # Solve the Unobserved Action model and save results into variables
    cvxpy_unobs.solve(solver = cp.CBC)
    sw_unobs[i] = obj_expr.value
    if cvxpy_unobs.status == "optimal":
        Ea_unobs[i] = np.sum([A[x]*eval("Pi_%i"%x).value[y,z] for x in X for y in Y for z in Z])
        Ec_unobs[i,:,:] = [[np.sum([C[z]*eval("Pi_%i"%x).value[y,z] for z in Z])/np.sum([eval("Pi_%i"%x).value[y,z] for z in Z])
                           for y in Y] for x in X]
```

```{code-cell} ipython3
# Plot Figure 1
plt.figure(figsize=(6.5, 6.5))
plt.plot(w_values, sw_full)
plt.plot(w_values, sw_unobs)
plt.hlines(0, 1.0, 5.0, linestyle="dashed")
plt.xlabel("w")
plt.ylabel("s(w)")
plt.xlim([1.0, 5.0])
plt.ylim([-1.5, 2.0])
plt.title("Figure 1\n Optimized surplus function", y=-0.2)
plt.text(2.5, 1.6, "Full Information", size=15)
plt.text(1.5, 0.8, "Unobserved Action", size=15)
plt.show()
```

```{code-cell} ipython3
# Plot Figure 2
plt.figure(figsize=(6.5, 6.5))
plt.plot(w_values, Ea_full)
plt.plot(w_values, Ea_unobs)
plt.xlabel("w")
plt.ylabel("E{a(w)}")
plt.xlim([1.0, 5.0])
plt.ylim([0.0, 0.8])
plt.title("Figure 2\n Actions", y=-0.2)
plt.text(2.3, 0.65, "Full Information", size=15)
plt.text(2.6, 0.15, "Unobserved Action", size=15)
plt.show()
```

```{code-cell} ipython3
# Plot Figure 3
plt.figure(figsize=(6.5, 6.5))
for x in X:
    for y in Y:
        plt.plot(w_values, Ec_unobs[:,x,y])

plt.xlabel("w")
plt.ylabel("E(c) given a, q, w")
plt.xlim([1.0, 5.0])
plt.ylim([0.0, 2.25])
plt.title("Figure 3\n Unobserved Action Consumption", y=-0.3)
plt.annotate("a=.4, q=2", xy=(2.5, 0.5), xytext=(1.3, 0.7), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2, q=2", xy=(3.7, 1.5), xytext=(2.2, 1.65), arrowprops={"arrowstyle":"-"})
plt.annotate("a=0, q=(1,2)", xy=(4.8, 2.05), xytext=(3.0, 2.15), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2,\nq=2", xy=(2.0, 0.15), xytext=(1.3, 0.24), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.4, q=1", xy=(3.0, 0.10), xytext=(3.6, 0.2), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2,\nq=1", xy=(4.0, 0.9), xytext=(4.3, 0.75), arrowprops={"arrowstyle":"-"})
plt.annotate("a=0, q=(1,2)\na=.2, q=1", xy=(2.1, 0), xytext=(1.8, -0.3), arrowprops={"arrowstyle":"-"})
plt.annotate(r"$\{$",fontsize=25, xy=(2.1, 0), xytext=(1.6, -0.3))
plt.annotate(r"$\}$",fontsize=25, xy=(2.1, 0), xytext=(2.5, -0.3))
plt.show()
```

```{code-cell} ipython3
plt.figure(figsize=(6.5, 6.5))
for x in X:
    for y in Y:
        plt.plot(w_values, Ec_full[:,x,y])
plt.xlabel("w")
plt.ylabel("E{c(w)}")
plt.xlim([1.0, 5.0])
plt.ylim([0.0, 2.25])
plt.title("Figure 4\n Full Information Consumption", y=-0.2)
plt.show()
```

# 3 The Static Unobserved-Action Economy(Based on Another Formlation)


## 3.1 Another Formulation 
    
**Formulation:**
$$
\begin{aligned}
\max_{\Pi^{w}} & \sum_{\{0,0.2,0.4,0.6\}\times\{1,2\}\times\{0.028125n, n=0,1,\cdots,80\}} (q-c)\Pi^{w}(a,q,c) \\
\end{aligned}
$$
    
s.t.
$$
\begin{aligned}
C1: \quad& w = \sum_{\mathbf{A}\times\mathbf{Q}\times\mathbf{C}}U(a,c)\Pi^{w}(a,q,c) \\
C2: \quad& \sum_{\mathbf{C}} \Pi^{w}(\bar{a}, \bar{q}, c)=P(\bar{q} \mid \bar{a}) \sum_{\mathbf{Q} \times \mathbf{C}} \Pi^{w}(\bar{a}, q, c) \\
C3: \quad& \sum_{\mathbf{A} \times \mathbf{Q} \times \mathbf{C}} \Pi^{w}(a, q, c)=1, \Pi^{w}(a, q, c) \geqq 0 \\
C4: \quad& \sum_{\mathbf{Q} \times \mathbf{C}} U[a, c] \Pi^{w}(a, q, c) \geqq \sum_{\mathbf{Q} \times \mathbf{C}} U[\hat{a}, c] \frac{P(q \mid \hat{a})}{P(q \mid a)} \Pi^{w}(a, q, c)
\end{aligned}
$$
    
Let 

$\Pi = \begin{bmatrix} 
        \Pi^{w}(a_1,q_1,c_1) & \Pi^{w}(a_1,q_1,c_2) & \cdots & \Pi^{w}(a_1,q_1,c_n) & \cdots & \Pi^{w}(a_1,q_m,c_1) & \Pi^{w}(a_l,q_m,c_2) & \cdots & \Pi^{w}(a_l,q_m,c_n) \\
        \Pi^{w}(a_2,q_1,c_1) & \Pi^{w}(a_2,q_1,c_2) & \cdots & \Pi^{w}(a_2,q_1,c_n) & \cdots & \Pi^{w}(a_2,q_m,c_1) & \Pi^{w}(a_2,q_m,c_2) & \cdots & \Pi^{w}(a_2,q_m,c_n) \\
        \vdots & \vdots & \ddots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
        \Pi^{w}(a_l,q_1,c_1) & \Pi^{w}(a_l,q_1,c_2) & \cdots & \Pi^{w}(a_l,q_1,c_n) & \cdots & \Pi^{w}(a_l,q_m,c_1) & \Pi^{w}(a_l,q_m,c_2) & \cdots & \Pi^{w}(a_l,q_m,c_n) \\
        \end{bmatrix}_{l\times (mn)}$,
        
$\Phi = \begin{bmatrix} 
        q_1-c_1 & q_1-c_2 & \cdots & q_1-c_n & \cdots & q_m-c_1 & q_m-c_2 & \cdots & q_m-c_n \\
        \end{bmatrix}_{1\times (mn)}$,

$U = \begin{bmatrix} 
        u(a_1,c_1) & u(a_1,c_2) & \cdots & u(a_1,c_n) & \cdots & u(a_1,c_1) & u(a_1,c_2) & \cdots & u(a_1,c_n) \\
        u(a_2,c_1) & u(a_2,c_2) & \cdots & u(a_2,c_n) & \cdots & u(a_2,c_1) & u(a_2,c_2) & \cdots & u(a_2,c_n) \\
        \vdots & \vdots & \ddots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
        u(a_l,c_1) & u(a_l,c_2) & \cdots & u(a_l,c_n) & \cdots & u(a_l,c_1) & u(a_l,c_2) & \cdots & u(a_l,c_n) \\
        \end{bmatrix}_{l\times (mn)}$,
        
$\mathbf{P} = \begin{bmatrix} \mathbf{P}(q_1|a_1) & \mathbf{P}(q_2|a_1) & \cdots & \mathbf{P}(q_m|a_1) & \mathbf{P}(q_1|a_2) & \mathbf{P}(q_2|a_2) & \cdots & \mathbf{P}(q_m|a_l) \end{bmatrix}_{1\times (lm)}$

$\tilde{\mathbf{P}} = \begin{bmatrix} \mathbf{P}(q_1|a_1) & \mathbf{P}(q_2|a_1) & \cdots & \mathbf{P}(q_m|a_1) \\ \mathbf{P}(q_1|a_2) & \mathbf{P}(q_2|a_2) & \cdots & \mathbf{P}(q_m|a_2) \\ \mathbf{P}(q_1|a_l) & \mathbf{P}(q_2|a_l) & \cdots & \mathbf{P}(q_m|a_l) \end{bmatrix}_{1\times m}$

Then the  problem is:
    
$$
\begin{aligned}
\max_{\Pi \geq 0} \quad & \mathbf{1}_{l\times1}^{'}\Pi\Phi' \\
\text{s.t.} \quad 
C1: \quad & \text{tr}(\Pi U') = w \\
C2: \quad & \Pi(I_{m\times m}\otimes \mathbf{1}_{n\times 1}) = \tilde{\mathbf{P}}\times(\Pi\mathbf{1}_{(mn)\times 1}\mathbf{1}_{m\times1}^{'})\\
C3: \quad & \mathbf{1}_{l\times 1}^{'}\Pi\mathbf{1}_{(mn)\times 1} = 1 \\
C4: \quad & (U\times\Pi)\mathbf{1}_{(mn)\times 1}\mathbf{1}_{l\times 1}^{'} \geqq \frac{\Pi}{\tilde{\mathbf{P}}\otimes\mathbf{1}_{n\times 1}^{'}} \bigg[ U\times(\tilde{\mathbf{P}}\otimes\mathbf{1}_{n\times 1}^{'}) \bigg]'
\end{aligned}
$$

where "$\times$" denotes elementwise multiplication or the Hadamard product.

## 3.2 Solve The Primal Problem

```{code-cell} ipython3
# Setting the variables
w = 3.0
A = np.array([0, 0.2, 0.4, 0.6])
Q = np.array([1, 2])
C = np.linspace(0, 2.25, 81)
l, m, n = len(A), len(Q), len(C)

Φ = (Q.reshape((m,1)) - C.reshape((1,n))).flatten().reshape((1, m*n))
U = np.kron(np.ones((1,2)), np.array([C[j]**0.5/0.5 + (1-A[i])**0.5/0.5 
                                      for i in range(l) 
                                      for j in range(n)]
                                    ).reshape((l,n)))
P_tilde = np.array([[0.9, 0.1], [0.6, 0.4], [0.4, 0.6], [0.25, 0.75]])
P = P_tilde.flatten().reshape((1, l*m))

E1 = np.ones((m*n, 1))
E2 = np.ones((l, 1))
I = np.eye(m)
E3 = np.ones((n, 1))
E4 = np.ones((m, 1))

Π = cp.Variable((l, m*n))
obj = cp.Maximize(E2.T@Π@Φ.T)
cons = [cp.trace(Π@U.T) == w, 
       Π@np.kron(I, E3) == cp.multiply(P_tilde, Π@E1@E4.T),
        E2.T@Π@E1 == 1,
        Π >= np.zeros((l, m*n)),
        Π <= np.ones((l, m*n)),
        cp.multiply(U, Π)@E1@E2.T >= cp.multiply(Π, 1/np.kron(P_tilde, E3.T))@cp.multiply(U, np.kron(P_tilde, E3.T)).T]

```

```{code-cell} ipython3
prob = cp.Problem(obj, cons)
prob.solve(solver=cp.ECOS)

print("DCP: ", prob.is_dcp())
print("status: ", prob.status)
print("The optimal value is ", prob.value)
print("The dual solution is ", cons[0].dual_value, cons[1].dual_value, cons[2].dual_value, cons[3].dual_value)

```

## 3.3 Figures

We then wrap everything into the function for convenience to plot.

```{code-cell} ipython3
def u(x, y):
    return x**0.5/0.5 + (1-y)**0.5/0.5
```

```{code-cell} ipython3
def LP(w, A, Q, C, P_tilde, u):
    
    l, m, n = len(A), len(Q), len(C)
    Φ = (Q.reshape((m,1)) - C.reshape((1,n))).flatten().reshape((1, m*n))
    U = np.kron(np.ones((1,2)), np.array([u(C[j], A[i]) 
                                          for i in range(l) 
                                          for j in range(n)]
                                        ).reshape((l,n)))
    P = P_tilde.flatten().reshape((1, l*m))
    
    E1 = np.ones((m*n, 1))
    E2 = np.ones((l, 1))
    I = np.eye(m)
    E3 = np.ones((n, 1))
    E4 = np.ones((m, 1))
    
    Π = cp.Variable((l, m*n))
    obj = cp.Maximize(E2.T@Π@Φ.T)
    cons = [cp.trace(Π@U.T) == w, 
           Π@np.kron(I, E3) == cp.multiply(P_tilde, Π@E1@E4.T),
            E2.T@Π@E1 == 1,
            Π >= np.zeros((l, m*n)),
            Π <= np.ones((l, m*n)),
            cp.multiply(U, Π)@E1@E2.T >= cp.multiply(Π, 1/np.kron(P_tilde, E3.T))@cp.multiply(U, np.kron(P_tilde, E3.T)).T]
    
    prob_full = cp.Problem(obj, cons[0:-1])
    prob_full.solve(solver=cp.CBC)
    if prob_full.status == "optimal":
        Π_full = Π.value
        dist_full = np.sum(Π_full,1)
        Ea_full = float(A.reshape((1,4))@dist_full.reshape((4,1)))
        Ec_full = Π_full@np.kron(I, C.reshape((n,1))) / (Π_full@np.kron(I, E3))
    else:
        Ea_full = float("-inf") 
        Ec_full = float("-inf")*np.ones((l,m))
    
    prob_unobs = cp.Problem(obj, cons) 
    prob_unobs.solve(solver=cp.CBC)
    if prob_unobs.status == "optimal":
        Π_unobs = Π.value
        dist_unobs = np.sum(Π_unobs,1)
        Ea_unobs = float(A.reshape((1,4))@dist_unobs.reshape((4,1)))
        Ec_unobs = Π_unobs@np.kron(I, C.reshape((n,1))) / (Π_unobs@np.kron(I, E3))
    else:
        Ea_unobs = float("-inf")
        Ec_unobs = float("-inf")*np.ones((l,m))
    
    optimal_value = np.array([prob_full.value, prob_unobs.value])
    Ea = np.array([Ea_full, Ea_unobs])
   
    return optimal_value, Ea, list(Ec_full), list(Ec_unobs)

```

## Plot Replication

```{code-cell} ipython3
w = np.linspace(1, 5, 100)
```

```{code-cell} ipython3
filterwarnings("ignore")
def LP_fig12(w):
    return LP(w, A, Q, C, P_tilde, u)[0:2]
rev_fig12 = np.array(list(map(LP_fig12, w)))
```

### Figure 1

```{code-cell} ipython3
plt.figure(figsize=(6.5, 6.5))
plt.plot(w, rev_fig12[:, 0][:, 0])
plt.plot(w, rev_fig12[:, 0][:, 1])
plt.hlines(0, 1.0, 5.0, linestyle="dashed")
plt.xlabel("w")
plt.ylabel("s(w)")
plt.xlim([1.0, 5.0])
plt.ylim([-1.5, 2.0])
plt.title("Figure 1\n Optimized surplus function", y=-0.2)
plt.text(2.5, 1.6, "Full Information", size=15)
plt.text(1.5, 0.8, "Unobserved Action", size=15)
plt.show()
```

### Figure 2

```{code-cell} ipython3
plt.figure(figsize=(6.5, 6.5))
plt.plot(w, rev_fig12[:, 1][:, 0])
plt.plot(w, rev_fig12[:, 1][:, 1])
plt.xlabel("w")
plt.ylabel("E{a(w)}")
plt.xlim([1.0, 5.0])
plt.ylim([0.0, 0.8])
plt.title("Figure 2\n Actions", y=-0.2)
plt.text(2.3, 0.65, "Full Information", size=15)
plt.text(2.6, 0.15, "Unobserved Action", size=15)
plt.show()
```

```{code-cell} ipython3
plt.figure(figsize=(6.5, 6.5))

plt.plot(w, rev_fig12[:, 1][:, 0])
plt.plot(w[rev_fig12[:, 1][:, 0] == 0.6], rev_fig12[:, 1][:, 0][rev_fig12[:, 1][:, 0] == 0.6], marker="v")
plt.plot(w[(rev_fig12[:, 1][:, 0] < 0.6) * (rev_fig12[:, 1][:, 0] > 0.4)], 
         rev_fig12[:, 1][:, 0][(rev_fig12[:, 1][:, 0] < 0.6) * (rev_fig12[:, 1][:, 0] > 0.4)], marker="^")
plt.plot(w[rev_fig12[:, 1][:, 0] == 0.4], rev_fig12[:, 1][:, 0][rev_fig12[:, 1][:, 0] == 0.4], marker="v")
plt.plot(w[(rev_fig12[:, 1][:, 0] < 0.4) * (rev_fig12[:, 1][:, 0] > 0.2)], 
         rev_fig12[:, 1][:, 0][(rev_fig12[:, 1][:, 0] < 0.4) * (rev_fig12[:, 1][:, 0] > 0.2)], marker="^")
plt.plot(w[rev_fig12[:, 1][:, 0] == 0.2], rev_fig12[:, 1][:, 0][rev_fig12[:, 1][:, 0] == 0.2], marker="v")
plt.plot(w[(rev_fig12[:, 1][:, 0] < 0.2) * (rev_fig12[:, 1][:, 0] > 0)], 
         rev_fig12[:, 1][:, 0][(rev_fig12[:, 1][:, 0] < 0.2) * (rev_fig12[:, 1][:, 0] > 0)], marker="^")
plt.plot(w[rev_fig12[:, 1][:, 0] == 0], rev_fig12[:, 1][:, 0][rev_fig12[:, 1][:, 0] == 0], marker="v")

plt.plot(w, rev_fig12[:, 1][:, 1])
plt.plot(w[rev_fig12[:, 1][:, 1] == 0.4], rev_fig12[:, 1][:, 1][rev_fig12[:, 1][:, 1] == 0.4], "k-")
plt.plot(w[rev_fig12[:, 1][:, 1] == 0.2], rev_fig12[:, 1][:, 1][rev_fig12[:, 1][:, 1] == 0.2], "k-")
plt.plot(w[rev_fig12[:, 1][:, 1] == 0], rev_fig12[:, 1][:, 1][rev_fig12[:, 1][:, 1] == 0], "k-")
plt.plot(w[(rev_fig12[:, 1][:, 1] < 0.4) * (w<2.3)], rev_fig12[:, 1][:, 1][(rev_fig12[:, 1][:, 1] < 0.4) * (w<2.3)], "o")
plt.plot(w[(rev_fig12[:, 1][:, 1] < 0.4) * (rev_fig12[:, 1][:, 1] > 0.2) * (w>2.7) * (w<4.0)], 
         rev_fig12[:, 1][:, 1][(rev_fig12[:, 1][:, 1] < 0.4) * (rev_fig12[:, 1][:, 1] > 0.2) * (w>2.7) * (w<4.0)], "o")
plt.plot(w[(rev_fig12[:, 1][:, 1] < 0.2) * (rev_fig12[:, 1][:, 1] > 0) * (w>4.3)], 
         rev_fig12[:, 1][:, 1][(rev_fig12[:, 1][:, 1] < 0.2) * (rev_fig12[:, 1][:, 1] > 0) * (w>4.3)], "o")

plt.xlabel("w")
plt.ylabel("E{a(w)}")
plt.xlim([1.0, 5.0])
plt.ylim([0.0, 0.8])
plt.title("Figure 2\n Actions", y=-0.2)
plt.text(2.3, 0.65, "Full Information", size=15)
plt.text(2.6, 0.15, "Unobserved Action", size=15)
plt.show()
```

### Figure 3 & 4

```{code-cell} ipython3
filterwarnings("ignore")
def LP_fig34(w):
    return LP(w, A, Q, C, P_tilde, u)[2:4]
rev_fig34 = np.array(list(map(LP_fig34, w)))
```

```{code-cell} ipython3
plt.figure(figsize=(6.5, 6.5))
for i in range(l):
    for j in range(m):
        plt.plot(w, rev_fig34[:, 1][:, i][:, j])
plt.xlabel("w")
plt.ylabel("E(c) given a, q, w")
plt.xlim([1.0, 5.0])
plt.ylim([0.0, 2.25])
plt.title("Figure 3\n Unobserved Action Consumption", y=-0.3)
plt.annotate("a=.4, q=2", xy=(2.5, 0.5), xytext=(1.3, 0.7), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2, q=2", xy=(3.7, 1.5), xytext=(2.2, 1.65), arrowprops={"arrowstyle":"-"})
plt.annotate("a=0, q=(1,2)", xy=(4.8, 2.05), xytext=(3.0, 2.15), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2,\nq=2", xy=(2.0, 0.15), xytext=(1.3, 0.24), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.4, q=1", xy=(3.0, 0.10), xytext=(3.6, 0.2), arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2,\nq=1", xy=(4.0, 0.9), xytext=(4.3, 0.75), arrowprops={"arrowstyle":"-"})
plt.annotate("a=0, q=(1,2)\na=.2, q=1", xy=(2.1, 0), xytext=(1.8, -0.3), arrowprops={"arrowstyle":"-"})
plt.annotate(r"$\{$",fontsize=25, xy=(2.1, 0), xytext=(1.6, -0.3))
plt.annotate(r"$\}$",fontsize=25, xy=(2.1, 0), xytext=(2.5, -0.3))
plt.show()
```

```{code-cell} ipython3
plt.figure(figsize=(6.5, 6.5))
for i in range(l):
    for j in range(m):
        plt.plot(w, rev_fig34[:, 0][:, i][:, j])
plt.xlabel("w")
plt.ylabel("E{c(w)}")
plt.xlim([1.0, 5.0])
plt.ylim([0.0, 2.25])
plt.title("Figure 4\n Full Information Consumption", y=-0.2)
plt.show()
```
