---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Repeated Moral Hazard

## Spear and Srivastava

Spear and Srivastava (1987) {cite}`Spear_Srivastava_87` presented a
recursive formulation of an infinitely repeated, discounted  repeated
principal-agent problem.

*  A **principal** owns a technology
that produces output $q_t$ at time $t$ according to a probability distribution $F$ that depends on the action $a_t$ of a worker we'll can an **agent**.
*  $q_t$ is determined by a family $F(q_t\vert a_t)$ of conditional c.d.f.'s  
*  $a_t$ is an action taken at the beginning of $t$ by the **agent**.  
*  The  agent thereby chooses the  c.d.f. that governs $y_t$.
*  The principal  does **not** observe $a_t$.
*  The principal  **does** observe $q_t$ at the end of period $t$.
*  The principle   remembers earlier $q_{t-j}$'s and so at the end of period $t$ knows  history $\{q_s\}_{s=0}^t$.
*  The principal is risk-neutral and  has access to an outside loan market with constant risk-free gross interest rate $\beta^{-1}$.
*  The agent is averse to consumption risk and has preferences random streams of consumption ordered by $E_0 \sum^\infty_{t=0} \beta^t u(c_t, a_t)$ 
*  $u$ is increasing in $c$ and decreasing in $a$.



The principal designs a  life-time contract. 

A **contract**  $\sigma$ is a sequence of functions  whose $t$th component $\sigma_t$ maps a history $q^{t-1} = q_{t-1}, q_{t-2}, \ldots , q_0$ of past outputs remembered at time $t$ into a time $t$ action $a_t$ that it asks the agent to perform and a time $t$  consumption $c_t$ that it pays the agent.  


The risk-neutral principal  designs the contract  to maximize $E_0 \sum^\infty_{t=0} \beta^t \{q_t - c_t\}$.



Spear and Srivastava (1987) {cite}`Spear_Srivastava_87` described a recursive representation  of an optimal contract.


## Recursive Contract

Let $w$ denote the  expected value of discounted  continuation utilities that the principal had   promised to the agent.
  

Given $w$, the principal designs three functions:

*  $a(w)$ determines a requested  action so that the action at $t$ is $a_t=a(w_t)$
*  $c(w,q)$ determines current consumption so that $c_t=c(w_t, q_t)$
*  $\tilde w(w,q)$  determines a promised next period expected discounted 
utility so that $w_{t+1} = \tilde w (w_t, q_t)$.

The functions $a(w)$, $c(w,q)$, and $\tilde w (w,q)$
must satisfy the following two sets of  constraints:

$$
w = \int \{ u[c(w,q), a(w)] + \beta \tilde w(w,q)\}\ dF[q | a(w)]
$$ (eq:eq1)

and

$$
\begin{aligned} & \int \left\{ u[c(w,q), a(w)] + \beta\tilde w (w,q)\right\}\
dF[q\vert a(w))\cr
&\geq \int \{u [c(w,q),\hat a] + \beta\tilde w
(w,q)\} dF(q\vert\hat a)\,, \hskip.5cm \forall\; \hat a \in A.
\end{aligned}
$$ (eq:eq2)

 * Equation {eq}`eq:eq1`  requires the contract to deliver the promised
discounted expected continuation utility. 
 * Equation {eq}`eq:eq2`  is the **incentive
compatibility** constraint that requires that the agent chooses to
deliver the amount of effort called for by the
contract. 

Let $v(w)$ be the value to the principal associated with promising discounted utility $w$ to the agent. 

The principal's optimal value function satisfies the Bellman equation 

$$ 
v(w) =\max_{a,c,\tilde w}\ \{q-c(w,q)+\beta \ v[\tilde w(w,q)]\}\
dF[q\vert a(w)]
$$ (eq:eq3)

where maximization is over functions $a(w)$, $c(w,q)$, and $\tilde w(w,q)$
and is subject to the constraints {eq}`eq:eq1` and {eq}`eq:eq2`.

Notice that constraint {eq}`eq:eq1` itself takes the form of a Bellman equation.

Thus,  a value function $w$ that satisfies a Bellman equation that describes the agent's continuation value    is an argument of Bellman equation {eq}`eq:eq3` that describes the principal's continuation value. 

The value function $v(w)$ and the associated optimum policy functions
are to be solved by iterating on the Bellman operator associated with Bellman equation {eq}`eq:eq3`.

## Lotteries

A difficulty in problems like these is that   incentive
constraints   the constraint set can  fail to
be convex. 

Phelan and
Townsend (1991) {cite}`Phelan_Townsend_91` circumvented this problem  by using **randomization** to  convexify the constraint set.

Phelan and Townsend restrict outputs $y$, consumption $c$, actions $a$, and continuation values $w$ each to be
in its own  discrete and finite  set.

They restrict the 
principal's choice to a space of lotteries
over actions $a$ and outcomes $c,w'$.

To describe Phelan and Townsend's formulation, let $P(q\vert a)$ be
a family of discrete conditional probability distributions
over discrete spaces of outputs and actions $Q,A$, respectively.

Suppose that consumption and  continuation values are  constrained to
lie in discrete spaces $C,W$, respectively.

Phelan and Townsend's principal 
chooses a probability distribution
 $\Pi(a,q,c,w^\prime)$ subject first to the constraint
 that for all fixed $(\bar a, \bar q)$

$$
\sum_{C\times W} \Pi (\bar a, \bar q, c, w^\prime) = P (\bar q\vert \bar a)
\sum_{Q\times C\times W}\ \Pi(\bar a, q,c,w')
$$ (eq:town1a)

$$
\Pi(a,q,c,w')\geq 0 
$$(eq:town1b)


$$
\sum_{A\times Q\times C\times W}\ \Pi(a,q,c,w^\prime)=1 .
$$ (eq:town1c)


Equation {eq}`eq:town1a`  states that
${\rm Prob} (\bar a, \bar q) = {\rm Prob}(\bar q \vert \bar a)
{\rm Prob}(\bar a)$.

Remaining parts of the preceding three equations  just
require that **probabilities are probabilities** in the sense that

* they are nonnegative
* sum to $1$

The counterpart of Spear-Srivastava's equation {eq}`eq:eq1`  is

$$
w=\sum_{A\times Q\times C\times W}\ \{u(c,a) +\beta w^\prime\}\
\Pi(a,q,c,w^\prime) . 
$$ (eq:eq1prime)

A counterpart to Spear-Srivastava's incentive constraint  {eq}`eq:eq2`   for each
$a,\hat a$ is


$$
\begin{aligned} & \sum_{Q\times C\times W}\ \{u(c,a)  + \beta w' \}\ \Pi (c,w' \vert q, a) P(q\vert a)\\
&\geq \sum_{Q\times C\times W}\ \{u(c,\hat a) + \beta w' \}\ \Pi(c,w' \vert q,a) P(q\vert\hat a).
\end{aligned}
$$ (eq:eq2prime)


Here $\Pi(c,w^\prime\vert q,a) P(q\vert \hat a)$ is the probability attached to  $(c,w^\prime, q)$ 
if the agent claims to be working $a$ but is actually working $\hat a$. 

Write

$$
\begin{aligned}\Pi(c,w^\prime\vert q,a) P(q\vert\hat a)  = 
\Pi(c,w^\prime\vert q,a) P(q\vert a)\ \frac{P(q\vert\hat a)}{P(q\vert a)} & =
\Pi(c,w^\prime,q\vert a)\ \cdot\ \frac{P(q\vert\hat a)}{P(q\vert a)}.
\end{aligned}
$$

Write the incentive constraints as

$$
\begin{aligned} \sum_{Q\times C\times W}\ &\{u(c,a)  +\beta w^\prime\} \Pi(c,w^\prime, q\vert a)\cr & \geq
\sum_{Q\times C\times W}\ \{u(c,\hat a) +\beta w^\prime\}\ \Pi(c,w^\prime, q\vert \hat a)\     \cdot\ {P(q\vert \hat a)\over P(q\vert a)}.
\end{aligned}
$$

Multiplying both sides by the unconditional probability $P(a)$ gives Phelan and Townsend's counterpart to Spear and Srivastava's  incentive constraint  {eq}`eq:eq2`.

$$
\begin{aligned} & \sum_{Q\times C\times W}\ \{u(c,a)+\beta w^\prime\}\
\Pi(a,q,c,w^\prime)\cr 
&\geq \sum_{Q\times C\times W}\ \{u(c,\hat a) + \beta w^\prime\} \
{P(q\vert\hat a)\over P(q\vert a)}\ \Pi (a,q,c,w^\prime)
\end{aligned}
$$

## Linear Programming Formulation


The value function  for the principal's problem satisfies the Bellman equation 

$$
v(w) =\max_{\Pi} \{(q -c) +
     \beta v(w')\} \Pi(a,q,c,w') , 
$$ (eq:bell2)

where  maximization is over the probabilities $\Pi(a,q,c,w')$
subject to equations {eq}`eq:town1a`, {eq}`eq:town1b`,  {eq}`eq:eq1prime`, and {eq}`eq:eq2prime`.
 

This is a **linear
programming** problem. 

Each  $(a,q,c,w')$ is constrained to a discrete grid of points.

The term $(q-c)+\beta v(w')$ on the right side of equation {eq}`eq:bell2` 
can be represented as a *fixed* vector that multiplies a vectorized
version of  the
probabilities $\Pi(a,q,c,w')$.  

Similarly, each of the
constraints  {eq}`eq:town1a`, {eq}`eq:town1b`,  {eq}`eq:eq1prime`, and {eq}`eq:eq2prime` can be represented
as a linear inequality in the choice variables, namely,  the
probabilities $\Pi$.  

An associated **Bellman operator** is also a linear program.

Phelan and Townsend iterated to convergence on this Bellman operator.  

At each step of the iteration on the  Bellman operator,
one linear program is to be solved for each point
$w$ in the space of grid values for $W$.

Phelan and Townsend found that
lotteries were often redundant in the sense that most of the
$\Pi(a,q,c,w')$'s  are  zero and only a few are $1$.


## Imported from Static version

Let's import some Python tools.

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

##  Static Version of Phelan-Townsend Model


This section  uses   several linear programming algorithms to solve the one-period version
of a model of  Phelan and
Townsend that appears in sections II and III of  {cite}`Phelan_Townsend_91`.  

We'll use Phelan and Townsend's  parameter values 


    
**Setting:**

 - Action $a$ from a finite set of possible actions $\mathbf{A} \subset \mathbf{R}_{+}$ 
 - Output $q \in \text{finite }\mathbf{Q} \subset \mathbf{R}_{+}$  described by conditional probability $\mathbf{P}(q|a)>0$ for all $q$
 - Consumption $c \in \text{finite } \mathbf{C} \subset \mathbf{R}_{+}$ 
 - Utility $U(a,c): \mathbf{A}\times\mathbf{C} \to \mathbf{R}_{+}$ where 
   - $U(a,c)$ is strictly concave over changes in $c$ holding $a$ constant
   - Higher actions induce higher expected utilities
 - **Ex ante** utility level:
   - Lowest ex ante utility level with certain highest labor assignment and lowest consumption receipt, $\underline{w}$
   - Highest ex ante utility level with certain lowest labor assignment and highest consumption receipt, $\overline{w}$
   - Arbitrary utility level in between, $w = \alpha\underline{w}+(1-\alpha)\overline{w}\in\mathbf{W}=[\underline{w},\overline{w}]$, $\alpha\in[0,1]$
 - $d_{0}(w)$ denotes the fraction of agents  whose initial promised utility is $w$
 - $\Pi^{w}(a,q,c)$ is the  probability  that an agent promised value   $w$ takes action $a$ receives  output $q$ and gets consumption  $c$  while running  his  own occur  production technology (and not the principal's technology) 
    
**Definitions:**
-   a **contract** with promised value   $w\in\mathbf{W}$  is a function $\Pi^{w}(a,q,c)$ that maps $w$ into a probability mass function  over   $(a, q, c)$. 
- an **allocation** is  a collection of contracts for each $w$ in the support of a probability mass function $d_{0}(w)$ 

**Full Information Problem (FIP)**

Section II of   {cite}`Phelan_Townsend_91` solves a full information, one-period version of their problem.

The purpose of this section is to establish a benchmark with which to compare their
section III one-period version of the private information problem.

In the section II full information problem, the principal observes $a$.

That eliminates the incentive constraint from the problem. 

The static version of full information problem is 
- 
$$
  \max_{\Pi^w(a,q,c)} s(w)=\sum_{\mathbf{A} \times \mathbf{Q} \times \mathbf{C}}(q-c)\Pi^{w}(a, q, c) 
$$

subject to

$$
\begin{aligned} \text{C1:} \quad & w = \sum_{\mathbf{A}\times\mathbf{Q}\times\mathbf{C}}U(a,c)\Pi^{w}(a,q,c) \\
\text{C2:} \quad & \sum_{\mathbf{C}} \Pi^{w}(\bar{a}, \bar{q}, c)=P(\bar{q} \mid \bar{a}) \sum_{\mathbf{Q} \times \mathbf{C}} \Pi^{w}(\bar{a}, q, c), \forall (\bar{a},\bar{q})\in\mathbf{A}\times\mathbf{Q} \\
\text{C3:} \quad  & \sum_{\mathbf{A} \times \mathbf{Q} \times \mathbf{C}} \Pi^{w}(a, q, c)=1 , \quad 
 \Pi^{w}(a, q, c) \geqq 0, \forall (a, q, c) \in \mathbf{A} \times \mathbf{Q} \times \mathbf{C} \\
\end{aligned}
$$

where 
* constraint C1 defines discounted expected utility
* constraint C restricts conditional probabilities
* constraint C3  resticts a probability measure
    
The total social surplus associated with  initial distribution $d_{0}$
is

$$
S^{*}(d_{0})\equiv \sum_{\mathbf{W}}s^{*}(w)d_{0}(w)
$$

We require that 

$$
S^{*}(d_{0}) \geq 0 
$$



**Definition;**: An initial distribution of utilities $d_{0}$ and an associated surplus-maximizing plan $\{\Pi^{w*}\}^{w^*\in\mathbf{W}}$ constitute a Pareto optimum if the support of $d_{0}$ lies within the non-increasing portion of s*(w).
</font>
    
Now we'll do an example.

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

Then full information problem (FIP) can be written as:

$$
\begin{aligned}
& \max_{\Pi_{xy} \ge 0} \ \sum_{xy} \Pi_{xy} \Phi_{xy} \\
&  \\ \text{C1:} & \ \sum_{x=1}^N \sum_{y=1}^M \Pi_{xy} \cdot u_y = w = 1.5\\
\text{C2:}  & \ \sum_{y=1}^M \Pi_{1 y} = P(\underline{q}|a=0) \sum_{xy}\Pi_{xy} \\
\text{C3:} & \sum_{y=1}^M \Pi_{2 y} = P(\overline{q}|a=0) \sum_{xy}\Pi_{xy}, \quad  \sum_{xy}\Pi_{xy} = 1 \\
\end{aligned}
$$

where $N=2,M=4$.

This is equivalent to:

$$
\begin{aligned}
& \max_{\Pi_{xy} \ge 0} \ \sum_{xy} \Pi_{xy} \Phi_{xy} \\
& \textrm{subject to } \\  \text{C1:} & \ \sum_{x=1}^N \sum_{y=1}^M \Pi_{xy} \cdot u_y = w = 1.5\\
\text{C2:} & \ \sum_{y=1}^M \Pi_{1 y} = P(\underline{q}|a=0) \cdot 1 = \frac{1}{2} \\
& \ \sum_{y=1}^M \Pi_{2 y} = P(\overline{q}|a=0) \cdot 1 = \frac{1}{2}\\
\text{C3:} & \  \sum_{xy}\Pi_{xy} = 1 \\
\end{aligned}
$$

Call this  the **"elementary version"**.

We can rewrite this in the compact matrix form

$$
\begin{aligned}
& \max_{\Pi_{xy} \ge 0} \ Tr (\Pi' \Phi)\\
& \textrm{subject to} \\ \text{C1:} & \ \mathbf{1}_2' \cdot \Pi \cdot u = w \\
\text{C2:} & \ \Pi \cdot \mathbf{1}_4 = p \\
\text{C3:} & \ \mathbf{1}_2' \cdot \Pi \cdot \mathbf{1}_4 = 1 \\
\end{aligned}
$$

where, $p=(P(\underline{q}|a=0),P(\overline{q}|a=0))=(\frac{1}{2},\frac{1}{2})$.

Using vectorizing and Kronecker product, we can formulate the problem as 
$$
\begin{aligned}
\max_{z \ge 0} & \ vec(\Phi)' z\\ 
& \textrm{subject  to}\\ 
\text{C1:} & \ (u' \otimes \mathbf{1}_2') \cdot z = w \\
\text{C2:} & \ (\mathbf{1}_4' \otimes \mathbf{I}_2) \cdot z = p \\
\text{C3:} & \ (\mathbf{1}_4' \otimes \mathbf{1}_2) \cdot z = 1 \\
\end{aligned}
$$

where $z = vec(\Pi)$.

The problem becomes

$$
\max_{z \ge 0} \ vec(\Phi)^\prime z 
$$

subject to

$$. 
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
=b
$$

Call this  the **"compact version"**.


### Module Details

We can  implement the "elemnentary version" by CVXPY and PuLP and the "compact version" by CVXPY and Scipy.linprog.


Later,   we will use Scipy.linprog to implement the "compact version" and CVXPY and PuLP to implement "elementary version".

**Dual Problem:**
    
The dual of FIP is:

$$
\min_{\nu_1, \mu, \nu_2} w\nu_1 + \sum_{x=1}^N p_x \mu_x + \nu_2
$$

subject to 
$$  u_y \nu_1 + \mu_x + \nu_2 \ge \Phi_{xy}, \forall x,y; 
\nu_1, \mu, \nu_2 \ \textrm{unrestricted}
$$

where N = 2, $\nu_1$ is the dual variable corresponding to C1, $\mu$ is the dual variable corresponding to C2 and $\nu_2$ is the dual variable corresponding to C3. Notice, here, $\mu$ is a vector with two entries while $\nu_1$ and $\nu_2$ are both scalars.
    
Call this the **"elementary version"** of the dual problem.
    
Based on the "compact version" of the primal problem, we can directly derive the "compact version" of the dual problem since the "compact version" of the primal problem is a standard form of LP.

The **"compact version"** of the dual problem is:

$$
\min_{\lambda} b' \lambda 
$$

subject to 

$$
A' \lambda \ge vec(\Phi) 
$$

where $\lambda$ is  unrestricted and 
where, $\lambda$ is the dual variable vector with $\lambda_1 = \nu_1$, $\left( \begin{array}\ \lambda_2 \\ \lambda_3 \end{array} \right)= \mu$, $\lambda_4 = \nu_2$.

Furthermore, by rewriting the "elementary version" of the dual problem, we can derive the same "compact version" of the dual problem as shown above.

Again, we can use CVXPY and Scipy.linprog to solve the "compact version" of the dual problem and CVXPY and PuLP to the "elementary version" of the dual problem.

Later   we will use Scipy.linprog and CVXPY to implement the "compact version" and CVXPY and PuLP to implement "elementary version".

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

##  The Primal Problem

### 1.2.1 SciPy


The three methods from `scipy.linprog` give different solutions.

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

### CVXPY

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

### PuLP

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

### QuantEcon

We use quantecon's `linprog_simplex` .

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

###  Time Comparison

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

## The Dual Problem

 

Let's use the program `scipy.linprog`

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



Now let's use CVXPY

First,  the "compact version".

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

Evidently, outcomes agree with those from  the primal problems.

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

###  PuLP

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

###  Time Comparison

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

##  Static Unobserved-Action Economy

Now we'll move on to replicate the static private information economy of section III of {cite}`Phelan_Townsend_91`.
    
**Setting:**
 - An agent's action is private.
 - That adds an incentive constraint to the full information setting of section II of {cite}`Phelan_Townsend_91`.
    - The lowest possible ex ante expected utility for the unobserved action economy is that of receiving the lowest consumption and the lowest labour amount
 - Additional constriants: for all assigned and possible alternative actions $(a,\hat{a})\in \mathbf{A}\times\mathbf{A}$
    - $\sum_{\mathbf{Q} \times \mathbf{C}} U[a, c]\left\{\Pi^{w}(c \mid q, a) P(q \mid a)\right\} \geqq \sum_{\mathbf{Q} \times \mathbf{C}} U[\hat{a}, c]\left\{\Pi^{w}(c \mid q, a) P(q \mid \hat{a})\right\}$
    - where $\Pi^{w}(c \mid q, a)$ is the conditional probability implied by $\Pi^{w}(a, q, c)$
    - $\Pi^{w}(c \mid q, a) P(q \mid a)$ is the probability of a given $(q, c)$ combination given that action $a$ is recommended and that this action $a$ is taken
    - $\Pi^{w}(c \mid q, a) P(q \mid \hat{a})$ is the probability of a given $(q, c)$ combination given that action $a$ is announced and deviation action $\hat{a}$ is taken instead 
    - $\Pi^{w}(q, c \mid a) = \Pi^{w}(c \mid q, a) P(q \mid a) \implies \sum_{\mathbf{Q} \times \mathbf{C}} U[a, c] \Pi^{w}(q, c \mid a) \geqq \sum_{\mathbf{Q} \times \mathbf{C}} U[\hat{a}, c] \frac{\mathbf{P}(q \mid \hat{a})}{\mathbf{P}(q \mid a)} \Pi^{w}(q, c \mid a) $  
    - $\implies \sum_{\mathbf{Q} \times \mathbf{C}} U[a, c] \Pi^{w}(a, q, c) \geqq \sum_{\mathbf{Q} \times \mathbf{C}} U[\hat{a}, c] \frac{P(q \mid \hat{a})}{P(q \mid a)} \Pi^{w}(a, q, c) \qquad \text{ (C4)}$ 
    - The ratio $\frac{P(q \mid \hat{a})}{P(q \mid a)}$ gives how many more times likely it is that output $q$ will occur given deviation action $\hat{a}$ as opposed to recommended action $a$, and thus updates the joint probability of observing recommended action $a$, output $q$, and consumption $c$.
    
**Dynamic Program**

$$
\begin{aligned}
& \max_{\Pi^{w}} \sum_{A \times Q \times C} (q-c)\Pi^{w}(a,q,c) \\
& \textrm{subject to}  \\
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
    
|   a   | P(q=1) | P(q=2) |
| :---: | :----: | :----: |
|   0   |  0.9   |  0.1   |
|  0.2  |  0.6   |  0.4   |
|  0.4  |  0.4   |  0.6   |
|  0.6  |  0.25  |  0.75  |
    
**Primal Problem:**
    
Define a three-dimension matrix $\Pi^w$ with a size of $(l \times m \times n) = (4 \times 2 \times 81)$, in which each dimension represents $A$, $Q$ or $C$, respectively. 

For example, $\Pi_{3,2,5}^w = \Pi^w(a=0.4,q=2,c=0.140625)$.

 In the same way, define a two-dimension matrix $\Phi$ with entries $\Phi_{yz} = q_y - c_z$, a two-dimension matrix $U$ with entries $U_{x,z} = U[a_x,c_z] = [(c_z^{0.5}/0.5+(1-a_x)^{0.5}/0.5]$ and a two-dimension matrix $P$ with entries $P_{xy} = P(q=q_y|a=a_x)$.

Then the problem is 

$$
\begin{aligned}
& \max_{\Pi_{xyz}^w \ge 0} \ \sum_{x=1}^4\sum_{y=1}^2\sum_{z=1}^{81} \Pi^w_{xyz} \Phi_{yz} \\
& \textrm{subject to} \\ 
     \text{C1:} & \ \sum_{x=1}^4\sum_{y=1}^2\sum_{z=1}^{81} U_{xz} \Pi^w_{xyz} = w\\
\text{C2:} & \ \sum_{z=1}^{81} \Pi^w_{xyz} = P_{xy} \sum_{y=1}^2 \sum_{z=1}^{81} \Pi^w_{xyz}, \forall x,y \\
\text{C3:} & \ \sum_{x=1}^4\sum_{y=1}^2\sum_{z=1}^{81} \Pi^w_{xyz} = 1 \\
\text{C4:} & \ \sum_{y=1}^2\sum_{z=1}^{81} U_{xz}\Pi^w_{x,y,z} \ge \sum_{y=1}^2\sum_{z=1}^{81} U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \Pi^w_{x,y,z}, \forall x, x^*
\end{aligned}
$$

Call this  the "elementary version" of the primal problem.

Note that this formulation contains a three-dimension matrix $\Pi$.

 We can expand matrix $\Pi$ by one axis and then convert it to a two-dimension matrix.
 
In this way, we obtain a new formulation of this problem below. 

We could  "vectorization" again  to convert two-dimension version of matrix $\Pi$ to a vector and then get a third formulation of the problem. 

However, we won't do that here.

Instead, we directly implement the "elementary version" of the primal problem by CVXPY.

Variables in CVXPY have at most two axes, so that it is impossible to create a three-dimension matrix of cp.Variable. 

The way we handle this situation  this is to create four two-dimension matrices of cp.Variable.

    
**Dual Problem:**
    
Here we only formulate the "elementary version" of the dual problem. 

We  have  to add **slack variables** $s_{xx^*}$ to convert constraint C4 into equality constraints.

For simplicity, we write $\Pi$ instead of $\Pi^w$. 

The standard form of primal problem is:
    
$$
\begin{aligned}
& \min_{\Pi_{xyz}} \ -\sum_x \sum_y \sum_z \Pi_{xyz} \Phi_{yz} \\
& \textrm{subject to} \\  \text{C1:} & \ \sum_x \sum_y \sum_z  U_{xz} \Pi_{xyz} = w\\
\text{C2:} & \ \sum_z \Pi_{xyz} - P_{xy} \sum_y \sum_z \Pi_{xyz} = 0, \forall x,y \\
\text{C3:} & \ \sum_x \sum_y \sum_z \Pi_{xyz} = 1 \\
\text{C4:} & \ \sum_y \sum_z U_{xz}\Pi_{xyz} = \sum_y \sum_z U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \Pi_{x,y,z} + s_{xx^*}, \forall x, x^* \\
& \ s_{xx^*} \ge 0, \forall x, x^* \\
& \ \Pi_{xyz} \ge 0, \forall x, y, z
\end{aligned}
$$
    
Denote $\alpha(scalar), \beta(l\times m = 4\times2), \gamma(scalar), \delta(l\times l =4\times4), \lambda(l\times l =4\times4)$ and $\mu(l\times m \times n=4\times 2 \times 81)$ are dual variables corresponding to C1, C2, C3, C4, $\ s_{xx^*} \ge 0$ and $\Pi_{xyz} \ge 0$, respectively.

The Langrangian is

$$
\begin{aligned}
& L(\Pi,s,\alpha,\beta,\gamma,\delta,\lambda,\mu) \\
= & -\sum_x \sum_y \sum_z \Pi_{xyz} \Phi_{yz} + (\sum_x \sum_y \sum_z U_{xz} \Pi_{xyz} - w)\alpha \\
&+ \sum_x \sum_y (\sum_z \Pi_{xyz} - P_{xy} \sum_y \sum_z \Pi_{xyz}) \beta_{xy} + (\sum_x \sum_y \sum_z \Pi_{xyz} -1)\gamma \\
&+ \sum_x \sum_{x^*} (\sum_y \sum_z U_{xz}\Pi_{xyz} - \sum_y \sum_z U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \Pi_{x,y,z} - s_{xx^*}) \delta_{xx^*}\\
&-  \sum_x \sum_{x^*} \lambda_{xx^*} s_{xx^*} - \sum_x \sum_y \sum_z \mu_{xyz} \Pi_{xyz} \\
= & -\sum_x \sum_y \sum_z \Pi_{xyz} \Phi_{yz} + \sum_x \sum_y \sum_z U_{xz} \Pi_{xyz}\alpha - w\alpha \\
&+ \sum_x \sum_y \sum_z \Pi_{xyz} \beta_{xy} - \sum_x \sum_y (P_{xy} \sum_y \sum_z \Pi_{xyz}) \beta_{xy} + \sum_x \sum_y \sum_z \Pi_{xyz} \gamma - \gamma \\
&+ \sum_x \sum_{x^*} (\sum_y \sum_z U_{xz}\Pi_{xyz})\delta_{xx^*} - \sum_x \sum_{x^*} (\sum_y \sum_z U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \Pi_{x,y,z}) \delta_{xx^*}  - \sum_x \sum_{x^*} s_{xx^*} \delta_{xx^*}\\
&-  \sum_x \sum_{x^*} \lambda_{xx^*} s_{xx^*} - \sum_x \sum_y \sum_z \mu_{xyz} \Pi_{xyz}
\end{aligned}
$$

We can simplify several terms of $L$.

$$
\begin{aligned}
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
\end{aligned}
$$

Substuting these  into $L$ gives

$$
\begin{aligned}
L = & \sum_x \sum_y \sum_z \Pi_{xyz} (-\Phi_{yz} + \alpha U_{xz} + \beta_{xy} - \sum_y \beta_{xy} P_{xy} + \gamma + \sum_{x^*} U_{xz} \delta_{xx^*} - \sum_{x^*} U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \delta_{xx^*} - \mu_{xyz})\\
&+ \sum_x \sum_{x^*} s_{xx^*} (-\delta_{xx^*} - \lambda_{xx^*})\\
&- \alpha w - \gamma \\
\end{aligned}
$$

Let $C = -\Phi_{yz} + \alpha U_{xz} + \beta_{xy} - \sum_y \beta_{xy} P_{xy} + \gamma + \sum_{x^*} U_{xz} \delta_{xx^*} - \sum_{x^*} U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \delta_{xx^*} - \mu_{xyz}$.

Then the dual criterion function is 

$$
g(\alpha,\beta,\gamma,\delta) 
= \inf_{\Pi,s}L 
= \begin{cases}
- \alpha w - \gamma, & \text{if } C = 0 \text{ and } -\delta_{xx^*}-\lambda_{xx^*} = 0 \\
- \infty, & \text{otherwise} 
\end{cases}
$$

The dual problem is


$$
\min_{\alpha,\beta,\gamma,\delta} \alpha w + \gamma
$$

subject to 

$$ -\Phi_{yz} + \alpha U_{xz} + \beta_{xy} - \sum_y \beta_{xy} P_{xy} + \gamma + \sum_{x^*} U_{xz} \delta_{xx^*} - \sum_{x^*} U_{x^* z} \frac{P_{x^* y}}{P_{xy}} \delta_{xx^*} \ge 0, \forall x,y,z \\
\delta_{xx^*} \le 0, \forall x,x^*
$$

We use CVXPY to solve the   dual problem.

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

## Solving The Primal Problem

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

## Solving the Dual Problem

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

##  Figures

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

## The Static Unobserved-Action Economy(Based on Another Formlation)


## 3.1 Another Formulation 
    
**Formulation:**
$$
\begin{aligned}
\max_{\Pi^{w}} & \sum_{\{0,0.2,0.4,0.6\}\times\{1,2\}\times\{0.028125n, n=0,1,\cdots,80\}} (q-c)\Pi^{w}(a,q,c) \\
\end{aligned}
$$

subject to    

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
& \textrm {subject to } \\  
C1: \quad & \text{tr}(\Pi U') = w \\
C2: \quad & \Pi(I_{m\times m}\otimes \mathbf{1}_{n\times 1}) = \tilde{\mathbf{P}}\times(\Pi\mathbf{1}_{(mn)\times 1}\mathbf{1}_{m\times1}^{'})\\
C3: \quad & \mathbf{1}_{l\times 1}^{'}\Pi\mathbf{1}_{(mn)\times 1} = 1 \\
C4: \quad & (U\times\Pi)\mathbf{1}_{(mn)\times 1}\mathbf{1}_{l\times 1}^{'} \geqq \frac{\Pi}{\tilde{\mathbf{P}}\otimes\mathbf{1}_{n\times 1}^{'}} \bigg[ U\times(\tilde{\mathbf{P}}\otimes\mathbf{1}_{n\times 1}^{'}) \bigg]'
\end{aligned}
$$

where "$\times$" denotes elementwise multiplication or the Hadamard product.

## Solve The Primal Problem

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
{
    "tags": [
        "hide-output"
    ]
}
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

# Dynamic Version of Phelan-Townsend (1991)

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

We are now set to compute    examples.

Let's begin with setting parameters.

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

So far, the algorithm is not fast enough. 

So, we relax our parameters as N=50 and tol=1e-5.

**Note to John and Smit:** We'll probably drop   this part because it takes a long time to finish. We provide an improved algorithm later.

```{code-cell} ipython3
# Solve the static unobserved-actions problem
w_l = u(A.min(), C.min())/(1-β)
w_u = u(A.min(), C.max())/(1-β)
W = np.linspace(w_l, w_u, N)
%time s_W_0, Pi_0 = solve_static_problem(W*(1-β), u, A, Q, C, P, "unobserved-actions")
```

```{code-cell} ipython3
{
    "tags": [
        "hide-output"
    ]
}
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

### Another  version

+++

Actually, each iteration of the algorithm takes about 12 seconds and the whole process takes about ten minutes. 

Maybe that does not  seem not too bad.

But we have set N as 50 and the tolerance as 1e-5.

This  N is not large enough to draw perfect pictures.

And the tolerance as 1e-5 is also not ideal. 

If we set N as 100 and tolerance as 1e-8, each iteration would take about 65 seconds and the process needs 81 iterations to convergence. 

The algorithm would take  one and a half hours to finish. 

So  inspired by the ection VI of {cite}`Phelan_Townsend_91`, we construc  afunction "solve_repeated_problem_v2" that  implements an algorithm that divides one period into two sub-periods with two sub-linear-programming problems. 

This function significantly shortens running times.

Here, we separate the utility function $U[a, c] = 2\sqrt{c} + 2\sqrt{1-a}$ into two independent parts.

So in the new function, we provide no argument "u".

For each period, $a, q, c, w'$ are random variables but $w$ is given.

We seek an the optimal probability $\Pi^w(a, q, c, w') = Pr(a, q, c, w'|w)$ and an optimal surplus $s(w)$.

Now, instead of directly solving the original problem, we solve two sub-problems. At each period, suppose the decision is made by two steps:

  - output is realized but the consumption is still undecided.
  - consumption is decided.

In the first step, $a, q, w^m$ are random variables but $w$ is given.

We seek an  optimal probability of $(a, q, w^m)$ that maximizes the surplus function $s(w)$, where $w^m$ is the expected utility of the second step. 

In the second step, $c, w'$ are random variables but $w^m$ is given.

We seek a   probability of $(c, w')$ that maximizes the surplus function $s^m(w^m)$.

The first step problem  is:

$$
\begin{aligned}
\max_{\Pi^{w}} & \ s(w) = \sum_{\mathbf{A} \times \mathbf{Q} \times \mathbf{W^m}} \{q + s^m(w^m)\} \Pi^w(a, q, w^m) \\
\textrm{subject to} \\
\mbox{C5:} & \ w = \sum_{\mathbf{A}\times\mathbf{Q}\times\mathbf{W^m}}\{2 \sqrt{1-a} +w^m\} \Pi^w(a, q, w^m) \\
\mbox{C6:} & \ \sum_{\mathbf{W^m}} \Pi^w(\bar{a}, \bar{q}, w^m)=P(\bar{q} \mid \bar{a}) \sum_{\mathbf{Q} \times \mathbf{W^m}} \Pi^w(\bar{a}, q, w^m) \\
\mbox{C7:} & \ \sum_{\mathbf{A} \times \mathbf{Q} \times \mathbf{W^m}} \Pi^w(a, q, w^m)=1, \Pi^w(a, q, w^m) \geqq 0 \\
\mbox{C8:} & \ \sum_{\mathbf{Q} \times \mathbf{W^m}} \{2 \sqrt{1-a}+w^m\} \Pi^w(a, q, w^m) \geqq \\
& \ \sum_{\mathbf{Q} \times \mathbf{W^m}} \{2 \sqrt{1-\hat{a}} + w^m \} \frac{P(q \mid \hat{a})}{P(q \mid a)} \Pi^w(a, q, w^m)
\end{aligned}
$$

The second step problem is:

$$
\begin{aligned}
\max_{\Pi^{w^m}} & \ s^m(w^m) = \sum_{\mathbf{C}  \times \mathbf{W'}} \{\beta s^*(w') - c\}\Pi^{w^m}(c, w') \\
\textrm{subject to} \\ 
\mbox{C5:} & \ w^m = \sum_{\mathbf{C} \times \mathbf{W'}} \{ 2\sqrt{c}+\beta w'\}\Pi^{w^m}(c, w') \\
\mbox{C7:} & \ \sum_{\mathbf{C} \times \mathbf{W'}} \Pi^{w^m}(c, w')=1, \Pi^{w^m}(c, w') \ge 0 \\
\end{aligned}
$$

We work backwards.

We solve the step 2 problem first  to get  $s^m(w^m)$. 

After that,  we also seek an  optimal probability $\Pi^{w^m}(c, w') = Pr(c, w'|w^m)$. 

Then we solve the step 1 problem for   $s(w)$ and $\Pi^{w}(a, q, w^m) = Pr(a, q, w^m|w)$.

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
{
    "tags": [
        "hide-output"
    ]
}
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

Let's plot figures to illustrate  solutions. 

Before that, we need to convert $\Pi^{w}(a, q, w^m)$ and $\Pi^{w^m}(c, w')$ into $\Pi^{w}(a, q, c, w')$.

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

For figure 9 to 12, Phelan and Townsend used $\beta = 0.95$. 

So, we ran our  function to get the solution to the problem with $\beta = 0.95$.

However, it  took many steps to converge.

With the improved algorithm, it didn't converge after 1000 iterations!

After 1000 iterations, we had a memory leak problem and Python was forced to restart.

The algorithm needs more further improvements to handle this problem.

We just presented the codes and running processing here. 

However, we recommend not to run this cell.

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
