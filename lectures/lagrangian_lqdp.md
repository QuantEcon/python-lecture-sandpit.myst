---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell} ipython3
import numpy as np
from quantecon import LQ
from scipy.linalg import schur
```

## Lagrangian formulation of LQ control problem

+++

This section describes a Lagrangian formulation of the optimal linear regulator.

Besides being useful computationally, this formulation carries insights about connections between
stability and optimality and also opens the way to constructing solutions of dynamic systems
not coming directly from an  intertemporal optimization problem.

The formulation is also the basis for constructing fast algorithms for solving Riccati equations.

+++

For the undiscounted optimal linear regulator problem, form the Lagrangian

\begin{equation*}
\cal L = - \sum^\infty_{t=0} \biggl\{ x^\prime_t R x_t + u_t^\prime Q u_t +
                                 2 \mu^\prime_{t+1} [A x_t + B u_t - x_{t+1}]\biggr\}.
                                 \label{eq1} \tag{1} 
\end{equation*}

First-order conditions for maximization with respect to $\{u_t,x_{t+1}\}_{t=0}^\infty$ are

$$
\eqalign{2 Q u_t &+ 2B^\prime \mu_{t+1} = 0 \ ,\ t \geq 0 \cr \mu_t &= R x_t + A^\prime \mu_{t+1}\ ,\ t\geq 1.\cr} \label{eq2} \tag{2}
$$

Define $\mu_0$ to be  a vector of shadow prices of $x_0$ and apply an envelope condition to
\eqref{eq1} (6.5.1) to deduce that

$$
\mu_0 = R x_0 + A' \mu_1,
$$

which is a time $t=0 $ counterpart to the second equation of system \eqref{eq2} (6.5.2).

Recall that  $\mu_{t+1} = P x_{t+1}$, where
$P$ is the matrix that solves the algebraic Riccati equation. 

Thus, $\mu_{t}$ is
the gradient of the value function.

The Lagrange multiplier vector $\mu_{t}$ is often called the *costate* vector
corresponding to the state vector $x_t$.

Solve the first equation of \eqref{eq2} (6.5.2) for $u_t$ in terms of $\mu_{t+1}$.

Substitute
into the law of motion $x_{t+1} = A x_t + B u_t$.

Then arrange the resulting
equation and the second equation of \eqref{eq2} (6.5.2) into the form

$$
L\ \pmatrix{x_{t+1}\cr \mu_{t+1}\cr}\ = \ N\ \pmatrix{x_t\cr \mu_t\cr}\
,\ t \geq 0,
$$

where

$$
L = \ \pmatrix{I & BQ^{-1} B^\prime \cr 0 & A^\prime\cr}, \quad N = \
\pmatrix{A & 0\cr -R & I\cr}.
$$

When $L$ is of full rank (i.e., when $A$ is of full rank), we can write
this system as

$$
\pmatrix{x_{t+1}\cr \mu_{t+1}\cr}\ = M\ \pmatrix{x_t\cr\mu_t\cr}
$$

where

$$
M\equiv L^{-1} N = \pmatrix{A+B Q^{-1} B^\prime A^{\prime-1}R &
-B Q^{-1} B^\prime A^{\prime-1}\cr -A^{\prime -1} R & A^{\prime -1}\cr}.
$$

+++

We seek to solve the difference equation system (6.5.4) for a sequence $\{x_t\}_{t=0}^\infty$
that satisfies the initial condition for $x_0$ and a terminal condition
$\lim_{t \rightarrow +\infty} x_t =0$ that expresses our wish for a *stable* solution.

We inherit our wish for stability of the $\{x_t\}$ sequence from a desire to maximize
$-\sum_{t=0}^\infty \bigl[ x_t ' R x_t + u_t' Q u_t \bigr]$, which requires that $x_t' R x_t$ converge to
zero.

+++

To proceed, we study properties of the $(2n \times 2n)$ matrix $M$. It is helpful to introduce
a $(2n \times 2n)$ matrix

$$
J= \pmatrix{0 & -I_n\cr I_n & 0\cr}.
$$

The rank of $J$ is $2n$.

*Definition:*  A matrix $M$ is called *symplectic* if

$$
MJM^\prime = J.
\label{eq3} \tag{3}
$$

It can be verified directly that $M$ in equation is symplectic.

It follows from equation \eqref{eq3} (6.5.6) and from the fact $J^{-1} = J^\prime = -J$ that for any symplectic
matrix $M$,

$$
M^\prime = J^{-1} M^{-1} J.
\label{eq4} \tag{4}
$$

Equation \eqref{eq4} (6.5.7) states that $M^\prime$ is related to the inverse of $M$
by a similarity transformation.

For square matrices, recall that  
  
  * similar matrices share eigenvalues
  
  *  eigenvalues of the inverse of a matrix are  inverses of  eigenvalues of the matrix
  
  * a matrix and its transpose have the same eigenvalues

It then follows from equation \eqref{eq4}  (6.5.7) that
the eigenvalues of $M$ occur in reciprocal pairs: if $\lambda$ is an
eigenvalue of $M$, so is $\lambda^{-1}$.

Write equation (6.5.4) as 

$$
y_{t+1} = M y_t
$$

where $y_t = \pmatrix{x_t\cr \mu_t\cr}$. 

Consider the following triangularization of $M$

$$
V^{-1} M V= \pmatrix{W_{11} & W_{12} \cr 0 & W_{22}\cr}
$$

where each block on the right side is $(n\times n)$, where $V$ is
nonsingular, and where $W_{22}$ has all its eigenvalues exceeding $1$ in modulus
and $W_{11}$ has all of its eigenvalues less than $1$ in modulus. 

The *Schur decomposition* and the *eigenvalue decomposition*
are two such decompositions. Write equation (6.5.8) as

$$
y_{t+1} = V W V^{-1} y_t.
$$

The solution of equation (6.5.9) for arbitrary initial condition $y_0$ is
evidently

$$
y_{t} = V \left[\matrix{W^t_{11} & W_{12,t}\cr 0 & W^t_{22}\cr}\right]
\ V^{-1} y_0
$$

where $W_{12,t} = W_{12}$ for $t=1$ and  for $t \geq 2$ obeys the recursion

$$
W_{12, t} = W^{t-1}_{11} W_{12,t-1} + W_{12,t-1} W^{t-1}_{22}
$$

and where $W^t_{ii}$ is $W_{ii}$ raised to the $t$th  power.

Write equation (6.5.10) as

$$
\pmatrix{y^\ast_{1t}\cr y^\ast_{2t}\cr}\ =\ \left[\matrix{W^t_{11} &
W_{12, t}\cr 0 & W^t_{22}\cr}\right]\quad \pmatrix{y^\ast_{10}\cr
y^\ast_{20}\cr}
$$

where $y^\ast_t = V^{-1} y_t$, and in particular where

$$
y^\ast_{2t} = V^{21} x_t + V^{22} \mu_t,
$$

and where $V^{ij}$ denotes the $(i,j)$ piece of
the partitioned $V^{-1}$ matrix.

Because $W_{22}$ is an unstable matrix, unless $y^\ast_{20} = 0$,
$y^\ast_t$ will diverge.

Let $V^{ij}$ denote the $(i,j)$ piece of the partitioned $V^{-1}$ matrix.

To attain stability, we must impose $y^\ast_{20} =0$, which from equation (6.5.11) implies

$$
V^{21} x_0 + V^{22} \mu_0 = 0
$$

or

$$
\mu_0 = - (V^{22})^{-1} V^{21} x_0.
$$

This equation replicates itself over
time in the sense that it implies

$$
\mu_t = - (V^{22})^{-1} V^{21} x_t.
$$

But notice that because $(V^{21}\ V^{22})$ is the second row block of
the inverse of $V,$

$$
(V^{21} \ V^{22})\quad \pmatrix{V_{11}\cr V_{21}\cr} = 0
$$

which implies

$$
V^{21} V_{11} + V^{22} V_{21} = 0.
$$

Therefore,

$$
-(V^{22})^{-1} V^{21} = V_{21} V^{-1}_{11}.
$$

So we can write

$$
\mu_0 = V_{21} V_{11}^{-1} x_0
$$

and

$$
\mu_t = V_{21} V^{-1}_{11} x_t.
$$

However, we know from equations (6.4.1) that $\mu_t = P x_t$,
where $P$ occurs in the matrix that solves the Riccati equation.


Thus, the preceding argument establishes that

$$
P = V_{21} V_{11}^{-1}.
$$

This formula provides us with an alternative, and typically computationally very
efficient, way of computing the matrix $P$.

This same method can be applied to compute the solution of
any system of the form (6.5.4) if a solution exists, even
if the eigenvalues of $M$ fail to occur in reciprocal pairs.

The method
will typically work so long as the eigenvalues of $M$ split   half
inside and half outside the unit circle.

Systems in which  eigenvalues (properly adjusted for discounting) fail
to occur in reciprocal pairs arise when the system being solved
is an equilibrium of a model in which there are distortions that
prevent there being any optimum problem that the equilibrium
solves. 

See Woodford (1999)  for an application of
such methods to solve for linear approximations
of equilibria of a monetary model with distortions.  

### Application

Here we demonstrate the computation with an example which is the deterministic version of the case borrowed from this [quantecon lecture](https://python.quantecon.org/lqcontrol.html).

```{code-cell} ipython3
# Model parameters
r = 0.05
c_bar = 2
μ = 1

# Formulate as an LQ problem
Q = np.array([[1]])
R = np.zeros((2, 2))
A = [[1 + r, -c_bar + μ],
     [0,              1]]
B = [[-1],
     [0]]

# Construct an LQ instance
lq = LQ(Q, R, A, B)
```

Given matrices $A$, $B$, $Q$, $R$, we can then compute $L$, $N$, and $M=L^{-1}N$.

```{code-cell} ipython3
def construct_LNM(A, B, Q, R):

    n, k = lq.n, lq.k

    # construct L and N
    L = np.zeros((2*n, 2*n))
    L[:n, :n] = np.eye(n)
    L[:n, n:] = B @ np.linalg.inv(Q) @ B.T
    L[n:, n:] = A.T

    N = np.zeros((2*n, 2*n))
    N[:n, :n] = A
    N[n:, :n] = -R
    N[n:, n:] = np.eye(n)

    # compute M
    M = np.linalg.inv(L) @ N

    return L, N, M
```

```{code-cell} ipython3
L, N, M = construct_LNM(lq.A, lq.B, lq.Q, lq.R)
```

```{code-cell} ipython3
M
```

Let's verify that $M$ is symplectic.

```{code-cell} ipython3
n = lq.n
J = np.zeros((2*n, 2*n))
J[n:, :n] = np.eye(n)
J[:n, n:] = -np.eye(n)

M @ J @ M.T - J
```

We can compute the eigenvalues of $M$ using `np.linalg.eigvals`, arranged in ascending order.

```{code-cell} ipython3
eigvals = sorted(np.linalg.eigvals(M))
eigvals
```

When we apply Schur decomposition such that $M=V W V^{-1}$, we want the upper left block of $W$, $W_{11}$, has all of its
eigenvalues less than 1 in modulus, and the lower right block $W_{22}$ has all its eigenvalues exceeding 1 in modulus. To do so, let's define a sorting function that tells `scipy.schur` to sort the corresponding eigenvalues with modulus smaller than 1 to the upper left.

```{code-cell} ipython3
stable_eigvals = eigvals[:n]

def sort_fun(x):
    "Sort the eigenvalues with modules smaller than 1 to the top-left."

    if x in stable_eigvals:
        stable_eigvals.pop(stable_eigvals.index(x))
        return True
    else:
        return False

W, V, _ = schur(M, sort=sort_fun)
```

```{code-cell} ipython3
W
```

```{code-cell} ipython3
V
```

We can check the modulus of eigenvalues of $W_{11}$ and $W_{22}$. Since they are both triangular matrices, the eigenvalues are the diagonals. 

```{code-cell} ipython3
# W11
np.diag(W[:n, :n])
```

```{code-cell} ipython3
# W22
np.diag(W[n:, n:])
```

The following functions wrap the procedure of $M$ matrix construction, Schur decomposition, and computation of $P$ by imposing stability on the solution.

```{code-cell} ipython3
def stable_solution(M, verbose=True):
    """
    Given a system of linear difference equations

        y' = |a b| y
        x' = |c d| x

    which is potentially unstable, find the solution
    by imposing stability.

    Parameter
    ---------
    M : np.ndarray(float)
        The matrix represents the linear difference equations system.
    """
    n = M.shape[0] // 2
    stable_eigvals = list(sorted(np.linalg.eigvals(M))[:n])

    def sort_fun(x):
        "Sort the eigenvalues with modules smaller than 1 to the top-left."

        if x in stable_eigvals:
            stable_eigvals.pop(stable_eigvals.index(x))
            return True
        else:
            return False

    W, V, _ = schur(M, sort=sort_fun)
    if verbose:
        print('eigenvalues:\n')
        print('    W11: {}'.format(np.diag(W[:n, :n])))
        print('    W22: {}'.format(np.diag(W[n:, n:])))

    # compute V21 V11^{-1}
    P = V[n:, :n] @ np.linalg.inv(V[:n, :n])

    return W, V, P

def stationary_P(lq, verbose=True):
    """
    Computes the matrix :math:`P` that represent the value function

         V(x) = x' P x

    in the infinite horizon case. Computation is via imposing stability
    on the solution path and using Schur decomposition.

    Parameters
    ----------
    lq : qe.LQ
        QuantEcon class for analyzing linear quadratic optimal control
        problems of infinite horizon form.

    Returns
    -------
    P : array_like(float)
        P matrix in the value function representation.
    """

    Q = lq.Q
    R = lq.R
    A = lq.A * lq.beta ** (1/2)
    B = lq.B * lq.beta ** (1/2)

    n, k = lq.n, lq.k

    L, N, M = construct_LNM(A, B, Q, R)
    W, V, P = stable_solution(M, verbose=verbose)

    return P
```

```{code-cell} ipython3
# compute P
stationary_P(lq)
```

Note the matrix $P$ computed by this way is close to what we get from the regular routine in quantecon that solves Riccati equation by iteration. The small difference comes from computational error and shall disappear as we increase the maximum number of iterations or decrease the tolerance for convergence.

```{code-cell} ipython3
lq.stationary_values()
```

The method of using Schur decomposition is much more efficient.

```{code-cell} ipython3
%%timeit
stationary_P(lq, verbose=False)
```

```{code-cell} ipython3
%%timeit
lq.stationary_values()
```

This way of finding the solution to a potentially unstable linear difference equations system is not necessarily restricted to the LQ problems. For example, this method is adopted in the [Stability in Linear Rational Expectations Models](https://python.quantecon.org/re_with_feedback.html#Another-perspective) lecture, and let's try to solve for the solution again using the `stable_solution` function defined above.

```{code-cell} ipython3
def construct_H(ρ, λ, δ):
    "contruct matrix H given parameters."

    H = np.empty((2, 2))
    H[0, :] = ρ,δ
    H[1, :] = - (1 - λ) / λ, 1 / λ

    return H

H = construct_H(ρ=.9, λ=.5, δ=0)
```

```{code-cell} ipython3
W, V, P = stable_solution(H)
P
```

## CH 6.6 More about the Lagrangian formulation

+++

We can use the  subsection 6.3.2 transformations to solve a discounted optimal
regulator problem using the Lagrangian and invariant subspace methods introduced
in section 6.5.

+++

For example, when $\beta=\frac{1}{1+r}$, we can solve for $P$ with $\hat{A}=\beta^{1/2} A$ and $\hat{B}=\beta^{1/2} B$. This is adopted by default in the function `stationary_P` defined above.

```{code-cell} ipython3
β = 1 / (1 + r)
lq.beta = β
```

```{code-cell} ipython3
stationary_P(lq)
```

We can verify that the solution is the same with the regular routine `LQ.stationary_values` in quantecon package.

```{code-cell} ipython3
lq.stationary_values()
```

For several purposes, it is useful  explicitly briefly to describe
a Lagrangian for a discounted problem. 

Thus, for the discounted optimal linear regulator problem,
form the Lagrangian

$$
\cal{L} = - \sum^\infty_{t=0} \beta^t \biggl\{ x^\prime_t R x_t + u_t^\prime Q u_t
+ 2 \beta \mu^\prime_{t+1} [A x_t + B u_t - x_{t+1}]\biggr\}
$$

where $2 \mu_{t+1}$ is a vector of Lagrange multipliers on the state vector $x_{t+1}$.

First-order conditions for maximization with respect
to $\{u_t,x_{t+1}\}_{t=0}^\infty$ are

$$
\eqalign{2 Q u_t &+ 2  \beta B^\prime \mu_{t+1} = 0 \ ,\ t \geq 0 \cr \mu_t &= R x_t + \beta A^\prime \mu_{t+1}\ ,\ t\geq 1.\cr}
$$

Define $2 \mu_0$ to be the vector of shadow prices of $x_0$ and apply an envelope condition to
(6.6.1) to  deduce that

$$
\mu_0 = R x_0 + \beta A' \mu_1 ,
$$

which is a time $t=0 $ counterpart to the second equation of system 6.6.2.

Proceeding as we did above with  the undiscounted system 6.5.3, we can rearrange the first-order conditions into the
system

$$
\left[\matrix{ I & \beta B Q^{-1} B' \cr
             0 & \beta A' }\right] 
\left[\matrix{ x_{t+1} \cr \mu_{t+1} }\right] =
\left[\matrix{ A & 0 \cr
             - R & I }\right] 
\left[\matrix{ x_t \cr \mu_t }\right]
$$

which in the special case that $\beta = 1$ agrees with equation (6.5.3), as expected.

+++

By staring at system 6.6.3, we can infer  identities that shed light on the structure of optimal linear regulator problems,
some of which will be useful THIS QUANTECON LECTURE when we apply and  extend the methods of this chapter to study Stackelberg and Ramsey problems.

First, note that the first block of equation 6.6.3 asserts that if when  $\mu_{t+1} = P x_{t+1}$, then    $(I + \beta Q^{-1} B' P B P ) x_{t+1} = A x_t$, which  can be rearranged to
be

$$
x_{t+1} = (I + \beta B Q^{-1} B' P)^{-1}  A x_t ,
$$

an expression for the optimal closed loop dynamics of the state that must agree with the alternative expression that we derived with dynamic programming,
namely,

$$
x_{t+1} = (A - BF) x_t .
$$

But using  $F=\beta (Q+\beta
B'PB)^{-1} B'PA$,  it follows that $A - B F = (I - \beta B (Q+ \beta B' P B)^{-1} B' P) A $. 

Thus, our two expressions for the
closed loop dynamics will agree if and only if

$$ 
(I + \beta B Q^{-1} B' P )^{-1} =    (I - \beta B (Q+\beta  B' P B)^{-1} B' P) ,
$$

a matrix  equation that can be verified by applying a partitioned inverse formula.

Next, note that an optimal $P$ obeys the following version of an algebraic matrix Ricatti equation:

$$
P = (R + F' Q F) + \beta (A - B F)' P (A - BF) .
$$

In addition, the second equation of system 6.6.3 implies the "forward looking" equation for the Lagrange multiplier $\mu_t = R x_t + \beta A' \mu_{t+1}$ whose
solution is $\mu_t = P x_t$, where

$$
P = R + \beta A' P (A - BF) . 
$$

A comparison of equations 6.6.6 and 6.6.7 is useful for bringing out features of the optimal value function for a discounted optimal linear regulator problem.

```{code-cell} ipython3

```
