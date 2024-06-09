---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"user_expressions": []}

# Computing Square Roots


## Introduction

This notebook provides a simple example of **invariant subspace** methods for analyzing linear difference equations. 

These methods are applied throughout applied economic dynamics, for example, in this quantecon lecture <https://intro.quantecon.org/money_inflation.html>

Our approach in this notebook is to illustrate the method with an ancient example, one that  ancient Greek mathematicians used to compute square roots of positive integers.

An integer is called a **perfect square** if its square root is also an integer.

An orderered sequence of  perfect squares starts with 

$$
4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, \ldots 
$$

If an integer is not a perfect square, then its square root is an irrational number -- i.e., it cannot be expressed as a ratio of two integers, and its decimal expansion is indefinite.

The ancient Greeks invented an algorithm to compute square roots of integers, including integers that are not perfect squares.

Their method involved

 * computing a sequence of integers $\{y_t\}_{t=0}^\infty$
 
 * computing $\lim_{t \rightarrow \infty} \left(\frac{y_{t+1}}{y_t}\right) = \bar r$
 
 * deducing the desired square root from $\bar r$
 
In this lecture, we'll describe this method.

We'll also use invariant subspaces to describe variations on this method that are faster.

In this lecture, we use the following imports:

```{code-cell} ipython3
:tags: []

import numpy as np
```

## Setup

Let $\sigma$ be a positive  integer greater than $1$

So $\sigma \in {\mathcal I} \equiv  \{2, 3, \ldots \}$ 

We want an algorithm to compute the square root of $\sigma \in {\mathcal I}$.

If $\sqrt{\sigma} \in {\mathcal I}$, $\sigma $ is said to be a **perfect square**.

If $\sqrt{\sigma} \not\in {\mathcal I}$, it turns out that it is irrational.

Ancient Greeks used a recursive algorithm to compute square roots of integers that are not perfect squares. 

The algorithm iterates on a  second order  linear  difference equation in the sequence $\{y_t\}_{t=0}^\infty$:

$$
y_{t} = 2 y_{t-1} + (1+ \sigma) y_{t-2}, \quad t \geq 0 \tag{1}
$$ 

**Humphrey: Dear Tom, should this be $y_{t} = 2 y_{t-1} - (1 - \sigma) y_{t-2}, \quad t \geq 0$? Please kindly correct me if I were wrong.**

together with a pair of integers that are  initial conditions for   $y_{-1}, y_{-2}$.

First, we'll deploy some techniques for solving difference equations that are also deployed in this quantecon lecture:
<https://python.quantecon.org/samuelson.html>

The characteristic equation associated with difference equation (1) is

$$
c(x) \equiv x^2 - 2 x - (1+\sigma) = 0 \tag{2}
$$

**Humphrey: Then this would be $c(x) \equiv x^2 - 2 x + (1 - \sigma) = 0$ with roots $1 + \sqrt{\sigma}$ and $1 - \sqrt{\sigma}$? Please find a function that computes roots for the original version and modified version below:**

```{code-cell} ipython3
def solve_lambdas(coefs):    
    # Calculate the roots using numpy.roots
    λs = np.roots(coefs)
    
    # Sort the roots for consistency
    λ_1, λ_2 = sorted(λs, reverse=False)
    
    return λ_1, λ_2

sigma = 2

# Current: c(x) \equiv x^2 - 2 x - (1+\sigma) = 0
print(f"Current: {solve_lambdas([1, -2, -(1 + sigma)])}")

# Suggested: c(x) \equiv x^2 - 2 x + (1 - \sigma) = 0
print(f"Suggested: {solve_lambdas([1, -2, (1 - sigma)])}")
```

or the factored form

$$
c(x)= (x - \lambda_1) (x-\lambda_2) = 0
$$(eq:cha_eq)


where 

$$ 
c(x) = 0 
$$

for $x = \lambda_1$ or $x = \lambda_2$.


By applying the quadratic formula to solve the characteristic equation we find 

$$
\lambda_1 = 1 + \sqrt{\sigma}, \quad \lambda_2 = 1 - \sqrt{\sigma} 
$$

Solutions  of our difference equation take the form

$$
y_t = \lambda_1^t \eta_1 + \lambda_2^t \eta_2
$$

where $\eta_1$ and $\eta_2$ are chosen to satisfy  prescribed initial conditions

$$
\begin{align}
\lambda_1^{-1} \eta_1 + \lambda_2^{-1} \eta_2 & =  y_{-1} \cr
\lambda_1^{-2} \eta_1 + \lambda_2^{-2} \eta_2 & =  y_{-2}
\end{align} \tag{3}
$$

System (3) of simultaneous linear equations will play a big role in the remainder of this lecture.  

Since $\lambda_1 = 1 + \sqrt{\sigma} > 1 > \lambda_2 = 1 - \sqrt{\sigma} $
it follows that for **almost all** initial conditions

$$
\lim_{t \rightarrow \infty} \left(\frac{y_{t+1}}{y_t}\right) = 1 + \sqrt{\sigma}
$$

Thus,

$$
\sqrt{\sigma} = \lim_{t \rightarrow \infty} \left(\frac{y_{t+1}}{y_t}\right) - 1
$$

However, notice that if $\eta_1 = 0$, then

$$
\lim_{t \rightarrow \infty} \left(\frac{y_{t+1}}{y_t}\right) = 1 - \sqrt{\sigma}
$$

so that 

$$
\sqrt{\sigma} = \lim_{t \rightarrow \infty} \left(\frac{y_{t+1}}{y_t}\right) + 1
$$

**Humphrey: Would this be $\sqrt{\sigma} = 1 - \lim_{t \rightarrow \infty} \left(\frac{y_{t+1}}{y_t}\right)$ and similarly the equation below be: $\sqrt{\sigma} = 1 - \left(\frac{y_{t+1}}{y_t}\right)$? Please kindly see the validation in the implementation section.**


Actually, if $\eta_1 =0$, it follows that

$$
\sqrt{\sigma} =\left(\frac{y_{t+1}}{y_t}\right) + 1 \quad \forall t \geq 0,
$$

so that convergence is immediate and there is no need to take limits.

Symmetrically, if $\eta_2 =0$, it follows that 


$$
\sqrt{\sigma} =  \left(\frac{y_{t+1}}{y_t}\right) - 1 \quad \forall t \geq 0
$$

so again, convergence is immediate, and we have no need to compute a limit.


System (3) of simultaneous linear equations can be used in various ways.

 * we can take $y_{-1}, y_{-2}$ as given initial conditions and solve for $\eta_1, \eta_2$
 
 * we can instead take $\eta_1, \eta_2$ as given and solve for initial conditions  $y_{-1}, y_{-2}$ 
 
Notice how we used the  second approach above when we set  $\eta_1, \eta_2$  either to $(0, 1)$, for example, or $(1, 0)$, for example.

In taking this second approach, we were in effect finding  an **invariant subspace** of ${\bf R}^2$. 

## Implementation

We now implement the above algorithm to compute the square root of $\sigma$

```{code-cell} ipython3
:tags: []

def solve_η(λ_1, λ_2, y_neg1, y_neg2):
    
    # Solve the system of linear equation (3)
    A = np.array([
        [1/λ_1, 1/λ_2],
        [1/(λ_1**2), 1/(λ_2**2)]
    ])
    b = np.array([y_neg1, y_neg2])
    ηs = np.linalg.solve(A, b)
    
    return ηs

def solve_sqrt(σ, y_neg1, y_neg2, t_max=100):
    
    # Ensure σ is greater than 1
    if σ <= 1:
        raise ValueError("σ must be greater than 1")
        
    # Characteristic roots
    λ_1 = 1 + np.sqrt(σ)
    λ_2 = 1 - np.sqrt(σ)
    
    # Solve for η_1 and η_2
    η_1, η_2 = solve_η(λ_1, λ_2, y_neg1, y_neg2)

    # Compute the sequence up to t_max
    t = np.arange(t_max + 1)
    y = (λ_1 ** t) * η_1 + (λ_2 ** t) * η_2
    
    # Compute the ratio y_{t+1} / y_t for large t
    sqrt_σ_estimate = (y[-1] / y[-2]) - 1
    
    return sqrt_σ_estimate

σ = 2
sqrt_σ = solve_sqrt(σ, y_neg1=2, y_neg2=1)
dev = abs(sqrt_σ-np.sqrt(σ))

print(f"sqrt({σ}) is approximately {sqrt_σ:.5f} (error: {dev:.5f})")
```

```{code-cell} ipython3
:tags: []

λ_1 = 1 + np.sqrt(σ)
λ_2 = 1 - np.sqrt(σ)

# Compute the sequence up to t_max
t = 1
y = lambda t, ηs: (λ_1 ** t) * ηs[0] + (λ_2 ** t) * ηs[1]

ηs = (0, 1)
sqrt_σ = 1 - y(2, ηs) / y(1, ηs) 
print(f"For η_1, η_2 = (0, 1), sqrt_σ = {sqrt_σ:.5f}")
```

```{code-cell} ipython3
:tags: []

ηs = (1, 0)
sqrt_σ = y(2, ηs) / y(1, ηs) - 1

print(f"For η_1, η_2 = (1, 0), sqrt_σ = {sqrt_σ:.5f}")
```

Let's represent the preceding analysis by vectorizing our second order difference equation and then using  eigendecompositions of a state transition matrix.

## Vectorizing the difference equation


Represent (1) with the first-order matrix difference equation

$$
\begin{bmatrix} y_{t+1} \cr y_{t} \end{bmatrix}
= \begin{bmatrix} 2 & -( 1+ \sigma) \cr 1 & 0 \end{bmatrix} \begin{bmatrix} y_{t} \cr y_{t-1} \end{bmatrix}
$$

or

$$
x_{t+1} = M x_t 
$$

where 

$$
M = \begin{bmatrix} 2 & - (1+ \sigma )  \cr 1 & 0 \end{bmatrix},  \quad x_t= \begin{bmatrix} y_{t} \cr y_{t-1} \end{bmatrix}
$$

Construct an eigendecomposition of $M$:

$$
M = V \begin{bmatrix} \lambda_1 & 0 \cr 0 & \lambda_2  \end{bmatrix} V^{-1} 
$$

where columns of $V$ are eigenvectors corresponding to  eigenvalues $\lambda_1$ and $\lambda_2$.

The eigenvalues can be ordered so that  $\lambda_1 > 1 > \lambda_2$.

Write equation (1) as

$$
x_{t+1} = V \Lambda V^{-1} x_t
$$

Define

$$
x_t^* = V^{-1} x_t
$$

We can recover $x_t$ from $x_t^*$:

$$
x_t = V x_t^*
$$


The following notations and equations will help us.

Let 

$$

V = \begin{bmatrix} V_{1,1} & V_{1,2} \cr 
                         V_{2,2} & V_{2,2} \end{bmatrix}, \quad
V^{-1} = \begin{bmatrix} V^{1,1} & V^{1,2} \cr 
                         V^{2,2} & V^{2,2} \end{bmatrix}
$$

Notice that it follows from

$$
 \begin{bmatrix} V^{1,1} & V^{1,2} \cr 
                         V^{2,2} & V^{2,2} \end{bmatrix} \begin{bmatrix} V_{1,1} & V_{1,2} \cr 
                         V_{2,2} & V_{2,2} \end{bmatrix} = \begin{bmatrix} 1  & 0 \cr 0 & 1 \end{bmatrix}
$$

that

 

$$
V^{2,1} V_{1,1} + V^{2,2} V_{2,1} = 0
$$

and

$$
V^{1,1}V_{1,2} + V^{1,2} V_{2,2} = 0
$$

These equations will be very useful soon.


Notice that

$$
\begin{bmatrix} x_{1,t+1}^* \cr x_{2,t+1}^* \end{bmatrix} = \begin{bmatrix} \lambda_1  & 0 \cr 0 & \lambda_2 \end{bmatrix}
\begin{bmatrix} x_{1,t}^* \cr x_{2,t}^* \end{bmatrix}
$$

To deactivate $\lambda_1$ we want to set

$$
x_{1,0}^* = 0
$$


This can be achieve by setting 

$$
x_{2,0} =  -( V^{1,2})^{-1} V^{1,1} = V_{2,1} V_{1,1}^{-1} x_{1,0}.
$$

To deactivate $\lambda_2$, we want to  set

$$
x_{2,0}^* = 0
$$

This can be achieved by setting 

$$
x_{2,0} = -(V^{2,2})^{-1} V^{2,1} = V_{2,1} V_{1,1}^{-1} x_{1,0}
$$

### Implementation

```{code-cell} ipython3
:tags: []

def iterate_H(y_0, M, num_steps):
    
    # Eigendecomposition of the matrix M
    Λ, V = np.linalg.eig(M)
    V_inv = np.linalg.inv(V)
    
    print(f"eigenvalue:\n{Λ}")
    print(f"eigenvector:\n{V}")
    
    # Initialize the array to store results
    x = np.zeros((y_0.shape[0], num_steps))
    
    # Perform the iterations
    for t in range(num_steps):
        x[:, t] = V @ np.diag(Λ**t) @ V_inv @ y_0
    
    return x

# Define the state transition matrix M
M = np.array([[2, -(1 - σ)],
              [1, 0]])

# Initial condition vector x_0
x_0 = np.array([1, 0])

# Perform the iteration
result = iterate_H(x_0, M, num_steps=100)
```

Compare the eigenvector to the roots we obtained above

```{code-cell} ipython3
:tags: []

roots = solve_lambdas([1, -2, (1 - sigma)])[::-1]
print(f"roots: {np.round(roots, 8)}")
```

## Thank you Humphrey!

Here is what i recommend doing ultimately

* write Python code to compute $\sqrt{\sigma}$ using the purely difference equation methods above, including solving the linear system
(3) that zero out either $\eta_1$ or $\eta_2$.  Use the code to illustrate the "three" ways of computing $\sqrt{\sigma}$. Use $\sqrt{2}$ as lead example.
DONE!

* then write Python code that does things using the matrix methods. I recommend just copying or adapting the code that you wrote for 
<https://intro.quantecon.org/money_inflation.html>
  
    * print out the eigenvalues and eigenvectors and describe quantitatively how they relate to our earlier having solved system (3) above
DONE