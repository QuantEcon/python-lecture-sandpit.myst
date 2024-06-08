---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
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

We'll also use invariant subspaces  to describe variations on this method that are faster.

+++ {"user_expressions": []}


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

together with a pair of integers that are  initial conditions for   $y_{-1}, y_{-2}$.

First, we'll deploy some techniques for solving difference equations that are also deployed in this quantecon lecture:
<https://python.quantecon.org/samuelson.html>

The characteristic equation associated with difference equation (1) is

$$
c(x) \equiv x^2 - 2 x - (1+\sigma) = 0 \tag{2}
$$

or the factored form

$$
c(x)= (x - \lambda_1) (x-\lambda_2) = 0
$$


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

Let's represent the preceding analysis by vectorizing our second order difference equation and then using  eigendecompositions of a state
transition matrix.

## Vectorizing the difference equation


Represent (1) with the first-order matrix difference equation

$$
\begin{bmatrix} y_{t+1} \cr y_{t} \end{bmatrix}
= \begin{bmatrix} 2 & 1+ \sigma \cr 1 & 0 \end{bmatrix} \begin{bmatrix} y_{t} \cr y_{t-1} \end{bmatrix}
$$
or
$$
x_{t+1} = M x_t 
$$

where 

$$
M = \begin{bmatrix} 2 & 1+ \sigma \cr 1 & 0 \end{bmatrix},  \quad x_t= \begin{bmatrix} y_{t} \cr y_{t-1} \end{bmatrix}
$$

Construct an eigendecomposition of $M$:

$$
M = V \begin{bmatrix} \lambda_1 & 0 \cr 0 & \lambda_2  \end{bmatrix} V^{-1} 
$$

where columns of $V$ are eigenvectors corresponding to  eigenvalues $\lambda_1$ and $\lambda_2$.

The eigenvalues can be ordered so that  $\lambda_1 > 1 > \lambda_2$.



+++ {"user_expressions": []}

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

+++ {"user_expressions": []}

## Requests for Humphrey

Here is what i recommend doing ultimately

* write Python code to compute $\sqrt{\sigma}$ using the purely difference equation methods above, including solving the linear system
(3) that zero out either $\eta_1$ or $\eta_2$.  Use the code to illustrate the "three" ways of computing $\sqrt{\sigma}$. Use $\sqrt{2}$ as lead example.

* then write Python code that does things using the matrix methods. I recommend just copying or adapting the code that you wrote for 
<https://intro.quantecon.org/money_inflation.html>
  
    * print out the eigenvalues and eigenvectors and describe quantitatively how they relate to our earlier having solved system (3) above
    
Perhaps we (i.e.,"you") might want first to convert the notebook into a *myst .md file, using the jupyter book technology, and then working in the sandpit.  What do you think?


```{code-cell} ipython3

```
