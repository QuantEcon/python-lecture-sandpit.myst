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

# Cross product trick

This notebooks describes formulas for eliminating 

  * cross products between states and control in linear quadratic dynamic programming  problems
  
  * covariances between state and measurement noises in  Kalman filtering  problems

+++

## Undiscounted dynamic programming problem

Here is a nonstochastic undiscounted LQ dynamic programming with cross products between
states and controls in the objective function.



The problem is defined by the 5-tuple of matrices $(A, B, R, Q, H)$
where  $R$ and $Q$ are positive definite symmetric matrices and 
$A \sim m \times m, B \sim m \times k,  Q \sim k \times k, R \sim m \times m$ and $H \sim k \times m$.


The problem is to choose $\{x_{t+1}, u_t\}_{t=0}^\infty$ to maximize 
$$
 - \sum_{t=0}^\infty (x_t' R x_t + u_t' Q u_t + 2 u_t H x_t) 
 $$
subject to the linear constraints 

$$ x_{t+1} = A x_t + B u_t,  \quad t \geq 0 $$

where $x_0$ is a given initial condition. 

The solution to this undiscounted infinite-horizon problem is a time-invariant feedback rule  

$$ u_t  = -F x_t $$

where

$$ F = -(Q + B'PB)^{-1} B'PA $$

and  $P \sim m \times m $ is a positive definite solution of the algebraic matrix Riccati equation

$$ P = R + A'PA - (A'PB + H')(Q + B'PB)^{-1}(B'PA + H).
$$

where .

where $A \sim m \times m, B \sim m \times k, P \sim m \times m, Q \sim k \times k, R \sim m \times m$ and $H \sim k \times m$.

+++

It can be verified that an **equivalent** problem without cross-products between states and controls
is  defined by  the following 4-tuple
of matrices : $(A^*, B, R^*, Q) $. 

That the omitted matrix $H=0$ indicates that there are no cross products between states and controls
in the equivalent problem. 

The matrices $(A^*, B, R^*, Q) $ defining the  equivalent problem and the matrices $P, F^*$ solving it are  related to the matrices $(A, B, R, Q, H)$ defining the original problem  and the matrices $P, F$ solving it by 

\begin{align*}
A^* & = A - B Q^{-1} H, \\
R^* & = R - H'Q^{-1} H, \\
P & = R^* + {A^*}' P A - ({A^*}' P B) (Q + B' P B)^{-1} B' P A^*, \\
F^* & = (Q + B' P B)^{-1} B' P A^*, \\
F & = F^* + Q^{-1} H.
\end{align*}

+++

## Kalman filter

Duality between linear-quadratic optimal control and Kalman filtering suggests that there
is an analogous transformation that allows us to transform a Kalman filtering problem
with covariance between between shocks to states and measurements to an equivalent
Kalman filtering problem with zero covariance between shocks to states and measurments.

There is.



First, let's recall the Kalman filter with covariance between noises to states and measurements.

The hidden Markov model is 

\begin{align*}
x_{t+1} & = A x_t + B w_{t+1},  \\
z_{t+1} & = D x_t + F w_{t+1},  
\end{align*}

where $A \sim m \times m, B \sim m \times p $ and $D \sim k \times m, F \sim k \times p $.

Thus, $x_t$ is $m \times 1$ and $z_t$ is $k \times 1$. 

The Kalman  filtering formulas are 


\begin{align*}
K(\Sigma_t) & = (A \Sigma_t D' + BF')(D \Sigma_t D' + FF')^{-1}, \\
\Sigma_{t+1}&  = A \Sigma_t A' + BB' - (A \Sigma_t D' + BF')(D \Sigma_t D' + FF')^{-1} (D \Sigma_t A' + FB').
\end{align*}
 

Define   tranformed matrices

\begin{align*}
A^* & = A - BF' (FF')^{-1} D, \\
B^* {B^*}' & = BB' - BF' (FF')^{-1} FB'.
\end{align*}

#### Algorithm


Compute $\Sigma, K^*$ using the ordinary Kalman filtering  formula with $BF' = 0$, i.e.,
with no covariance between noises to  states and  measurements. 

That is, compute  $K^*$ and $\Sigma$ that  satisfy

\begin{align*}
K^* & = (A^* \Sigma D')(D \Sigma D' + FF')^{-1} \\
\Sigma & = A^* \Sigma {A^*}' + B^* {B^*}' - (A^* \Sigma D')(D \Sigma D' + FF')^{-1} (D \Sigma {A^*}').
\end{align*}

The Kalman gain for the original problem **with covariance** between noises to states and measurements is then

$$
K = K^* + BF' (FF')^{-1},
$$

The state reconstruction covariance matrix $\Sigma$ for the orignal and transformed problems are equal.

+++

## Duality table

Here is a handy table to remember how the Kalman filter and dynamic program are related.


|Dynamic Program|  Kalman Filter |
|:-------------:|:--------:|
|      $A$      |   $A'$   |
|      $B$      |   $D'$   |
|      $H$      |   $FB'$  |
|      $Q$      |   $FF'$  |
|      $R$      |   $BB'$  |
|      $F$      |   $K'$   |
|      $P$      | $\Sigma$ |

+++

## Duality table - cross-product elimination

Here is a table that states the transformations for the dual problems


<table>
    <tr>
	    <th width = 150><center>Dynamic Program</center></th> 
        <th width = 300><center>Kalman Filter</center></th>
	</tr>
    <tr>
        <td><center>$A^* = A - B Q^{-1} H$</center></td> 
        <td><center>${A^*}' = A' - D' (FF')^{-1} F B'$</center></td>
	</tr>
    <tr>
        <td><center>$B$</center></td> 
        <td><center>$D'$</center></td>
	</tr>
    <tr>
        <td><center>$Q$</center></td> 
        <td><center>$FF'$</center></td>
	</tr>
    <tr>
        <td><center>$R^* = R - H' Q^{-1} H$</center></td> 
        <td><center>$BB' - BF'(FF')^{-1}FB' = B^*{B^*}'$</center></td>
	</tr>
    <tr>
        <td><center>$F^*$</center></td> 
        <td><center>${K^*}'$</center></td>
	</tr>
    <tr>
        <td><center>$P$</center></td> 
        <td><center>$\Sigma$</center></td>
	</tr>
    <tr>
        <td><center>$F = F^* + Q^{-1} H$</center></td> 
        <td><center>$K' = {K^*}' + (FF')^{-1} FB'$ <br/> or <br/> $K = K^* + BF' (FF')^{-1}$</center></td>
	</tr>
</table>


```{code-cell} ipython3

```
