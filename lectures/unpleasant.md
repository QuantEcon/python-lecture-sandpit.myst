---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region user_expressions=[] -->
# Unpleasant Monetarist Arithmetic 

## Overview


This lecture starts where our lecture on Money Supplies and Price Levels ended.

That lecture describes stationary equilibria that reveal a **Laffer curve** in the inflation tax rate and the associated  stationary rate of return 
on currency.  

We describe a setting in which from date $T > 0$, such  a stationary equilibrium prevails.  

But for $t=0, \ldots, T-1$, the money supply, the price level, and interest-bearing government debt vary along a transition path that ends at $t=T$.

We use this setting to describe Sargent and Wallace's **unpleasant monetarist arithmetic**.

**Reader's Guide:** Please read our lecture on Money Supplies and Price levels before diving into this lecture.

That lecture  described  supplies and demands for money that appear in lecture, and it  characterized the steady state equilibrium from which we work backwards in this lecture. 
<!-- #endregion -->

<!-- #region user_expressions=[] -->

## Setup

Let's start with quick reminders of the model's components.

For $t \geq 1$, **real balances** evolve according to


$$
\frac{m_{t+1}}{p_t} - \frac{m_{t}}{p_{t-1}} \frac{p_{t-1}}{p_t} = g
$$

or

$$
b_t - b_{t-1} R_{t-1} = g
$$ (eq:bmotion)

where

  * $b_t = \frac{m_{t+1}}{p_t}$ is real balances at the end of period $t$
  * $R_{t-1} = \frac{p_{t-1}}{p_t}$ is the gross rate of return on real balances held from $t-1$ to $t$

The demand for real balances is 

$$
b_t = \gamma_1 - \gamma_2 R_t^{-1} . 
$$ (eq:bdemand)



## Monetary-Fiscal Policy

To the basic model of lecture XXX, as a way for the government to finance government expenditures, we add inflation-indexed one-period government bonds. 

Let $\widetilde R > 1$ be the gross real rate of return on government one-period inflation-indexed bonds.

The government's budget constraint at time $t \geq 0$ is

$$
B_t + \frac{m_{t+1}}{p_t} = \widetilde B_{t-1} + \frac{m_t}{p_t} + g
$$ 


Just before the beginning of time $0$, the  public owns  $\check m_0$ units of currency
and $\widetilde R \check R_{-1}$ units of one-period indexed bonds (TOM ADD UNITS)


At time $0$, government can rearrange its portolio of debts with subject to the following constraint (on open-market operations):

$$
\widetilde R B_{-1} + \frac{m_0}{p_0} = \widetilde R \check B_{-1} + \frac{\check m_0}{p_0}
$$

or

$$
B_{-1} - \check B_{-1} = \frac{\widetilde R}{p_0} \left( \check m_0 - m_0 \right)  
$$ (eq:openmarketconstraint)

This equation says that the government (e.g., the central bank) can **decrease** $m_0$ relative to 
$\check m_0$ by **increasing** $B_{-1}$ relative to $\check B_{-1}$. 

Just before time $0$, the government chooses $(m_0, B_{-1})$  subject to constraint
{eq}`eq:openmarketconstraint`.

For $t =0, 1, \ldots, T-1$,

$$
\begin{align}
B_t & = \widetilde R B_{t-1} + g \cr
m_{t+1} &  = m_0 
\end{align}
$$

while for $t \geq T$,

$$
\begin{align}
B_t & = B_{T-1} \cr
m_{t+1} & = m_t + p_t \overline g
\end{align}
$$

where 

$$
\overline g = \left[(\tilde R -1) B_{T-1} +  g \right]
$$ (eq:overlineg)


## Algorithm (basic idea)


We work backwards from $t=T$ and first compute $p_T, R_u$ associated with the  low-inflation, low-inflation-tax-rate   stationary equilibrium of our lecture on the dynamic Laffer curve for the inflation tax.

In particular, we compute 

$$
\begin{align}
R_T & = R_u \cr
b_T & = \gamma_1 - \gamma_2 R_T \cr
p_T & = \frac{m_0}{\gamma_1 - \overline g - \gamma_2 R_T^{-1}}
\end{align}
$$ (eq:LafferTstationary)


We compute continuation sequences $\{R_t, b_t\}_{t=T+1}^\infty$ of rates of return and real balances that are associated with an equilibrium by solving equation {eq}`eq:bmotion` and {eq}`eq:bdemand` sequentially  for $t \geq 1$:  
   \begin{align}
b_t & = b_{t-1} R_{t-1} + \overline g \cr
R_t^{-1} & = \frac{\gamma_1}{\gamma_2} - \gamma_2^{-1} b_t \cr
p_t & = R_t p_{t-1} \cr
   m_t & = b_{t-1} p_t 
\end{align}

   

## Earlier dates

Define 
$$
\lambda \equiv \frac{\gamma_2}{\gamma_1}
$$

We want to compute

$$ 
\begin{align}
p_0 &  = \gamma_1^{-1} \left[ \sum_{j=0}^\infty \lambda^j m_{1+j} \right] \cr
& = \gamma_1^{-1} \left[ \sum_{j=0}^{T-1} \lambda^j m_{0} + \sum_{j=T}^\infty \lambda^j m_{1+j} \right]
\end{align}
$$

Thus,

$$
\begin{align}
p_0 & = \gamma_1^{-1} m_0  \left\{ \frac{1 - \lambda^T}{1-\lambda} +  \frac{\lambda^T}{1 - R_u^{-1}\lambda}   \right\} \cr
p_1 & = \gamma_1^{-1} m_0  \left\{ \frac{1 - \lambda^{T-1}}{1-\lambda} +  \frac{\lambda^{T-1}}{1 - R_u^{-1}\lambda}   \right\} \cr
\quad \vdots  & \quad \quad \vdots \cr
p_{T-1} & = \gamma_1^{-1} m_0  \left\{ \frac{1 - \lambda}{1-\lambda} +  \frac{\lambda}{1 - R_u^{-1}\lambda}   \right\}  \cr
p_T & = \gamma_1^{-1} m_0  \left\{\frac{1}{1 - R_u^{-1}\lambda}   \right\}
\end{align}
$$ (eq:allts)

We can code  the preceding formulas by iterating on

$$
p_t = \gamma_1^{-1} m_0 + \lambda p_{t+1}, \quad t = T-1, T-2, \ldots, 0
$$

starting from  

$$
p_T =   \frac{m_0}{\gamma_1 - \overline g - \gamma_2 R_u^{-1}}  = \gamma_1^{-1} m_0  \left\{\frac{1}{1 - R_u^{-1}\lambda} \right\}
$$ (eq:pTformula)

**Remark:**
We can verify the equivalence of the two formulas on the right sides of {eq}`eq:pTformula` by recalling that 
$R_u$ is a root of the quadratic equation that determines steady state rates of return on currency,
namely,

$$
-\gamma_2 + (\gamma_1 + \gamma_2 + g) \bar R - \gamma_1 \bar R^2 = 0
$$ (eq:steadyquadratic)
 
 
 
## Algorithm (pseudo code)

To compute an equilibrium, we deploy the following algorithm.

Given **parameters** include $g, \check m_0, \check B_{-1}, \widetilde R >1, T $

We define a mappying from $p_0$ to $p_0$ as follows.

* Set $m_0$ and then compute $B_{-1}$ to satisfy the constraint on time $0$ **open market operations**

$$
B_{-1}- \check B_{-1}  = \frac{\widetilde R}{p_0} \left( \check m_0 - m_0) \right)
$$

* Compute $B_{T-1}$ from

$$
B_{T-1} = \widetilde R^T B_{-1} + \left( \frac{1 - \widetilde R^T}{1-\widetilde R} \right) g
$$

* Compute 

$$
\overline g = g + \left[ \tilde R - 1\right] B_{T-1}
$$

* Compute $R_u, p_T$ from formulas {eq}`eq:LafferTstationary` above

* Compute a new estimate of $p_0$, call it $\widehat p_0$,  from equation {eq}`eq:allts` above


* Note that the preceding steps define a mapping

$$
\widehat p_0 = {\mathcal S}(p_0)
$$

* We seek a fixed point of ${\mathcal S}$, i.e., a solution of $p_0 = {\mathcal S}(p_0)$.

* Compute a fixed point by iterating to convergence on the relaxation algorithm

$$
p_{0,j+1} = (1-\theta)  {\mathcal S}(p_{0,j})  + \theta  p_{0,j}, 
$$

where $\theta \in [0,1)$ is a relaxation parameter.











 

<!-- #endregion -->


