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

# Money Supplies and Price Levels

+++ {"user_expressions": []}

## Overview

This lecture describes a theory of price level variations that consists of two components

 * a demand function for money 
 * a law of motion for the supply of money
 
The demand function describes the public's demand for "real balances", defined as the ratio of nominal money balances to the price level

The law of motion for the supply of money assumes that the government prints money to finance government expenditures

Our model equates the demand for money to the supply at each time $t \geq 0$.

Equality between those demands and supply gives in a **dynamic** model in which   money supply
and  price level **sequences** are simultaneously deteR_mined by a special  set of simultaneous linear  
equations.

These equations take the form of what are often called vector linear **difference equations**.  

In this lecture, we'll roll up our sleeves and solve those equations in a couple of different ways.

As we'll see, Python is good at solving them.

Let's start with some imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams["figure.figsize"] = (11, 5)
from collections import namedtuple
```

## Demands  for and Supplies of Money

We say demand**s** and supp**ies** (plurals) because there is one of each for each $t \geq 0$.

Let 

  * $m_{t+1}$ be the supply of currency at the end of time $t \geq 0$
  * $m_{t}$ be the supply  of currency brought into time $t$ from time $t-1$
  * $g$ be the government deficit that is financed by printing currency at $t \geq 1$
  * $m_{t+1}^d$ be the demand at time $t$ for currency  to bring into time $t+1$
  * $p_t$ be  the price level at time $t$
  * $b_t = \frac{m_{t+1}}{p_t}$ is real balances at the end of time $t$ 
  * $R_t = \frac{p_t}{p_{t+1}} $ be the gross rate of return on currency held from time $t$ to time $t+1$
  
It is often helpful  to state units in which quantitities are measured:

  * $m_t$ and $m_t^d$ are measured in dollars
  * $g$ is measured in time $t$ goods 
  * $p_t$ is measured in dollars per time $t$ goods
  * $R_t$ is measured in time $t+1$ goods per unit of time $t$ goods
  * $b_t$ is measured in time $t$ goods
   
  
Our job now is to specify demand and supply functions for money. 

We assume that the demand for  currency satisfies the Cagan-like demand function

$$
m_{t+1}^d/p_t =\gamma_1 - \gamma_2 \frac{p_{t+1}}{p_t}, \quad t \geq 0
$$ (eq:demandmoney)
  
  
Now we turn to the supply of money.

We assume that $m_0 >0$ is an "initial condition" deteR_mined outside the model. 

We set $m_0$ at some arbitrary positive value, say \$100.
  
For $ t \geq 1$, we assume that the supply of money is deteR_mined by the government's budget constraint

$$
m_{t+1} - m_{t} = p_t g , \quad t \geq 0
$$ (eq:budgcontraint)

According to this equation, each period, the government prints money to pay for quantity $g$ of goods. 

In an **equilibrium**, the demand for currency equals the supply:

$$
m_{t+1}^d = m_{t+1}, \quad t \geq 0
$$ (eq:syeqdemand)

Let's take a moment to think  about what equation {eq}`eq:syeqdemand` tells us.

The demand for money at any time $t$ depends on the price level at time $t$ and the price level at time $t+1$.

The supply of money at time $t+1$ depends on the money supply at time $t$ and the price level at time $t$.

So the infinite sequence  of equations {eq}`eq:syeqdemand` for $ t \geq 0$ imply that the **sequences** $\{p_t\}_{t=0}^\infty$ and $\{m_t\}_{t=0}^\infty$ are tied together and ultimately simulataneously deteR_mined.


## Equilibrium price and money supply sequences


The preceding specifications imply that for $t \geq 1$, **real balances** evolve according to


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
  
We'll restrict our attention to  parameter values and  associated gross real rates of return on real balances that assure that the demand for real balances is positive, which according to {eq}`eq:bdemand` means that

$$
b_t = \gamma_1 - \gamma_2 R_t^{-1} > 0 
$$ 

which implies that 

$$
R_t \geq \left( \frac{\gamma_2}{\gamma_1} \right) \equiv \underline R
$$ (eq:Requation)

Gross real rate of return $\underline R$ is the smallest rate of return on currency 
that is consistent with a nonnegative demand for real balances.



We shall describe two distinct but closely related ways of computing a pair   $\{p_t, m_t\}_{t=0}^\infty$ of sequences for the price level and money supply.

But first it is instructive to describe a special type of equilibrium known as a **steady state**.

### Steady States

In a **steady state**

\begin{align}
R_t & = \bar R \cr
b_t & = \bar b
\end{align}

for $t \geq 0$.  

To compute a steady state, we seek steady state values $\bar R, \bar b$ that satisfy steady-state versions of  both the government budget constraint and the demand function for real balances:

\begin{align}
g & = \bar b ( 1 - \bar R)  \cr
\bar b & = \gamma_1- \gamma_2 \bar R^{-1}
\end{align}

Together these equations  imply

$$
(\gamma_1 + \gamma_2) - \frac{\gamma_2}{\bar R} - \gamma_1 \bar R = g
$$ (eq:seignsteady)


The left side is the steady-state amount of **seigniorage** or government revenues that the government gathers by paying a gross rate of return $\bar R < 1$ on currency. 

The right side is government expenditures.

Define steady-state seigniorage as

$$
S(\bar R) = (\gamma_1 + \gamma_2) - \frac{\gamma_2}{\bar R} - \gamma_1 \bar R
$$ (eq:SSsigng)

Notice that $S(\bar R) \geq 0$ only when $\bar R \in [\frac{\gamma2}{\gamma1}, 1] 
\equiv [\underline R, \overline R]$ and that $S(\bar R) = 0$ if $\bar R  = \underline R$
or if $\bar R  = \overline R$.

We shall study equilibrium sequences that  satisfy

$$
R_t \in \bar R  = [\underline R, \overline R],  \quad t \geq 0. 
$$

Maximizing steady state seigniorage  {eq}`eq:SSsigng` with respect to $\bar R$, we find that the maximizing rate of return on currency is 

$$
\bar R_{\rm max} = \sqrt{\frac{\gamma_2}{\gamma_1}}
$$

and that the associated maximum seigniorage revenue that the government can gather from printing money is

$$
(\gamma_1 + \gamma_2) - \frac{\gamma_2}{\bar R_{\rm max}} - \gamma_1 \bar R_{\rm max}
$$

It is useful to rewrite  equation {eq}`eq:seignsteady` as

$$
-\gamma_2 + (\gamma_1 + \gamma_2 + g) \bar R - \gamma_1 \bar R^2 = 0
$$ (eq:steadyquadratic)

A steady state value $\bar R$ solves quadratic equation {eq}`eq:steadyquadratic`.

So two steady states typically exist. 


Let's set some parameter values and compute possible steady state rates of return on currency $\bar R$, the  signiorage maximizing rate of return on currency, and an object that we'll discuss later, namely, an initial price level $p_0$ associated with the maximum steady state rate of return on currency.

+++

First, we create a `namedtuple` to store pararmeters.

We can reuse this `namedtuple` in our functions throughout this lecture

```{code-cell} ipython3
γ1 = 100
γ2 = 50
g = 3.0
M0 = 100

# Create a namedtuple that contains parameters
SeignModel = namedtuple("SeignModel", 
                        ["γ1", "γ2", "g", 
                         "M0", "R_u", "R_l"])

def create_model(γ1=100, γ2=50, g=3.0, M0=100):
    
    # Calculate the steady states for R
    R_steady = np.roots((-γ1, γ1 + γ2 - g, -γ2))
    R_u, R_l = R_steady
    print("[R_u, R_l] =", R_steady)
    
    return SeignModel(γ1=γ1, γ2=γ2, g=g, M0=M0, R_u=R_u, R_l=R_l)
```

Now we compute the $\bar R_{\rm max}$ and corresponding revenue

```{code-cell} ipython3
def seign(R, model):
    γ1, γ2, g = model.γ1, model.γ2, model.g
    return -γ2/R + (γ1 + γ2)  - γ1 * R

seign_model = create_model()
R_u, R_l = seign_model.R_u, seign_model.R_l

# Calculate initial guess for p0
p0_min = M0 / (γ1 - g - γ2 / R_u)
print("p0_min =", p0_min)

# Calculate signiorage maximizing rate of return
R_max = np.sqrt(γ2 / γ1)

# Calculate seigniorage revenue
max_seign = seign(R_max, seign_model)

print("R_max =", R_max)
print("Max seigniorage revenue =", max_seign)
```

Now let's plot seigniorage as a function of alternative potential steady-state values of $R$.

We'll see that there are two values of $R$ that attain seigniorage levels equal to $g$,
one that we'll denote $R_l$, another that we'll denote $R_u$.

They satisfy $R_l < R_u$ and are affiliated with a higher inflation tax rate $(1-R_l)$ and a lower
inflation tax rate $1 - R_u$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Revenue from inflation tax
    name: infl_tax
    width: 500px
---
# Generate values for R
R_values = np.linspace(γ2/γ1, 1, 250)

# Calculate the function values
seign_values = seign(R_values, seign_model)

# Visualize seign_values against R values
fig, ax = plt.subplots(dpi=300)
plt.plot(R_values, seign_values, label='inflation tax revenue')
plt.axhline(y=g, color='red', linestyle='--', label='government deficit')
plt.xlabel('$R$')
plt.ylabel('seigniorage')

#plt.title('Steady state revenue from inflation tax')
plt.legend()
plt.grid(True)
plt.show()
```

Let's print the two steady-state rates of return $\bar R$ and the associated seigniorage revenues that the government collects.

(By contruction, both steady state rates of return should raise the same amounts real revenue).

We hope that the following code will  confirm this.

```{code-cell} ipython3
g1 = seign(R_u, seign_model)
print(f"R_u, g_u = {R_u:.4f}, {g1:.4f}")

g2 = seign(R_l, seign_model)
print(f"R_l, g_l = {R_l:.4f}, {g2:.4f}")
```

Now let's compute the maximum steady state amount of seigniorage that could be gathered by printing money and the state state rate of return on money that attains it.

```{code-cell} ipython3
R_max = np.sqrt(γ2/γ1)
g_max = seign(R_max, seign_model)
print(f"R_max, g_max = {R_max:.4f}, {g_max:.4f}")
```

## Two equilibrium computation strategies


We now proceed to compute equilibria, not necessarily steady states.

We shall  deploy two distinct computation strategies.

### Method 1 

   * set $R_0 \in [\frac{\gamma_2}{\gamma_1}, R_u]$ and compute $b_0 = \gamma_1 - \gamma_2/R_0$.

   * compute sequences $\{R_t, b_t\}_{t=1}^\infty$ of rates of return and real balances that are associated with an equilibrium by solving equation {eq}`eq:bmotion` and {eq}`eq:bdemand` sequentially  for $t \geq 1$:  
   \begin{align}
b_t & = b_{t-1} R_{t-1} + g \cr
R_t^{-1} & = \frac{\gamma_1}{\gamma_2} - \gamma_2^{-1} b_t 
\end{align}

   * Construct the associated equilibrium $p_0$ from 
  
   $$
   p_0 = \frac{m_0}{\gamma_1 - g - \gamma_2/R_0}
   $$ (eq:p0fromR0)
   
   * compute $\{p_t, m_t\}_{t=1}^\infty$  by solving the following equations sequentially
  
  $$
   \begin{align}
   p_t & = R_t p_{t-1} \cr
   m_t & = b_{t-1} p_t 
   \end{align}
  $$ (eq:method1) 
   
**Remark 1:** method 1 uses an indirect approach to computing an equilibrium by first computing an equilbrium  $\{R_t, b_t\}_{t=0}^\infty$ sequence and then using it to back out an equilibrium  $\{p_t, m_t\}_{t=0}^\infty$  sequence.


 **Remark 2:** notice that  method 1 starts by picking an **initial condition** $R_0$ from a set $[\frac{\gamma_2}{\gamma_1}, R_u]$. An equilibrium $\{p_t, m_t\}_{t=0}^\infty$ sequences are not unique.  There is actually a continuum of equilibria indexed by a choice of $R_0$ from the set $[\frac{\gamma_2}{\gamma_1}, R_u]$. 

 **Remark 3:** associated with each selection of $R_0$ there is a unique $p_0$ described by
 equation {eq}`eq:p0fromR0`.
 
### Method 2

   This method deploys a direct approach. 
   It defines a "state vector" 
    $y_t = \begin{bmatrix} m_t \cr p_t\end{bmatrix} $
   and formulates  equilibrium conditions {eq}`eq:demandmoney`, {eq}`eq:budgcontraint`, and
   {eq}`eq:syeqdemand`
 in terms of a first-order vector difference equation

   $$
   y_{t+1} = M y_t, \quad t \geq 0 ,
   $$

   where we temporarily take $y_0 = \begin{bmatrix} m_0 \cr p_0 \end{bmatrix}$ as an **initial condition**. 
   
   The solution is 
   
   $$
   y_t = M^t y_0 .
   $$

   Now let's think about the initial condition $y_0$. 
   
   It is natural to take the initial stock of money $m_0 >0$ as an initial condition.
   
   But what about $p_0$?  
   
   Isn't it  something that we want  to be **determined** by our model?

   Yes, but sometimes we want too much, because there is actually a continuum of initial $p_0$ levels that are compatible with the existence of an equilibrium.  
   
   As we shall see soon, selecting an initial $p_0$ in method 2 is intimately tied to selecting an initial rate of return on currency $R_0$ in method 1. 
   
## Computation Method 1  

%We start from an arbitrary $R_0$ and  $b_t = \frac{m_{t+1}}{p_t}$, we have 

%$$
%b_0 = \gamma_1 - \gamma_0 R_0^{-1} 
%$$

Remember that there exist  two steady state equilibrium  values $ R_l <  R_u$  of the rate of return on currency  $R_t$.

We proceed as follows.

Start at $t=0$ 
 * select a  $R_0 \in [\frac{\gamma_2}{\gamma_1}, R_u]$  
 * compute   $b_0 = \gamma_1 - \gamma_0 R_0^{-1} $ 
 
Then  for $t \geq 1$ construct $(b_t, R_t)$ by
iterating  on the system 
\begin{align}
b_t & = b_{t-1} R_{t-1} + g \cr
R_t^{-1} & = \frac{\gamma_1}{\gamma_2} - \gamma_2^{-1} b_t
\end{align}


When we implement this part of method 1, we shall discover the following  striking 
outcome:

 * starting from an $R_0$ in  $[\frac{\gamma_2}{\gamma_1}, R_u]$, we shall find that 
$\{R_t\}$ always converges to a limiting "steady state" value  $\bar R$ that depends on the initial
condition $R_0$.

  * there are only two possible limit points $\{ R_l, R_u\}$. 
  
  * for almost every initial condition $R_0$, $\lim_{t \rightarrow +\infty} R_t = R_l$.
  
  * if and only if $R_0 = R_u$, $\lim_{t \rightarrow +\infty} R_t = R_u$.
  
The quantity   $1 - R_t$ can be interpreted as an **inflation tax rate** that the government imposes on holders of its currency.


We shall soon  see that the existence of two steady state rates of return on currency
that serve to finance the government deficit of $g$ indicates the presence of a **Laffer curve** in the inflation tax rate.  

```{note}
Arthur Laffer's curve plots a hump shaped curve of revenue raised from a tax against the tax rate.  
Its hump shape indicates that there are typically two tax rates that yield the same amount of revenue. This is due to two countervailing courses, one being that raising a tax rate typically decreases the **base** of the tax as people take decisions to reduce their exposure to the tax.
```

```{code-cell} ipython3
def simulate_system(R0, model, num_steps):
    
    γ1, γ2, g = model.γ1, model.γ2, model.g
    
    # Initialize lists to store results
    b_values = [γ1 - γ2 / R0]
    R_values = [1 / ((γ1 / γ2) - (γ2**(-1) * b_values[0]))]

    # Iterate over time steps
    for t in range(1, num_steps):
        # Calculate b_t and R_t
        b_t = b_values[t - 1] * R_values[t - 1] + g
        R_t_inverse = (γ1 / γ2) - γ2**(-1) * b_t
        R_values.append(1 / R_t_inverse)
        b_values.append(b_t)

    return b_values, R_values
```

Let's write some code plot outcomes for several possible initial values $R_0$.

```{code-cell} ipython3
:tags: [hide-cell]

dashed_param = {'color':'grey', 
                'linestyle': '--',
                'lw': 1.5,
                'alpha': 0.6}

label_param = {'verticalalignment': 'center', 
               'color': 'grey',
               'size': 12}

line_param = {'lw': 1.5, 
              'marker': 'o',
              'markersize': 3}

graph_params = [dashed_param, label_param, line_param]

def draw_paths(R0_values, model, graph_params, num_steps):
    
    R_u, R_l, γ1, γ2, g = (model.R_u, model.R_l, 
                           model.γ1, model.γ2, model.g)
    
    dashed_param, label_param, line_param = graph_params

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Iterate over R_0s and simulate the system 
    for R0 in R0_values:
        b_values, R_values = simulate_system(R0, model, num_steps)
        
        # Plot R_t against time
        axes[0].plot(range(num_steps), R_values, 
                    **line_param)
        
        # Plot b_t against time
        axes[1].plot(range(num_steps), b_values, 
                    **line_param)
        
    # Add line and text annotations to subgraph 
    annotate_graph(axes[0], model, 
                   num_steps, 
                   dashed_param, label_param)
    
    axes[0].set_ylabel('$R_t$')
    
    axes[1].set_xlabel('timestep')
    axes[1].set_ylabel('$b_t$')
    
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.show()
    
def annotate_graph(ax, model, num_steps, 
                   dashed_param, label_param):
    
    R_u, R_l, γ1, γ2 = (model.R_u, model.R_l, 
                        model.γ1, model.γ2)
    
    # Add dashed lines for R_u, R_l, and γ2/γ1
    ax.axhline(y=R_u, **dashed_param)
    ax.axhline(y=R_l, **dashed_param)
    ax.axhline(y=γ2/γ1, **dashed_param)

    # Add text annotations for dashed lines
    ax.text(num_steps * 1.02, R_u, 
            r'$R_u$', **label_param)
    ax.text(num_steps * 1.02, R_l, 
            r'$R_l$', **label_param)
    ax.text(num_steps * 1.02, γ2/γ1, 
            r'$\frac{\gamma_2}{\gamma_1}$', 
            **label_param)
```

Let's plot  distinct outcomes  associated with several  $R_0 \in [\frac{\gamma_2}{\gamma_1}, R_u]$.

Each line below shows a path associated with a different $R_0$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Paths of $R_t$ (top panel) and $b_t$ (bottom panel) starting from different
      initial condition $R_0$
    name: R0_path
    width: 500px
---
# Create a grid of R_0s
R0s = np.linspace(γ2/γ1, R_u, 9)
R0s = np.append(R_l, R0s)
draw_paths(R0s, seign_model, 
           graph_params, num_steps=20)
```

Notice how sequences that  start from $R_0$ in the half-open interval $[R_l, R_u)$ converge to the steady state  associated with  to $ R_l$.

+++ {"user_expressions": []}

## Computation method 2 

Set $m_t = m_t^d $ for all $t \geq -1$. 

Let 

$$
  y_t =  \begin{bmatrix} m_{t} \cr p_{t} \end{bmatrix} .
$$

Represent  equilibrium conditions {eq}`eq:demandmoney`, {eq}`eq:budgcontraint`, and    {eq}`eq:syeqdemand` as

$$
\begin{bmatrix} 1 & \gamma_2 \cr
                 1 & 0 \end{bmatrix} \begin{bmatrix} m_{t+1} \cr p_{t+1} \end{bmatrix} =
                 \begin{bmatrix} 0 & 1 \cr
                 1 & g \end{bmatrix} \begin{bmatrix} m_{t} \cr p_{t} \end{bmatrix} 
$$ (eq:sytem101)

or

$$ 
H_1 y_t = H_2  y_{t-1} 
$$

where 

\begin{align} H_1 & = \begin{bmatrix} 1 & \gamma_2 \cr
                 1 & 0 \end{bmatrix} \cr
                H_2 & = \begin{bmatrix} 0 & 1 \cr
                 1 & g \end{bmatrix}  
\end{align}

Define

$$
H = H_1^{-1} H_2
$$

and write the system  {eq}`eq:sytem101` as

$$
y_{t+1} = H y_t, \quad t \geq 0 
$$ (eq:Vaughn)

so that $\{y_t\}_{t=0}$ can be computed from

$$
y_t = H^t y_0, t \geq 0
$$ (eq:ytiterate)

where 

$$
y_0 = \begin{bmatrix} m_{0} \cr p_0 \end{bmatrix} .
$$


It is natural to take  $m_0$ as an initial condition determined outside the model.

The mathematics seems to tell us that $p_0$ must also be determined outside the model, even though
it is something that we actually wanted to be determined by the model.

(As usual, we should listen when mathematics talks to us.)

For now, let's just proceed mechanically on faith. 

Compute the eigenvector decomposition 

$$
H=  Q \Lambda Q^{-1} 
$$ 

where $\Lambda$ is a diagonal matrix of eigenvalues and the columns of $Q$ are eigenvectors correspondng to those eigenvalues.

It turns out that  


$$
\begin{bmatrix} {R_l}^{-1} & 0 \cr 
                0 & {R_u}^{-1} \end{bmatrix}
$$

where $R_l$ and $R_u$ are the lower and higher steady-state rates of return on currency that we computed above.  



Partition $Q$ as

$$ 
Q =\begin{bmatrix} Q_{11} & Q_{12} \cr
                   Q_{21} & Q_{22} \end{bmatrix}
$$

Below we shall verify the following claims: 


**Claims:** If we set 

$$
p_0 = \overline p_0 \equiv Q_{21} Q_{11}^{-1}  m_{0} ,
$$ (eq:magicp0)

it turns out that 

$$ 
\frac{p_{t+1}}{p_t} = {R_u}^{-1}, \quad t \geq 0
$$


However, if we set 

$$ 
p_0 > \bar p_0
$$

then

$$
\lim_{t\rightarrow + \infty} \frac{p_{t+1}}{p_t} = {R_l}^{-1}.
$$

Let's verify these claims step by step.



Note that

$$
H^t = Q \Lambda^t Q^{-1}
$$

so that

$$
y_t = Q \Lambda^t Q^{-1} y_0
$$

For almost all initial vectors $y_0$, the gross rate of inflation $\frac{p_{t+1}}{p_t}$ eventually converges to  the larger eigenvalue ${R_l}^{-1}$.

The only way to avoid this outcome is for  $p_0$ to take  the specific value described by {eq}`eq:magicp0`.

To understand  this situation,  we  use the following
transformation

$$
y^*_t = Q^{-1} y_t . 
$$

Dynamics of $y^*_t$ are evidently governed by 

$$
y^*_{t+1} = \Lambda^t y^*_t .
$$ (eq:stardynamics)

This equation represents the dynamics of our system  in a way that lets us  isolate the
force that causes  gross inflation to converge to the inverse of the lower steady state rate
of inflation $R_l$ that we discovered earlier. 

Staring at  equation {eq}`eq:stardynamics` indicates that unless

```{math}
:label: equation_11

y^*_0 = \begin{bmatrix} y^*_{1,0} \cr 0 \end{bmatrix}
```

the path of $y^*_t$,  and therefore the paths of both $m_t$ and $p_t$ given by
$y_t = Q y^*_t$ will eventually grow at gross rates ${R_l}^{-1}$ as 
$t \rightarrow +\infty$. 

Equation {eq}`equation_11` also leads us to conclude that there is a unique setting
for the initial vector $y_0$ for which both components forever grow at the lower rate ${R_u}^{-1}$. 


For this to occur, the required setting of $y_0$ must evidently have the property
that

$$
Q y_0 =  y^*_0 = \begin{bmatrix} y^*_{1,0} \cr 0 \end{bmatrix} .
$$

But note that since
$y_0 = \begin{bmatrix} m_0 \cr p_0 \end{bmatrix}$ and $m_0$
is given to us an initial condition,  $p_0$ has to do all the adjusting to satisfy this equation.

Sometimes this situation is described informally  by saying that while $m_0$
is truly a **state** variable, $p_0$ is a **jump** variable that
must adjust at $t=0$ in order to satisfy the equation.

Thus, in a nutshell the unique value of the vector $y_0$ for which
the paths of $y_t$ **don't** eventually grow at rate ${R_l}^{-1}$ requires  setting the second component
of $y^*_0$ equal to zero.

The component $p_0$ of the initial vector
$y_0 = \begin{bmatrix} m_0 \cr p_0 \end{bmatrix}$ must evidently
satisfy

$$
Q^{\{2\}} y_0 =0
$$

where $Q^{\{2\}}$ denotes the second row of $Q^{-1}$, a
restriction that is equivalent to

```{math}
:label: equation_12

Q^{21} m_0 + Q^{22} p_0 = 0
```

where $Q^{ij}$ denotes the $(i,j)$ component of
$Q^{-1}$.

Solving this equation for $p_0$, we find

```{math}
:label: equation_13

p_0 = - (Q^{22})^{-1} Q^{21} m_0.
```


#### A more convenient formula 

We can get the equivalent but perhaps more convenient formula {eq}`eq:magicp0` for $p_0$ that is cast
in terms of components of $Q$ instead of components of
$Q^{-1}$.

To get this formula, first note that because $(Q^{21}\ Q^{22})$ is
the second row of the inverse of $Q$ and because
$Q^{-1} Q = I$, it follows that

$$
\begin{bmatrix} Q^{21} & Q^{22} \end{bmatrix}  \begin{bmatrix} Q_{11}\cr Q_{21} \end{bmatrix} = 0
$$

which implies that

$$
Q^{21} Q_{11} + Q^{22} Q_{21} = 0.
$$

Therefore,

$$
-(Q^{22})^{-1} Q^{21} = Q_{21} Q^{-1}_{11}.
$$

So we can write

```{math}

p_0 = Q_{21} Q_{11}^{-1} m_0 .
```

whic is our formula {eq}`eq:magicp0`.

It can be verified that this formula replicates itself over time in the sense  that

```{math}
:label: equation_15

p_t = Q_{21} Q^{-1}_{11} m_t.
```



Let's compute $p_0$ in the code below.

```{code-cell} ipython3
:user_expressions: []

# Define H1 and H2
H1 = np.array([[1, γ2], 
               [1, 0]])
H2 = np.array([[0, γ1], 
               [1, g]]) 

H = np.linalg.inv(H1) @ H2
print("H = ", H)

Λ, Q = np.linalg.eig(H)
print("Λ = ", Λ)
print("Q = ", Q)
```

```{code-cell} ipython3
R_bar1 = 1 / Λ[0]
R_bar2 = 1 / Λ[1]

print(f'R_bar_1 = {R_bar1:.4f}')
print(f'R_bar_2 = {R_bar2:.4f}')
```

```{code-cell} ipython3
p0_bar = (Q[1, 0]/Q[0, 0]) * M0

print('p0_bar = ', p0_bar)
print('p0_bar == p0_min:', np.isclose(p0_bar, p0_min))
```

```{code-cell} ipython3
def iterate_H(y_0, H, num_steps):
    y = np.empty((2, num_steps))
    y[:, 0] = y_0
    for t in range(num_steps-1):
        y[:, t+1] = H @ y[:, t]
    return y
```

Let's draw the dynamics of $m_t$, $p_t$, and $R_t$ starting from different $p_0$ values.

We create a function `draw_iterations` to generate the plot

```{code-cell} ipython3
:tags: [hide-cell]

def draw_iterations(p0s, model, graph_params,
                    num_steps):
    
    M0, R_u, R_l, γ1, γ2 = (model.M0, model.R_u, 
                            model.R_l, model.γ1, 
                            model.γ2)

    dashed_param, label_param, line_param = graph_params

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Iterate over R_0s and simulate the system 
    for p0 in p0s:
        y0 = np.array([M0, p0])
        y_series = iterate_H(y0, H, num_steps)
        M, P = y_series[0, :], y_series[1, :]

        # Plot R_t against time
        axes[0].plot(range(num_steps), M, 
                    **line_param)
        axes[0].set_yscale('log')

        # Plot b_t against time
        axes[1].plot(range(num_steps), P, 
                    **line_param)
        axes[1].set_yscale('log')

        R = [P[i]/P[i+1] for i in range(len(P)-1)]

        axes[2].plot(range(num_steps-1), R, 
                **line_param)
        
    # Add line and text annotations to subgraph 
    annotate_graph(axes[2], model, 
                   num_steps, 
                   dashed_param, label_param)
    
    # Draw labels
    axes[0].set_ylabel('$m_t$')
    axes[1].set_ylabel('$p_t$')
    axes[2].set_ylabel('$R_t$')
    axes[2].set_xlabel('timestep')
    
    # Enforce integar axis label
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Paths of $m_t$ (top panel), $p_t$ (middle panel), $R_t$ (bottom panel)
      starting from different initial condition $p_0$
    name: R0_path
    width: 500px
---
p0s = [p0_bar, 2.34, 2.5, 3, 4, 7, 30, 100_000]

draw_iterations(p0s, seign_model, graph_params, num_steps=20)
```

NOTE TO HUMPRHEY.  WHAT I'D LIKE TO DO IS WRITE SOME CODE TO ITERATE ON EQUATION SYSTEM 
{eq}`eq:Vaughn` from initial condition $m_0$ and VARIOUS $p_0$ values. 

ONLY FOR THE particular value of $p_0$ computed in the above code will the the inflation rate state at the inverse of the ** highe**r steady state rate of return on currency. 

FOR ALL OTHER ADMISSIBLE VALUES OF $p_0$, rates of return on currency will converge to the lower steady-state rate of return on currency.

I SUSPECT THAT SOME NICE GRAPHS COULD BE GOTTEN BY PLOTTING LOGARITHMS OF M_0 AND p_0 -- EACH SHOULD EVENTUALLY START GROWING AT A CONSTANT RATE.  

OR MAYBE INSTEAD JUST PLOT THE GROSS RATES OF GROWTH, LIKE YOU HAVE ABOVE.  

ARE YOU WILLING TO EXPERIMENT WITH THIS?
