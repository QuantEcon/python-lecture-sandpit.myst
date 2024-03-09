---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"user_expressions": []}

# Laffer curve

This notebook graphs the steady state Laffer curve for the nonlinear version of the 
Cagan plus Laffer curve model.

Let's start by importing libraries necessory for this lecture

```{code-cell} ipython3
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import fsolve 
```

+++ {"user_expressions": []}

Let  

  * $m_t$ be the log of the money supply at the beginning of time $t$
  * $p_t$ be the log of the price level at time $t$
  
The demand function for money is 

$$
m_{t+1} - p_t = -\alpha (p_{t+1} - p_t) \tag{1}
$$

where $\alpha \geq 0$.  

The law of motion of the money supply is

$$ 
\exp(m_{t+1}) - \exp(m_t) = g \exp(p_t) \tag{2}
$$

where $g$ is the part of government expenditures financed by printing money.

**Remark:** Please notice that while equation (1) is linear in logs of the money supply and price level, equation (2) is linear in levels. This will require adapting the equilibrium computation methods that we deployed in lecture **money_inflation**.

## Computing An Equilibrium Sequence 

We'll deploy a method similar to **Method 2** used in the "money_inflation" lecture.  

We'll take the time $t$ state vector to be $m_t, p_t$.

  * we'll treat $m_t$ as a ''natural state variable'' and $p_t$ as a ''jump'' variable.
  
Let

$$
\lambda \equiv \frac{\alpha}{1+ \alpha}
$$

Let's rewrite equations (2) and (1) as


$$ 
\exp(m_{t+1}) - \exp(m_t) = g \exp(p_t) \tag{10}
$$

and 

$$
p_t = (1-\lambda) m_{t+1} + \lambda p_{t+1} \tag{11}
$$

We'll summarize our algorithm with the following pseudo-code.

**Pseudo-code**

  * start for $m_0, p0$ at time $t =0$

  * solve (10) for $m_{t+1}$
  
  * solve (11) for $p_{t+1} = \lambda^{-1} p_t + (1 - \lambda^{-1}) m_{t+1}$

  * compute $\pi_t = p_{t+1} - p_t$ and $\mu_t = m_{t+1} - m_t $
  
  * iterate on $t$ to convergence of $\pi_t \rightarrow \overline \pi$ and $\mu_t \rightarrow \overline \mu$
  
  
It will turn out that 

 * limiting values $\overline \pi$ and $\overline \mu$ will be equal
 
 * if a limiting value exists, there are two possible limiting values, one high, one low
 
 * for almost all initial log price levels $p_0$, the limiting $\overline \pi = \overline \mu$ is 
 the higher value
 
 * there is a unique initial log price level $p_0$ that implies that $\pi_t = \mu_t = \bar \mu$ for all 
 $t \geq 0$
 
    * this unique initial log price level solves $\log(\exp(m_0) + g \exp(p_0)) - p_0 = - \alpha \bar \pi $
    
    * the preceding equation for $p_0$ comes from $m_1 - p_0 = -  \alpha \bar \pi$

+++ {"user_expressions": []}

We can compute the two prospective limiting values for $\bar \pi$ by studying the steady-state Laffer curve.

Thus, in a  **steady state** 

$$
m_{t+1} - m_t = p_{t+1} - p_t =  x \quad \forall t ,
$$

where $x > 0 $ is a common rate of growth of logarithms of the money supply and price level.

A few lines of algebra yields the following equation that $x$ satisfies

$$
\exp(-\alpha x) - \exp(-(1 + \alpha) x) = g \tag{3}
$$

where we require that

$$
g \leq \max_{x: x \geq 0} \exp(-\alpha x) - \exp(-(1 + \alpha) x) ,  \tag{4}
$$

so that it is feasible to finance $g$ by printing money.

The left side of (3) is steady state revenue raised by printing money.

The right side of (3) is the amount of goods  that the government raises by printing money. 

Later we'll plot  the left and right sides of (3).

But first we'll write code that computes a steady-state
$\bar \pi$.

Let's create a `namedtuple` to store the parameters of the model

```{code-cell} ipython3
CaganLaffer = namedtuple('CaganLaffer', 
                        ["m0",  # log of the money supply at t=0
                         "α",   # sensitivity of money demand
                         "λ",
                         "g" ])

# Create a CaganLaffer model
def create_model(α=0.5, m0=np.log(100), g=0.35):
    return CaganLaffer(α=α, m0=m0, λ=α/(1+α), g=g)

model = create_model()
```

+++ {"user_expressions": []}

Now we write code that computes steady-state $\bar \pi$s.

```{code-cell} ipython3
# Define formula for π_bar
def solve_π(x, α, g):
    return np.exp(-α * x) - np.exp(-(1 + α) * x) - g

def solve_π_bar(model, x0):
    π_bar = fsolve(solve_π, x0=x0, xtol=1e-10, args=(model.α, model.g))[0]
    return π_bar

# Solve for the two steady state of π
π_l = solve_π_bar(model, x0=0.6)
π_u = solve_π_bar(model, x0=3.0)
print(f'The two steady state of π are: {π_l, π_u}')
```

+++ {"user_expressions": []}

We find two steady state $\bar \pi$ values

+++ {"user_expressions": []}

## Request for Humphrey

Is it possible for you please improve the code in the previous cell for computing a steady state $\bar \pi$ (chatgpt and I wrote the code)?

In particular, i'd like to compute **both** steady states.  I experimented a little with the preceding cell and found that if you initialize near one of the fixed points, the algorithm goes there.  

Please note that in the above cell, I initialized things so that the algorithm computed the lower $\bar \pi$. 

I'd  like you please to adapt and improve the code in the following cell to compute initial price levels $p_0$ that would be associated with the two possible $\bar \pi$ stationary inflation rates.

```{code-cell} ipython3
def solve_p0(p0, m0, α, g, π):
    return np.log(np.exp(m0) + g * np.exp(p0)) + α * π - p0

def solve_p0_bar(model, x0, π_bar):
    p0_bar = fsolve(solve_p0, x0=x0, xtol=1e-20, args=(model.m0, 
                                                       model.α, 
                                                       model.g, 
                                                       π_bar))[0]
    return p0_bar

# Compute two initial price levels associated with π_l and π_u
p0_l = solve_p0_bar(model, 
                    x0=np.log(220), 
                    π_bar=π_l)
p0_u = solve_p0_bar(model, 
                    x0=np.log(220), 
                    π_bar=π_u)
print(f'The two initial price levels p_0s are: {π_l, π_u}')
```

+++ {"user_expressions": []}

## Explanation

Now we proceed with Humphrey's high quality code

+++ {"user_expressions": []}

## Request for Humphrey

I'd like to subdivide this code to allow a user to compute and plot equilibrium paths for $\pi_t$ and $\mu_t$ starting from given values of $p_0$ -- this will help us prepare the way for  plot graphs like those in the **money_inflation** lecture.

```{code-cell} ipython3
# Implement pseudo-code above
def simulate_seq(p0, model, num_steps):
    λ, g = model.λ, model.g
    π_seq, μ_seq, m_seq, p_seq = [], [], [model.m0], [p0]

    for t in range(num_steps):
        
        m_seq.append(np.log(np.exp(m_seq[t]) + g * np.exp(p_seq[t])))
        p_seq.append(1/λ * p_seq[t] + (1 - 1/λ) * m_seq[t+1])

        μ_seq.append(m_seq[t+1]-m_seq[t])
        π_seq.append(p_seq[t+1]-p_seq[t])

    return π_seq, μ_seq, m_seq, p_seq
```

```{code-cell} ipython3
π_seq, μ_seq, m_seq, p_seq = simulate_seq(p0_l, model, 150)

# Check π and μ at steady state
print('π_bar == μ_bar:', π_seq[-1] == μ_seq[-1])

# Check steady state m_{t+1} - m_t and p_{t+1} - p_t 
print('m_{t+1} - m_t:', m_seq[-1] - m_seq[-2])
print('p_{t+1} - p_t:', p_seq[-1] - p_seq[-2])

# Check if exp(-αx) - exp(-(1 + α)x) = g
eq_g = lambda x: np.exp(-model.α * x) - np.exp(-(1 + model.α) * x)

print('eq_g == g:', np.isclose(eq_g(m_seq[-1] - m_seq[-2]), model.g))
```

+++ {"user_expressions": []}

## Another request for Humphrey

Now we are all set up to compute some graphs of time series starting from different $p_0$ settings, like those in the
**money_inflation** lecture.

```{code-cell} ipython3
def draw_iterations(p0s, model, line_params, p0_bars, num_steps):

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    
    # Pre-compute time steps
    time_steps = np.arange(num_steps) 
    
    # Plot the first two y-axes in log scale
    for ax in axes[:2]:
        ax.set_yscale('log')

    # Iterate over p_0s and calculate a series of y_t
    for p0 in p0s:
        π_seq, μ_seq, m_seq, p_seq = simulate_seq(p0, model, num_steps)

        # Plot m_t
        axes[0].plot(time_steps, m_seq[1:], **line_params)

        # Plot p_t
        axes[1].plot(time_steps, p_seq[1:], **line_params)
        
        # Plot π_t
        axes[2].plot(time_steps, π_seq, **line_params)
        
        # Plot μ_t
        axes[3].plot(time_steps, μ_seq, **line_params)
    
    # Draw labels
    axes[0].set_ylabel('$m_t$')
    axes[1].set_ylabel('$p_t$')
    axes[2].set_ylabel('$\pi_t$')
    axes[3].set_ylabel('$\mu_t$')
    axes[3].set_xlabel('timestep')
    
    for p_0, label in [(p0_bars[0], '$p_0=p_l$'), (p0_bars[1], '$p_0=p_u$')]:
        y = simulate_seq(p_0, model, 1)[0]
        for ax in axes[2:]:
            ax.axhline(y=y[0], color='grey', linestyle='--', lw=1.5, alpha=0.6)
            ax.text(num_steps * 1.02, y[0], label, verticalalignment='center', 
                         color='grey', size=10)
    
    # Enforce integar axis label
    axes[3].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()
```

```{code-cell} ipython3
# Generate a sequence from p0_l to p0_u
p0s = np.arange(p0_l, p0_u, 0.1) 

line_params = {'lw': 1.5, 
              'marker': 'o',
              'markersize': 3}

p0_bars = (p0_l, p0_u)
              
draw_iterations(p0s, model, line_params, p0_bars, num_steps=20)
```
