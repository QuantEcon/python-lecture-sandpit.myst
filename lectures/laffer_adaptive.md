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

+++ {"user_expressions": [], "tags": []}

# Inflation Rate Laffer Curves  with Adaptive Expectations 

## Overview

This lecture studies stationary and dynamic **Laffer curves** in the inflation tax rate in a non-linear version of the model studied in this XXXX lecture.

As in lecture XXXXX, this lecture uses the log-linear version of the demand function for money that Cagan {cite}`Cagan` used in his classic paper in place of the linear demand function used in this XXXX lecture.

But now, instead of assuming  ''rational expectations'' in the form of ''perfect foresight'',
we'll adopt the ''adaptive expectations'' assumption used by Cagan {cite}`Cagan`.



This means that instead of assuming that expected inflation $\pi_t^*$ is described by

$$
\pi_t^* = p_{t+1} - p_t
$$ 

as we assumed in lecture XXXX, we'll now assume that $\pi_t^*$ is determined by the adaptive expectations scheme described in equation {eq}`eq:adaptex`  reported below. 







## The Model

Let  

  * $m_t$ be the log of the money supply at the beginning of time $t$
  * $p_t$ be the log of the price level at time $t$
  * $\pi_t^*$ be the public's expectation of the rate of inflation between $t$ and $t+1$ 
  

The law of motion of the money supply is

$$ 
\exp(m_{t+1}) - \exp(m_t) = g \exp(p_t) 
$$ (eq:msupply)


where $g$ is the part of government expenditures financed by printing money.

Notice that equation {eq}`eq:msupply` implies that

$$
m_{t+1} = \log[ \exp(m_t) + g \exp(p_t)]
$$ (eq:msupply2)

The demand function for money is 

$$
m_{t+1} - p_t = -\alpha \pi_t^* 
$$ (eq:mdemand)

where $\alpha \geq 0$.  



Expectations of inflation are governed by 

HUMPHREY: I CHANGED THE FOLLOWING EQUATION (MARCH 15)

$$
\pi_{t}^* = (1-\delta) (p_t - p_{t-1}) + \delta \pi_{t-1}^*
$$ (eq:adaptex)

where $\delta \in (0,1)$


## Computing An Equilibrium Sequence 

Equation the expressions for $m_{t+1}$ promided  by {eq}`eq:mdemand` and {eq}`eq:msupply2` and use equation {eq}`eq:adaptex` to eliminate $\pi_t^*$ to obtain
the following equation for $p_t$:

$$
\log[ \exp(m_t) + g \exp(p_t)] - p_t = -\alpha [(1-\delta) (p_t - p_{t-1}) + \delta \pi_{t-1}^*]
$$ (eq:pequation)


**Pseudo-code**

Here is pseudo code for our algorithm.

Starting at time $0$ with initial conditions $(m_0, \pi_{-1}^*, p_{-1})$, for each $t \geq 0$
deploy the following steps in order:

* solve {eq}`eq:pequation` for $p_t$
* solve equation {eq}`eq:adaptex` for $\pi_t^*$ 
* solve equation {eq}`eq:msupply2` for $m_{t+1}$

This completes the algorithm.


## Claims or Conjectures
  
  
It will turn out that 

 * if they exist, limiting values $\overline \pi$ and $\overline \mu$ will be equal
 
 * if  limiting values exists, there are two possible limiting values, one high, one low
 
 * unlike the outcome in lecture XXXX, for almost all initial log price levels and expected inflation rates $p_0, \pi_{t}^*$, the limiting $\overline \pi = \overline \mu$ is  the **lower** steady state  value
 
 * for each of the two possible limiting values $\bar \pi$ ,there is a unique initial log price level $p_0$ that implies that $\pi_t = \mu_t = \bar \mu$ for all  $t \geq 0$
 
    * this unique initial log price level solves $\log(\exp(m_0) + g \exp(p_0)) - p_0 = - \alpha \bar \pi $
    
    * the preceding equation for $p_0$ comes from $m_1 - p_0 = -  \alpha \bar \pi$

+++ {"user_expressions": []}

## Limiting Values of Inflation Rate

As in our earlier lecture XXXX, we can compute the two prospective limiting values for $\bar \pi$ by studying the steady-state Laffer curve.

Thus, in a  **steady state** 

$$
m_{t+1} - m_t = p_{t+1} - p_t =  x \quad \forall t ,
$$

where $x > 0 $ is a common rate of growth of logarithms of the money supply and price level.

A few lines of algebra yields the following equation that $x$ satisfies

$$
\exp(-\alpha x) - \exp(-(1 + \alpha) x) = g 
$$ (eq:steadypi)

where we require that

$$
g \leq \max_{x: x \geq 0} \exp(-\alpha x) - \exp(-(1 + \alpha) x) ,  
$$ (eq:revmax)

so that it is feasible to finance $g$ by printing money.

The left side of {eq}`eq:steadypi` is steady state revenue raised by printing money.

The right side of {eq}`eq:steadypi` is the quantity  of time $t$ goods  that the government raises by printing money. 

Soon  we'll plot  the left and right sides of equation {eq}`eq:steadypi`.

But first we'll write code that computes a steady-state
$\bar \pi$.



Let's start by importing some  libraries

```{code-cell} ipython3
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import get_cmap
from matplotlib.colors import to_rgba
import matplotlib
from scipy.optimize import fsolve 
```

+++ {"user_expressions": []}

Let's create a `namedtuple` to store the parameters of the model

```{code-cell} ipython3
LafferAdaptive = namedtuple('LafferAdaptive', 
                        ["m0",  # log of the money supply at t=0
                         "α",   # sensitivity of money demand
                         "g",   # government expenditure
                         "δ"])

# Create a Cagan Laffer model
def create_model(α=0.5, m0=np.log(200), g=0.35, δ=0.5):
    return LafferAdaptive(α=α, m0=m0, g=g, δ=δ)

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

## Steady State Laffer Curve

The following figure plots the steady state Laffer curve together with the two stationary inflation rates.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Seigniorage as function of steady state inflation. The dashed brown lines
      indicate $\pi_l$ and $\pi_u$.
    name: laffer_curve_nonlinear
    width: 500px
---
def compute_seign(x, α):
    return np.exp(-α * x) - np.exp(-(1 + α) * x) 

def plot_laffer(model, πs):
    α, g = model.α, model.g
    
    # Generate π values
    x_values = np.linspace(0, 5, 1000)

    # Compute corresponding seigniorage values for the function
    y_values = compute_seign(x_values, α)

    # Plot the function
    plt.plot(x_values, y_values, 
            label=f'$exp((-{α})x) - exp(- (1- {α}) x)$')
    for π, label in zip(πs, ['$\pi_l$', '$\pi_u$']):
        plt.text(π, plt.gca().get_ylim()[0]*2, 
                 label, horizontalalignment='center',
                 color='brown', size=10)
        plt.axvline(π, color='brown', linestyle='--')
    plt.axhline(g, color='red', linewidth=0.5, 
                linestyle='--', label='g')
    plt.xlabel('$\pi$')
    plt.ylabel('seigniorage')
    plt.legend()
    plt.grid(True)
    plt.show()

# Steady state Laffer curve
plot_laffer(model, (π_l, π_u))
```

+++ {"user_expressions": []}

## Associated Initial Price Levels

 Now that we have our hands on the two possible steady states, we can compute two initial log price levels $p_0$, which as initial conditions, imply that $\pi_t = \bar \pi $ for all $t \geq 0$.

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
print(f'Associated initial p_0s are: {p0_l, p0_u}')
```

+++ {"user_expressions": []}

### Verification 



To start, let's write some code to verify that if the initial log price level $p_0$ takes one
of the two values we just calculated, the inflation rate $\pi_t$ will be constant for all $t \geq 0$.

The following code verifies this.

HUMPRHEY - HOPEFULLY THINGS WILL WORK BETTER WITH THE ALTERED MARCH 15 TWEAKS DESCRIBED ABOVE

```{code-cell} ipython3
def solve_laffer_adapt(p0, π0, model, num_steps):
    m0, α, δ, g = model.m0, model.α, model.δ, model.g
    π_seq, μ_seq, m_seq, p_seq = [π0], [], [m0], [p0]
    for t in range(num_steps):
        
        # Solve for p_t
        def p_t(p):
            return np.log(np.exp(m_seq[t]) + g * np.exp(p)) - p + α * π_seq[t]
        p_seq.append(fsolve(p_t, p_seq[t])[0])
        
        # Solve for m_{t+1}
        m_seq.append(p_seq[t] - α * π_seq[t])
        
        # Solve for π_{t+1}^*
        π_seq.append((1 - δ) * (p_seq[t] - p_seq[t-1]) + δ * π_seq[t])

        # Solve for μ_{t+1}
        μ_seq.append(m_seq[t] - m_seq[t-1])
        
    return π_seq, μ_seq, m_seq, p_seq
```

```{code-cell} ipython3
π_seq, μ_seq, m_seq, p_seq = solve_laffer_adapt(p0_l, np.log(200), model, 200)

# Check π and μ at steady state
print('π_bar == μ_bar:', np.isclose(π_seq[-1], μ_seq[-1]))

print(π_seq[-1], μ_seq[-1])

# Check steady state m_{t+1} - m_t and p_{t+1} - p_t 
print('m_{t+1} - m_t:', m_seq[-1] - m_seq[-2])
print('p_{t+1} - p_t:', p_seq[-1] - p_seq[-2])

# Check if exp(-αx) - exp(-(1 + α)x) = g
eq_g = lambda x: np.exp(-model.α * x) - np.exp(-(1 + model.α) * x)

print('eq_g == g:', np.isclose(eq_g(m_seq[-1] - m_seq[-2]), model.g))
```

+++ {"user_expressions": []}

## Slippery Side of Laffer Curve Dynamics



We are now equipped  to compute  time series starting from different $p_{-1}, \pi_{-1}^*$ settings, analogous to those in the this XXXX **money_inflation** lecture. 

HUMPHREY -- PLEASE NOTE HOW I EDITED THE PREVIOUS SENTENCE TO RECOGNIZE THE NEW STATE VECTOR WE NOW HAVE.  

```{code-cell} ipython3
:tags: [hide-cell]

p0_color_map = matplotlib.colormaps['viridis']
m0_color_map = matplotlib.colormaps['magma']

def draw_iterations(p0s, m0s, model, line_params, p0_bars, num_steps):
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    
    for ax in axes[:2]:
        ax.set_yscale('log')
    
    p0_colors = [p0_color_map(i) for i in np.linspace(0, 1, len(p0s))]
    m0_colors = [m0_color_map(i) for i in np.linspace(0, 1, len(m0s))]
    
    for i, p0 in enumerate(p0s):
        for j, m0 in enumerate(m0s):
            π_seq, μ_seq, m_seq, p_seq = solve_laffer_adapt(p0, m0, model, num_steps)
            
            # Corrected color blending
            avg_rgb = [(p + m) / 2 for p, m in zip(p0_colors[i][:3], m0_colors[j][:3])]
            line_color = (*avg_rgb, 0.5)

            updated_line_params = dict(line_params, color=line_color)
            
            axes[0].plot(np.arange(num_steps), m_seq[1:], **updated_line_params)
            axes[1].plot(np.arange(num_steps), p_seq[1:], **updated_line_params)
            axes[2].plot(np.arange(num_steps), π_seq[1:], **updated_line_params)
            axes[3].plot(np.arange(num_steps), μ_seq, **updated_line_params)
    
    axes[0].set_ylabel('$m_t$')
    axes[1].set_ylabel('$p_t$')
    axes[2].set_ylabel('$\pi_t$')
    axes[3].set_ylabel('$\mu_t$')
    axes[3].set_xlabel('timestep')
    axes[3].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()
```

+++ {"user_expressions": []}

Let's simulate the result generated by varying initial $p_0s$

```{code-cell} ipython3
# Generate a sequence from p0_l to p0_u
p0s = np.arange(p0_l-10, p0_u+10, 0.2) 
π0s = np.arange(π_l-10, π_u+10, 0.1)

line_params = {'lw': 1.5, 
              'marker': 'o',
              'markersize': 3}

p0_bars = (p0_l, p0_u)
              
draw_iterations(p0s, [π_u], model, line_params, p0_bars, num_steps=20)
```

+++ {"user_expressions": []}

Let's now simulate the result generated by varying initial $\pi_0s$

```{code-cell} ipython3
draw_iterations([p0_u], π0s, model, line_params, p0_bars, num_steps=20)
```
