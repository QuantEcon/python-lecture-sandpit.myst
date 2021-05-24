---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

## Five Preference Orderings

This lecture describes   static representations of five classes of preferences over risky prospects. All  incorporate **risk aversion**, meaning displeasure from  risks governed by  well known probability distributions.  Two of them also incorporate  **uncertainty aversion**, meaning  dislike of not knowing a  probability distribution.



### Basic objects

Basic ingredients are 

* a set of states of the world
*  plans describing outcomes as functions of the state of the world,
*  a  utility  function mapping outcomes into utilities
*  either a probability distribution or a **set** of probability distributions over states of the world; and 
*  a way of measuring a discrepancy between two probability distributions.


In more detail, we'll work with the following setting.

 *  A  finite set of possible **states** ${\cal I} = \{i= 1, \ldots, I\}$.
 *  A (consumption) **plan** is a function $c: {\cal I} \rightarrow {\mathbb R}$.  
 * $u: {\mathbb R} \rightarrow {\mathbb R}$ is a **utility function**.
 * $\pi$ is an $I \times 1$ vector of nonnegative **probabilities** over  states, with $\pi_ i \geq 0, \sum_{i=1}^I \pi_i = 1$.
 * **Relative entropy** of a probability vector  $\hat \pi$ with respect to a probability vector $\pi$ is the expected value of the logarithm of the  likelihood ratio $m_i \doteq \Bigl( \frac{\hat \pi_i}{\pi_i} \Bigr) $  under   distribution $\hat \pi$:  $
 \textrm{ent}(\pi, \hat \pi) = \sum_{i=1}^I \hat \pi_i  \log \Bigl( \frac{\hat \pi_i}{\pi_i} \Bigr)   = \sum_{i=1}^I \pi_i \Bigl( \frac{\hat \pi_i}{\pi_i} \Bigr) \log \Bigl( \frac{\hat \pi_i}{\pi_i} \Bigr)  $ or $
    \textrm{ent}(\pi, \hat \pi) = \sum_{i=1}^I \pi_i m_i \log m_i  . $


**Remark** The likelihood ratio $m_i$ is a discrete random variable. For any discrete random variable $\{x_i\}_{i=1}^I$, the expected  value under the $\hat \pi_i$ distribution can be represented as the expected  value of  $x_i$ times the `shock'  $m_i$ under the $\pi$ distribution:
$ \hat E x = \sum_{i=1}^I x_i \hat \pi_i = \sum_{i=1}^I m_i x_i  \pi_i = E m x ,$ where $\hat E$ is the mathematical  expectation under the $\hat \pi$ distribution and $E$ is the expectation under the $\pi$ distribution. Evidently, $ \hat E 1 = E m = 1$ and relative entropy is $ E m \log m  = \hat E \log m .$

 Figure 1 XXXX depicts  entropy as a function of $\hat \pi_1$ when $I=2$ and $\pi_1 = .5$. 
 
 When $\pi_1 \in (0,1)$, entropy is finite for both $\hat \pi_1 = 0$  and $\hat \pi_1 = 1$ because $\lim_{x\rightarrow 0} x \log x = 0$  
 
 
 However, when $\pi_1=0$ or $\pi_1=1$, entropy  is infinite. 



GGHH


```{code-cell} ipython3
# Package imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc, cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize, stats
from scipy.io import loadmat
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from numba import njit


# Plotting parameters
%matplotlib inline
%config InlineBackend.figure_format='retina'

rc('text', usetex=True)

label_size = 20
label_tick_size = 18
title_size = 24
legend_size = 16
text_size = 18

mpl.rcParams['axes.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_tick_size 
mpl.rcParams['xtick.labelsize'] = label_tick_size 
mpl.rcParams['axes.titlesize'] = title_size
mpl.rcParams['legend.fontsize'] = legend_size
mpl.rcParams['font.size'] = text_size
```

```{code-cell} ipython3
# Useful functions
@njit
def ent(π, π_hat):
    """
    Compute the relative entropy of a probability vector `π_hat` with respect to `π`. JIT-compiled using Numba.
    
    """
    ent_val = -np.sum(π_hat * (np.log(π) - np.log(π_hat)))
    
    return ent_val


def T_θ_factory(θ, π):
    """
    Return an operator `T_θ` for a given penalty parameter `θ` and probability distribution vector `π`.
    
    """
    def T_θ(u):  
        """
        Risk-sensitivity operator of Jacobson (1973) and Whittle (1981) taking a function `u` as argument.
        
        """
        return lambda c: -θ * np.log(np.sum(π * np.exp(-u(c) / θ)))
    
    return T_θ


def compute_change_measure(u, c, θ, π):
    """
    Compute the channge of measure given a utility function `u`, a consumption vector `c`,
    a penalty parameter `θ` and a baseline probability vector `π` 
    
    """
    
    m_unnormalized = np.exp(-u(c) / θ)
    m =  m_unnormalized / (π * m_unnormalized).sum()
    return m


def utility_function_factory(α):
    """
    Return a CRRA utility function parametrized by `α` 
    
    """
    if α == 1.:
        return lambda c: np.log(c)
    else: 
        return lambda c: c ** (1 - α) / (1 - α)
```

## Figure 2.1

- We remove the dashed line given that it corresponds to the threshold level of entropy $\eta$ in figure 2.3 and is not mentioned up to this point.
- The figure doesn't contain much information: we propose to vary both $\hat{\pi}_1$ and $\pi_1$ and create an entropy heat map. We create a figure for both entropy and the logarithm of entropy below.

```{code-cell} ipython3
# Specify baseline probability vector `π`
π = np.array([0.5, 0.5])

# Construct grid for `π_hat_0` values
min_prob = 1e-2
π_hat_0_nb = 201
π_hat_0_vals = np.linspace(min_prob, 1 - min_prob, num=π_hat_0_nb)

# Initialize `π_hat` vector with arbitrary values
π_hat = np.empty(2)

# Initialize `ent_vals` to store entropy values
ent_vals = np.empty(π_hat_0_vals.size)

for i in range(π_hat_0_vals.size):  # Loop over all possible values for `π_hat_0` 
    # Update `π_hat` values
    π_hat[0] = π_hat_0_vals[i]
    π_hat[1] = 1 - π_hat_0_vals[i]
    
    # Compute and store entropy value
    ent_vals[i] = ent(π, π_hat)
```

```{code-cell} ipython3
plt.figure(figsize=(5, 3))
plt.plot(π_hat_0_vals, ent_vals, color='blue');
plt.ylabel(r'entropy ($\pi_{1}=%.2f$)' % π[0] );
plt.xlabel(r'$\hat{\pi}_1$');
plt.show()
```

```{code-cell} ipython3
# Use same grid for `π_0_vals` as for `π_hat_0_vals` 
π_0_vals = π_hat_0_vals.copy() 

# Initialize matrix of entropy values
ent_vals_mat = np.empty((π_0_vals.size, π_hat_0_vals.size))

for i in range(π_0_vals.size):  # Loop over all possible values for `π_0` 
    # Update `π` values
    π[0] = π_0_vals[i]
    π[1] = 1 - π_0_vals[i]
    
    for j in range(π_hat_0_vals.size):  # Loop over all possible values for `π_hat_0` 
        # Update `π_hat` values
        π_hat[0] = π_hat_0_vals[j]
        π_hat[1] = 1 - π_hat_0_vals[j]
        
        # Compute and store entropy value
        ent_vals_mat[i, j] = ent(π, π_hat)
```

```{code-cell} ipython3
x, y = np.meshgrid(π_0_vals, π_hat_0_vals)

plt.figure(figsize=(10, 8))
plt.pcolormesh(x, y, ent_vals_mat.T, cmap='seismic', shading='gouraud')
plt.colorbar();
plt.ylabel(r'$\hat{\pi}_1$');
plt.xlabel(r'$\pi_1$');
plt.title('Entropy Heat Map');
plt.show()
```

```{code-cell} ipython3
# Check the point (0.01, 0.9)
π = np.array([0.01, 0.99])
π_hat = np.array([0.9, 0.1])
ent(π, π_hat)
```

```{code-cell} ipython3
plt.figure(figsize=(10, 8))
plt.pcolormesh(x, y, np.log(ent_vals_mat.T), shading='gouraud', cmap='seismic')
plt.colorbar()
plt.ylabel(r'$\hat{\pi}_1$');
plt.xlabel(r'$\pi_1$');
plt.title('Log Entropy Heat Map');
plt.show()
```

## Figure 2.2

- We add a new figure (see below for description) to provide additional insight into the workings of $\textbf{T}$

```{code-cell} ipython3
c_bundle = np.array([2., 1.])  # Consumption bundle
θ_vals = np.array([100, 0.6])  # Array containing the different values of θ
u = utility_function_factory(1.)  # Utility function

# Intialize arrays containing values for Tu(c)
Tuc_vals = np.empty((θ_vals.size, π_hat_0_vals.size))

for i in range(θ_vals.size):  # Loop over θ values
    for j in range(π_hat_0_vals.size):  # Loop over `π_hat_0` values
        # Update `π` values
        π[0] = π_0_vals[j]
        π[1] = 1 - π_0_vals[j]

        # Create T operator
        T = T_θ_factory(θ_vals[i], π)
        
        # Apply T operator to utility function
        Tu = T(u)
        
        # Compute and store Tu(c)
        Tuc_vals[i, j] = Tu(c_bundle)
```

```{code-cell} ipython3
plt.figure(figsize=(10, 8))
plt.plot(π_0_vals, Tuc_vals[0], label=r'$\theta=100$', color='blue');
plt.plot(π_0_vals, Tuc_vals[1], label=r'$\theta=0.6$', color='red');
plt.ylabel(r'$\mathbf{T}u\left(c\right)$');
plt.xlabel(r'$\pi_1$');
plt.legend();
```

We break down the transformation that $\mathbf{T}$ produces in the two plots below. First, $\exp\left(\frac{-u\left(c\right)}{\theta}\right)$ sends $u\left(c\right)$ to a different space where (i) signs are flipped and (ii) curvature is increased in proportion to $\theta$. Expectations are then computed in this transformed space. Notice that the distance between the expectation and the curve is greater in the transformed space than the original space as a result of additional curvature. Finally, $\theta\log E\left[\exp\left(\frac{-u\left(c\right)}{\theta}\right)\right]$ sends the computed expectation back to the original space. Relative to the expected utility case, the distance between the green dot and the orange line reflects the additional adjustment.

```{code-cell} ipython3
# Parameter values
θ= 0.8
π = np.array([0.5, 0.5])
c_bundle = np.array([2., 1.])  # Consumption bundle

# Compute the average consumption level implied by `c_bundle` wrt to `π` 
avg_c_bundle = np.sum(π * c_bundle)

# Construct grid for consumption values
c_grid_nb = 101
c_grid = np.linspace(0.5, 2.5, num=c_grid_nb)

# Evaluate utility function on the grid
u_c_grid = u(c_grid)

# Evaluate utility function at bundle points
u_c_bundle = u(c_bundle)

# Compute the average utility level implied by `c_bundle` wrt to `π` 
avg_u = np.sum(π * u_c_bundle)

# Compute the first transformation exp(-u(c) / θ) for grid values
first_trnsf_u_c_grid = np.exp(-u_c_grid / θ) 

# Compute the first transformation exp(-u(c) / θ) for bundle values
first_trnsf_u_c_bundle = np.exp(-u_c_bundle/θ)

# Take expectations in the transformed space for bundle values (second transformation)
second_trnsf_u_c_bundle = np.sum(π * first_trnsf_u_c_bundle)

# Send expectation back to the original space (third transformation)
third_trnsf_u_c_bundle = -θ * np.log(second_trnsf_u_c_bundle)
```

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

ax1.plot(c_grid, u_c_grid, label=r'$\log\left(c\right)$', color='blue')
ax1.plot(c_bundle, u_c_bundle, color='red')
ax1.plot(avg_c_bundle, third_trnsf_u_c_bundle, 'o', color='green');
ax1.annotate(r'$-\theta\log E\left[\exp\left(\frac{-\log\left(c\right)}{\theta}\right)\right]$',
             (avg_c_bundle+0.05, third_trnsf_u_c_bundle-0.15))

# Consumption dots
ax1.plot(c_bundle[0], u_c_bundle[0], 'o', color='black')
ax1.plot(c_bundle[1], u_c_bundle[1], 'o', color='black')
ax1.annotate(r'$c_1$', (c_bundle[0]-0.08, u_c_bundle[0]+0.03))
ax1.annotate(r'$c_2$', (c_bundle[1]-0.08, u_c_bundle[1]+0.03))
ax1.set_xlabel(r'$c$')

ax1.set_title('Original space')
ax1.legend()

ax2.plot(c_grid, first_trnsf_u_c_grid, label=r'$\exp\left(\frac{-\log\left(c\right)}{\theta}\right)$', color='blue');
ax2.plot(c_bundle, first_trnsf_u_c_bundle, color='red')
ax2.plot(avg_c_bundle, second_trnsf_u_c_bundle, 'o', color='green')
ax2.annotate(r'$E\left[\exp\left(\frac{-\log\left(c\right)}{\theta}\right)\right]$',
             (avg_c_bundle+0.05, second_trnsf_u_c_bundle+0.08))

# Consumption dots
ax2.plot(c_bundle[0], first_trnsf_u_c_bundle[0], 'o', color='black')
ax2.plot(c_bundle[1], first_trnsf_u_c_bundle[1], 'o', color='black')
ax2.annotate(r'$c_1$', (c_bundle[0], first_trnsf_u_c_bundle[0]+0.05))
ax2.annotate(r'$c_2$', (c_bundle[1], first_trnsf_u_c_bundle[1]+0.05))
ax2.set_xlabel(r'$c$')

ax2.set_title('Transformed space')
ax2.legend();
```

## Figure 2.3

- Instead of a dashed line, we use an area fill to visualize the zone where the entropy constraint is satisfied
- We added a dot to represent the solution to the optimization problem

```{code-cell} ipython3
# Parameter 
η = 0.25
π = np.array([0.5, 0.5])
c_bundle = np.array([2., 1.])  # Consumption bundle

# Create η array for putting an upper bound on entropy values
η_line = np.ones_like(π_hat_0_vals.size) * η

# Initialize array for storing values for \hat{E}[u(c)]
E_hat_uc = np.empty(π_hat_0_vals.size)

for i in range(π_hat_0_vals.size):  # Loop over π_hat_0
    # Compute and store \hat{E}[u(c)]
    E_hat_uc[i] = u(c_bundle[1]) + π_hat_0_vals[i] * (u(c_bundle[0]) - u(c_bundle[1]))
    

# Set up a root finding problem to find the solution to the constraint problem
# First argument to `root_scalar` is a function that takes a value for π_hat_0 and returns 
# the entropy of π_hat wrt π minus η
root = optimize.root_scalar(lambda x: ent(π, np.array([x, 1-x])) - η, bracket=[1e-4, 0.5]).root
```

```{code-cell} ipython3
plt.figure(figsize=(12, 8))
plt.plot(π_hat_0_vals, E_hat_uc, label=r'$\hat{E}u\left(c\right)$', color='blue')
plt.fill_between(π_hat_0_vals, η_line, alpha=0.3, label=r'$\mathrm{ent}\left(\pi,\hat{\pi}\right)\leq\eta$',
                 color='gray',
                 linewidth=0.0)
plt.plot(π_hat_0_vals, ent_vals, label=r'$\mathrm{ent}\left(\pi,\hat{\pi}\right)$', color='red')
plt.plot(root, η, 'o', color='black', label='constraint problem solution')
plt.xlabel(r'$\hat{\pi}_1$');
plt.legend();
```

## Figure 2.4

- We add dots for problem solutions

```{code-cell} ipython3
# Parameter values
θ_vals = np.array([0.42, 1.])

# Initialize values for storing the multiplier criterion values
multi_crit_vals = np.empty((θ_vals.size, π_hat_0_vals.size))

for i in range(θ_vals.size):  # Loop over θ values
    for j in range(π_hat_0_vals.size):  # Loop over π_hat_0 values
        # Update `π_hat` values
        π_hat[0] = π_hat_0_vals[j]
        π_hat[1] = 1 - π_hat_0_vals[j]
        
        # Compute distorting measure
        m_i = π_hat / π
        
        # Compute and store multiplier criterion objective value
        multi_crit_vals[i, j] = np.sum(π_hat * (u(c_bundle) + θ_vals[i] * np.log(m_i)))
```

```{code-cell} ipython3
plt.figure(figsize=(12, 8))

# Expected utility values
plt.plot(π_hat_0_vals, E_hat_uc, label=r'$\hat{E}u\left(c\right)$', color='blue')

# First multiplier criterion objective values
plt.plot(π_hat_0_vals, multi_crit_vals[0],
         label=r'$\sum_{i=1}^{I}\pi_{i}m_{i}\left[u\left(c_{i}\right)+0.42\log\left(m_{i}\right)\right]$',
         color='green')

# Second multiplier criterion objective values
plt.plot(π_hat_0_vals, multi_crit_vals[1],
         label=r'$\sum_{i=1}^{I}\pi_{i}m_{i}\left[u\left(c_{i}\right)+\log\left(m_{i}\right)\right]$',
         color='purple')

# Entropy values
plt.plot(π_hat_0_vals, ent_vals, label=r'$\mathrm{ent}\left(\pi,\hat{\pi}\right)$', color='red')

# Area fill
plt.fill_between(π_hat_0_vals, η_line, alpha=0.3, label=r'$\mathrm{ent}\left(\pi,\hat{\pi}\right)\leq\eta$', color='gray')

# Problem solution dots
plt.plot(root, η, 'o', color='black', label='constraint problem solution')
plt.plot(π_hat_0_vals[multi_crit_vals[0].argmin()], multi_crit_vals[0].min(), 'o', label='multiplier problem solution', color='darkred')
plt.plot(π_hat_0_vals[multi_crit_vals[1].argmin()], multi_crit_vals[1].min(), 'o', color='darkred')

plt.xlabel(r'$\hat{\pi}_1$');
plt.legend();
```

## Figure 2.5

The code for this figure is more involved because I formulate a root finding problem for finding indifference curves. Below is a description of the method I used:

**Parameters**

- Consumption bundle $c=\left(1,1\right)$
- Penalty parameter $θ=2$
- Utility function $u=\log$
- Probability vector $\pi=\left(0.5,0.5\right)$

**Algorithm:**
- Compute $\bar{u}=\pi_{1}u\left(c_{1}\right)+\pi_{2}u\left(c_{2}\right)$
- Given values for $c_{1}$, solve for values of $c_{2}$ such that $\bar{u}=u\left(c_{1},c_{2}\right)$:
     - Expected utility: $c_{2,EU}=u^{-1}\left(\frac{\bar{u}-\pi_{1}u\left(c_{1}\right)}{\pi_{2}}\right)$
     - Multiplier preferences: solve $\bar{u}-\sum_{i}\pi_{i}\frac{\exp\left(\frac{-u\left(c_{i}\right)}{\theta}\right)}{\sum_{j}\exp\left(\frac{-u\left(c_{j}\right)}{\theta}\right)}\left(u\left(c_{i}\right)+\theta\log\left(\frac{\exp\left(\frac{-u\left(c_{i}\right)}{\theta}\right)}{\sum_{j}\exp\left(\frac{-u\left(c_{j}\right)}{\theta}\right)}\right)\right)=0$ numerically
     - Constraint preference: solve $\bar{u}-\sum_{i}\pi_{i}\frac{\exp\left(\frac{-u\left(c_{i}\right)}{\theta^{*}}\right)}{\sum_{j}\exp\left(\frac{-u\left(c_{j}\right)}{\theta^{*}}\right)}u\left(c_{i}\right)=0$ numerically where $\theta^{*}$ solves $\sum_{i}\pi_{i}\frac{\exp\left(\frac{-u\left(c_{i}\right)}{\theta^{*}}\right)}{\sum_{j}\exp\left(\frac{-u\left(c_{j}\right)}{\theta^{*}}\right)}\log\left(\frac{\exp\left(\frac{-u\left(c_{i}\right)}{\theta^{*}}\right)}{\sum_{j}\exp\left(\frac{-u\left(c_{j}\right)}{\theta^{*}}\right)}\right)-\eta=0$ numerically.
     

**Note:** It seems that the constraint problem is hard to solve in its original form, i.e. by finding the distorting measure that minimizes the expected utility. It seems that viewing (2.5) as a root finding problem works much better. Notice that 2.5 does not always have a solution. Under $u=\log$, $c_{1}=c_{2}=1$, we have: 

$$\sum_{i}\pi_{i}\frac{\exp\left(\frac{-u\left(c_{i}\right)}{\tilde{\theta}}\right)}{\sum_{j}\pi_{j}\exp\left(\frac{-u\left(c_{j}\right)}{\tilde{\theta}}\right)}\log\left(\frac{\exp\left(\frac{-u\left(c_{i}\right)}{\tilde{\theta}}\right)}{\sum_{j}\pi_{j}\exp\left(\frac{-u\left(c_{j}\right)}{\tilde{\theta}}\right)}\right)=0$$

Guess: the method fails because the derivative of the objective doesn't exist for these parameter choices

**Note 2:** Algorithm is tricky to get to work properly for all values of $c_{1}$. In particular, parameters were chosen with [graduate student descent](https://sciencedryad.wordpress.com/2014/01/25/grad-student-descent/).

```{code-cell} ipython3
def multiplier_criterion_factory(θ, π, u):
    """
    Return a function to compute the multiplier preferences objective function parametrized 
    by a penalty parameter `θ`, a probability vector `π` and a utility function `u`
    
    """
    def criterion(c_1, c_2, return_entropy=False):
        """
        Compute the multiplier preferences objective function and
        associated entropy level if return_entropy=True for a
        consumption bundle (c_1, c_2).
        
        """
        # Compute the distorting measure
        m = compute_change_measure(u, np.array([c_1, c_2]), θ, π)
        
        # Compute objective value
        obj = π[0] * m[0] * (u(c_1) + θ * np.log(m[0])) + π[1] * m[1] * (u(c_2) + θ * np.log(m[1]))
        
        if return_entropy:        
            # Compute associated entropy value
            π_hat = np.array([π[0] * m[0], π[1] * m[1]])
            ent_val = ent(π, π_hat)
        
            return ent_val
        
        else:
            return obj
    
    return criterion


def constraint_criterion_factory(η, π, u):
    """
    Return a function to compute the constraint preferences objective function parametrized 
    by a penalty parameter `η`, a probability vector `π` and a utility function `u`
    
    """
    
    def inner_root_problem(θ_tilde, c_1, c_2):
        """
        Inner root problem associated with the constraint preferences objective function. 
        
        """
        # Compute the change of measure
        m = compute_change_measure(u, np.array([c_1, c_2]), θ_tilde, π)
            
        # Compute the associated entropy value
        π_hat = np.array([π[0] * m[0], π[1] * m[1]])
        ent_val = ent(π, π_hat)
        
        # Compute the error
        Δ = ent_val - η
        
        return ent_val - η
    
    def criterion(c_1, c_2, return_θ_tilde=False):
        try:
            # Solve for the Lagrange multiplier
            res = optimize.root_scalar(inner_root_problem, args=(c_1, c_2), bracket=[3e-3, 10.], method='bisect')
            
            if res.converged:
                θ_tilde = res.root

                # Compute change of measure
                m = compute_change_measure(u, np.array([c_1, c_2]), θ_tilde, π)

                obj = π[0] * m[0] * u(c_1) + π[1] * m[1] * u(c_2)

                if return_θ_tilde: 
                    return θ_tilde

                else: 
                    return obj

            else: 
                return np.nan
            
        except: 
            return np.nan
        
    return criterion


def solve_root_problem(problem, u_bar, c_1_grid, method='bisect', bracket=[0.5, 3.]):
    """
    Solve root finding problem `problem` for all values in `c_1_grid` taking `u_bar` as
    given and using `method`.
    
    """
    
    # Initialize array to be filled with c_2 values
    c_2_grid = np.empty(c_1_grid.size)
    
    for i in range(c_1_grid.size):  # Loop over c_1 values
        c_1 = c_1_grid[i]
        
        try: 
            # Solve root problem given c_1 and u_bar
            res = optimize.root_scalar(problem, args=(c_1, u_bar), bracket=bracket, method=method)

            if res.converged:  # Store values if successfully converged
                c_2_grid[i] = res.root
            else:  # Store np.nan otherwise
                c_2_grid[i] = np.nan
                
        except:
            c_2_grid[i] = np.nan
            
    return c_2_grid
```

```{code-cell} ipython3
# Parameters
c_bundle = np.array([1., 1.])  # Consumption bundle
u_inv = lambda x: np.exp(x)  # Inverse of the utility function
θ = 1.
η = 0.12


# Conustruct grid for c_1
c_1_grid_nb = 102
c_1_grid = np.linspace(0.5, 2., num=c_1_grid_nb)

# Compute \bar{u}
u_bar = u(c_bundle) @ π

# Compute c_2 values for the expected utility case
c_2_grid_EU = u_inv((u_bar - u(c_1_grid) * π[0]) / π[1])

# Compute c_2 values for the multiplier preferences case
multi_pref_crit = multiplier_criterion_factory(θ, π, u)  # Create criterion
multi_pref_root_problem = lambda c_2, c_1, u_bar: u_bar - multi_pref_crit(c_1, c_2)  # Formulate root problem
c_2_grid_mult = solve_root_problem(multi_pref_root_problem, u_bar, c_1_grid)  # Solve root problem for all c_1 values

# Compute c_2 values for the constraint preferences case
constraint_pref_crit = constraint_criterion_factory(η, π, u)  # Create criterion
constraint_pref_root_problem = lambda c_2, c_1, u_bar: u_bar - constraint_pref_crit(c_1, c_2)  # Formulate root problem
# Solve root problem for all c_1 values
c_2_grid_cons = solve_root_problem(constraint_pref_root_problem, u_bar, c_1_grid, method='bisect', bracket=[0.5, 2.5])  

# Compute associated η and θ values
ηs = np.empty(c_1_grid.size)
θs = np.empty(c_1_grid.size)

for i in range(c_1_grid.size):
    ηs[i] = multi_pref_crit(c_1_grid[i], c_2_grid_mult[i], return_entropy=True)
    θs[i] = constraint_pref_crit(c_1_grid[i], c_2_grid_cons[i], return_θ_tilde=True)
    
θs[~np.isfinite(c_2_grid_cons)] = np.nan
```

```{code-cell} ipython3
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True)

ax1.plot(c_1_grid, c_1_grid, '--', color='black')
ax1.plot(c_1_grid, c_2_grid_EU, label='Expected Utility', color='blue')
ax1.plot(c_1_grid, c_2_grid_mult, label='Multiplier Preferences', color='red')
ax1.plot(c_1_grid, c_2_grid_cons, label='Constraint Preferences', color='green')
ax1.plot(1., 1., 'o', color='black')
ax1.set_xlabel(r'$c_1$')
ax1.set_ylabel(r'$c_2$')
ax1.annotate('(1, 1)', (1-0.16, 1-0.02))
ax1.set_ylim(0.5, 2.)
ax1.set_xlim(0.5, 2.)
ax1.legend();

ax2.plot(c_1_grid, ηs, label=r'$\eta^{*}$', color='red')
ax2.plot(c_1_grid, θs, label=r'$\theta^{*}$', color='green')
ax2.set_xlabel(r'$c_1$')
ax2.legend();
```

## Figure 2.6

```{code-cell} ipython3
# Parameters
θ = 2.
η = 0.036
c_bundle = np.array([3., 1.])

# Conustruct grid for c_1
c_1_grid_num = 101
c_1_grid = np.linspace(0.5, 4., num=c_1_grid_num)

# Compute u_bar
u_bar = u(c_bundle) @ π

# Compute c_2 values for the expected utility case
c_2_grid_EU = u_inv((u_bar - u(c_1_grid) * π[0]) / π[1])

# Compute c_2 values for the multiplier preferences case
multi_pref_crit = multiplier_criterion_factory(θ, π, u)  # Create criterion
multi_crit_bar = multi_pref_crit(*c_bundle)  # Evaluate criterion at consumption bundle
multi_pref_root_problem = lambda c_2, c_1, u_bar: u_bar - multi_pref_crit(c_1, c_2)  # Formulate root problem
# Solve root problem for all c_1 values
c_2_grid_mult = solve_root_problem(multi_pref_root_problem, multi_crit_bar, c_1_grid, bracket=[1e-5, 5.])  

# Compute c_2 values for the constraint preferences case
constraint_pref_crit = constraint_criterion_factory(η, π, u)  # Create criterion
cons_crit_bar = constraint_pref_crit(*c_bundle)  # Evaluate criterion at consumption bundle
constraint_pref_root_problem = lambda c_2, c_1, u_bar: u_bar - constraint_pref_crit(c_1, c_2)  # Formulate root problem
# Solve root problem for all c_1 values
c_2_grid_cons = solve_root_problem(constraint_pref_root_problem, cons_crit_bar, c_1_grid, method='bisect', bracket=[0.3, 4.4])  

# Compute associated η and θ values
ηs = np.empty(c_1_grid.size)
θs = np.empty(c_1_grid.size)

for i in range(c_1_grid.size):
    ηs[i] = multi_pref_crit(c_1_grid[i], c_2_grid_mult[i], return_entropy=True)
    θs[i] = constraint_pref_crit(c_1_grid[i], c_2_grid_cons[i], return_θ_tilde=True)
    
θs[~np.isfinite(c_2_grid_cons)] = np.nan
```

```{code-cell} ipython3
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True)

ax1.plot(c_1_grid, c_1_grid, '--', color='black')
ax1.plot(c_1_grid, c_2_grid_EU, label='Expected Utility', color='blue')
ax1.plot(c_1_grid, c_2_grid_mult, label='Multiplier Preferences', color='red')
ax1.plot(c_1_grid, c_2_grid_cons, label='Constraint Preferences', color='green')
ax1.plot(3., 1., 'o', color='black')
ax1.set_xlabel(r'$c_1$')
ax1.set_ylabel(r'$c_2$')
ax1.annotate('(3, 1)', (3, 1+0.07))
ax1.set_ylim(0.75, 4.)
ax1.set_xlim(0.75, 4.)
ax1.legend();

ax2.plot(c_1_grid, ηs, label=r'$\eta^{*}$', color='red')
ax2.plot(c_1_grid, θs, label=r'$\theta^{*}$', color='green')
ax2.set_xlabel(r'$c_1$')
ax2.legend();
```

Note that all three lines of the left graph intersect at (1, 3). While the intersection at (3, 1) is hard-coded, the intersection at (1,3) arises from the computation which is a good sign.

+++

## Figure 2.7

```{code-cell} ipython3
# Parameters
θ = 2.
η = 0.036
c_bundle = np.array([3., 1.])
u = utility_function_factory(1.)
u_prime = lambda c: 1 / c  # Derivative of the utility function

# Compute value of θ at c_1=3
θ = np.interp(3., c_1_grid, θs)

# Compute change of measure
m = compute_change_measure(u, c_bundle, θ, π)

# Compute budget constraint
q = π * m * u_prime(c_bundle)
endowment = (c_bundle * q).sum()
intercept = endowment / q[1]
slope = -q[0] / q[1]
budget_constraint = slope * c_1_grid + intercept
```

```{code-cell} ipython3
plt.figure(figsize=(10, 8))

plt.plot(c_1_grid, c_1_grid, '--', color='black')
plt.plot(c_1_grid, c_2_grid_mult, label='Multiplier Preferences', color='red')
plt.plot(c_1_grid, c_2_grid_cons, label='Constraint Preferences', color='green')
plt.plot(c_1_grid, budget_constraint, label='Budget Constraint', color='darkblue')
plt.plot(3., 1., 'o', color='black')
plt.xlabel(r'$c_1$')
plt.ylabel(r'$c_2$')
plt.annotate('(3, 1)', (3, 1+0.07))
plt.ylim(0.75, 4.)
plt.xlim(0.75, 4.)
plt.legend();
```

## Figure 2.8

```{code-cell} ipython3
# Compute values for the certainty equivalent line 
intercept = 4.  # Intercept value
mask = (1. <= c_1_grid) & (c_1_grid <= 3.)  # Mask to keep only data between c_1=1 and c_1=3

# Initialize array 
certainty_equiv = np.ones(c_1_grid.size) * np.nan

# Fill relevant locations
certainty_equiv[mask] = (intercept - c_1_grid)[mask]


# Set up a fixed point problem to find intersections with x=x line
# Function used to approximate indifference curves using linear interpolation
func_approx = lambda c_1, fp: np.interp(c_1, c_1_grid, fp)  
x0 = 2.  # Initial guess

# Solve for fixed points
fp_CE = optimize.fixed_point(func_approx, x0, args=([certainty_equiv]))
fp_EU = optimize.fixed_point(func_approx, x0, args=([c_2_grid_EU]))
fp_mult = optimize.fixed_point(func_approx, x0, args=([c_2_grid_mult]))
fp_cons = optimize.fixed_point(func_approx, x0, args=([c_2_grid_cons]))
```

```{code-cell} ipython3
plt.figure(figsize=(8, 8))

plt.plot(c_1_grid, c_1_grid, '--', color='black')
plt.plot(c_1_grid, c_2_grid_EU, label='Expected Utility', color='blue')
plt.plot(c_1_grid, c_2_grid_mult, label='Multiplier Preferences', color='red')
plt.plot(c_1_grid, c_2_grid_cons, label='Constraint Preferences', color='green')
plt.plot(c_1_grid, certainty_equiv, color='black')
plt.plot(3., 1., 'o', color='black')
plt.plot(1., 3., 'o', color='black')
plt.xlabel(r'$c_1$')
plt.ylabel(r'$c_2$')
plt.annotate('(3, 1)', (3, 1+0.07))
plt.annotate('(1, 3)', (1+0.02, 3+0.07))
plt.ylim(0.75, 4.)
plt.xlim(0.75, 4.)

plt.plot(fp_CE, fp_CE, 'o')
plt.plot(fp_EU, fp_EU, 'o')
plt.plot(fp_mult, fp_mult, 'o')
plt.plot(fp_cons, fp_cons, 'o')

plt.annotate('A', (fp_CE-0.01, fp_CE+0.06))
plt.annotate('B', (fp_EU-0.01, fp_EU+0.06))
plt.annotate('C', (fp_mult-0.01, fp_mult+0.06))
plt.annotate('D', (fp_cons-0.01, fp_cons+0.06))

plt.legend();
```

## Figure 2.9

- Comment: This figure only uses half of the available space which is inefficient.

```{code-cell} ipython3
# Plotting functions
def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)
    plt.colorbar(lc)
    
    return lc
```

```{code-cell} ipython3
# Parameters
π_1 = 0.3
π_2 = 0.4
π_3 = 1 - π_1 - π_2
π_base = np.array([π_1, π_2, π_3])
c_bundle = np.array([1., 2., 3.])


def contour_plot(α, π_vals_nb=200, levels_nb=20, min_π_val=1e-8):
    """
    Create a contour plot containing iso-utility and iso-entropy curves given utility
    function parameter `α`.
    
    """
    
    # Create utility function
    u = utility_function_factory(α)

    # Initialize arrays 
    π_hat = np.empty(3)
    EU_levels = np.empty((π_vals_nb, π_vals_nb))
    ent_levels = np.empty_like(EU_levels)

    # Create grids for π_hat_1 and π_hat_2 values
    π_hat_1_vals = np.linspace(min_π_val, 1.-min_π_val, π_vals_nb)
    π_hat_2_vals = np.linspace(min_π_val, 1.-min_π_val, π_vals_nb)

    # Evaluate utility function at consumption bundle
    u_c_bundle = u(c_bundle)

    # Loop over all (π_hat_1,  π_hat_2) pairs
    for i, π_hat_1 in enumerate(π_hat_1_vals):
        for j, π_hat_2 in enumerate(π_hat_2_vals):
            # Update π_hat vector with current values
            π_hat[0] = π_hat_1
            π_hat[1] = π_hat_2
            π_hat[2] = 1 - π_hat[0] - π_hat[1]

            # Compute and store expected utility and entropy level if π_hat is a valid probability vector
            if 0. <= π_hat[2] <= 1:
                EU_levels[j, i] = np.sum(π_hat * u_c_bundle)
                ent_levels[j, i] = ent(π_base, π_hat)
            else: 
                EU_levels[j, i] = np.nan
                ent_levels[j, i] = np.nan
                
    # Create grid of θ values
    θ_vals = np.linspace(1e-4, 2., num=50)
    π_hat_coord = np.empty((π_base.size, θ_vals.size))
    
    # For each θ, compute distorted probability distribution
    for i in range(θ_vals.size):  
        m = compute_change_measure(u, c_bundle, θ_vals[i], π_base)
        π_hat_coord[:, i] = π_base * m
    
    # Create contour plot
    plt.figure(figsize=(14, 6))
    plt.contour(π_hat_1_vals, π_hat_2_vals, EU_levels, levels=levels_nb, cmap='spring')
    plt.colorbar()
    plt.contour(π_hat_1_vals, π_hat_2_vals, ent_levels, levels=levels_nb, cmap='winter')
    plt.colorbar()
    colorline(π_hat_coord[0, :], π_hat_coord[1, :], z=θ_vals)
    plt.xlabel(r'$\hat{\pi}_{1}$')
    plt.ylabel(r'$\hat{\pi}_{2}$')
```

```{code-cell} ipython3
α = 0.

contour_plot(α)
```

#### Description of color bars 

First color bar: variation in $\theta$  
Second color bar: variation in utility levels  
Third color bar: variation in entropy levels

+++

Comment: This plot looks quite different from the original one. However, I think this one is more sensible. My reasoning is as follows. When $\theta$ is close to 0, the penalty for distorting the baseline probability measure is very small. Therefore, the optimal distortion should put a lot of weight on the worst-case state. In this case, this translates to a high $\hat{\pi}_{1}$. As this penalty increases, the optimal distortion gets smaller and small and as such, the distorted probability measure is closer` to the baseline one.

+++

## Figure 2.10

```{code-cell} ipython3
α = 3.

contour_plot(α)
```

## Figure 2.11

I compute the best-case and worst-case expected utility by numerically solving optimization problems with respect to the change of measure.

```{code-cell} ipython3
# Parameters
α = 3
u = utility_function_factory(α)
u_c_bundle = u(c_bundle)

# Create grid for η values 
η_vals_nb = 100
η_vals = np.linspace(1e-10, 0.08, η_vals_nb)


# Initialize arrays to be filled by minimum and maximum expected utility values
min_EU = np.empty(η_vals_nb)
min_EU[:] = np.nan
max_EU = min_EU.copy()


@njit
def objective(m_0_and_1, η):
    """
    Compute expected utility with respect to the distorted probability measure
    given the first two values of the change of measure `m_0_and_1`. 
    
    """
    # Back out third implied value for the change of measure
    m_2 = (1 - (π_base[:2] * m_0_and_1).sum()) / π_base[2]
    
    # Compute distorted probability measure π_hat
    m = np.array([m_0_and_1[0], m_0_and_1[1], m_2])
    π_hat = π_base * m
    
    # Compute expected utility wrt π_hat
    EU = np.sum(π_hat * u_c_bundle)
    
    # Return np.inf if entropy constraint is violated
    if ent(π_base, π_hat) > η:
        return np.inf
    
    # Return np.inf if π_hat is not a valid probability vector
    if not ((0. <= π_hat) & (π_hat <= 1.)).all():
        return np.inf
    
    return EU


@njit
def max_obj_wrapper(m_0_and_1, η):
    """
    Wrap `objective` to make it suitable for maximization using minimization routines.
    
    """
    obj_val = objective(m_0_and_1, η)
    if np.isfinite(obj_val):
        return -obj_val
    else:
        return obj_val
```

```{code-cell} ipython3
method = 'Nelder-Mead'
m_0_and_1 = np.ones(2)  # Initial guess

# Compute worst-case expected utility values
for i in range(η_vals_nb):
    opt_res = optimize.minimize(objective, m_0_and_1, method=method, args=(η_vals[i]))
    opt_res = optimize.minimize(objective, opt_res.x, method=method, args=(η_vals[i]))
    if opt_res.success:
        min_EU[i] = opt_res.fun
         
# Compute best-case expected utility values
for i in range(η_vals_nb):
    opt_res = optimize.minimize(max_obj_wrapper, m_0_and_1, method=method, args=(η_vals[i]))
    opt_res = optimize.minimize(max_obj_wrapper, opt_res.x, method=method, args=(η_vals[i]))
    if opt_res.success:
        max_EU[i] = -opt_res.fun
```

```{code-cell} ipython3
# Compute lower bound line
θ = 1.269230769133136
T_θ = T_θ_factory(θ, π_base)
intercept = T_θ(u)(c_bundle)
lower_bound = intercept - θ * η_vals
```

```{code-cell} ipython3
plt.figure(figsize=(8, 6))
plt.plot(η_vals, min_EU, color='blue')
plt.plot(η_vals, max_EU, color='blue')
plt.fill_between(η_vals, min_EU, max_EU, color='lightgray');
plt.plot(η_vals, lower_bound, color='green')
plt.ylabel(r'$E\left[mu\left(c\right)\right]$');
plt.xlabel(r'$\eta$');
```
<!-- 
## Figure 2.12

Density is originally scaled by a number `intconstant`

```{code-cell} ipython3
# Load data
data = loadmat('dataBHS.mat')

# Set parameter values
μ_c = 0.004952
σ_c = 0.005050;
μ_c_tilde = μ_c - σ_c * 0.304569723799467
```

```{code-cell} ipython3
# Compute consumption growth
c = data['c']
c_growth = c[1:] - c[:-1]

# Create histogram of consumption growth
nb_bins = 30
cnt, bins = np.histogram(c_growth, bins=nb_bins)

bins_min = bins.min()
bins_max = bins.max()

# Create grid for PDF values
pdf_x = np.linspace(bins_min, bins_max, num=100)

# Evaluate PDF at grid points
approx = stats.norm(loc=μ_c, scale=σ_c).pdf(pdf_x)
worst_case = stats.norm(loc=μ_c_tilde, scale=σ_c).pdf(pdf_x)
```

```{code-cell} ipython3
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
lns1 = ax.hist(c_growth, bins=bins, alpha=0.5, label='consumption growth (LHS)')
ax.set_ylim(0, 30.)
ax2 = ax.twinx()
lns2 = ax2.plot(pdf_x, approx, color='blue', label='approximating model (RHS)')
lns3 = ax2.plot(pdf_x, worst_case, color='green', label='worst case model (RHS)')
ax2.set_ylim(0, 90.)

lns = [lns1[2][0]]+lns2+lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0);
```

```{code-cell} ipython3
rc('text',usetex=True)
```

```{code-cell} ipython3

``` -->