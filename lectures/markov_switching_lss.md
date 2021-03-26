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

### Copied from v9orig, Jan 6

+++

## Markov Switching Model

$$\begin{eqnarray}
x_{t+1}&=&A_{s\left(t\right)}x_{t}+C_{s\left(t\right)}w_{t+1}\\y_{t}&=&G_{s\left(t\right)}x_{t}\\z_{t}&=&H_{s\left(t\right)}x_{t}
\end{eqnarray}$$

```{code-cell} ipython3
from ipywidgets import interactive_output
import ipywidgets as widgets
from IPython.core.display import display
from IPython.display import display, Latex
import array_to_latex as a2l
import numpy as np
from quantecon import MarkovChain
```

```{code-cell} ipython3
frmt = '{:1.2f}'
TOL = 1e-16
max_itr = 1000


def create_lss_matrices(ρ_00=1.2, ρ_10=-0.3, ρ_01=0.8, ρ_11=0., c_0=0.5, c_1=0.8):
    "Create the Linear State Space system matrices for the Markov switching model"
    
    A_0 = np.array([[ρ_00, ρ_10],
                   [1, 0.]])
    A_1 = np.array([[ρ_01, ρ_11],
                   [1., 0.]])

    C_0 = np.array([[c_0, 0.], 
                   [0., 0.]])

    C_1 = np.array([[c_1, 0.], 
                   [0., 0.]])

    G_0 = np.array([[1., 0.]])

    G_1 = np.array([[1., 0.]])
    
    return A_0, A_1, C_0, C_1, G_0, G_1 


def display_parametrization(lss_matrices):
    "Display a view of the Markov switching model parametrization"
    
    A_0, A_1, C_0, C_1, G_0, G_1 = lss_matrices
    
    A_0_ltx = a2l.to_ltx(A_0, frmt=frmt, print_out=False)
    C_0_ltx = a2l.to_ltx(C_0, frmt=frmt, print_out=False)

    A_1_ltx = a2l.to_ltx(A_1, frmt=frmt, print_out=False)
    C_1_ltx = a2l.to_ltx(C_1, frmt=frmt, print_out=False)

    G_0_ltx = a2l.to_ltx(G_0, frmt=frmt, print_out=False)
    G_1_ltx = a2l.to_ltx(G_1, frmt=frmt, print_out=False)

    lss_ltx = ('\\begin{eqnarray*} x_{t+1} &=& \\begin{cases}' +
        ' %sx_{t}+%sw_{t+1} & \mathrm{if}\,s_{t}=1 \\\  ' +
        ' %sx_{t}+%sw_{t+1} & \mathrm{if}\,s_{t}=2' +
        ' \end{cases} \\\ ' + 
        'y_{t}&=&\\begin{cases} ' + 
        '%sx_{t} & \mathrm{if}\,s_{t}=1 \\\ ' +
        ' %sx_{t} & \mathrm{if}\,s_{t}=2 \\\ ' + 
        ' \end{cases} \\\ ' + 
        ' \end{eqnarray*}') % (A_0_ltx, C_0_ltx, A_1_ltx, C_1_ltx, G_0_ltx, G_1_ltx)

    display(Latex('$\\textit{Linear State Space System}$'))
    display(Latex(lss_ltx))

    
def create_P(P_00=0.5, P_11=0.5):
    "Create the transition matrix P"
    
    P = np.empty((2, 2))
    P[0, 0] = P_00
    P[0, 1] = 1 - P_00
    P[1, 0] = 1 - P_11
    P[1, 1] = P_11
    
    return P


def display_P(P):
    "Display the transition matrix P"

    P_ltx = '$P=' + a2l.to_ltx(P, frmt=frmt, print_out=False) + '$'
    
    display(Latex('$\\textit{Transition Matrix}$'))
    display(Latex(P_ltx))
    
    mc = MarkovChain(P)
    
    display(Latex('$\\textit{Stationary Distributions of } P$'))
    π_infty_ltx = '$\pi_{\infty}=' + a2l.to_ltx(mc.stationary_distributions, frmt=frmt, print_out=False) + '$'
    display(Latex(π_infty_ltx))
    
    
def compute_H(lss_matrices, P, β=0.9):
    "Compute H matrices"
    
    A_0, A_1, C_0, C_1, G_0, G_1 = lss_matrices
    
    H = np.zeros((1, 4))

    G = np.block([[G_0, G_1]])

    P_tilde = np.block([[A_0 * P[0, 0], A_1 * P[1, 0]],
                       [A_0 * P[0, 1], A_1 * P[1, 1]]])
    
    a = (np.eye(4) - β * P_tilde).T
    b = G.T
    H = np.linalg.solve(a, b).T
    
    H_0 = H[:, :2]
    H_1 = H[:, 2:]
    
    return H_0, H_1


def display_dist_xs(lss_matrices, H_0, H_1):
    "Display distributions conditional on x_t and  s_t."
    
    A_0, A_1, C_0, C_1, G_0, G_1 = lss_matrices
    
    Σ_0 = C_0 @ C_0.T
    Σ_1 = C_1 @ C_1.T
    
    # Transform to latex string
    A_0_ltx = a2l.to_ltx(A_0, frmt=frmt, print_out=False)
    C_0_ltx = a2l.to_ltx(C_0, frmt=frmt, print_out=False)

    A_1_ltx = a2l.to_ltx(A_1, frmt=frmt, print_out=False)
    C_1_ltx = a2l.to_ltx(C_1, frmt=frmt, print_out=False)

    G_0_ltx = a2l.to_ltx(G_0, frmt=frmt, print_out=False)
    G_1_ltx = a2l.to_ltx(G_1, frmt=frmt, print_out=False)
    
    H_0_ltx = a2l.to_ltx(H_0, frmt=frmt, print_out=False)
    H_1_ltx = a2l.to_ltx(H_1, frmt=frmt, print_out=False)
    
    Σ_0_ltx = a2l.to_ltx(Σ_0, frmt=frmt, print_out=False)
    Σ_1_ltx = a2l.to_ltx(Σ_1, frmt=frmt, print_out=False)
    
    dist_ltx = ('$\\begin{eqnarray*}' +
            '\\left.x_{t+1}\\right|x_{t},s_{t}&\sim& \\begin{cases}' + 
            '\mathcal{N}\left(%sx_{t},%s\\right) & \mathrm{if}\,s_{t}=1 \\\ ' +
            '\mathcal{N}\left(%sx_{t},%s\\right) & \mathrm{if}\,s_{t}=2' +
            '\end{cases} \\\ ' + 
            '\\left.y_{t}\\right|x_{t},s_{t}&\sim&\\begin{cases}' +
            '\mathcal{N}\left(%sx_{t},0\\right) & \mathrm{if}\,s_{t}=1 \\\ ' + 
            '\mathcal{N}\left(%sx_{t},0\\right) & \mathrm{if}\,s_{t}=2' +
            '\end{cases} \\\ ' + 
            '\\left.z_{t}\\right|x_{t},s_{t}&\sim&\\begin{cases}' +
            '\mathcal{N}\left(%sx_{t},0\\right) & \mathrm{if}\,s_{t}=1 \\\ ' +
            '\mathcal{N}\left(%sx_{t},0\\right) & \mathrm{if}\,s_{t}=2' +
            '\end{cases}' + 
            '\end{eqnarray*}$'
           ) % (A_0_ltx, Σ_0_ltx, A_1_ltx, Σ_1_ltx, G_0_ltx, G_1_ltx, H_0_ltx, H_1_ltx)
    
    display(Latex(dist_ltx))
    
    
def display_π_0(π_0):
    "Display the initial distribution"
    
    π_0_ltx = a2l.to_ltx(π_0, frmt=frmt, print_out=False)
    π_0_ltx = '$\\pi_0=' + π_0_ltx + '$'
    display(Latex('$\\textit{Initial Distribution}$'))
    display(Latex(π_0_ltx))
    
    
def display_π_t(π_t):
    "Display the time t distribution"
    
    π_t_ltx = a2l.to_ltx(π_t, frmt=frmt, print_out=False)
    π_t_ltx = '$\\pi_t=' + π_t_ltx + '$'
    display(Latex('$\\textit{Time } t \\textit{ Distribution}$'))
    display(Latex(π_t_ltx))
    
    
def compute_bars(lss_matrices, H_0, H_1, π_t):
    A_0, A_1, C_0, C_1, G_0, G_1 = lss_matrices
    
    A_bar = π_t[0] * A_0 + π_t[1] * A_1
    G_bar = π_t[0] * G_0 + π_t[1] * G_1
    H_bar = π_t[0] * H_0 + π_t[1] * H_1
    C_bar = π_t[0] * C_0 + π_t[1] * C_1
    
    bars = (A_bar, G_bar, H_bar, C_bar)
    
    return bars


def display_means_x(bars):
    "Display the means of the distributions conditional on x_t only."
    A_bar, G_bar, H_bar, C_bar = bars
                  
    A_bar_ltx = a2l.to_ltx(A_bar, frmt=frmt, print_out=False)
    G_bar_ltx = a2l.to_ltx(G_bar, frmt=frmt, print_out=False)
    H_bar_ltx = a2l.to_ltx(H_bar, frmt=frmt, print_out=False)
    
    mean_ltx = ('$\\begin{eqnarray*}' +
              'E\left.x_{t+1}\\right|x_{t}&=&%sx_{t} \\\ ' +  
              ' E\left.y_{t}\\right|x_{t}&=&%sx_{t} \\\ ' + 
              'E\left.z_{t}\\right|x_{t}&=&%sx_{t}' +
              '\end{eqnarray*}$') % (A_bar_ltx, G_bar_ltx, H_bar_ltx)
    
    display(Latex('$\\textit{Conditional means}$'))
    display(Latex(mean_ltx))


def display_conditional_var_x(lss_matrices, H_0, H_1, bars, π_t):
    "Display the variances of distributions conditional on x_t only."
    A_0, A_1, C_0, C_1, G_0, G_1 = lss_matrices
    A_bar, G_bar, H_bar, C_bar = bars
    
    C_bar_C_bar_t = C_bar @ C_bar.T
    
    dev_A_0 = A_0 - A_bar
    dev_A_1 = A_1 - A_bar
    
    dev_G_0 = G_0 - G_bar
    dev_G_1 = G_1 - G_bar
    
    dev_H_0 = H_0 - H_bar
    dev_H_1 = H_1 - H_bar
    
    dev_A_0_ltx = a2l.to_ltx(dev_A_0, frmt=frmt, print_out=False)
    dev_A_1_ltx = a2l.to_ltx(dev_A_1, frmt=frmt, print_out=False)
    
    dev_G_0_ltx = a2l.to_ltx(dev_G_0, frmt=frmt, print_out=False)
    dev_G_1_ltx = a2l.to_ltx(dev_G_1, frmt=frmt, print_out=False)
    
    dev_H_0_ltx = a2l.to_ltx(dev_H_0, frmt=frmt, print_out=False)
    dev_H_1_ltx = a2l.to_ltx(dev_H_1, frmt=frmt, print_out=False)
    
    C_bar_C_bar_t_ltx = a2l.to_ltx(C_bar_C_bar_t, frmt=frmt, print_out=False)
    
    π_t_0 = π_t[0].round(2)
    π_t_1 = π_t[1].round(2)
    
    var_ltx = ('$\\begin{eqnarray*}' +
          "E\left.\left(x_{t+1}-\\bar{A}_{t}x_{t}\\right)\left(x_{t+1}-\\bar{A}_{t}x_{t}\\right)' \\right|x_{t}&=&" +
          "%s%sx_{t}x_{t}'%s'+%s%sx_{t}x_{t}'%s'+%s \\\ " +  
          "E\left.\left(y_{t}-\\bar{G}_{t}x_{t}\\right)\left(y_{t}-\\bar{G}_{t}x_{t}\\right)' \\right|x_{t}&=&" +
          "%s%sx_{t}x_{t}'%s'+%s%sx_{t}x_{t}'%s' \\\ " + 
          "E\left.\left(z_{t}-\\bar{H}_{t}x_{t}\\right)\left(z_{t}-\\bar{H}_{t}x_{t}\\right)' \\right|x_{t}&=&" +
           "%s%sx_{t}x_{t}'%s'+%s%sx_{t}x_{t}'%s' \\\ "  +
          '\end{eqnarray*}$') % \
    (π_t_0, dev_A_0_ltx, dev_A_0_ltx, π_t_1, dev_A_1_ltx, dev_A_1_ltx, C_bar_C_bar_t_ltx,
     π_t_0, dev_G_0_ltx, dev_G_0_ltx, π_t_1, dev_G_1_ltx, dev_G_1_ltx,
     π_t_0, dev_H_0_ltx, dev_H_0_ltx, π_t_1, dev_H_1_ltx, dev_H_1_ltx)
    
    display(Latex('$\\textit{Conditional variances}$'))
    display(Latex(var_ltx))
    
    
def check_H_bar(bars, β):
    "Check `H_bar == G_bar @ (np.eye(2) - β * A_bar)`"    
    A_bar, G_bar, H_bar, C_bar = bars
    
    H_bar_ltx = a2l.to_ltx(H_bar, frmt=frmt, print_out=False)
    H_bar_ltx = '$\\bar{H}_{t}=' + H_bar_ltx + '$'
    
    RHS = np.linalg.solve((np.eye(2) - β * A_bar).T, G_bar.T).T
    RHS_ltx = a2l.to_ltx(RHS, frmt=frmt, print_out=False)
    RHS_ltx = '$\\bar{G}_{t}\left(I-\\beta\\bar{A}_{t}\\right)^{-1}=' + RHS_ltx + '$'
    
    display(Latex(r'$\textit{Check } \bar{H}_{t}=\bar{G_t}\left(I-\beta\bar{A}_{t}\right)^{-1}$'))
    display(Latex(H_bar_ltx))
    display(Latex(RHS_ltx))    

    
def display_dist_x(lss_matrices, H_0, H_1, P, t, β, π_t):
    "Display distributions conditional on x_t only"
    display_π_t(π_t)
    
    bars = compute_bars(lss_matrices, H_0, H_1, π_t)
    
    display_means_x(bars)
    
    display_conditional_var_x(lss_matrices, H_0, H_1, bars, π_t)
    
    check_H_bar(bars, β)
    
    
def compute_Σ(lss_matrices, π):
    "Compute the covariance matrix Σ given π."
    A_0, A_1, C_0, C_1, G_0, G_1 = lss_matrices
    A_bar = π[0] * A_0 + π[1] * A_1
    
    error = TOL + 1
    i = 0
    Σ = np.zeros((2, 2))

    while error > TOL and i < max_itr:
        Σ_new = (π[0] * ((A_0 - A_bar) @ Σ @ (A_0 - A_bar).T + C_0 @ C_0.T)
                 + π[1] * ((A_1 - A_bar) @ Σ @ (A_1 - A_bar).T + C_1 @ C_1.T))
        
        error = np.max(np.abs(Σ_new - Σ))
        Σ[:] = Σ_new[:]
        i += 1

    if i == max_itr:
        display(Latex("Computation of $\Sigma$ failed to converge"))
        
    return Σ

    
def display_Σ_bar(Σ_bar, C_bar):
    "Display the variance of the stationary distribution"
                  
    Σ_bar_ltx = a2l.to_ltx(Σ_bar, frmt=frmt, print_out=False)
    Σ_bar_ltx = '$\\bar{\Sigma}=' + Σ_bar_ltx + '$'
    display(Latex('$\\textit{Stationary Covariance Matrix}$'))
    display(Latex(Σ_bar_ltx))
    
    C_bar_C_bar = C_bar @ C_bar.T
    C_bar_ltx = a2l.to_ltx(C_bar_C_bar, frmt=frmt, print_out=False)
    C_bar_ltx = "$\\bar{C}\\bar{C}'=" + C_bar_ltx + "$"
    display(Latex(C_bar_ltx))
    
    
def plot(ρ_00, ρ_10, ρ_01, ρ_11, c_0, c_1, P_00, P_11, β, π_00, t):
    "Update the Markov switching model dashboard"
    
    ### View of the model parametrization
    display(Latex('$\\textbf{View of the Markov Switching Model Parametrization}$'))
    
    lss_matrices = create_lss_matrices(ρ_00, ρ_10, ρ_01, ρ_11, c_0, c_1)
    A_0, A_1, C_0, C_1, G_0, G_1 = lss_matrices
    display_parametrization(lss_matrices)
    
    # Transition matrix
    P = create_P(P_00, P_11)
    display_P(P)
    
    # Initial distribution
    π_0 = np.array([π_00, 1 - π_00])
    display_π_0(π_0)
        
    # Compute H_0 and H_1
    H_0, H_1 = compute_H(lss_matrices, P, β)
    
    ### Distributions conditional on (x_t, s_t)
    display(Latex('$\\textbf{Distributions Conditional on } (x_{t},s_{t})$'))
    display_dist_xs(lss_matrices, H_0, H_1)
    
    ### Distributions conditional on (x_t) only
    display(Latex('$\\textbf{Distributions Conditional on } x_{t} \\textbf{ Only}$'))
    display(Latex('(Computed using the time $t$ distribution)'))
    P_t = np.linalg.matrix_power(P, t)
    π_t = π_0 @ P_t
    display_dist_x(lss_matrices, H_0, H_1, P, t, β, π_t)

    ### Stationary distributions
    display(Latex('$\\textbf{Stationary Distribution}$'))
    
    # Use first stationary distribution
    display(Latex('(Computed using the first stationary distribution)'))
    π_infty = MarkovChain(P).stationary_distributions[0]
    
    # Stationary mean μ
    A_bar_infty, _, _, C_bar_infty = compute_bars(lss_matrices, H_0, H_1, π_infty)
    μ = np.linalg.solve((np.eye(2) - A_bar_infty), np.zeros(2))
    μ_ltx = a2l.to_ltx(μ, frmt=frmt, print_out=False)
    μ_ltx = '$\\mu=' + μ_ltx + '$'
    display(Latex('$\\textit{Stationary Mean}$'))
    display(Latex(μ_ltx))
    
    # Stationary covariance matrix Σ_bar
    Σ_bar = compute_Σ(lss_matrices, π_infty)
    display_Σ_bar(Σ_bar, C_bar_infty)
```

```{code-cell} ipython3
# Setup dashboard
ρ_00 = widgets.FloatText(value=1.2, step=0.05, description='$ρ_{1,1}$')
ρ_01 = widgets.FloatText(value=-0.3, step=0.05, description='$ρ_{1,2}$')
ρ_10 = widgets.FloatText(value=0.8, step=0.05, description='$ρ_{2,1}$')
ρ_11 = widgets.FloatText(value=0., step=0.05, description='$ρ_{2,2}$')
c_0 = widgets.FloatText(value=0.5, step=0.05, description='$c_1$')
c_1 = widgets.FloatText(value=0.8, step=0.05, description='$c_2$')
P_00 = widgets.FloatSlider(min=0., max=1, step=0.01, value=0.5, description='$P_{11}$')
P_11 = widgets.FloatSlider(min=0., max=1., step=0.01, value=0.5, description='$P_{22}$')
β = widgets.FloatSlider(min=0., max=1., step=0.01, value=0.95, description='$\\beta$')
π_00 = widgets.FloatSlider(min=0., max=1., step=0.01, value=0.5, description='$\pi_{0,1}$')
t = widgets.IntSlider(min=0., max=20, value=0., description='$t$')

left_box = widgets.VBox([widgets.Label('Coefficient Parameters'), ρ_00, ρ_01, ρ_10, ρ_11])
middle_box = widgets.VBox([widgets.Label('Variance Parameters'), c_0, c_1, widgets.Label('Discount Factor'), β])
right_box = widgets.VBox([widgets.Label('Markov Chain Parameters'), P_00, P_11, π_00, widgets.Label('Time Period'), t])
ui = widgets.HBox([left_box, middle_box, right_box])
        
controls = {'ρ_00': ρ_00,
            'ρ_10': ρ_01,
            'ρ_01': ρ_10,
            'ρ_11': ρ_11,
            'c_0': c_0,
            'c_1': c_1,
            'P_00': P_00,
            'P_11': P_11,
            'β': β,
            'π_00': π_00,
            't': t
           }

hh = interactive_output(plot, controls)
```

```{code-cell} ipython3
# Display dashboard
display(Latex('$\large\\textbf{Markov Switching Model Dashboard}$'))
display(Latex('$\\textbf{Model Parameters}$'))
display(ui, hh)
```

## Tests

The sections below compute various objects using different methods to ensure that the computation is correct.

+++

### Computing $H_0$ and $H_1$

#### Solve by iteration

Let $H=\left[\begin{array}{cc}
H_{1} & H_{2}\end{array}\right]$, $G=\left[\begin{array}{cc}
G_{1} & G_{2}\end{array}\right] $ and $\tilde{P}=\left[\begin{array}{cc}
A_{1}P_{11} & A_{2}P_{21}\\
A_{1}P_{12} & A_{2}P_{22}
\end{array}\right]$. We iterate on: 

$$H_{j+1}=G+\beta H_{j}\tilde{P}$$

until $\left\Vert H_{j+1}-H_{j}\right\Vert _{\infty}\leq10^{-16}$.

```{code-cell} ipython3
lss_matrices = create_lss_matrices()
A_0, A_1, C_0, C_1, G_0, G_1 = lss_matrices
P = create_P()
β = 0.95

TOL = 1e-16
max_itr = 1000 

H = np.zeros((1, 4))

G = np.block([[G_0, G_1]])

P_tilde = np.block([[A_0 * P[0, 0], A_1 * P[1, 0]],
                   [A_0 * P[0, 1], A_1 * P[1, 1]]])

for i in range(max_itr):
    H_next = G + β * H @ P_tilde

    Δ = np.max(np.abs(H_next - H))
    
    H = H_next
    
    if Δ < TOL:
        break
    
H_0 = H[:, :2]
H_1 = H[:, 2:]
    
# Print
display(Latex(r'$H_1=' + a2l.to_ltx(H_0, frmt='{:1.4f}', print_out=False) + '$'))
display(Latex(r'$H_2=' + a2l.to_ltx(H_1, frmt='{:1.4f}', print_out=False) + '$'))
```

#### Solve linear system

For numerical reasons, we solve:

$$\left(I-\beta\tilde{P}\right)'H'=G'$$

```{code-cell} ipython3
a = (np.eye(4) - β * P_tilde).T
b = G.T
H_alt = np.linalg.solve(a, b).T
```

```{code-cell} ipython3
# Check that both solutions give the same result
np.max(np.abs(H - H_alt))
```

## Why naive vertical stacking doesn't work

Let $H=\left[\begin{array}{c}
H_{1}\\
H_{2}
\end{array}\right]=\left[\begin{array}{cc}
H_{11} & H_{12}\\
H_{21} & H_{22}
\end{array}\right]$, $A_{i}=\left[\begin{array}{cc}
A_{11}^{i} & A_{12}^{i}\\
A_{21}^{i} & A_{22}^{i}
\end{array}\right]$ for $i\in\left\{ 1,2\right\} $ and $P=\left[\begin{array}{cc}
P_{11} & P_{12}\\
P_{21} & P_{22}
\end{array}\right]$. 

Consider only what would need to happen to $H_{11}$ when we apply a transformation in a given iteration. Essentially, what we are looking for is a transformation that yields:

$$\left(H_{11}A_{11}^{1}+H_{12}A_{21}^{1}\right)P_{11}+\left(H_{21}A_{11}^{2}+H_{22}A_{21}^{2}\right)P_{12}$$

Notice that both $H_{11}$ and $H_{22}$ show up here. Pre-multiplying $H$ by some matrix will yield a transformation of the form $aH_{11}+bH_{21}$. Meanwhile, post-multiplication will yield a transformation of the form $cH_{11}+dH_{12}$. Therefore, it's clear that none of these would work because none contain $H_{22}$. However, when we flatten the $H$ matrix, we can apply a transformation of the form $a'H_{11}+b'H_{12}+c'H_{21}+d'H_{22}$. In this sense, vertical and horizontal stacking are not symmetric operations because horizontal stacking gives us more degrees of freedom for applying linear transformations.

+++

## Simulation

```{code-cell} ipython3
from quantecon import MarkovChain

mc = MarkovChain(P)

T = 1_000_000

init = 0

s = mc.simulate(T, init=init, random_state=1)

random_state = np.random.RandomState(1)

w = random_state.randn(T, 2, 1)

π_0 = mc.stationary_distributions[0]
```

### Compute objects

```{code-cell} ipython3
from numba import njit


@njit
def simulate(lss_matrices, π_0, s): 
    A_0, A_1, C_0, C_1, G_0, G_1 = lss_matrices
    
    x = np.zeros((T, 2, 1))
    z = np.zeros((T))
    y = np.zeros(T)
    π = np.zeros((T, 2))
    π[0] = π_0

    A_bar = np.zeros((T, *A_0.shape))
    C_bar = np.zeros((T, *C_0.shape))
    G_bar = np.zeros((T, *G_0.shape))
    H_bar = np.zeros((T, *H_0.shape))

    E_x = np.zeros_like(x)
    E_y = np.zeros_like(y)
    E_z = np.zeros_like(z)

    Σ_x = np.zeros((T, 2, 2))
    Σ_y = np.zeros(T)
    Σ_z = np.zeros(T)

    for t in range(T):
        if s[t] == 0:
            A = A_0
            C = C_0
            G = G_0
            H = H_0
        else:
            A = A_1
            C = C_1
            G = G_1
            H = H_1

        A_bar[t] = π[t, 0] * A_0 + π[t, 1] * A_1
        C_bar[t] = π[t, 0] * C_0 + π[t, 1] * C_1
        G_bar[t] = π[t, 0] * G_0 + π[t, 1] * G_1
        H_bar[t] = π[t, 0] * H_0 + π[t, 1] * H_1

        if t < T - 1:
            x[t+1] = A @ x[t] + C @ w[t+1]
            π[t+1] = π[t] @ P
            E_x[t+1] = A_bar[t] @ x[t]

        y[t] = (G @ x[t])[0, 0]
        z[t] = (H @ x[t])[0, 0]

        E_y[t] = (G_bar[t] @ x[t])[0, 0]
        E_z[t] = (H_bar[t] @ x[t])[0, 0]

        Σ_x[t] = (π[t, 0] * ((A_0 - A_bar[t]) @ x[t] @ x[t].T @ (A_0 - A_bar[t]).T + C_0 @ C_0.T) +
               π[t, 1] * ((A_1 - A_bar[t]) @ x[t] @ x[t].T @ (A_1 - A_bar[t]).T + C_1 @ C_1.T)
              )

        Σ_y[t] = (π[t, 0] * (G_0 - G_bar[t]) @ x[t] @ x[t].T @ (G_0 - G_bar[t]).T +
               π[t, 1] * (G_1 - G_bar[t]) @ x[t] @ x[t].T @ (G_1 - G_bar[t]).T
              )[0, 0]

        Σ_z[t] = (π[t, 0] * (H_0 - H_bar[t]) @ x[t] @ x[t].T @ (H_0 - H_bar[t]).T +
               π[t, 1] * (H_1 - H_bar[t]) @ x[t] @ x[t].T @ (H_1 - H_bar[t]).T
              )[0, 0]
    
    variables = (x, y, z)
    matrices = (A_bar, C_bar, G_bar, H_bar)
    expectations = (E_x, E_y, E_z)
    covariances = (Σ_x, Σ_y, Σ_z)
    
    return (variables, matrices, expectations, covariances)
```

```{code-cell} ipython3
((x, y, z), (A_bar, C_bar, G_bar, H_bar), (E_x, E_y, E_z), (Σ_x, Σ_y, Σ_z)) = simulate(lss_matrices, π_0, s)
```

## Convergence of $\left(T+1\right)^{-1}\sum_{t=0}^{T+1}\Sigma_{t,x}$

```{code-cell} ipython3
import matplotlib.pyplot as plt


Σ_x_rolling_mean = Σ_x.cumsum(axis=0) / np.arange(1, T+1).reshape((-1, 1, 1))
Ω_bar = Σ_x_rolling_mean[-1]

Δ = np.max(np.abs(Σ_x_rolling_mean[1:] - Σ_x_rolling_mean[:-1]), axis=(1, 2))
plt.plot(Δ);
plt.title(r'$\left\Vert \left(T+1\right)^{-1}\sum_{t=0}^{T+1}\Sigma_{t,x}-T^{-1}\sum_{t=0}^{T}\Sigma_{t,x}\right\Vert _{\infty}$',
          fontsize=22);
```

# Plot simulated time paths

```{code-cell} ipython3
import matplotlib.pyplot as plt


%config InlineBackend.figure_format = 'retina'
%matplotlib inline

fig, ax = plt.subplots(3, 2, figsize=(16, 14))

ax[0, 0].plot(y, label=r'$y$');
ax[0, 0].plot(E_y, label=r'$E\left[\left.y_{t}\right|x_{t}\right]$')
ax[0, 0].set_title(r'Time path of $y$');
ax[0, 0].legend();

ax[0, 1].plot(s + 1, 'o', markersize=1.);
ax[0, 1].set_title(r'Time path of $s$');

ax[1, 0].plot(x[1:, 0, 0], label=r'$x_{1, t}$');
ax[1, 0].plot(E_x[1:, 0, 0], label=r'$E\left[\left.x_{1,t}\right|x_{t-1}\right]$')
ax[1, 0].set_title(r'Time path of $x_1$');
ax[1, 0].legend();

ax[1, 1].plot(x[1:, 1, 0], label=r'$x_{2, t}$');
ax[1, 1].plot(E_x[:, 1, 0], label=r'$E\left[\left.x_{2,t}\right|x_{t-1}\right]$')
ax[1, 1].set_title(r'Time path of $x_2$');
ax[1, 1].legend();

ax[2, 0].plot(z, label=r'$z_t$')
ax[2, 0].plot(E_z, label=r'$E\left[\left.z_{t}\right|x_{t}\right]$');
ax[2, 0].set_title(r'Time path of $z$');
ax[2, 0].legend();

ax[2, 1].plot(w[:, :, 0]);
ax[2, 1].set_title(r'Time path of $w$');

```

```{code-cell} ipython3
fig, ax = plt.subplots(3, 2, figsize=(16, 14))

ax[0, 0].plot(Σ_x[:, 0, 0]);
ax[0, 0].set_title(r'Time path of $\Sigma_{x}^{\left(1,1\right)}$');

ax[0, 1].plot(Σ_x[:, 1, 0]);
ax[0, 1].set_title(r'Time path of $\Sigma_{x}^{\left(2,1\right)}$');

ax[1, 0].plot(Σ_x[:, 1, 0]);
ax[1, 0].set_title(r'Time path of $\Sigma_{x}^{\left(1,2\right)}$');

ax[1, 1].plot(Σ_x[:, 1, 1]);
ax[1, 1].set_title(r'Time path of $\Sigma_{x}^{\left(2,2\right)}$');

ax[2, 0].plot(Σ_y);
ax[2, 0].set_title(r'Time path of $\Sigma_{y}$');

ax[2, 1].plot(Σ_z);
ax[2, 1].set_title(r'Time path of $\Sigma_{z}$');
```

# Least Squares Regressions

+++

## Regression of $y_t$ on $x_t$

```{code-cell} ipython3
from statsmodels.regression.linear_model import OLS

mask = s == 0

reg = OLS(y[mask], x[mask, :, 0])
result = reg.fit()

# Test regression against population value
print('Estimated G_0:', result.params)
print('True G_0:', G_0[0])
```

```{code-cell} ipython3
mask = s == 1

reg = OLS(y[mask], x[mask, :, 0])
result = reg.fit()

# Test regression against population value
print('Estimated G_1:', result.params)
print('True G_1:', G_1[0])
```

```{code-cell} ipython3
reg = OLS(y, x[:, :, 0])
result = reg.fit()

# Test regression against population value
print('Estimated G_bar:', result.params)
print('True G_bar:', G_bar[0, 0])
```

## Regression of $z_t$ on $x_t$

```{code-cell} ipython3
mask = s == 0

reg = OLS(z[mask], x[mask, :, 0])
result = reg.fit()

# Test regression against population value
print('Estimated H_0:', result.params)
print('True H_0:', H_0[0])
```

```{code-cell} ipython3
mask = s == 1

reg = OLS(z[mask], x[mask, :, 0])
result = reg.fit()

# Test regression against population value
print('Estimated H_1:', result.params)
print('True H_1:', H_1[0])
```

```{code-cell} ipython3
reg = OLS(z, x[:, :, 0])
result = reg.fit()

# Test regression against population value
print('Estimated H_bar:', result.params)
print('True H_bar:', H_bar[0, 0])
```

## Regression of $x_{t+1}$ on $x_t$

```{code-cell} ipython3
from statsmodels.tsa.api import VAR

np.set_printoptions(suppress=True)

var = VAR(x[:, :, 0])
result = var.fit(maxlags=1, trend='nc')
Ω_bar_hat = result.resid.T @ result.resid / (T - 1 - x.shape[1])

# Test regression against population value
print('Estimated Ω_bar:', Ω_bar_hat)
print('Approximate Ω_bar:', Ω_bar)
```

### VAR Results

L1.y1 is the first order coefficient estimate and L1.y2 is the second order coefficient estimate.

```{code-cell} ipython3
result.summary()
```

#### OLS Regression of the state $s_t$ (scaled by a factor of 39) on $x_t$

```{code-cell} ipython3
ols = OLS(s * 39, x[:, :, 0])
result = ols.fit()

result.summary()
```

### Sample stationary covariance

```{code-cell} ipython3
sample_resid = x[1:] - A_bar[-1] @ x[:-1]

np.cov(sample_resid[:, 0, 0], sample_resid[:, 1, 0])
```

Note to Tom: there seems to be some small notational issues in the textbook. On page 127, you define

\begin{eqnarray*}
z_{t} & = & E\left[\left.\sum_{t=0}^{\infty}\delta^{t}y_{t+j}\right|x_{t},s_{t}\right]\\
\tilde{z}_{t} & = & E\left[\left.\sum_{t=0}^{\infty}\delta^{t}y_{t+j}\right|x_{t}\right]
\end{eqnarray*}

The index of the sum should be $j$.

I think it is somewhat tricky to keep track of what is a random variable and what is fixed. On page 129, you compute the distribution of $z_t$ conditional on $x_t$ and $s_t$, but $z_t$ already is an expectation conditional on $x_t$ and $s_t$. On page 130, you compute $\mathrm{var}\left(\left.z_{t}\right|x_{t}\right)$. In this computation, $s_t$ is a random variable, but $x_t$ is fixed. Some warnings might be helpful.

On page 130, there's a missing $t$ subscript in the first equation. The same is true for the analog equation for $y_t$. 

It might be useful to elaborate on what is meant by "well approximated". For instance, one criterion for measuring the quality of the approximation could be the proportion of time periods $t$ such that $A_{s_{t}}=\bar{A}$. Under this criterion and a nontrivial parametrization, you could not do worse than the suggested approximation.

+++

### Quentin -some answers

I have tried to fix some of my errors and added corrections to the text. I will put a revised version in the Quention_2020 folder this evening.  

I couldn't find the missing subscript t that you spotted on page 130.  Maybe we can discuss this
Tuesday (tomorrow) morning if you still feel that is worthwhile.  It might help me to be talked through a few things in this notebook.


```{code-cell} ipython3
# LSS Matrices
lss_matrices = create_lss_matrices()
A_0, A_1, C_0, C_1, G_0, G_1 = lss_matrices

# Transition matrix
P = create_P()

# Initial distribution
π_infty = MarkovChain(P).stationary_distributions[0]

# Compute H_0 and H_1
H_0, H_1 = compute_H(lss_matrices, P, 0.95)

# Stationary mean μ
A_bar_infty, _, _, C_bar_infty = compute_bars(lss_matrices, H_0, H_1, π_infty)
μ = np.linalg.solve((np.eye(2) - A_bar_infty), np.zeros(2))

# Stationary covariance matrix Σ_bar
Σ_bar = compute_Σ(lss_matrices, π_infty)
```

## Mean of $\Sigma_{z}\left(x_{t}\right)$

```{code-cell} ipython3
Σ_z.mean()
```

```{code-cell} ipython3
Σ_bar 
```

## Autoregression of $m_t$

```{code-cell} ipython3
from statsmodels.tsa.api import AutoReg

lags = 6  # Number of lags
ar = AutoReg(x[:, 0, 0], lags)
result = ar.fit()
```

```{code-cell} ipython3
# A_0
lss_matrices[0]
```

```{code-cell} ipython3
# A_1
lss_matrices[1]
```

```{code-cell} ipython3
result.summary()
```

```{code-cell} ipython3
lss_matrices = create_lss_matrices(ρ_00=1.2, ρ_10=-0.2, ρ_01=0.2, ρ_11=0.)
((x, y, z), (A_bar, C_bar, G_bar, H_bar), (E_x, E_y, E_z), (Σ_x, Σ_y, Σ_z)) = simulate(lss_matrices, π_0, s)
```

```{code-cell} ipython3
# A_0
lss_matrices[0]
```

```{code-cell} ipython3
# A_1
lss_matrices[1]
```

```{code-cell} ipython3
lags = 12  # Number of lags
ar = AutoReg(x[:, 0, 0], lags)
result = ar.fit()
```

```{code-cell} ipython3
result.summary()
```

```{code-cell} ipython3
P = create_P(P_00=0.3, P_11=0.1)
mc = MarkovChain(P)

T = 1_000_000

init = 0

s = mc.simulate(T, init=init, random_state=0)
π_0 = mc.stationary_distributions[0]
```

```{code-cell} ipython3
# Transition matrix
P
```

```{code-cell} ipython3
((x, y, z), (A_bar, C_bar, G_bar, H_bar), (E_x, E_y, E_z), (Σ_x, Σ_y, Σ_z)) = simulate(lss_matrices, π_0, s)
```

```{code-cell} ipython3
lags = 12  # Number of lags
ar = AutoReg(x[:, 0, 0], lags)
result = ar.fit()
```

```{code-cell} ipython3
result.summary()
```

Under the default parametrization, the higher order coefficients appear to be zero. We can obtain non-zero coefficients by modifying the transition matrix. 

To understand this result, notice that an $s_t$ process driven by $P=\left[\begin{array}{cc}
0.5 & 0.5\\
0.5 & 0.5
\end{array}\right]$ is statistically indistinguishable from an i.i.d process. In other words, $s_{t-1}$ contains no useful information for predicting $s_{t}$. As such, any information that the history of $x_t$ might reveal about $s_{t-1}$ is useless for predicting $s_{t}$. 

This is not the case when the transition matrix does not contain identical rows, that is, when the distribution of $s_t$ depends on $s_{t-1}$. In this case, information about $s_{t-1}$ is useful for predicting $s_t$.

```{code-cell} ipython3
lss_matrices = create_lss_matrices(ρ_00=0.9, ρ_10=-0.3, ρ_01=0.9, ρ_11=-0.3, c_0=1e-4, c_1=100.)
((x, y, z), (A_bar, C_bar, G_bar, H_bar), (E_x, E_y, E_z), (Σ_x, Σ_y, Σ_z)) = simulate(lss_matrices, π_0, s)
```

```{code-cell} ipython3
# A_0
lss_matrices[0]
```

```{code-cell} ipython3
# A_1
lss_matrices[1]
```

```{code-cell} ipython3
# C_0
lss_matrices[2]
```

```{code-cell} ipython3
# C_1
lss_matrices[3]
```

```{code-cell} ipython3
lags = 12  # Number of lags
ar = AutoReg(x[:, 0, 0], lags)
result = ar.fit()
```

```{code-cell} ipython3
result.summary()
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
