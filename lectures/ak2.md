---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Auerbach-Kotlikoff 2 period model

```{code-cell} ipython3
!pip install numba
!pip install quantecon
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from quantecon.optimize import brent_max
```



# Introduction



This lecture computes  transition paths of the two-period life cycle OLG economy described in chapter 2 of Auerback and 
Kotlikoff (1987)
{cite}`auerbach1987dynamic`.



We will first solve for equilibrium paths using the closed form solution that we derived in the class. And then, let's pretend that we don't know the closed form solution, and solve for the transitions paths by iterating over the guesses of price sequences and tax rate sequence. The equilibrium paths will be found as a fixed point.



## Closed form dynamics

Auerback and 
Kotlikoff (1987)
{cite}`auerbach1987dynamic` construct a 
a two-period model
in which both the utility and production functions are Cobb-Douglas, so that 


$$
U_t  = C_{yt}^\beta C_{o,t+1}^{1-\beta}, \quad \beta \in (0,1)
$$ (eq:utilfn)

$$
Y_t  = K_t^\alpha L_t^{1-\alpha}, \quad \alpha \in (0,1)
$$ (eq:prodfn)



Equation {eq}`eq:utilfn`  expresses the lifetime utility of a person who is young at time $t$ 
as a function of consumption  $C_{yt} $ when young and consumption $C_{o,t+1}$ when
old.  


Production function {eq}`eq:prodfn` relates output $Y_t$ per young
worker  to capital  $K_t$ per young worker and labor $L$ per young worker; 
 $L$ is  supplied  inelasticallly by each young worker and is measured in
units that make  $L = 1$. 

The lifetime budget constraint
of a young person at time $t$ is

$$ 
C_{yt} + \frac{C_{ot+1}}{1 + r_{t+1}} = W_t
$$ (eq:lifebudget)

where $W_t$ is the wage rate at time  $t$  and $r_{t+1}$  is the net  return
on savings between $t$ and $t+1$. 

Equation {eq}`eq:lifebudget` states that the present value of consumption
equals the present value of labor earnings.

Another way to write the lifetime budget constraint at equality is  



$$ 
C_{ot+1} = A_{t+1} (1 + r_{t+1}) 
$$ (eq:lifbudget2)

where assets $A_{t+1}$ accumulated by old people at the beginning of time $t+1$   equals their savings $W_t - C_{yt}$ at time $t$  when they were young. 


Maximization of  utility function {eq}`eq:utilfn` subject to budget constraint {eq}`eq:lifebudget` implies that consumption when young is
 
 $$ 
 C_{yt} = \beta W_t 
 $$ 
 
and that savings when young are  

$$
A_{t+1} = (1-\beta) W_t.
$$


The young consumer allocates his/her savings are  entirely to physical capital. 

Profit maximization by representative firms in the economy implies the
that the real wage $W_t$ and the return on capital $r_t$ satisfy

\begin{align}
W_t & = (1-\alpha) K_t^\alpha \\
r_t & = \alpha K_t^{\alpha -1}
\end{align}


The condition for equilibrium in the market for capital is given by

$$
K_t = A_t.
$$

## Zejin Start

In this simple two period life cycle model, we have a closed form solution for the transition dynamics of the aggregate capital level

$$
K_{t+1}=K_{t}^{\alpha}\left(1-\tau_{t}\right)\left(1-\alpha\right)\left(1-\beta\right)+A_{t}^{g} \\
r_{t}=\alpha K_{t}^{\alpha-1} \\
$$

And the government budget constraint implies

$$
\tau_{t}=\left(G_{t}-r_{t}A_{t}^{g}\right)/\left(Y_{t} - r_t A_{t}^g\right) \\
$$

Let the initial steady state be the case:
1. there is no government debt, $A^g_t=0$,
2. government consumption equals $15\%$ of the initial steady state output $Y$

which implies the steady state values

$$
\hat{\tau} = 0.15 \\
\hat{K} = \hat{K} ^ \alpha (1 - \hat{\tau}) (1 - \alpha) (1 - \beta)
$$

so that

$$
\hat{K}=\left[\left(1-\hat{\tau}\right)\left(1-\alpha\right)\left(1-\beta\right)\right]^{\frac{1}{1-\alpha}}
$$
Let $\alpha = 0.3$, $\beta = 0.5$, solve for $\hat{K}$.

```{code-cell} ipython3
@njit
def K_to_Y(K, α):

    return K ** α

@njit
def K_to_r(K, α):

    return α * K ** (α - 1)

@njit
def K_to_W(K, α):

    return (1 - α) * K ** α

@njit
def K_to_C(K, Ag, τ, r, α, β):

    # consumption for old
    Ap = K - Ag
    Co = Ap * (1 + r * (1 - τ))

    # consumption for young
    W = K_to_W(K, α)
    Cy = β * W * (1 - τ)

    return Cy, Co
```

```{code-cell} ipython3
# parameters
α = 0.3
β = 0.5

# steady state ̂τ
τ_hat = 0.15
Ag_hat = 0.

# solve for steady state
K_hat = ((1 - τ_hat) * (1 - α) * (1 - β)) ** (1 / (1 - α))
K_hat
```

```{code-cell} ipython3
Y_hat, r_hat, W_hat = K_to_Y(K_hat, α), K_to_r(K_hat, α), K_to_W(K_hat, α)
Y_hat, r_hat, W_hat
```

```{code-cell} ipython3
G_hat = τ_hat * Y_hat
G_hat
```

```{code-cell} ipython3
Cy_hat, Co_hat = K_to_C(K_hat, Ag_hat, τ_hat, r_hat, α, β)
Cy_hat, Co_hat
```

```{code-cell} ipython3
init_ss = np.array([K_hat, Y_hat, Cy_hat, Co_hat,
                    W_hat, r_hat,
                    τ_hat, Ag_hat, G_hat])
```



Let's consider the following fiscal policy change:

1. at $t=0$, unexpectedly announce a one-period tax cut ($\tau_0 \   0.15 \rightarrow 0.1$) by issuing government debt $A^g$
2. from $t=1$, adjust $\tau_t$ and use tax revenues to pay for government consumption and interest payments on the official debt
3. government consumption $G_t$ will be constant and equal $0.15 \hat{Y}$

The implied transition dynamics will be

$$
K_{t+1}=K_{t}^{\alpha}\left(1-\tau_{t}\right)\left(1-\alpha\right)\left(1-\beta\right)+A^{g} \\
A^{g}=\tau_0\hat{Y}-\hat{G} \\
\hat{\tau}_{0}=0.1,\quad\tau_{t}=\frac{\hat{G}-r_{t}A^{g}}{\hat{Y}-r_{t}A^{g}}
$$

```{code-cell} ipython3
@njit
def closed_form_transition(T, init_ss, tax_cut, α, β):

    # unpack the steady state variables
    K_hat, Y_hat, Cy_hat, Co_hat = init_ss[:4]
    W_hat, r_hat = init_ss[4:6]
    τ_hat, Ag_hat, G_hat = init_ss[6:9]

    # initialize array containers
    # (note that Python is row-major)
    # K, Y, Cy, Co
    quant_seq = np.empty((T+1, 4))

    # W, r
    price_seq = np.empty((T+1, 2))

    # τ, Ag, G
    policy_seq = np.empty((T+1, 3))

    # t=0, starting from steady state
    K0, Y0 = K_hat, Y_hat
    W0, r0 = W_hat, r_hat
    Ag0 = 0.

    # tax cut
    τ0 = τ_hat * (1 - tax_cut)
    Ag1 = Ag0 * (1 + r0 * (1 - τ0)) + τ0 * Y0 - G_hat

    # immediate consumption increase
    Cy0, Co0 = K_to_C(K0, Ag0, τ0, r0, α, β)

    # t=0 economy
    quant_seq[0, :] = K0, Y0, Cy0, Co0
    price_seq[0, :] = W0, r0
    policy_seq[0, :] = τ0, Ag0, G_hat

    # starting from t=1 to T
    for t in range(1, T+1):

        # transition dynamics of K_t
        K_old, τ_old = quant_seq[t-1, 0], policy_seq[t-1, 0]

        # transition of K
        K = K_old ** α * (1 - τ_old) * (1 - α) * (1 - β) + Ag1

        # output, capital return, wage
        Y, r, W = K_to_Y(K, α), K_to_r(K, α), K_to_W(K, α)

        # tax rate
        τ = (G_hat - r * Ag1) / (Y - r * Ag1)

        # consumption
        Cy, Co = K_to_C(K, Ag1, τ, r, α, β)

        quant_seq[t, :] = K, Y, Cy, Co
        price_seq[t, :] = W, r
        policy_seq[t, :] = τ, Ag1, G_hat

    return quant_seq, price_seq, policy_seq
```

```{code-cell} ipython3
T = 20
tax_cut = 1 / 3

quant_seq, price_seq, policy_seq = closed_form_transition(T, init_ss, tax_cut, α, β)
```

```{code-cell} ipython3
fig, axs = plt.subplots(3, 3, figsize=(14, 10))

# quantities
for i, name in enumerate(['K', 'Y', 'Cy', 'Co']):
    ax = axs[i//3, i%3]
    ax.plot(range(T+1), quant_seq[:, i])
    ax.hlines(init_ss[i], 0, T+1, color='r', linestyle='--')
    ax.set_title(name)

# prices
for i, name in enumerate(['W', 'r']):
    ax = axs[(i+4)//3, (i+4)%3]
    ax.plot(range(T+1), price_seq[:, i])
    ax.hlines(init_ss[i+4], 0, T+1, color='r', linestyle='--')
    ax.set_title(name)

# policies
for i, name in enumerate(['τ', 'Ag', 'G']):
    ax = axs[(i+6)//3, (i+6)%3]
    ax.plot(range(T+1), policy_seq[:, i])
    ax.hlines(init_ss[i+6], 0, T+1, color='r', linestyle='--')
    ax.set_title(name)
```



Above we did an experiment where the tax cut rate is $1/3$. Here we can also try to let the tax cut rate be $0.2$.

```{code-cell} ipython3
tax_cut2 = 0.2
quant_seq2, price_seq2, policy_seq2 = closed_form_transition(T, init_ss, tax_cut2, α, β)
```

```{code-cell} ipython3
fig, axs = plt.subplots(3, 3, figsize=(14, 10))

# quantities
for i, name in enumerate(['K', 'Y', 'Cy', 'Co']):
    ax = axs[i//3, i%3]
    ax.plot(range(T+1), quant_seq[:, i])
    ax.plot(range(T+1), quant_seq2[:, i])
    ax.hlines(init_ss[i], 0, T+1, color='r', linestyle='--')
    ax.set_title(name)

# prices
for i, name in enumerate(['W', 'r']):
    ax = axs[(i+4)//3, (i+4)%3]
    ax.plot(range(T+1), price_seq[:, i])
    ax.plot(range(T+1), price_seq2[:, i])
    ax.hlines(init_ss[i+4], 0, T+1, color='r', linestyle='--')
    ax.set_title(name)

# prices
for i, name in enumerate(['τ', 'Ag', 'G']):
    ax = axs[(i+6)//3, (i+6)%3]
    ax.plot(range(T+1), policy_seq[:, i])
    ax.plot(range(T+1), policy_seq2[:, i])
    ax.hlines(init_ss[i+6], 0, T+1, color='r', linestyle='--')
    ax.set_title(name)
```



### Look at another policy experiment



The same initial steady state: $\hat{\tau}=0.15$, $\hat{G}=0.15\hat{Y}$, $\hat{A^g}=0$

But assume that from $t=0$, the government decreases its spending on services and goods by $\gamma$ fraction, $G_t=\left(1-\gamma\right) \hat{G} \  \forall t \geq 0$.

The government wants to keep the same $\tau_t=\hat{\tau}$ and accumulate assets $A^g_t$ over time.

```{code-cell} ipython3
T = 20

quant_seq3 = np.empty((T+1, 4))
price_seq3 = np.empty((T+1, 2))
policy_seq3 = np.empty((T+1, 3))

# t=0, starting from steady state
K0, Y0 = K_hat, Y_hat
W0, r0 = W_hat, r_hat
Ag0 = 0.

# remove government consumption
γ = 0.5
G0 = G_hat * (1 - γ)
# keep the same tax rate
τ0 = τ_hat
# government net worth at t=0 is predetermined
Ag0 = Ag_hat

Cy0, Co0 = K_to_C(K0, Ag0, τ0, r0, α, β)

# t=0 economy
quant_seq3[0, :] = K0, Y0, Cy0, Co0
price_seq3[0, :] = W0, r0
policy_seq3[0, :] = τ0, Ag0, G0

# starting from t=1 to T
for t in range(1, T+1):

    # from last period
    K_old, Y_old = quant_seq3[t-1, :2]
    W_old, r_old = price_seq3[t-1, :]
    τ_old, Ag_old, G_old = policy_seq3[t-1, :]

    # transition of government assets
    Ag = Ag_old * (1 + r_old * (1 - τ_old)) + τ_old * Y_old - G_old

    # transition of K
    K = K_old ** α * (1 - τ_hat) * (1 - α) * (1 - β) + Ag

    # output, capital return, wage
    Y, r, W = K_to_Y(K, α), K_to_r(K, α), K_to_W(K, α)

    # tax rate
    τ = τ_hat

    # consumption
    Cy, Co = K_to_C(K, Ag, τ, r, α, β)

    quant_seq3[t, :] = K, Y, Cy, Co
    price_seq3[t, :] = W, r
    policy_seq3[t, :] = τ, Ag, G0
```

```{code-cell} ipython3
fig, axs = plt.subplots(3, 3, figsize=(14, 10))

# quantities
for i, name in enumerate(['K', 'Y', 'Cy', 'Co']):
    ax = axs[i//3, i%3]
    ax.plot(range(T+1), quant_seq3[:, i])
    ax.hlines(init_ss[i], 0, T+1, color='r', linestyle='--')
    ax.set_title(name)

# prices
for i, name in enumerate(['W', 'r']):
    ax = axs[(i+4)//3, (i+4)%3]
    ax.plot(range(T+1), price_seq3[:, i])
    ax.hlines(init_ss[i+4], 0, T+1, color='r', linestyle='--')
    ax.set_title(name)

# policies
for i, name in enumerate(['τ', 'Ag', 'G']):
    ax = axs[(i+6)//3, (i+6)%3]
    ax.plot(range(T+1), policy_seq3[:, i])
    ax.hlines(init_ss[i+6], 0, T+1, color='r', linestyle='--')
    ax.set_title(name)
```



It will be useful for understanding the transition paths by looking at the ratio of government asset to the output, $\frac{A^g_t}{Y_t}$

```{code-cell} ipython3
plt.plot(range(T+1), policy_seq3[:, 1] / quant_seq3[:, 0])
plt.xlabel('t')
plt.title('Ag/Y');
```



### Another interesting policy



Again, the economy was in the same initial steady state.

The government spend $G_0=0$ for only one period, and accumulate $A^g_1$ asset. From $t \geq 1$, the government will choose the same level of consumption as before $\hat{G}$, and will adjust $\tau_t$ to keep the same level of asset $A^g_1$.

```{code-cell} ipython3
quant_seq4 = np.empty((T+1, 4))
price_seq4 = np.empty((T+1, 2))
policy_seq4 = np.empty((T+1, 3))

# t=0, starting from steady state
K0, Y0 = K_hat, Y_hat
W0, r0 = W_hat, r_hat
Ag0 = 0.

# remove government consumption
G0 = 0.
# keep the same tax rate
τ0 = τ_hat
# government net worth at t=0 is predetermined
Ag0 = Ag_hat
Ag1 = Ag0 * (1 + r0 * (1 - τ0)) + τ0 * Y0 - G0

Cy0, Co0 = K_to_C(K0, Ag0, τ0, r0, α, β)

# t=0 economy
quant_seq4[0, :] = K0, Y0, Cy0, Co0
price_seq4[0, :] = W0, r0
policy_seq4[0, :] = τ0, Ag0, G0

# starting from t=1 to T
for t in range(1, T+1):

    # from last period
    K_old, Y_old = quant_seq4[t-1, :2]
    W_old, r_old = price_seq4[t-1, :]
    τ_old, Ag_old, G_old = policy_seq4[t-1, :]

    # transition of government assets
    Ag = Ag_old * (1 + r_old * (1 - τ_old)) + τ_old * Y_old - G_old

    # transition of K
    K = K_old ** α * (1 - τ_hat) * (1 - α) * (1 - β) + Ag

    # output, capital return, wage
    Y, r, W = K_to_Y(K, α), K_to_r(K, α), K_to_W(K, α)

    # tax rate
    τ = (G_hat - r * Ag1) / (Y - r * Ag1)

    # consumption
    Cy, Co = K_to_C(K, Ag, τ, r, α, β)

    quant_seq4[t, :] = K, Y, Cy, Co
    price_seq4[t, :] = W, r
    policy_seq4[t, :] = τ, Ag, G_hat
```

```{code-cell} ipython3
fig, axs = plt.subplots(3, 3, figsize=(14, 10))

# quantities
for i, name in enumerate(['K', 'Y', 'Cy', 'Co']):
    ax = axs[i//3, i%3]
    ax.plot(range(T+1), quant_seq4[:, i])
    ax.hlines(init_ss[i], 0, T+1, color='r', linestyle='--')
    ax.set_title(name)

# prices
for i, name in enumerate(['W', 'r']):
    ax = axs[(i+4)//3, (i+4)%3]
    ax.plot(range(T+1), price_seq4[:, i])
    ax.hlines(init_ss[i+4], 0, T+1, color='r', linestyle='--')
    ax.set_title(name)

# policies
for i, name in enumerate(['τ', 'Ag', 'G']):
    ax = axs[(i+6)//3, (i+6)%3]
    ax.plot(range(T+1), policy_seq4[:, i])
    ax.hlines(init_ss[i+6], 0, T+1, color='r', linestyle='--')
    ax.set_title(name)
```



## A general method of computation



Given model parameters {$\alpha$, $\beta$}, a competitive equilibrium is characterized by

1. sequences of optimal consumptions $\{C_{yt}, C_{ot}\}$
2. sequences of prices $\{W_t, r_t\}$
3. sequences of aggregate capital and output $\{K_t, Y_t\}$
4. sequences of tax rates, assets (debt), government consumption $\{\tau_t, A_t^g, G_t\}$

such that

1. given the price sequences and government policy, the consumption choices maximize the household utility
2. the consumption and the government policy satisfy the government budget constraints



Focus on a particular fiscal policy experiment that we consider here

1. $\tau_0 = 0.1$
2. $A_t^g = A^g_1$
3. $G_t = \hat{G}$

The equilibrium transition path can be found by

1. giving guesses on the prices $\{W_t, r_t\}$ and tax rates $\{\tau_t\}$
2. solve for individual optimization problem
3. solve for transition of aggregate capital
4. update the guesses for prices and tax rates
5. iterate until convergence

```{code-cell} ipython3
@njit
def U(Cy, Co, β):

    return (Cy ** β) * (Co ** (1-β))
```



`quantecon.optimize.brent_max`:

1. quantecon source code: https://github.com/QuantEcon/QuantEcon.py/blob/8dbd7b2b4063f2caa89230fa6481b7eae5a91dec/quantecon/optimize/scalar_maximization.py#L5

```{code-cell} ipython3
brent_max?
```

```{code-cell} ipython3
@njit
def Cy_val(Cy, W, r_next, τ, τ_next, β):

    # Co given by the budget constraint
    Co = (W * (1 - τ) - Cy) * (1 + r_next * (1 - τ_next))

    return U(Cy, Co, β)
```

```{code-cell} ipython3
W, r_next, τ, τ_next = W_hat, r_hat, τ_hat, τ_hat
Cy_opt, U_opt, _ = brent_max(Cy_val,         # maximand
                             1e-3,           # lower bound
                             W*(1-τ)-1e-3,   # upper bound
                             args=(W, r_next, τ, τ_next, β))

Cy_opt, U_opt
```



Compare with the closed form solution.

```{code-cell} ipython3
Cy_hat
```



which is

```{code-cell} ipython3
W * β * (1 - τ)
```



Verify that the optimal $C_{y,t}$ does not depend on future prices or policies (but $C_{o,t+1}$ will).

```{code-cell} ipython3
r_next, τ_next = r_hat * 0.5, τ_hat * 0.5
brent_max(Cy_val, 1e-3, W*(1-τ)-1e-3, args=(W, r_next, τ, τ_next, β))
```

```{code-cell} ipython3
T = 20
tax_cut = 1 / 3

K_hat, Y_hat, Cy_hat, Co_hat = init_ss[:4]
W_hat, r_hat = init_ss[4:6]
τ_hat, Ag_hat, G_hat = init_ss[6:9]

# initial guesses of prices
W_seq = np.ones(T+2) * W_hat
r_seq = np.ones(T+2) * r_hat

# initial guesses of policies
τ_seq = np.ones(T+2) * τ_hat

Ag_seq = np.zeros(T+1)
G_seq = np.ones(T+1) * G_hat

# containers
K_seq = np.empty(T+2)
Y_seq = np.empty(T+2)
C_seq = np.empty((T+1, 2))

# t=0, starting from steady state
K_seq[0], Y_seq[0] = K_hat, Y_hat
W_seq[0], r_seq[0] = W_hat, r_hat

# tax cut
τ_seq[0] = τ_hat * (1 - tax_cut)
Ag1 = Ag_hat * (1 + r_seq[0] * (1 - τ_seq[0])) + τ_seq[0] * Y_hat - G_hat
Ag_seq[1:] = Ag1

# prepare to plot iterations until convergence
fig, axs = plt.subplots(1, 3, figsize=(14, 4))

# containers for checking convergence (Don't use np.copy)
W_seq_old = np.empty_like(W_seq)
r_seq_old = np.empty_like(r_seq)
τ_seq_old = np.empty_like(τ_seq)

max_iter = 500
i_iter = 0
tol = 1e-5 # tolerance for convergence

# start iteration
while True:

    # plot current prices at ith iteration
    for i, seq in enumerate([W_seq, r_seq, τ_seq]):
        axs[i].plot(range(T+2), seq)

    # store old prices from last iteration
    W_seq_old[:] = W_seq
    r_seq_old[:] = r_seq
    τ_seq_old[:] = τ_seq

    # start update quantities and prices
    for t in range(T+1):

        # note that r_seq[t+1] and τ_seq[t+1] are guesses!
        W, r_next, τ, τ_next = W_seq[t], r_seq[t+1], τ_seq[t], τ_seq[t+1]

        # consumption optimization
        out = brent_max(Cy_val, 1e-3, W*(1-τ)-1e-3, args=(W, r_next, τ, τ_next, β))
        Cy = out[0]

        # private saving, Ap[t+1]
        Ap_next = W * (1 - τ) - Cy

        # asset next period
        K_next = Ap_next + Ag1
        W_next, r_next, Y_next = K_to_W(K_next, α), K_to_r(K_next, α), K_to_Y(K_next, α)

        K_seq[t+1] = K_next
        # note that here the updated guesses will be used immediately!
        W_seq[t+1] = W_next
        r_seq[t+1] = r_next
        τ_seq[t+1] = (G_hat - r_next * Ag1) / (Y_next - r_next * Ag1)

    # one iteration finishes
    i_iter += 1

    # check convergence
    if (np.max(np.abs(W_seq_old - W_seq)) < tol) & \
       (np.max(np.abs(r_seq_old - r_seq)) < tol) & \
       (np.max(np.abs(τ_seq_old - τ_seq)) < tol):
        print(f"Converge using {i_iter} iterations")
        break

    if i_iter > max_iter:
        print(f"Fail to converge using {i_iter} iterations")
        break

# compare to the closed form solutions
axs[0].plot(range(T+1), price_seq[:, 0], 'r-*')
axs[0].set_title('W')
axs[1].plot(range(T+1), price_seq[:, 1], 'r-*')
axs[1].set_title('r')
axs[2].plot(range(T+1), policy_seq[:, 0], 'r-*')
axs[2].set_title('τ')

plt.show();
```



## Work in two periods



Changing the assumption that agents only supply $1$ labor unit when young, but assume that they supply $1/2$ labor unit when young and old.

The aggregate labor supply in the economy won't change.

Now the lifetime budget constraint becomes

$$
C_{yt}+\frac{C_{ot+1}}{1+r_{t+1}\left(1-\tau_{t+1}\right)}=\frac{1}{2}W_{t}\left(1-\tau_{t}\right)+\frac{1}{2}\frac{W_{t+1}\left(1-\tau_{t+1}\right)}{1+r_{t+1}\left(1-\tau_{t+1}\right)}
$$

```{code-cell} ipython3
@njit
def Cy_val2(Cy, W, W_next, r_next, τ, τ_next, β):

    # Co given by the budget constraint
    Co = (W / 2 * (1 - τ) - Cy) * (1 + r_next * (1 - τ_next)) + (W_next / 2) * (1 - τ_next)

    return U(Cy, Co, β)
```

```{code-cell} ipython3
W, W_next, r_next, τ, τ_next = W_hat, W_hat, r_hat, τ_hat, τ_hat

# lifetime budget
C_ub = (W / 2) * (1 - τ) + (W_next / 2) * (1 - τ_next) / (1 + r_next * (1 - τ_next))

Cy_opt, U_opt, _ = brent_max(Cy_val2,      # maximand
                             1e-3,         # lower bound
                             C_ub-1e-3,    # upper bound
                             args=(W, W_next, r_next, τ, τ_next, β))

Cy_opt
```



Does the optimal consumption for the young now depend on the future wage $W_{t+1}$?

```{code-cell} ipython3
W, W_next, r_next, τ, τ_next = W_hat, W_hat, r_hat, τ_hat, τ_hat

W_next = W_hat / 2

# what's the new lifetime income?
C_ub = (W / 2) * (1 - τ) + (W_next / 2) * (1 - τ_next) / (1 + r_next * (1 - τ_next))

Cy_opt, U_opt, _ = brent_max(Cy_val2,   # maximand
                             1e-3,      # lower bound
                             C_ub-1e-3,    # upper bound
                             args=(W, W_next, r_next, τ, τ_next, β))

Cy_opt
```



Does it depend on the future interest rate $r_{t+1}$?

```{code-cell} ipython3
W, W_next, r_next, τ, τ_next = W_hat, W_hat, r_hat, τ_hat, τ_hat

r_next = r_hat / 2

# what's the new lifetime income?
C_ub = (W / 2) * (1 - τ) + (W_next / 2) * (1 - τ_next) / (1 + r_next * (1 - τ_next))

Cy_opt, U_opt, _ = brent_max(Cy_val2,   # maximand
                             1e-3,      # lower bound
                             C_ub-1e-3,    # upper bound
                             args=(W, W_next, r_next, τ, τ_next, β))

Cy_opt
```



Does it depend on the future tax rate $\tau_{t+1}$?

```{code-cell} ipython3
W, W_next, r_next, τ, τ_next = W_hat, W_hat, r_hat, τ_hat, τ_hat

τ_next = τ_hat / 2

# what's the new lifetime income?
C_ub = (W / 2) * (1 - τ) + (W_next / 2) * (1 - τ_next) / (1 + r_next * (1 - τ_next))

Cy_opt, U_opt, _ = brent_max(Cy_val2,   # maximand
                             1e-3,      # lower bound
                             C_ub-1e-3,    # upper bound
                             args=(W, W_next, r_next, τ, τ_next, β))

Cy_opt
```

```{code-cell} ipython3
T = 20
tax_cut = 1 / 3

K_hat, Y_hat, Cy_hat, Co_hat = init_ss[:4]
W_hat, r_hat = init_ss[4:6]
τ_hat, Ag_hat, G_hat = init_ss[6:9]

# initial guesses of prices
W_seq = np.ones(T+2) * W_hat
r_seq = np.ones(T+2) * r_hat

# initial guesses of policies
τ_seq = np.ones(T+2) * τ_hat

Ag_seq = np.zeros(T+1)
G_seq = np.ones(T+1) * G_hat

# containers
K_seq = np.empty(T+2)
Y_seq = np.empty(T+2)
C_seq = np.empty((T+1, 2))

# t=0, starting from steady state
K_seq[0], Y_seq[0] = K_hat, Y_hat
W_seq[0], r_seq[0] = W_hat, r_hat

# tax cut
τ_seq[0] = τ_hat * (1 - tax_cut)
Ag1 = Ag_hat * (1 + r_seq[0] * (1 - τ_seq[0])) + τ_seq[0] * Y_hat - G_hat
Ag_seq[1:] = Ag1

# immediate effect on consumption
# only know about Co,0 but not Cy,0
C_seq[0, 1] = (W_hat / 2) + K_seq[0] * (1 + r_hat * (1 - τ_hat))

# prepare to plot iterations until convergence
fig, axs = plt.subplots(1, 3, figsize=(14, 4))

# containers for checking convergence (Don't use np.copy)
W_seq_old = np.empty_like(W_seq)
r_seq_old = np.empty_like(r_seq)
τ_seq_old = np.empty_like(τ_seq)

max_iter = 500
i_iter = 0
tol = 1e-5 # tolerance for convergence

# start iteration
while True:

    # plot current prices at ith iteration
    for i, seq in enumerate([W_seq, r_seq, τ_seq]):
            axs[i].plot(range(T+2), seq)

    # store old prices from last iteration
    W_seq_old[:] = W_seq
    r_seq_old[:] = r_seq
    τ_seq_old[:] = τ_seq

    # start update quantities and prices
    for t in range(T+1):

        # note that r_seq[t+1] and τ_seq[t+1] are guesses!
        W, W_next, r_next, τ, τ_next = W_seq[t], W_seq[t+1], r_seq[t+1], τ_seq[t], τ_seq[t+1]
        
        C_ub = (W / 2) * (1 - τ) + (W_next / 2) * (1 - τ_next) / (1 + r_next * (1 - τ_next))

        out = brent_max(Cy_val2,   # maximand
                        1e-3,      # lower bound
                        C_ub-1e-3,    # upper bound
                        args=(W, W_next, r_next, τ, τ_next, β))
        Cy = out[0]

        # private saving, Ap[t+1]
        Ap_next = W * (1 - τ) - Cy

        # asset next period
        K_next = Ap_next + Ag1
        W_next, r_next, Y_next = K_to_W(K_next, α), K_to_r(K_next, α), K_to_Y(K_next, α)

        K_seq[t+1] = K_next
        W_seq[t+1] = W_next
        r_seq[t+1] = r_next
        τ_seq[t+1] = (G_hat - r_next * Ag1) / (Y_next - r_next * Ag1)

        # record consumption
        C_seq[t, 0] = Cy
        if t < T:
            C_seq[t+1, 1] = (W / 2 * (1 - τ) - Cy) * (1 + r_next * (1 - τ_next)) + W_next / 2

    # one iteration finishes
    i_iter += 1

    # check convergence
    if (np.max(np.abs(W_seq_old - W_seq)) < tol) & \
       (np.max(np.abs(r_seq_old - r_seq)) < tol) & \
       (np.max(np.abs(τ_seq_old - τ_seq)) < tol):
        print(f"Converge using {i_iter} iterations")
        break

    if i_iter > max_iter:
        print(f"Fail to converge using {i_iter} iterations")
        break

axs[0].set_title('W')
axs[1].set_title('r')
axs[2].set_title('τ')

plt.show();
```

```{code-cell} ipython3
plt.plot(K_seq)
plt.title('K')
```



## Interpolation



Here is a demonstration of how to use `interp`. This jitted function is very useful for solving optimal consumption and saving problems using Bellman equations.

```{code-cell} ipython3
!pip install interpolation
```

```{code-cell} ipython3
from interpolation import interp
```

```{code-cell} ipython3
# x_arr, V_arr, x => return V(x)
interp(np.array([0., 1.]), np.array([0.2, 0.8]), 0.2)
```

```{code-cell} ipython3
interp(np.array([0., 1.]), np.array([0.2, 0.8]), np.array([0.2, 0.3, 0.4]))
```

```{code-cell} ipython3

```
