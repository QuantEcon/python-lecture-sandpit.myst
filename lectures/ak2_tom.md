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

# Tom_Auerbach-Kotlikoff

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

This lecture is presents the life cycle model consisting of overlapping generations of two-period lived people proposed  by Peter Diamond
{cite}`diamond1965national` and  analyzed  in chapter 2 of Auerbach and 
Kotlikoff (1987) {cite}`auerbach1987dynamic`.

Auerbach and 
Kotlikoff (1987) use the two period model as a warm-up for their analysis of  overlapping generation models of long-lived people that is the main topic of their book.

Their model of two-period lived overlapping generations is a useful warmup because 

* it sets forth the structure of interactions between generations of different agents who are alive at a given date
* it activates key forces and tradeoffs confronting the government and successive generations of people
* interesting experiments involving transitions from one steady state to another can be computed by hand
 ```{note}
Auerbach and Kotlikoff use computer code to calculate transition paths for their models with long-lived people.
``` 

 





## Setting

Time is discrete and is indexed by $t=0, 1, 2, \ldots$.  

The economy lives forever, but the people living in it do not.  

At each time $t, t \geq 0$ a representative old person and a representative young person are alive.

Thus, at time $t$ a representative old person coexists with a representative young person who will become an old person at time $t+1$. 

A young person works, saves, and consumes.

An old person dissaves and consumes but does not work, 

There is a government that lives forever, i.e., at $t=0, 1, 2, \ldots $.

Each period $t \geq 0$, the government taxes, spends, transfers, and borrows.  




Initial conditions set from outside the model at time $t=0$ are

* $K_0$ -- initial capital stock  brought into time $t=0$ by a representative  initial old person
* $D_0$ government debt falling due at $t=0$ and owned by a representative old person at time $t=0$
  
$K_0$ and $D_0$ are both measured in units of time $0$ goods.

A government **policy** is a collection of sequences $\{G_t, D_t, \tau_t, \delta_o, \delta_y,\}_{t=0}^\infty $,
where  

 * $\tau_t$ -- flat rate tax on wages and earnings from capital and government bonds
 * $\delta_y$ -- lump sum tax on each young person
 * $\delta_o$ -- lump sum tax on each old person
 * $D_t$ -- one-period government bond principal due at time $t$, per capita
 * $G_t$ -- government purchases of goods (`thrown into ocean'), per capita
  
An **allocation** is a collection of sequences $\{C_{yt}, C_{ot}, K_{t+1}, Y_t, G_t\}_{t=0}^\infty $, where constituents
of the sequence include output and factors of production

 * $K_t$ -- physical capital per capita
 * $L_t$ -- labor per capita
 * $Y_t$ -- output per capita

and also consumption and physical  investment

* $C_{yt}$ -- consumption of young person at time $t \geq 0$
* $C_{ot}$ -- consumption of old person at time $t \geq 0$
* $K_{t+1} - K_t \equiv I_t $ -- investment in physical capital at time $t \geq 0$

The national income and product accounts for the economy are described by a sequence of equalities

* $Y_t = C_{yt} + C_{ot} + (K_{t+1} - K_t) + G_t, \quad t \geq 0$ 

A **price system** is a pair of sequences $\{W_t, r_t\}_{t=0}^\infty$, where constituents of the sequence include rental rates for the factors of production

* $W_t$ -- rental rate for labor at time $t \geq 0$
* $r_t$ -- rental rate for capital at time $t \geq 0$


## Production

There are two factors of production, physical capital $K_t$ and labor $L_t$.  

Capital does not depreciate.  

The initial capital stock $K_0$ is owned by the initial old person, who rents it to the firm at time $0$.

The economy's net investment rate $I_t$ at time $t$ is 

$$
I_t = K_{t+1} - K_t
$$

The economy's capital stock at time $t$ emerges from cumulating past rates of investment:

$$
K_t = K_0 + \sum_{s=0}^{t-1} I_s 
$$

There is  a Cobb-Douglas technology that  converts physical capital $K_t$ and labor services $L_t$ into 
output $Y_t$

$$
Y_t  = K_t^\alpha L_t^{1-\alpha}, \quad \alpha \in (0,1)
$$ (eq:prodfn)


## Government

The government at time  $t-1$   issues one-period risk-free debt promising to pay $D_t$ time $t$  goods per capita at time $t$.

Young people at time $t$ purchase government debt $D_{t+1}$ maturing at time $t+1$. 

The government budget constraint at time $t \geq 0$ is

$$
D_{t+1} - D_t = r_t D_t + G_t - T_t
$$

or 

$$
D_{t+1} = (1 + r_t)  D_t - T_t 
$$ 

<font color='red'>Zejin: should it be</font>

$$
D_{t+1} = (1 + r_t)  D_t + G_t - T_t 
$$ 

where total tax collections net of transfers are given by $T_t$ satisfying

$$
T_t = \tau_t Y_t + \delta_y + \delta_o
$$

or

$$
T_t = \tau_t W + \tau_t (D_t + K_t) + \delta_y + \delta_o
$$

<font color='red'>Zejin: should it be</font>

$$
T_t = \tau_t Y_t + \tau_t r_t D_t + \delta_y + \delta_o
$$

<font color='red'>and</font>

$$
T_t = \tau_t W + \tau_t r_t (D_t + K_t) + \delta_y + \delta_o
$$

<font color='red'>Zejin: Also, by writing $\tau_t W_t$ instead of $\tau_t W_t L_t$, we are implicitly using the fact that $L_t = 1$. We need to mention somewhere above that the population size of the young worker is 1, and that each young worker supply one unit of labor inelastically. Alternatively, we can keep $L_t$ here.</font>

**Note to Zejin and Tom: I have assumed that the goverment taxes interest on government debt. Do AK also assume that -- we can do
what we want here**



## Households' Activities in Factor Markets

At time $t \geq 0$, an old person brings $K_t$ into the period, rents it to a representative  firm for $r_{t+1} K_t$, collects these rents, pays a lump sum tax or receives 
receives a lump sum subsidy from the government, then sells whatever is left over to a young person.  

At each $t \geq 0$, a  young person sells one unit of labor services to a representative firm for $W_t$ in wages, pays taxes to the goverment, then divides the remainder between acquiring assets $A_{t+1}$ consisting of a sum of physical capital $K_{t+1}$ and government bonds $D_{t+1}$  maturiting at $t+1$.




## Representative firm's problem 

The firm hires labor services from  young households and capital from old  households at competitive rental rates,
$W_t$ for labor service, $r_t$ for capital. 

The units of these rental rates are:

* for $W_t$, output at time $t$ per unit of labor at time $t$  
* for $r_t$,  output at time $t$  per unit of capitalat time $t$ 


We take output at time $t$ as *numeraire*, so the price of output at time $t$ is one.

The firm's profits at time $t$ are thus

$$
K_t^\alpha L_t^{1-\alpha} - r_t K_t - W_t L_t . 
$$

To maximize profits the firms equates marginal products to rental rates:

$$
\begin{align}
W_t & = (1-\alpha) K_t^\alpha L_t^{-\alpha} \\
r_t & = \alpha K_t^\alpha L_t^{1-\alpha}
\end{align}
$$  (eq:firmfonc)

Output can either be consumed by old or young households, taken by the government for its own uses (e.g., throwing into the ocean),
or used to augment the capital stock.  


The firm  sells output to old households, young households, and the government.









## Households' problems

### Initial old household 

At time $t=0$, a representative initial old household is endowed with $(1 - \tau_0) (1 - r_0) A_0$ (<font color='red'>Zejin: should this be $(1 + r_0(1 - \tau_0)) A_0$</font>?) in initial assets, and must pay a lump sum tax to (if positive) or receive a subsidy from  (if negative)
$\delta_o$ the government.  The   households' budget constraint is

$$
C_{o0} = (1 - \tau_0) (1 - r_0) A_0 - \delta_o .
$$ (eq:hbudgetold)

<font color='red'>Zejin: and accordingly, this will be</font>

$$
C_{o0} = (1 + r_0 (1 - \tau_0)) A_0 - \delta_o .
$$ (eq:hbudgetold)

An initial old household's utility function is $C_{o0}$, so the household's optimal consumption plan
is provided by equation {eq}`eq:hbudgetold`.

### Young household

At each $t \geq 0$, a  young household inelastically supplies one unit of labor and in return
receives pre-tax labor earnings of $W_t$ units of output.  

A young-household's post-tax-and-transfer earnings are $W_t (1 - \tau_t) - \delta_y$.  

At each $t \geq 0$, a young household chooses a consumption plan  $C_{yt}, C_{ot+1}$ 
to maximize

$$
U_t  = C_{yt}^\beta C_{o,t+1}^{1-\beta}, \quad \beta \in (0,1)
$$ (eq:utilfn)

subject to the budget constraints

$$
\begin{align}
C_{yt} + A_{t+1} & =  W_t (1 - \tau_t) - \delta_y \\
C_{ot+1} & = (1+ r_{t+1})A_{t+1} - \delta_o
\end{align}
$$ (eq:twobudgetc)

<font color='red'>Zejin: the capital return for the representative old will be taxed</font>

$$
\begin{align}
C_{ot+1} & = (1+ r_{t+1} (1 - \tau_{t+1}))A_{t+1} - \delta_o
\end{align}
$$

Solving the second equation of {eq}`eq:twobudgetc` for savings  $A_{t+1}$ and substituting it into the first equation implies the present value budget constraint

$$
C_{yt} + \frac{C_{ot+1}}{1 + r_{t+1}(1 - \tau_{t+1})} = W_t (1 - \tau_t) - \delta_y - \frac{\delta_o}{1 + r_{t+1}(1 - \tau_{t+1})}
$$ (eq:onebudgetc)

Form a Lagrangian 

$$ 
\begin{align}
L  & = C_{yt}^\beta C_{o,t+1}^{1-\beta} \\ &  + \lambda \Bigl[ C_{yt} + \frac{C_{ot+1}}{1 + r_{t+1}(1 - \tau_{t+1})} - W_t (1 - \tau_t) + \delta_y + \frac{\delta_o}{1 + r_{t+1}(1 - \tau_{t+1})}\Bigr],
\end{align}
$$ (eq:lagC)

where $\lambda$ is a Lagrange multiplier on the intertemporal budget constraint {eq}`eq:onebudgetc`.


After several lines of algebra, first-order conditions for maximizing $L$ with respect to $C_{yt}, C_{ot+1}$ 
imply that an optimal consumption plan satisfies

$$
\begin{align}
C_{yt} & = \beta \Bigl[ W_t (1 - \tau_t) - \delta_y - \frac{\delta_o}{1 + r_{t+1}(1 - \tau_{t+1})}\Bigr] \\
\frac{C_{0t+1}}{1 + r_{t+1}(1-\tau_{t+1})  } & = (1-\beta)   \Bigl[ W_t (1 - \tau_t) - \delta_y - \frac{\delta_o}{1 + r_{t+1}(1 - \tau_{t+1})}\Bigr] 
\end{align}
$$ (eq:optconsplan)

The first-order condition for minimizing Lagrangian {eq}`eq:lagC` with respect to the Lagrange multipler $\lambda$ recovers the budget constraint {eq}`eq:onebudgetc`,
which, using {eq}`eq:optconsplan` gives the optimal savings plan

$$
A_{t+1} = (1-\beta) [ (1- \tau_t) W_t - \delta_y] + \beta \frac{\delta_o}{1 + r_{t+1}(1 - \tau_{t+1})} 
$$ (eq:optsavingsplan)



## Equilbrium 

**Definition:** An equilibrium is an allocation,  a government policy, and a price system with the properties that
* given the price system and the government policy, the allocation solves
    * represenative firms' problems for $t \geq 0$
    * households problems for $t \geq 0$
* given the price system and the allocation, the government budget constraint is satisfies for all $t \geq 0$.


**Tom's part stops, Zejin's part starts here.**


We will first solve for equilibrium paths using the closed form solution that we derived in the class. And then, let's pretend that we don't know the closed form solution, and solve for the transitions paths by iterating over the guesses of price sequences and tax rate sequence. The equilibrium paths will be found as a fixed point.


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
