# Econ 33580 -- Homework 3

**Lars and Tom**

**Due Friday April 30**

## Overview

This exercise modifies a  celebrated model of optimal fiscal policy by Robert E.Lucas, Jr., and Nancy Stokey (1983).

The model revisits classic issues about how to pay for a war.

Here a *war* means a more  or less temporary surge in an exogenous government expenditure process.

The model features

- a government that must finance an exogenous stream of government expenditures with  either
  - a flat rate tax on labor, or
  - purchases and sales from a full array of Arrow-Debreu history-contingent securities
- a representative household that values consumption and leisure
- a linear production function mapping labor into a single good
- a Ramsey planner who at time $t=0$ chooses a plan for taxes and   trades of history,date- contingent Arrow Debreu securities.

One message  of this exercise is that **timing protocols matter**.

Relative to Lucas and Stokey, we change timing protocols by  eliminating what amounts to an important  **participation constraint** that Lucas and Stokey imposed.

*  They assumed that while at time $0$ households have access to complete markets in state contingent securities indexed by time and histories,   the government does not -- it cannot instantaneously **reschedule** an   *ex ante* fixed stream of state contingent obligations by making time $0$ trades at competitive equilibrium prices before those prices have been equilibrated.

* By way of contrast, we allow the government completely to **schedule** a stream of  history-contingent obligatons *ex ante*, **before** time $0$.

* We are tempted to name  this altered timing protocol after Secretary of Treasury **Alexander Hamilton** because it seems to approximate the operations that he carried out for the U.S. at its version of time $0$, namely, 1790.

  * Hamilton more or less simultaneuosly designed a tax plan and debt structure for the U.S. federal government.

## A Competitive Equilibrium with Distorting Taxes

At time $t \geq 0$, a random variable $s_t$ belongs to a time-invariant set ${\cal S} = [1, 2, \ldots, S]$.

For $t \geq 0$, a history $s^t = [s_t, s_{t-1}, \ldots, s_0]$ of an exogenous state $s_t$ has joint probability density $\pi_t(s^t)$.

We begin by assuming that government purchases $g_t(s^t)$ at time $t \geq 0$ depend on $s^t$.

Let $c_t(s^t)$,  $\ell_t(s^t)$, and $n_t(s^t)$ denote consumption, leisure, and labor supply, respectively, at history $s^t$ and date $t$.

A representative  household is endowed with one unit of time that can be divided between leisure $\ell_t$ and labor $n_t$:


<a id='equation-feas1-opt-tax'></a>
$$
n_t(s^t) + \ell_t(s^t) = 1 \tag{1}
$$

Output equals $n_t(s^t)$ and can be divided between $c_t(s^t)$ and $g_t(s^t)$

<a id='equation-tss-techr-opt-tax'></a>
$$
c_t(s^t) + g_t(s^t) = n_t(s^t) \tag{2}
$$

A representative household’s preferences over $\{c_t(s^t), \ell_t(s^t)\}_{t=0}^\infty$ are ordered by

<a id='equation-ts-prefr-opt-tax'></a>
$$
\sum_{t=0}^\infty \sum_{s^t} \beta^t \pi_t(s^t) u[c_t(s^t), \ell_t(s^t)] \tag{3}
$$

where the utility function $u$ is  increasing, strictly concave, and three times continuously differentiable in both arguments.

The technology pins down a pre-tax wage rate to unity for all $t, s^t$.

The government imposes a flat-rate tax $\tau_t(s^t)$ on labor income at time $t$, history $s^t$.


The government starts time $0$ with an exogenous stream of history, date contingent debt $\{\overline b_t(s^t)\}_{t, s^t}$.

There are complete markets in date, history contingent securities open each period and available to both to the representative household and the government.

## Timing Protocol

At the **end** of time $t=-1$ (or at time $0_{0}$ **before** time $0$ has started, a Ramsey planner chooses a tax-debt plan  $\{\tau_t(s^t)\}, \{\overline b_t(s^t)\}_{t, s^t}$.

  * Thus, the time $0$ government arrives at time $0$ with a tax-debt plan that a Ramsey planner had previously designed.

  * At time $0$, the Ramsey planner is out of the picture.
 
 ## More Details

It is convenient to work with an Arrow-Debreu price system.

Let $q_t^0(s^t)$ be the price at time $0$, measured in time $0$ consumption goods, of one unit of consumption at time $t$, history $s^t$.

The household’s present-value budget constraint:

<a id='equation-ts-bcpv2'></a>
$$
\sum_{t=0}^\infty \sum_{s^t} q^0_t(s^t) c_t(s^t) \leq
\sum_{t=0}^\infty \sum_{s^t} q^0_t(s^t) [1-\tau_t(s^t)] n_t(s^t) \tag{7}
$$
where $\{q^0_t(s^t)\}_{t=1}^\infty$ is  a time $0$ Arrow-Debreu price system.

We can use feasibility at each date, history pair to rewrite  the above inequality as the government's present-value budget constraint is:


<a id='equation-ts-bgpv3'></a>
$$
- \sum_{t=0}^\infty \sum_{s^t} q^0_t(s^t)  g_t(s^t) +
\sum_{t=0}^\infty \sum_{s^t} q^0_t(s^t) [1-\tau_t(s^t)] n_t(s^t) \leq
0 \tag{7a}
$$

## Boiler Plate Definitions

Some helpful reminders.

A  **government policy** is an exogenous sequence $\{g(s_t)\}_{t=0}^\infty$, a tax rate sequence $\{\tau_t(s^t)\}_{t=0}^\infty$, and a government debt sequence $\{b_{t}(s^{t})\}_{t=0}^\infty$.

A **feasible allocation** is a consumption-labor supply plan $\{c_t(s^t), n_t(s^t)\}_{t=0}^\infty$
that satisfies [(2)](#equation-tss-techr-opt-tax) at all $t, s^t$.

A time $0$ Arrow Debreu **price system** is a sequence of  prices $\{q^0_{t}(s^t) \}_{t \ge0, s^t}$.

The household faces the price system as a price-taker and takes the government policy as given.

The household chooses $\{c_t(s^t), \ell_t(s^t)\}_{t=0}^\infty$ to maximize [(3)](#equation-ts-prefr-opt-tax) subject to [(5)](#equation-ts-bcr) and [(1)](#equation-feas1-opt-tax) for all $t, s^t$.

A **competitive equilibrium with distorting taxes** is a feasible allocation,
a price system, and a government policy such that

- Given the price system and the government policy, the allocation solves the
  household’s optimization problem.
- Given the allocation, government policy, and  price system, the government’s
  budget constraint is satisfied for all $t, s^t$.


**Note:** There are many competitive equilibria with distorting taxes. They are indexed by different government policies.

## A Hamilton-Ramsey plan 

A **Hamilton-Ramsey problem** or **optimal taxation, debt scheduling problem** is to choose a competitive equilibrium with distorting taxes that maximizes [(3)](#equation-ts-prefr-opt-tax).

### Primal Approach

Please deploy a *primal approach*.

The idea is to use first-order conditions for  household optimization to eliminate taxes and prices in favor of quantities, then to pose an optimization problem cast entirely in terms of quantities.

After Hamilton-Ramsey quantities have been found, taxes and prices can  be unwound from an allocation.

The primal approach uses four steps:

1. Obtain  first-order conditions of the household’s problem and solve them for $\{q^0_t(s^t), \tau_t(s^t)\}_{t=0}^\infty$ as functions of the allocation $\{c_t(s^t), n_t(s^t)\}_{t=0}^\infty$.
2. Substitute these expressions for taxes and prices in terms of the allocation  into the household’s present-value budget constraint.
  - This intertemporal constraint involves only the allocation and is regarded     as an *implementability constraint*.
3. Find the allocation that maximizes the utility of the representative household   [(3)](#equation-ts-prefr-opt-tax) subject to  the feasibility constraints [(1)](#equation-feas1-opt-tax)  and [(2)](#equation-tss-techr-opt-tax)  and the implementability condition derived in step 2.  
  - This optimal allocation is called the **Hamilton-Ramsey allocation**.  
4. Use the allocation together with the formulas from step 1 to find  taxes and prices.  



### The Implementability Constraint


To approach the Hamilton-Ramsey problem, we study the household’s optimization problem.

First-order conditions for the household’s problem for $\ell_t(s^t)$
and $b_t(s_{t+1}| s^t)$, respectively, imply


<a id='equation-lsa-taxr'></a>
$$
(1 - \tau_t(s^t)) = {\frac{u_l(s^t)}{u_c(s^t)}} \tag{8}
$$

and

<a id='equation-ls102'></a>
$$
q_t^0(s^t) = \beta^{t} \pi_{t}(s^{t})
                            {u_c(s^{t})  \over u_c(s^0)} \tag{9}
$$

(The stochastic process $\{q_t^0(s^t)\}$ is an instance of what finance economists call a **stochastic discount factor** process.)

Using the first-order conditions [(8)](#equation-lsa-taxr) and [(9)](#equation-ls101) to eliminate taxes and prices from [(7)](#equation-ts-bcpv2), we derive the *implementability condition*


<a id='equation-tss-cham1'></a>
$$
\sum_{t=0}^\infty  \sum_{s^t} \beta^t \pi_t(s^t)
         [u_c(s^t) c_t(s^t) - u_\ell(s^t) n_t(s^t)] \leq 0 .
         \tag{11}
$$


The **Hamilton-Ramsey problem** is to choose a feasible  allocation  that maximizes

<a id='equation-ts-prefr2'></a>
$$
\sum_{t=0}^\infty \sum_{s^t} \beta^t \pi_t(s^t) u[c_t(s^t), 1 - n_t(s^t)] \tag{12}
$$

subject to  [(11)](#equation-tss-cham1).

**Exercise 1:**  Obtain first-order necessary conditions for a Hamilton-Ramsey allocation.  Please compare them with analogous conditions for a Ramsey allocation under Lucas and Stokey's timing protocol.

**Exercise 2:** Please verify that the following debt plan implements a Hamilton-Ramsey plan

$$
b_t(s^t) = \tau_t(s^t) n_t(s^t) - g_t(s^t) \tag{13}
$$

where the net-of-interest or *primary* government surplus $\tau_t(s^t) n_t(s^t) - g_t(s^t)$ equals $c_t(s^t) - \frac{u_\ell(s^t)}{u_c(s^t)  }n_t(s^t)$.

**Exercise 3:** Please interpret inequality (11) and rule (13) as asserting that at each date and history, the value of government debt equals the  present value of government surpluses, so that large government debts **now** signal large government **surpluses** later.

**Exercise 4:** Please interpret the government debt plan (3) as an insurance contract in which the government has purchased  insurance against **fiscal risk**.

**Exercise 5:** Suppose that at time $t\geq 0$ history $s^t$ **after** $s_t$ has been realized, someone asks you to solve a Hamilton-Ramsey plan anew.  That is, in light of how things have unfolded under the original Hamilton-Ramsey plan,
you are to pretend that time $-1$ is **now**, meaning time $t$ after $s_t$ has been realized and time time $t$ history $s^t$ provisions of the original Hamilton-Ramsey plan have been implemented.  Please find the new Hamilton-Ramsey plan.

**Exercise 6:** Please compare the Hamilton-Ramsey plan that you derived in exercise 5 with the time $t$ history $s^t$ **continuation** of the original Hamilton-Ramsey plan.

**Exercise 7:** Is the original Hamilton-Ramsey plan **time consistent**?
