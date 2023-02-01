---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Repeated Moral Hazard

## Spear and Srivastava

Spear and Srivastava (1987) {cite}`Spear_Srivastava_87` presented a
recursive formulation of an infinitely repeated, discounted  repeated
principal-agent problem.

*  A **principal** owns a technology
that produces output $q_t$ at time $t$ according to a probability distribution $F$ that depends on the action $a_t$ of a worker we'll can an **agent**.
*  $q_t$ is determined by a family $F(q_t\vert a_t)$ of conditional c.d.f.'s  
*  $a_t$ is an action taken at the beginning of $t$ by the **agent**. 
*  The principal  does not observe $a_t$.
*  The principal  does observe $q_t$ at the end of period $t$.
*  The principle   remembers earlier $q_{t-j}$'s and so at the end of period $t$ knows  history $\{q_s\}_{s=0}^t$.
*  The principal is risk-neutral and  has access to an outside loan market with constant risk-free gross interest rate $\beta^{-1}$.
*  The agent is averse to consumption risk and has preferences random streams of consumption ordered by $E_0 \sum^\infty_{t=0} \beta^t u(c_t, a_t)$ 
*  $u$ is increasing in $c$ and decreasing in $a$.



The principal designs a  life-time contract. 

A **contract**  $\sigma$ is a sequence of functions  whose $t$th component $\sigma_t$ maps a history $q^{t-1} = q_{t-1}, q_{t-2}, \ldots , q_0$ of past outputs remembered at time $t$ into a time $t$ action $a_t$ that it asks the agent to perform and a time $t$  consumption $c_t$ that it pays the agent.  


The risk-neutral principal  designs the contract  to maximize $E_0 \sum^\infty_{t=0} \beta^t \{q_t - c_t\}$.



Spear and Srivastava (1987) {cite}`Spear_Srivastava_87` described a recursive representation  of an optimal contract.


## Recursive Contract

Let $w$ denote the  expected value of discounted  continuation utilities that the principal had   promised to the agent.
  

Given $w$, the principal designs three functions:

*  $a(w)$ determines a requested  action so that the action at $t$ is $a_t=a(w_t)$
*  $c(w,q)$ determines current consumption so that $c_t=c(w_t, q_t)$
*  $\tilde w(w,q)$  determines a promised next period expected discounted 
utility so that $w_{t+1} = \tilde w (w_t, q_t)$.

The functions $a(w)$, $c(w,q)$, and $\tilde w (w,q)$
must satisfy the following two sets of  constraints:

$$
w = \int \{ u[c(w,q), a(w)] + \beta \tilde w(w,q)\}\ dF[q | a(w)]
$$ (eq:eq1)

and

$$
\begin{aligned} & \int \left\{ u[c(w,q), a(w)] + \beta\tilde w (w,q)\right\}\
dF[q\vert a(w))\cr
&\geq \int \{u [c(w,q),\hat a] + \beta\tilde w
(w,q)\} dF(q\vert\hat a)\,, \hskip.5cm \forall\; \hat a \in A.
\end{aligned}
$$ (eq:eq2)

 * Equation {eq}`eq:eq1`  requires the contract to deliver the promised
discounted expected continuation utility. 
 * Equation {eq}`eq:eq2`  is the **incentive
compatibility** constraint that requires that the agent chooses to
deliver the amount of effort called for by the
contract. 

Let $v(w)$ be the value to the principal associated with promising discounted utility $w$ to the agent. 

The principal's optimal value function satisfies the Bellman equation 

$$ 
v(w) =\max_{a,c,\tilde w}\ \{q-c(w,q)+\beta \ v[\tilde w(w,q)]\}\
dF[q\vert a(w)]
$$ (eq:eq3)

where maximization is over functions $a(w)$, $c(w,q)$, and $\tilde w(w,q)$
and is subject to the constraints {eq}`eq:eq1` and {eq}`eq:eq2`.

Notice that constraint {eq}`eq:eq1` itself takes the form of a Bellman equation.

Thus,  a value function $w$ that satisfies a Bellman equation that describes the agent's continuation value    is an argument of Bellman equation {eq}`eq:eq3` that describes the principal's continuation value. 

The value function $v(w)$ and the associated optimum policy functions
are to be solved by iterating on the Bellman operator associated with Bellman equation {eq}`eq:eq3`.

## Lotteries

A difficulty in problems like these is that   incentive
constraints   the constraint set can  fail to
be convex. 

Phelan and
Townsend (1991) {cite}`Phelan_Townsend_91` circumvented this problem  by using **randomization** to  convexify the constraint set.

Phelan and Townsend restrict outputs $y$, consumption $c$, actions $a$, and continuation values $w$ each to be
in its own  discrete and finite  set.

They restrict the 
principal's choice to a space of lotteries
over actions $a$ and outcomes $c,w'$.

To describe Phelan and Townsend's formulation, let $P(q\vert a)$ be
a family of discrete conditional probability distributions
over discrete spaces of outputs and actions $Q,A$, respectively.

Suppose that consumption and  continuation values are  constrained to
lie in discrete spaces $C,W$, respectively.

Phelan and Townsend's principal 
chooses a probability distribution
 $\Pi(a,q,c,w^\prime)$ subject first to the constraint
 that for all fixed $(\bar a, \bar q)$

$$
\sum_{C\times W} \Pi (\bar a, \bar q, c, w^\prime) = P (\bar q\vert \bar a)
\sum_{Q\times C\times W}\ \Pi(\bar a, q,c,w')
$$ (eq:town1a)

$$
\Pi(a,q,c,w')\geq 0 
$$(eq:town1b)


$$
\sum_{A\times Q\times C\times W}\ \Pi(a,q,c,w^\prime)=1 .
$$ (eq:town1c)


Equation {eq}`eq:town1a`  states that
${\rm Prob} (\bar a, \bar q) = {\rm Prob}(\bar q \vert \bar a)
{\rm Prob}(\bar a)$.

Remaining parts of the preceding three equations  just
require that **probabilities are probabilities** in the sense that

* they are nonnegative
* sum to $1$

The counterpart of Spear-Srivastava's equation {eq}`eq:eq1`  is

$$
w=\sum_{A\times Q\times C\times W}\ \{u(c,a) +\beta w^\prime\}\
\Pi(a,q,c,w^\prime) . 
$$ (eq:eq1prime)

A counterpart to Spear-Srivastava's incentive constraint  {eq}`eq:eq2`   for each
$a,\hat a$ is


$$
\begin{aligned} & \sum_{Q\times C\times W}\ \{u(c,a)  + \beta w' \}\ \Pi (c,w' \vert q, a) P(q\vert a)\\
&\geq \sum_{Q\times C\times W}\ \{u(c,\hat a) + \beta w' \}\ \Pi(c,w' \vert q,a) P(q\vert\hat a).
\end{aligned}
$$ (eq:eq2prime)


Here $\Pi(c,w^\prime\vert q,a) P(q\vert \hat a)$ is the probability attached to  $(c,w^\prime, q)$ 
if the agent claims to be working $a$ but is actually working $\hat a$. 

Write

$$
\begin{aligned}\Pi(c,w^\prime\vert q,a) P(q\vert\hat a)  = 
\Pi(c,w^\prime\vert q,a) P(q\vert a)\ \frac{P(q\vert\hat a)}{P(q\vert a)} & =
\Pi(c,w^\prime,q\vert a)\ \cdot\ \frac{P(q\vert\hat a)}{P(q\vert a)}.
\end{aligned}
$$

Write the incentive constraints as

$$
\begin{aligned} \sum_{Q\times C\times W}\ &\{u(c,a)  +\beta w^\prime\} \Pi(c,w^\prime, q\vert a)\cr & \geq
\sum_{Q\times C\times W}\ \{u(c,\hat a) +\beta w^\prime\}\ \Pi(c,w^\prime, q\vert \hat a)\     \cdot\ {P(q\vert \hat a)\over P(q\vert a)}.
\end{aligned}
$$

Multiplying both sides by the unconditional probability $P(a)$ gives Phelan and Townsend's counterpart to Spear and Srivastava's  incentive constraint  {eq}`eq:eq2`.

$$
\begin{aligned} & \sum_{Q\times C\times W}\ \{u(c,a)+\beta w^\prime\}\
\Pi(a,q,c,w^\prime)\cr 
&\geq \sum_{Q\times C\times W}\ \{u(c,\hat a) + \beta w^\prime\} \
{P(q\vert\hat a)\over P(q\vert a)}\ \Pi (a,q,c,w^\prime)
\end{aligned}
$$

## Linear Programming Formulation


The value function  for the principal's problem satisfies the Bellman equation 

$$
v(w) =\max_{\Pi} \{(q -c) +
     \beta v(w')\} \Pi(a,q,c,w') , 
$$ (eq:bell2)

where  maximization is over the probabilities $\Pi(a,q,c,w')$
subject to equations {eq}`eq:town1a`, {eq}`eq:town1b`,  {eq}`eq:eq1prime`, and {eq}`eq:eq2prime`.
 

This is a **linear
programming** problem. 

Each  $(a,q,c,w')$ is constrained to a discrete grid of points.

The term $(q-c)+\beta v(w')$ on the right side of equation {eq}`eq:bell2` 
can be represented as a *fixed* vector that multiplies a vectorized
version of  the
probabilities $\Pi(a,q,c,w')$.  

Similarly, each of the
constraints  {eq}`eq:town1a`, {eq}`eq:town1b`,  {eq}`eq:eq1prime`, and {eq}`eq:eq2prime` can be represented
as a linear inequality in the choice variables, namely,  the
probabilities $\Pi$.  

An associated **Bellman operator** is also a linear program.

Phelan and Townsend iterated to convergence on this Bellman operator.  

At each step of the iteration on the  Bellman operator,
one linear program is to be solved for each point
$w$ in the space of grid values for $W$.

Phelan and Townsend found that
lotteries were often redundant in the sense that most of the
$\Pi(a,q,c,w')$'s  are  zero and only a few are $1$.
