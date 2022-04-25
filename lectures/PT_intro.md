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

Spear and Srivastava (1987) {cite}`Spear_Srivastava_87` presented the following
recursive formulation of an infinitely repeated, discounted  repeated
principal-agent problem.

*  A **principal** owns a technology
that produces output $q_t$ at time $t$ according to a probability distribution $F$ that depends on the action $a_t$ of an **agent**.
*  $q_t$ is determined by a family of c.d.f.'s  $F(q_t\vert a_t)$
*  $a_t$ is an action taken at the beginning of $t$ by the **agent** who
operates the technology.  
*  The principal  does not observe $a_t$ but does observe $q_t$ at the end of period $t$, ane the principal remembers earlier $q_{t-j}$'s.
*  The principal has access to an outside loan market with constant risk-free gross interest rate $\beta^{-1}$.
*  The agent has preferences over consumption streams ordered by $E_0 \sum^\infty_{t=0} \beta^t u(c_t, a_t).$
*  The principal is risk neutral and offers a contract to
the agent  that the principal designs to maximize $E_0 \sum^\infty_{t=0} \beta^t \{q_t - c_t\}$
where $c_t$ is the principal's payment to the agent at $t$.

The principal designs a  contract that he offers to offer an agent.  

A **contract**  $\sigma$ is a sequence of functions $\sigma_t$ whose $t$th component maps the history $q^{t-1} = q_{t-1}, q_{t-2}, \ldots , q_0$ of past outputs observed at time $t$ into a recommended time $t+1$ action $a_t$.  

Spear and Srivastava (1987) {cite}`Spear_Srivastava_87` described a recursive formulation of an optimal contract.



### Timing
Let $w$ denote the discounted present value of expected utility promised to the agent
at the beginning of a period.  

Given $w$, the principal designs three functions $a(w)$, $c(w,q)$, and $\tilde w(w,q)$ that 
determine  current action $a_t=a(w_t)$,
 current consumption $c_t=c(w_t, q_t)$, and a promised next period expected discounted 
utility $w_{t+1} = \tilde w (w_t, q_t)$.

The principal's choice of the three functions $a(w)$, $c(w,q)$, and $\tilde w (w,q)$
must satisfy the following two sets of constraints:

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
level of discounted utility. 
 * Equation {eq}`eq:eq2`  is the **incentive
compatibility** constraint that requires that the agent choose to
deliver the amount of effort called for in the
contract. 

Let $v(w)$ be the value to the principal associated with promising discounted utility $w$ to the agent. 

The principal's optimal value function satisfies the Bellman equation 

$$ 
v(w) =\max_{a,c,\tilde w}\ \{q-c(w,q)+\beta \ v[\tilde w(w,q)]\}\
dF[q\vert a(w)]
$$ (eq:eq3)

where the maximization is over functions $a(w)$, $c(w,q)$, and $\tilde w(w,q)$
and is subject to the constraints {eq}`eq:eq1` and {eq}`eq:eq2`.

Notice that constraint {eq}`eq:eq1` itself takes the form of a Bellman equation.

So a value function $w$ that satisfies a Bellman equation   is an argument of Bellman equation {eq}`eq:eq3`.

This value function $v(w)$ and the associated optimum policy functions
are to be solved by iterating on the Bellman equation {eq}`eq:eq3`.

## Lotteries

A difficulty in problems like these is that   incentive
constraints might make the constraint set fail to
be convex. 

Phelan and
Townsend (1991) {cite}`Phelan_Townsend_91` circumvented this problem  by using **randomization** to  convexify the constraint set.

Phelan and Townsend alter the problem by extending the
principal's choice to a space of lotteries
over actions $a$ and outcomes $c,w'$.

To introduce Phelan and Townsend's formulation, let $P(q\vert a)$ be
a family of discrete conditional probability distributions
over discrete spaces of outputs and actions $Q,A$, and
suppose that consumption and  continuation values are also both constrained to
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
require that *probabilities are probabilities*.

The counterpart of Spear-Srivastava's equation {eq}`eq:eq1`  is

$$
w=\sum_{A\times Q\times C\times W}\ \{u(c,a) +\beta w^\prime\}\
\Pi(a,q,c,w^\prime) . 
$$ (eq:eq1prime)

The counterpart to Spear-Srivastava's equation {eq}`eq:eq2`   for each
$a,\hat a$ is


$$
\begin{aligned} & \sum_{Q\times C\times W}\ \{u(c,a)  + \beta w' \}\ \Pi (c,w' \vert q, a) P(q\vert a)\\
&\geq \sum_{Q\times C\times W}\ \{u(c,\hat a) + \beta w' \}\ \Pi(c,w' \vert q,a) P(q\vert\hat a).
\end{aligned}
$$ (eq:eq2prime)


Here $\Pi(c,w^\prime\vert q,a) P(q\vert \hat a)$ is the probability of $(c,w^\prime, q)$ 
if the agent claims to be working $a$ but is actually working $\hat a$. 

Write

$$
\begin{aligned}\Pi(c,w^\prime\vert q,a) P(q\vert\hat a) & = \\
\Pi(c,w^\prime\vert q,a) P(q\vert a)\ \frac{P(q\vert\hat a)}{P(q\vert a)} & =
\Pi(c,w^\prime,q\vert a)\ \cdot\ \frac{P(q\vert\hat a)}{P(q\vert a)}.
\end{aligned}
$$

Write the incentive constraint as

$$
\begin{aligned} \sum_{Q\times C\times W}\ &\{u(c,a)  +\beta w^\prime\} \Pi(c,w^\prime, q\vert a)\cr & \geq
\sum_{Q\times C\times W}\ \{u(c,\hat a) +\beta w^\prime\}\ \Pi(c,w^\prime, q\vert \hat a)\     \cdot\ {P(q\vert \hat a)\over P(q\vert a)}.
\end{aligned}
$$

Multiplying both sides by the unconditional probability $P(a)$ gives expression {eq}`eq:eq2`.

$$
\begin{aligned} & \sum_{Q\times C\times W}\ \{u(c,a)+\beta w^\prime\}\
\Pi(a,q,c,w^\prime)\cr 
&\geq \sum_{Q\times C\times W}\ \{u(c,\hat a) + \beta w^\prime\} \
{P(q\vert\hat a)\over P(q\vert a)}\ \Pi (a,q,c,w^\prime)
\end{aligned}
$$

The value function  for the principal's problem satisfies the Bellman equation 

$$
v(w) =\max_{\Pi} \{(q -c) +
     \beta v(w')\} \Pi(a,q,c,w') , 
$$ (eq:bell2)

where  maximization is over the probabilities $\Pi(a,q,c,w')$
subject to equations {eq}`eq:town1a`, {eq}`eq:town1b`,  {eq}`eq:eq1prime`, and {eq}`eq:eq2prime`.
 

This is a **linear
programming** problem. 

Think of each of $(a,q,c,w')$
being constrained to a discrete grid of points.

The term $(q-c)+\beta v(w')$ on the right side of equation {eq}`eq:bell2` 
can be represented as a *fixed* vector that multiplies a vectorized
version of  the
probabilities $\Pi(a,q,c,w')$.  

Similarly, each of the
constraints  {eq}`eq:town1a`, {eq}`eq:town1b`,  {eq}`eq:eq1prime`, and {eq}`eq:eq2prime` can be represented
as a linear inequality in the choice variables, namely,  the
probabilities $\Pi$.  

Phelan and Townsend compute solutions
of these linear programs to
iterate on the Bellman equation {eq}`eq:bell2`.  

At each step of the iteration on the  Bellman equation,
one linear program is to be solved for each point
$w$ in the space of grid values for $W$.

Phelan and Townsend found that
lotteries were often redundant in the sense that most of the
$\Pi(a,q,c,w')$'s  are  zero and only a few are $1$.
