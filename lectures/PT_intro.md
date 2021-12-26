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

# Title

## Spear and Srivastava

Spear and Srivastava (1987) introduced the following
recursive formulation of an infinitely repeated, discounted  repeated
principal-agent problem:  A **principal** owns a technology
that produces output $q_t$ at time $t$, where $q_t$ is determined
by a family of c.d.f.'s  $F(q_t\vert a_t)$, and $a_t$ is an
action taken at the beginning of $t$ by an **agent** who
operates the technology.  The principal has access to an outside
loan market with constant risk-free gross interest rate $\beta^{-1}$.
The agent has preferences over consumption streams ordered by
$E_0 \sum^\infty_{t=0} \beta^t u(c_t, a_t).$
The principal is risk neutral and offers a contract to
the agent designed to maximize
$E_0 \sum^\infty_{t=0} \beta^t \{q_t - c_t\}$
where $c_t$ is the principal's payment to the agent at $t$.


### Timing
Let $w$ denote the discounted utility promised to the agent
at the beginning of the period.  Given $w$, the principal
selects three functions $a(w)$, $c(w,q)$, and $\tilde w(w,q)$
determining the current action $a_t=a(w_t)$,
the current consumption $c_t=c(w_t, q_t)$, and a promised
utility $w_{t+1} = \tilde w (w_t, q_t)$.
The choice of the three functions $a(w)$, $c(w,q)$, and $\tilde w (w,q)$
must satisfy the following two sets of constraints:

$$
w = \int \{ u[c(w,q), a(w)] + \beta \tilde w(w,q)\}\ dF[q\vert a(w)]
$$ (eq:eq1}

and

$$
\begin{align}\int &\{ u[c(w,q), a(w)] + \beta\tilde w (w,q)\}\
dF[q\vert a(w)]\cr
&\geq \int \{u [c(w,q),\hat a] + \beta\tilde w
(w,q)\} dF(q\vert\hat a)\,, \hskip.5cm \forall\; \hat a \in A.
\end{align}
$$ (eq:eq2)

Equation \Ep{1} requires the contract to deliver the promised
level of discounted utility. Equation \Ep{2} is the **incentive
compatibility** constraint requiring the agent to want to
deliver the amount of effort called for in the
contract. 

Let $v(w)$ be the value to the principal associated with promising discounted utility $w$ to the agent.  The principal's Bellman equation is

$$ 
v(w) =\max_{a,c,\tilde w}\ \{q-c(w,q)+\beta \ v[\tilde w(w,q)]\}\
dF[q\vert a(w)]
$$ (eq:eq3}

where the maximization is over functions $a(w)$, $c(w,q)$, and $\tilde w(w,q)$
and is subject to the constraints \Ep{1} and \Ep{2}.
This value function $v(w)$ and the associated optimum policy functions
are to be solved by iterating on the Bellman equation \Ep{3}.

## Use of lotteries

A difficulty in problems like thes can be that the the structure of the incentive
constraints makes the constraint set fail to
be convex.  This problem has been overcome by Phelan and
Townsend (1991) by convexifying the constraint set through **randomization**.
Phelan and Townsend alter the problem by extending the
principal's choice to the space of lotteries
over actions $a$ and outcomes $c,w'$.

To introduce Phelan and Townsend's formulation, let $P(q\vert a)$ be
a family of discrete conditional probability distributions
over discrete spaces of outputs and actions $Q,A$, and
imagine that consumption and values are also constrained to
lie in discrete spaces $C,W$, respectively.
Phelan and Townsend instruct  the principal to
choose a probability distribution
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


Equation \Ep{town1;a}  states that
${\rm Prob} (\bar a, \bar q) = {\rm Prob}(\bar q \vert \bar a)
{\rm Prob}(\bar a)$.
The remaining pieces of the preceding three equations  just
require that ``probabilities are probabilities.''
The counterpart of Spear-Srivastava's equation \Ep{1} is

$$
w=\sum_{A\times Q\times C\times W}\ \{u(c,a) +\beta w^\prime\}\
\Pi(a,q,c,w^\prime) . 
$$ (eq:eq1prime)

The counterpart to Spear-Srivastava's equation \Ep{2}  for each
$a,\hat a$ is

$$
\begin{align}\sum_{Q\times C\times W}\ &\{u(c,a) + \beta w' \}\ \Pi (c,w' \vert q, a) P(q\vert a)\cr
  &\geq \sum_{Q\times C\times W}
\ \{u(c,\hat a) + \beta w' \}\ \Pi(c,w' \vert q,a) P(q\vert\hat a).
\end{align)
$$

Here $\Pi(c,w^\prime\vert q,a) P(q\vert \hat a)$ is the probability of $(c,w^\prime, q)$ 
if the agent claims to be working $a$ but is actually working $\hat a$.  Express

$$
\begin{aligned}\Pi(c,w^\prime\vert q,a) P(q\vert\hat a) & = \cr
\Pi(c,w^\prime\vert q,a) P(q\vert a)\ {P(q\vert\hat a)\over P(q\vert a)} & =
\Pi(c,w^\prime,q\vert a)\ \cdot\ {P(q\vert\hat a)\over P(q\vert a).
\end{aligned)
$$

Write the incentive constraint as

$$
\begin{aligned} \sum_{Q\times C\times W}\ &\{u(c,a)
 +\beta w^\prime\} \Pi(c,w^\prime, q\vert a)\cr & \geq
\sum_{Q\times C\times W}\ \{u(c,\hat a) +\beta w^\prime\}\
\Pi(c,w^\prime, q\vert \hat a)\     \cdot\ {P(q\vert \hat a)\over P(q\vert a)}.
\end{aligned}
$$

Multiplying both sides by the unconditional probability $P(a)$ gives expression \Ep{2'}.

$$
\begin{aligned} & \sum_{Q\times C\times W}\ \{u(c,a)+\beta w^\prime\}\
\Pi(a,q,c,w^\prime)\cr 
&\geq \sum_{Q\times C\times W}\ \{u(c,\hat a) + \beta w^\prime\} \
{P(q\vert\hat a)\over P(q\vert a)}\ \Pi (a,q,c,w^\prime)
\end{aligned}
$$

The Bellman equation for the principal's problem is

$$
v(w) =\max_{\Pi} \{(q -c) +
     \beta v(w')\} \Pi(a,q,c,w') , 
$$ (eq:bell2)

where  maximization is over the probabilities $\Pi(a,q,c,w')$
subject to equations \Ep{town1}, \Ep{1'}, and \Ep{2'}.
%The problem on the right side of equation \Ep{bell2} 
This is a linear
programming problem.  Think of each of $(a,q,c,w')$
being constrained to a discrete grid of points.  Then, for example,
the term $(q-c)+\beta v(w')$ on the right side of equation \Ep{bell2}
can be represented as a {\it fixed} vector that multiplies a vectorized
version of  the
probabilities $\Pi(a,q,c,w')$.  Similarly, each of the
constraints \Ep{town1}, \Ep{1'}, and \Ep{2'} can be represented
as a linear inequality in the choice variables, the
probabilities $\Pi$.   Phelan and Townsend compute solutions
of these linear programs to
iterate on the Bellman equation \Ep{bell2}.   Note that
at each step of the iteration on the  Bellman equation,
there is  one linear program to be solved for each point
$w$ in the space of grid values for $W$.

In practice, Phelan and Townsend have found that
lotteries are often redundant in the sense that most of the
$\Pi(a,q,c,w')$'s  are  zero and only a few are $1$.
