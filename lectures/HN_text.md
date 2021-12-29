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

# Optimal Unemployment Compensation


This notebook describes a model of optimal unemployment
compensation along the lines of Shavell and Weiss (1979) and
Hopenhayn and Nicolini (1997).

We shall use the techniques of
Hopenhayn and Nicolini to analyze a model closer to Shavell and
Weiss's. 

An unemployed worker orders stochastic processes of
consumption and  search effort $\{c_t , a_t\}_{t=0}^\infty$
according to

$$ 
E \sum_{t=0}^\infty \beta^t \left[ u(c_t) - a_t \right]  
$$ (eq:hugo1)
%\EQN hugo1

where $\beta \in (0,1)$ and $u(c)$ is strictly increasing, twice differentiable,
and strictly concave.  

We assume that $u(0)$ is well defined.

We require that $c_t \geq 0$ and $ a_t \geq 0$.

All jobs are alike and pay wage
$w >0$ units of the consumption good each period forever. 

An unemployed
worker searches with effort $a$ and with probability
$p(a)$ receives a permanent job at the beginning
of the next period. 

Once
a worker has found a job, he is beyond the planner's grasp.

(This is Shavell and
Weiss's assumption, but not Hopenhayn and Nicolini's. Hopenhayn
and Nicolini allow the unemployment insurance agency to
impose history-dependent taxes on previously unemployed workers.
Since there is no incentive problem after the worker has found
a job, it is optimal for the agency to provide an employed worker with
a constant level of consumption, and hence, the agency imposes
a permanent per-period history-dependent tax on a previously
unemployed worker.)

Furthermore, $a=0$ when the worker is
employed. 

The probability
of finding a job  is $p(a)$ where $p$ is an increasing, strictly concave,
 and twice differentiable function of $a$ that satisfies
$p(a)   \in [0,1]$ for  $a \geq 0$, $p(0)=0$.

The consumption good is nonstorable.

The unemployed worker has no savings and cannot borrow or lend.
The  insurance agency is the unemployed worker's only source of consumption
smoothing over time and across states.


### autarky problem

As a benchmark, we first study the fate of the unemployed worker
who has no access to unemployment insurance. 

Because employment
is an absorbing state for the worker, we work backward from that
state.

Let $V^e$ be the expected sum of discounted one-period utilities of an employed worker.

Once the worker is employed, $a=0$, making  his period utility
be $u(c)-a = u(w)$ forever.

Therefore,

$$  V^e = {u(w) \over (1-\beta)}  . 
$$ (eq:hugo2)
% \EQN hugo2

Now let $V^u$ be the expected present value of utility for an
unemployed worker who chooses  current effort $a$ %period pair $(c,a)$
optimally. 

It satisfies the Bellman equation  

$$ 
V^u = \max_{a \geq 0} \biggl\{ u(0) - a + \beta \left[
   p(a) V^e + (1-p(a)) V^u \right] \biggr\} .  
$$ (eq:hugo3)
%EQN hugo3
   
The first-order condition for this problem is

$$ 
\beta p'(a) \left[V^e - V^u \right] \leq 1 , 
$$ (eq:hugo4)
%\EQN hugo4 

with equality if $a>0$. 

Since there is no state variable in this
infinite horizon problem, there is a time-invariant optimal
search intensity $a$ and an associated value of being unemployed $V^u$.

Let $V_{\rm aut} = V^u$ denote the solution of Bellman equation {eq}`eq:hugo3`. 

Equations {eq}`eq:hugo3` 
 and {eq}`eq:hugo4` 
form the basis for
an iterative algorithm for computing $V^u = V_{\rm aut}$. 

 * Let $V^u_j$ be
the estimate of $V_{\rm aut}$ at the $j$th iteration.   

 * Use this value
in equation {eq}`eq:hugo4}  and solve
for an estimate of effort  $a_j$. 

 * Use this value in a version of equation
{eq}`eq:hugo3` with $V^u_j$ on the right side
to compute $V^u_{j+1}$.  

 * Iterate to convergence.

### Unemployment insurance with full information

As another benchmark, we study the provision of insurance with
full information.  

An insurance agency can observe and control
the unemployed person's consumption and search effort. 

The
agency wants to design an unemployment insurance contract to give
the unemployed worker expected discounted utility $V > V_{\rm aut}$.

The planner wants to deliver value $V$ in the most efficient way,
meaning the way that minimizes expected
 discounted cost, using $\beta$ as the discount factor.
 
We formulate the optimal insurance problem
recursively.  

Let $C(V)$ be the expected discounted cost of giving
the worker expected discounted utility
$V$.

The cost function is strictly convex because
a higher $V$ implies a lower marginal utility of the worker;
that is, additional expected utils can be awarded to the worker
only at an increasing marginal cost in terms of the consumption good.


Given $V$, the planner assigns first-period pair   $(c,a)$ and promised
continuation value $V^u$, should  the worker  be unlucky
and not find a job; $(c, a, V^u)$ will all be
chosen to be functions of $V$ and to
satisfy the Bellman equation 

$$ 
C(V) = \min_{c, a, V^u} \biggl\{ c  + \beta [1 - p(a)] C(V^u) \biggr\} ,
$$ (eq:hugo5)
%\EQN hugo5
where  minimization is subject to the promise-keeping constraint

$$
 V \leq u(c) - a + \beta
\left\{ p(a) V^e + [1-p(a)] V^u \right\}.
$$ (eq:hugo6)
%\EQN hugo6

Here $V^e$ is given by equation {eq}`eq:hugo2`, which reflects the
assumption that once the worker is employed, he is beyond the
reach of the unemployment insurance agency.


The right side of  Bellman equation \Ep{hugo5} is attained by
policy functions $c=c(V), a=a(V)$, and $V^u=V^u(V)$.

The promise-keeping constraint,
 equation  {eq}`eq:hugo6`, 
asserts that the 3-tuple $(c, a, V^u)$ attains
at least $V$.   

Let $\theta$ be a Lagrange multiplier
on constraint {eq}`eq:hugo6`. 

At an interior solution, the first-order
conditions with
respect to $c, a$, and $V^u$, respectively, are

$$
\begin{align} \theta & = {1 \over u'(c)}\,,   \cr
             C(V^u) & = \theta \left[ {1 \over \beta p'(a)} -
                           (V^e - V^u) \right]\,,    \cr
             C'(V^u) & = \theta\,.   
\end{align}
$$ (eq:hugo7)

The envelope condition   $C'(V) = \theta$ and the third equation
of {eq}`eq:hugo7`  imply that $C'(V^u) =C'(V)$. 

Strict convexity of $C$ then
implies that $V^u =V$

Applied repeatedly over time,
$V^u=V$ makes
the continuation value remain constant during the entire
spell of unemployment.  

The first equation of {eq}`eq:hugo7`
determines $c$, and the second equation of {eq}`eq:hugo7`  determines
$a$, both as functions of the promised $V$. 

That $V^u = V$ then
implies that $c$ and $a$ are held constant during the unemployment
spell. 

Thus, the unemployed worker's consumption $c$ and search effort $a$ are both fully smoothed
 during the unemployment
spell.

But
the worker's consumption is not  smoothed across states of
employment and unemployment unless $V=V^e$.

### The incentive problem

The preceding efficient insurance scheme requires that the insurance agency
control both $c$ and $a$. 

It will not do for the insurance agency
simply to announce $c$ and then allow the worker to choose $a$.

Here is why.

The  agency delivers a value $V^u$  higher than
the autarky value $V_{\rm aut}$ by doing two things.

It *increases* the unemployed worker's consumption $c$ and *decreases* his search
effort $a$.

But the prescribed
search effort is *higher* than what the worker would choose
if he were to be guaranteed consumption level $c$ while he
remains unemployed.

This follows from the first two equations of {eq}`eq:hugo7` and the
fact that the insurance scheme is costly, $C(V^u)>0$, which imply
$[ \beta p'(a) ]^{-1} > (V^e - V^u)$.

But look at the worker's
first-order condition {eq}`eq:hugo4`  under autarky.

It implies that if search effort $a>0$, then
$[\beta p'(a)]^{-1} = [V^e - V^u]$, which is inconsistent
with the preceding inequality
$[ \beta p'(a) ]^{-1} > (V^e - V^u)$ that prevails when $a >0$ under
the social
insurance arrangement.

If he were free to choose $a$, the worker would therefore want to
fulfill {eq}`eq:hugo4`, either at equality so long as $a >0$, or by setting
$a=0$ otherwise.  

Starting from the  $a$ associated with
the social insurance scheme,
he  would establish the desired equality
in {eq}`eq:hugo4` by *lowering* $a$, thereby decreasing
the term $[ \beta p'(a) ]^{-1}$ (which also lowers $(V^e - V^u)$
when the value of being
unemployed $V^u$ increases]). 


If an equality can be established before
$a$ reaches zero, this would be the worker's preferred search effort;
otherwise the worker would find it optimal to accept the insurance
payment, set $a=0$,  and  never work again.

Thus, since the worker does not take the
cost of the insurance scheme into account, he would choose a search
effort below the socially optimal one.

The efficient contract
relies on  the agency's ability to control *both* the unemployed
worker's consumption *and* his search effort.

## Unemployment insurance with asymmetric information}

Following Shavell and Weiss (1979) and Hopenhayn and Nicolini
(1997), now assume that  the unemployment insurance agency cannot
observe or enforce $a$, though it can observe and control $c$.

The worker is free to choose $a$, which puts expression {eq}`eq:hugo4`, the worker's first-order condition under autarky,
back in the picture.

(We are assuming that the worker's
best response to the unemployment insurance arrangement is
completely characterized by the first-order condition {eq}`eq:hugo4`,
an instance of the so-called first-order approach to incentive problems.)

Given a contract, the individual will choose search effort according to
first-order condition {eq}`eq:hugo4`. 

This fact leads the insurance agency
to design the unemployment insurance contract to respect this restriction.

Thus, the recursive contract design problem is now to minimize the right side of equation
{eq}`eq:hugo5` subject to expression {eq}`eq:hugo6` and the incentive constraint {eq}`eq:hugo4`.

Since the restrictions {eq}`eq:hugo4` and {eq}`eq:hugo6` are not linear
and generally do not define a convex set, it becomes difficult
to provide conditions under which the solution to the dynamic
programming problem results in a convex function $C(V)$.

As
discussed in XXXX Appendix A of chapter socialinsurance,
this complication can be handled by convexifying
the constraint set through the introduction of lotteries.

However, a common finding is that optimal plans do not involve
lotteries, because convexity of the constraint set is a sufficient
but not necessary condition for convexity of the cost function.
Following Hopenhayn and Nicolini (1997), we therefore proceed
under the assumption that $C(V)$ is strictly convex in order to
characterize the optimal solution.

Let $\eta$ be the multiplier on constraint {eq}`eq:hugo4`, while
$\theta$ continues to denote the multiplier on constraint {eq}`eq:hugo6`.

But now we replace the weak inequality in {eq}`eq:hugo6` by an equality.

The unemployment insurance agency cannot award a higher utility than
$V$ because that might violate an incentive-compatibility constraint
for exerting the proper search effort in earlier periods.

At an interior solution,  first-order conditions with
respect to $c, a$, and $V^u$, respectively, are

$$
\begin{align} \theta & = {1 \over u'(c)}\,,   \cr
 C(V^u)  & = \theta \left[ {1 \over \beta p'(a)} - (V^e - V^u) \right]
            \,-\, \eta {p''(a) \over p'(a)} (V^e - V^u)                  \cr
         & = \,- \eta {p''(a) \over p'(a)} (V^e - V^u) \,,   \cr
 C'(V^u) & = \theta \,-\, \eta {p'(a) \over 1-p(a)}\, ,  
\end{align} 
$$ (eq:hugo8)

where the second equality in the second equation in {eq}`eq:hugo8`  follows from strict equality
of the incentive constraint {eq}`eq:hugo4` when $a>0$.

As long as the
insurance scheme is associated with costs, so that $C(V^u)>0$, first-order
condition in the second equation of {eq}`eq:hugo8` implies that the multiplier $\eta$ is strictly
positive. 

The first-order condition in the second equation of the third equality in {eq}`eq:hugo8`  and the
envelope condition $C'(V) = \theta$ together allow us to conclude that
$C'(V^u) < C'(V)$. 

Convexity of $C$ then implies that $V^u < V$.


After we have also used e the first equation of {eq}`eq:hugo8`, it follows that
in order to provide  the proper incentives, the consumption
of the unemployed worker must decrease as the duration of the unemployment
spell lengthens. 

It also follows from {eq}`eq:hugo4` at equality that
search effort $a$ rises as $V^u$ falls, i.e., it rises with the duration
of unemployment.

The duration dependence of benefits is  designed to provide
incentives to search.  

To see this, from  the third equation of {eq}`eq:hugo8`, notice how
the conclusion that consumption falls with the duration of
unemployment depends on the assumption that more search effort
raises the prospect of finding a job, i.e., that $p'(a) > 0$. 

If
$p'(a) =0$, then  the third equation of {eq}`eq:hugo8` and the strict convexity of $C$ imply that
$V^u =V$. 

Thus, when $p'(a) =0$, there is no reason for the
planner to make consumption fall with the duration of
unemployment.

```{code-cell} ipython3

```

or parameters chosen by Hopenhayn and Nicolini, Figure \Fg{hugoreplnewf} %15.5
displays the
replacement ratio $c / w$ as a function of  the duration of the unemployment
spell.\NFootnote{This figure was computed using the Matlab programs
{\tt hugo.m}, {\tt hugo1a.m}, {\tt hugofoc1.m}, {\tt valhugo.m}.
These are available in the subdirectory {\tt hugo}, which
contains a readme file.  These programs were composed
by various members of Economics 233 at Stanford in 1998,
especially Eva Nagypal, Laura Veldkamp, and Chao Wei.}
 This schedule was computed by finding the optimal policy functions
$$ \eqalign{ V^u_{t+1} & = f(V^u_t) \cr
             c_t & = g(V^u_t). \cr} $$
and iterating on them, starting from some initial $V^u_0 > V_{\rm aut}$,
where $V_{\rm aut}$ is the autarky level for an unemployed worker.
Notice how the replacement ratio declines with duration.
Figure \Fg{hugoreplnewf} %15.5
sets $V^u_0$ at 16,942, a number that
has to be interpreted in the context of Hopenhayn and Nicolini's
parameter settings.

We computed these numbers using the parametric version studied by Hopenhayn
and Nicolini.\NFootnote{In  section \use{HNalgorithm}, %of chapter \use{practical},
we described a computational strategy
of iterating to convergence on the Bellman equation \Ep{hugo5}, subject
to expressions \Ep{hugo6} at equality, and \Ep{hugo4}.}
  Hopenhayn and Nicolini chose parameterizations and parameters as follows:
They interpreted one  period as one week, which led them
to set $\beta=.999$.  They took  $u(c) = {c^{(1-\sigma)} \over 1 - \sigma}$
and set
$\sigma=.5$.  They set the wage $w=100$ and
specified the hazard function to be  $p(a) = 1 - \exp(-ra)$, with $r$ chosen
to give a hazard rate $ p(a^*) = .1$, where
$a^*$ is the optimal search  effort under autarky. To compute the numbers
in Figure \Fg{hugoreplnewf} we used
these same settings.

+++


\subsection{Computational details}
Exercise {\it \the\chapternum.1\/} asks the reader  to solve the Bellman equation
numerically. \index{Bellman equation}%
In doing so, it is useful to note that there
%%is a natural upper bound to the set
%%of continuation values $V^u$.  To  compute it,
are natural lower and upper bounds to the set
of continuation values $V^u$. The lower bound is
the expected lifetime utility in autarky,
$V_{\rm aut}$. To compute the upper bound,
represent condition \Ep{hugo4} as
$$ V^u \geq V^e  - [\beta  p'(a)]^{-1},$$
with equality if $ a > 0$.
If there is zero search effort, then $V^u \geq V^e -[\beta p'(0)]^{-1}$.  %% >
Therefore, to rule out zero search effort we require
$$ V^u < V^e - [\beta p'(0)]^{-1} .$$                                     %% \leq
(Remember that $p''(a) < 0$.)  This step gives  our upper bound
for $V^u$.

To formulate the Bellman equation numerically,
we suggest using the constraints to eliminate $c$ and $a$ as choice
variables, thereby reducing the Bellman equation to
a minimization over the one choice variable $V^u$.
First express the promise-keeping constraint \Ep{hugo6} as
$u(c) = V + a - \beta \{p(a) V^e +[1-p(a)] V^u \}$.                       %% \geq
That is, consumption is equal to
$$ c = u^{-1}\left(
     V+a -\beta [p(a)V^e + (1-p(a))V^u] \right). \EQN hugo21$$
%%For the preceding utility function,
%%whenever the right side of this
%%inequality is negative, then this  promise-keeping constraint  is
%%not binding and can be satisfied with $c=0$.   This observation
%%allows us to write
%%$$ c = u^{-1}\left( \max\left\{0,
%%     V+a -\beta [p(a)V^e + (1-p(a))V^u] \right\} \right). \EQN hugo21$$
Similarly, solving the inequality \Ep{hugo4} for $a$ and using the
assumed  functional
form for $p(a)$  leads to
$$ a = \max\left\{0, {\log[r \beta (V^e - V^u)] \over r } \right\}.
             \EQN hugo22 $$
Formulas \Ep{hugo21} and \Ep{hugo22} express $(c,a)$ as functions
of $V$ and  the continuation value $V^u$.  Using these functions
allows us to write the Bellman equation in $C(V)$ as
$$ C(V)  = \min_{V^u} \left\{ c + \beta [1 - p(a)] C(V^u) \right\} \EQN hugo23 $$
where $c$ and $a$ are given by equations \Ep{hugo21} and \Ep{hugo22}.


### Interpretations
The substantial downward slope in the replacement ratio in Figure \Fg{hugoreplnewf}
comes entirely  from the incentive constraints facing the
planner.   
We saw earlier that without private information, the
planner would smooth consumption
over an unemployment spell by
keeping the  replacement ratio constant. 

In the situation depicted in
Figure \Fg{hugoreplnewf}, the planner can't observe the worker's search effort
and therefore makes   the replacement ratio  fall and search
effort rise as the duration of unemployment increases, especially
early in an unemployment spell.   

There is a **carrot-and-stick**
aspect to the replacement rate and search effort  schedules:

 * the **carrot** occurs in the forms of high compensation and low search
effort early in an unemployment spell.  

 * the **stick** occurs in the low compensation and high effort  later in
the spell.   

XXXXXXWe shall see  this carrot-and-stick feature in some of the
credible government policies analyzed in chapters \use{credible}, \use{chang}, and
\use{wldtrade}.

The planner offers declining benefits and asks for increased search
effort as the duration of an unemployment spell rises in order to provide an
unemployed worker with proper incentives, not to punish an unlucky worker
who has been unemployed for a long time.  

The planner believes that a
worker who has been unemployed a long time is unlucky, not that he has
done anything wrong (i.e., not lived up to the contract).  

Indeed, the
contract is designed to induce the unemployed workers to search in
the way the planner expects.  

The falling consumption and rising
search effort of the unlucky ones with long unemployment spells are
simply  prices that have to be paid for the common good of providing
proper incentives.

```{code-cell} ipython3

```
