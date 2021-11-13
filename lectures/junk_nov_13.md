



Thus, the panel on the right portrays how the transformation $\exp\left(\frac{-u\left(c\right)}{\theta}\right)$ sends $u\left(c\right)$ to a new function by  (i)  flipping the  sign,  and (ii) increasing curvature in proportion to $\theta$.

In the left panel, the red line is our tool for computing  the mathematical expectation for different
values  of $\pi$.

The green lot indicates the mathematical expectation of $\exp\left(\frac{-u\left(c\right)}{\theta}\right)$ 
when $\pi = .5$.  

Notice that the distance between the green dot  and the curve is greater in the transformed space than the original space as a result of additional curvature. 

The inverse transformation  $\theta\log E\left[\exp\left(\frac{-u\left(c\right)}{\theta}\right)\right]$ generates  the green dot on the left panel that constitutes the risk-sensitive utility  index.  

The gap between the green dot and the red line on the left panel measures the additional adjustment for risk
that risk-sensitive preferences make relative to plain vanilla expected utility preferences. 


CAPTIONS

${\sf T} u(c)$ as a function of $\pi_1$ for $\theta=100$ (nearly linear line) and $\theta=.6$ (convex curved line). Here  $I=2, c_1=2, c_2=1$, $u(c) = \ln c$.



Indifference curves for expected logarithmic utility (solid and
smooth), multiplier (dotted and smooth), and constraint (kinked at 45
degree line) preferences. The worst-case probability $\hat \pi_1 < .5$
when $c_1 > c_2$ and $\hat \pi_1 > .5$ when
$c_1 < c_2$.



Indifference curves through point $(c_1,c_2)=(3,1)$ for expected
logarithmic utility, multiplier, constraint (kinked at 45 degree line),
and *ex post* Bayesian (dotted lines) preferences. Certainty equivalents
for risk-neutrality (point A), expected utility with log preferences
(point B), multiplier preferences (point C), and constraint preferences
(point D).](indifference4.eps){#fig_indifference4 height="2in"}

HERE IS FIGURE 1.8, 2.8

END OF CAPTIONS



Evidently, for a given $\eta$ and a given $(c_1, c_2)$ off the 45 degree line, by solving
equations {eq}`tom7` and {eq}`tom20`, we can find $\tilde \theta (\eta, c)$
and $\tilde \eta(\theta,c)$ that make the indifference curves for 
multiplier and constraint preferences be tangent to one another at a
given allocation $c$.



For fixed $\eta$, a given plan $c$, and
a utility function increasing in $c$, the worst case probabilities are
$\hat \pi_1 < .5$ when $c_1 > c_2$ and $\hat \pi_1 > .5$ when
$c_2 > c_1$. 

The discontinuity in the worst case $\hat \pi_1$ at the 45
degree line accounts for the kink in the indifference curve for
constraint preferences associated with a particular positive entropy
$\eta$. 





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
     

**Note:** It seems that the constraint problem is hard to solve in its original form, i.e. by finding the distorting measure that minimizes the expected utility. 

It seems that viewing (2.5) as a root finding problem works much better. 

Notice that 2.5 does not always have a solution. 

Under $u=\log$, $c_{1}=c_{2}=1$, we have: 

$$\sum_{i}\pi_{i}\frac{\exp\left(\frac{-u\left(c_{i}\right)}{\tilde{\theta}}\right)}{\sum_{j}\pi_{j}\exp\left(\frac{-u\left(c_{j}\right)}{\tilde{\theta}}\right)}\log\left(\frac{\exp\left(\frac{-u\left(c_{i}\right)}{\tilde{\theta}}\right)}{\sum_{j}\pi_{j}\exp\left(\frac{-u\left(c_{j}\right)}{\tilde{\theta}}\right)}\right)=0$$

Guess: the method fails because the derivative of the objective doesn't exist for these parameter choices

**Note 2:** Algorithm is tricky to get to work properly for all values of $c_{1}$. In particular, parameters were chosen with [graduate student descent](https://sciencedryad.wordpress.com/2014/01/25/grad-student-descent/).
