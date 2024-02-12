---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"user_expressions": []}

# Money Supplies and Price Levels

+++ {"user_expressions": []}

## Overview

This notebook describes a theory of price level variations that consists of two components

 * a demand function for money 
 * a law of motion for the supply of money
 
The demand function describes the public's demand for "real balances", defined as the ratio of nominal money balances to the price level

The law of motion for the supply of money assumes that the government prints money to finance government expenditures

Our model equates the demand for money to the supply at each time $t \geq 0$.

Equality between those demands and supply gives in a **dynamic** model in which   money supply
and  price level **sequences** are simultaneously determined by a special  set of simultaneous linear  
equations.

These equations take the form of what are often called vector linear **difference equations**.  

In this notebook, we'll roll up our sleeves and solve those equations in a couple of different ways.

As we'll see, Python is good at solving them.


## Demands  for and Supplies of Money

We say demand**s** and supp**ies** (plural) because there is one of each for each $t \geq 0$.

Let 

  * $m_{t+1}$ be the supply of currency at the end of time $t \geq 0$
  * $m_{t}$ be the supply  of currency brought into time $t$ from time $t-1$
  * $g$ be the government deficit that is financed by printing currency at $t \geq 1$
  * $m_{t+1}^d$ be the demand at time $t$ for currency  to bring into time $t+1$
  * $p_t$ be  the price level at time $t$
  * $R_t = \frac{p_t}{p_{t+1}} $ be the gross rate of return on currency held from time $t$ to time $t+1$
  
It is often helpful  to state units in which quantitities are measured:

  * $m_t$ and $m_t^d$ are measured in dollars
  * $g$ is measured in time $t$ goods 
  * $p_t$ is measured in dollars per time $t$ goods
  * $R_t$ is measured in time $t+1$ goods per unit of time $t$ goods
  
  
  
  
Our job now is to specify demand and supply functions for money. 

We assume that the demand for  currency satisfies the Cagan-like demand function

$$
m_{t+1}^d/p_t =\gamma_1 - \gamma_2 \frac{p_{t+1}}{p_t}, \quad t \geq 0
$$
  
  
Now we turn to the supply of money.

We assume that $m_0 >0$ is an "initial condition" determined outside the model. 

We just set it at some arbitrary value, say \$100.
  
For $ t \geq 1$, we assume that the supply of money is determined by the government's budget constraint

$$
m_{t+1} - m_{t} = p_t g , \quad t \geq 0
$$

According to this equation, each period, the government prints money to pay for quantity $g$ of goods. 

In an **equilibrium**, the demand for currency equals the supply:

$$
m_{t+1}^d = m_{t+1}, \quad t \geq 0
$$

Let's take a moment to reflect about what this equation tells us.

The demand for money at any time $t$ depends on the price level at time $t$ and the price level at time $t+1$.

The supply of money at time $t+1$ depends on the money supply at time $t$ and the price level at time $t$.

So the infinite set of equations XX for $ t \geq 0$ imply that the **sequences** $\{p_t\}_{t=0}^\infty$
and $\{m_t\}_{t=0}^\infty$ are tied together and ultimately simulataneously determined.


## Equilibrium price and money supply sequences


The preceding specifications imply that for $t \geq 1$, **real balances** evolve according to


$$
\frac{m_{t+1}}{p_t} - \frac{m_{t}}{p_{t-1}} \frac{p_{t-1}}{p_t} = g
$$

or

$$
b_t - b_{t-1} R_{t-1} = g
$$

where

  * $b_t = \frac{m_{t+1}}{p_t}$ is real balances at the end of period $t$
  * $R_{t-1} = \frac{p_{t-1}}{p_t}$ is the gross rate of return on real balances held from $t-1$ to $t$
  
We'll restrict our attention to  parameter values and  associated rates of return on real balances that assure that the demand for real balances is positive, which according to (1) means that

$$
b_t = \gamma_1 - \gamma_2 R_t^{-1} > 0 
$$

which implies that

$$
R_t \geq \left( \frac{\gamma_2}{\gamma_1} \right)
$$


We shall describe two distinct ways of computing an equilibrium $\{p_t, m_t\}_{t=0}$ sequence.

## Two equilibrium computation strategies

**Method 1:** 

   * set $R_0 \in [\underline R, \overline R]$ and compute $b_0 = \gamma_1 - \gamma_2/R_0$.

   * compute sequences $\{R_t, b_t\}_{t=1}^\infty$ of rates of return and real balances that are associated with an equilibrium by solving the following equations sequentially for $t \geq 1$:
   \begin{align}
b_t & = b_{t-1} R_{t-1} + p_t \cr
R_t^{-1} & = \frac{\gamma_1}{\gamma_2} - \gamma_2^{-1} b_t 
\end{align}

   
   * Construct an equilibrium $p_0$ from 
   $$
   p_0 = \frac{m_0}{\gamma_1 - g - \gamma_2/R_0}
   $$
   
   * compute $\{p_t, m_t\}_{t=1}^\infty$  by solving the following equations sequentially
   \begin{align}
   p_t & = R_t p_{t-1} \cr
   m_t & = b_t p_t 
   \end{align}
   
   
**Remark:** method 1 uses an indirect approach to computing an equilibrium by first computing an equilbrium 
 $\{R_t, b_t\}_{t=0}^\infty$ sequence and then using it to back out an equilibrium  $\{p_t, m_t\}_{t=0}^\infty$  sequence.
 
 **Remark:** notice that  method 1 starts by picking an **initial condition** $R_0$ from a set $[\underline R, \overline R]$. That we have to do this is a symptom that equilibrium $\{p_t, m_t\}_{t=0}^\infty$ sequences are not unique.  There is actually a continuum of them indexed by a choice of $R_0$ from the set $[\underline R, \overline R]$ that we shall describe soon. 
 
 
   
**Method 2:** 

   This method deploys a direct approach.  It defines a "state vector" 
    $y_t = \begin{bmatrix} m_t \cr p_t\end{bmatrix} $
   and formulates  equilibrium conditions in terms of a first-order vector difference equation
   $$
   y_{t+1} = M y_t, \quad t \geq 0 ,
   $$
   where $y_0 = \begin{bmatrix} m_0 \cr p_0 \end{bmatrix}$ is an **initial condition**. 
   
   The solution is 
   
   $$
   y_t = M^t y_0 .
   $$
   
   It is natural to take the initial stock of money $m_0 >0$ as an initial condition.
   
   But what about $p_0$?  
   
   It is something to be **determined** by our model.  
   
   However, there is actually a continuum of initial $p_0$ levels that are compatible with the existence of an equilibrium.  
   
   As we shall see soon, selecting an initial $p_0$ in method 2 is intimately tied to selecting an initial rate of return on currency $R_0$ in method 1. 
   
   
   
   An equilibrium $\{p_t, m_t\}_{t=0}^\infty $ sequence is generated by







## Computation Method 1  

%We start from an arbitrary $R_0$ and  $b_t = \frac{m_{t+1}}{p_t}$, we have 

%$$
%b_0 = \gamma_1 - \gamma_0 R_0^{-1} 
%$$

We proceed as follows:

Start at $t=0$ 
 * take a   given $R_0 \in  
 * compute   $b_0 = \gamma_1 - \gamma_0 R_0^{-1} $ 
 
Then  for $t \geq 1$ construct $(b_t, R_t)$ by
iterating  on the system 
\begin{align}
b_t & = b_{t-1} R_{t-1} + g \cr
R_t^{-1} & = \frac{\gamma_1}{\gamma_2} - \gamma_2^{-1} b_t
\end{align}


When we implement this part of method 1 below, we shall discover the following  striking 
outcome:

 * starting from an $R_0$ in an admissible set $[\underline R, \overline R]$, we shall find that 
$\{R_t\}$ always converges to a limiting "steady state" value  $\bar R$ that depends on the initial
condition $R_0$.

  * there are only two possible limit points $\{ \bar R_{\rm min}, \bar R_{\rm max}\}$. 
  
  * for almost every initial condition $R_0$, $\lim_{t \rightarrow +\infty} R_t = \bar R_{\rm min}$.
  
  * if and only if $R_0 = \bar R_{\rm max}$, $\lim_{t \rightarrow +\infty} R_t = \bar R_{\rm max}$.
  
When we recognize that $1 - R_t$ can be interpreted as an **inflation tax rate** that the government imposes on holders of its currency, we shall see that the existence of two steady state rates of return on currency
that serve to finance the government deficit of $g$ indicates the presence of a **Laffer curve** in the inflation tax rate.






### Steady States

In a **steady state**

\begin{align}
R_t & = \bar R \cr
b_t & = \bar b
\end{align}

for $t \geq 0$.  

To compute a steady state, we seek steady state values $\bar R, \bar b$ that satisfy steady-state versions of  both the government budget constraint and the demand function for real balances:

\begin{align}
g & = \bar b ( 1 - \bar R)  \cr
\bar b & = \gamma_1- \gamma_2 \bar R^{-1}
\end{align}

Together these equations  imply

$$
(\gamma_1 + \gamma_2) - \frac{\gamma_2}{\bar R} - \gamma_1 \bar R = g
$$


The left side is the steady-state amount of **seigniorage** or government revenues that the government gathers by paying a gross rate of return $\bar R < 1$ on currency. 

The right side is government expenditures.

If we maximize seigniorage with respect to $\bar R$, we find maximizing rate of return on currency is 

$$
\bar R_{\rm max} = \sqrt{\frac{\gamma_2}{\gamma_1}}
$$

and that the associated maximum seigniorage revenue that the government can gather from printing money is

$$
(\gamma_1 + \gamma_2) - \frac{\gamma_2}{\bar R_{\rm max}} - \gamma_1 \bar R_{\rm max}
$$

It is useful to rewrite   equation GGHH as

$$
-\gamma_2 + (\gamma_1 + \gamma_2 + g) \bar R - \gamma_1 \bar R^2 = 0
$$

A steady state value $\bar R$ is a zero of this quadratic equation.

If a steady state exists, there  are typically two steady state values of $\bar R$. 

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

import numpy as np
import math

gam1 = 100
gam2 = 50
g = 3.0
M0 = 100

coeff = (-gam1, (gam1+gam2-g), -gam2)


RR = np.roots(coeff)
print("RR = ", RR)

Rmaxguess = RR[0]
Rminguess = RR[1]
print("Rmaxguess = ", Rmaxguess)
print("Rminguess = ", Rminguess)
p0guess = M0/(gam1 - g - gam2/Rmaxguess)  # revised formula -- see notes and bring into markdown text
print("p0guess =", p0guess)  
RRt = math.sqrt(gam2/gam1) 

seignmax = (gam1 + gam2) - gam2/RRt - gam1*RRt

print("RRt = ", RRt)
print("seignmax = ",seignmax)
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

def seign(R, gam1=1, gam2=2, g=3):
    return -gam2/R + (gam1 + gam2 )  - gam1 * R

# Generate values for R
R_values = np.linspace(gam2/gam1, 1, 250)  # Adjust the range and number of points as needed

# Default parameter values
gam1_default = 100
gam2_default = 50
g_default = 3

# Calculate the function values
y_values = seign(R_values, gam1_default, gam2_default, g_default)

# Plot the function
plt.plot(R_values, y_values, label='Inflation tax revenue')
plt.axhline(y=g_default, color='red', linestyle='--', label='government deficit')
plt.xlabel('R')
plt.ylabel('Seigniorage')
plt.title('Revenue from inflation tax')
plt.legend()
plt.grid(True)
plt.show()
```

```{code-cell} ipython3
g1 = seign(Rmaxguess, gam1_default, gam2_default, g_default)
g1
print("Rmax, g = ", Rmaxguess, g1)

g2 = seign(Rminguess, gam1_default, gam2_default, g_default)
print("Rmin, g = ", Rminguess, g2)
print("g_default = ", g_default)
```

```{code-cell} ipython3
Rmax = np.sqrt(gam2/gam1)
print("Rmax = ", Rmax)
seign(Rmax, gam1_default, gam2_default, g_default)

```

```{code-cell} ipython3
#gam1 = 100
#gam2 = 50
#g = 3.0
#M0 = 100



def simulate_system(R0, gam1, gam2, g, num_steps):
    # Initialize arrays to store results
    b_values = [gam1 - gam2 / R0]
    R_values = [1 / ((gam1 / gam2) - (gam2**(-1) * b_values[0]))]

    # Iterate over time steps
    for t in range(1, num_steps):
        # Calculate b_t and R_t based on the given formulas
        b_t = b_values[t - 1] * R_values[t - 1] + g
        R_t_inverse = (gam1 / gam2) - gam2**(-1) * b_t
        R_values.append(1 / R_t_inverse)
        b_values.append(b_t)

    return b_values, R_values

# Parameters
R0 = Rmaxguess 

num_steps = 25

# Run simulation
b_values, R_values = simulate_system(R0, gam1, gam2, g, num_steps)

# Print results
for t in range(num_steps):
    print(f"Time step {t}: b_t = {b_values[t]}, R_t = {R_values[t]}")
```

```{code-cell} ipython3
R0 = Rminguess
b_values, R_values = simulate_system(R0, gam1, gam2, g, num_steps)

# Print results
for t in range(num_steps):
    print(f"Time step {t}: b_t = {b_values[t]}, R_t = {R_values[t]}")
```

+++ {"user_expressions": []}



## Dynamics
 
An **equilibrium** is pair of sequences $(\vec p, \vec m ) = \{p_t, m_t\}_{t=0}^\infty$ satisfying (1) and (2) and $m_{t+1} = m_{t+1}^d$ for all $t \geq 0$. 


Set $m_t = m_t^d $ for all $t \geq -1$, and represent equations (1) and (2) as

$$
\begin{bmatrix} 1 & \gamma_2 \cr
                 1 & 0 \end{bmatrix} \begin{bmatrix} m_{t+1} \cr p_{t+1} \end{bmatrix} =
                 \begin{bmatrix} 0 & 1 \cr
                 1 & g \end{bmatrix} \begin{bmatrix} m_{t} \cr p_{t} \end{bmatrix} 
$$

or

$$ 
L y_t = N y_{t-1} 
$$

where 

\begin{align} L & = \begin{bmatrix} 1 & \gamma_2 \cr
                 1 & 0 \end{bmatrix} \cr
                N & = \begin{bmatrix} 0 & 1 \cr
                 1 & g \end{bmatrix}  \cr
                 y_t & = \begin{bmatrix} m_{t} \cr p_{t} \end{bmatrix}
\end{align}

Define

$$
M = L^{-1} N
$$

and write the system as

$$
y_{t+1} = M y_t, \quad t \geq 0
$$

where 

$$
y_0 = \begin{bmatrix} m_{0} \cr p_0 \end{bmatrix}
$$


To find the smallest equilibrium  $p_0$, we use the invariant subspace methods described in section 5.6 of RMT5.  

Compute the eigenvector decomposition 

$$
M = V D V^{-1}
$$ 

where $D$ is a diagonal matrix of eigenvalues and the columns of $V$ are eigenvectors correspondng to those eigenvalues.

Partition $V$ as

$$ 
V =\begin{bmatrix} V_{11} & V_{12} \cr
                   V_{21} & V_{22} \end{bmatrix}
$$

Then set 


$$
p_0 = V_{21} V_{11}^{-1}  m_{0}
$$

```{code-cell} ipython3
:user_expressions: []

gam1 = 100
gam2 = 50
g = 3.0
M0 = 100

m1 = np.array([[1, gam2], [1, 0]])  # This is $L$
m2 = np.array([[0, gam1], [1, g]])  # This is $N$

m1, m2

print("m1 = ", m1)

print("m2 = ", m2)

M = np.linalg.inv(m1) @ m2

print("M = ", M)

d, v = np.linalg.eig(np.linalg.inv(m1) @ m2)
v, d


d = np.diag(d)

#v = np.linalg.inv(w)

#print("w = ", w)
print("d = ", d)

#print("v = ", v)


print (d)


Rsteady1 = 1/d[0]
Rsteady2 = 1/d[1]

print("Rsteady1 =", Rsteady1)
print("Rsteady2 =", Rsteady2)


d = np.diag(d)


a1 = v @ d @ np.linalg.inv(v)

print("M = ", M)
print("a1 = ", a1)


gam1 = 100
gam2 = 50
g = 3.0
M0 = 100

m1 = np.array([[1, gam2], [1, 0]])  # This is $L$
m2 = np.array([[0, gam1], [1, g]])  # This is $N$

m1, m2

print("m1 = ", m1)

print("m2 = ", m2)

M = np.linalg.inv(m1) @ m2

print("M = ", M)

d, v = np.linalg.eig(np.linalg.inv(m1) @ m2)
v, d


Rsteady1 = 1/d[0]
Rsteady2 = 1/d[1]

print("Rsteady1 =", Rsteady1)
print("Rsteady2 =", Rsteady2)
      


d = np.diag(d)




#print("d = ", d)




#a1 = v @ d @ np.linalg.inv(v)


#a10 = v @ np.linalg.matrix_power(d, 10) @ np.linalg.inv(v)
#M10 = np.linalg.matrix_power(M,10)

#print("M10= ",M10)
#print("a10= ", a10)

p0 = (v[1, 0] / v[0, 0]) * M0



y0 = np.array([M0, p0])

print("p0 =", p0)
print("p0guess = ", p0guess)

print("y0 =", y0)
   



print("M = ", M)
print("a1 = ", a1)
```

```{code-cell} ipython3

```
