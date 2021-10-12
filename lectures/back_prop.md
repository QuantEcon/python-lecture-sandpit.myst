---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# A Neural Network

```{code-cell} python3
:tags: [hide-output]
!pip install --upgrade jax jaxlib
```

We describe a simple "deep" neural network of "width" one.  

The "width" being one means that we are dealing with scalar functions at critical places.

We let $x \in \mathbb{R}$ be a scalar and $y \in \mathbb{R}$ be another scalar that is a function of $x$:

$$
y = f(x)
$$

We want to approximate the function $f(x)$ with another function that we construct in the following iterative
or recursive way.

For a network of depth $N \geq 1$, each layer $i =1, \ldots N$ consists of 

* an input $x_i$

* an affine function $w_i x_i + bI$, where $w_i$ is a scalar "weight" placed on the input and $b_i$ is a scalar "bias"

* an activation function $h_i$ that takes $(w_i x_i + b_i)$ as an argument and produces an output $x_{i+1}$
   
     
An example of an activation function $h$ is the **sigmoid** function

$$
h (z) = \frac{1}{1 + e^{-z}} 
$$

Below, we'll use the sigmoid function for layers $1$ to $N-1$ and the identity function for  layer $N$.

To approximate the function $f(x)$ we create a new function $\hat f(x)$ by proceeding as follows.

Let $l_{i}\left(x\right)=w_{i}x+b_{i}$. 

We construct  $\hat f$ by iterating on compositions of functions $h_i \circ l_i$:

$$
f(x)\approx\hat{f}(x)=h_{N}\circ l_{N}\circ h_{N-1}\circ l_{1}\circ\cdots\circ h_{1}\circ l_{1}(x)
$$

If $N >1$, we call the right side a "deep" neural net.

The larger is the integer $N$, the "deeper" is the neural net.

Evidently,  if we know the values of the parameters $\{w_i, b_i\}_{i=1}^N$, then we can compute
$\hat f(x)$ for a given $x = \tilde x$ by iterating on the recursion

$$
x_{i+1} = h_i \circ l_i(x_i) , \quad, i = 1, \ldots N
$$ (eq:recursion)

starting from $x_1 = \tilde x$.  

The value of $x_{N+1}$ that emerges from this iterative scheme
equals $\hat f(\tilde x)$.

## Calibrating  parameters


We now consider a  neural network like the one describe above  with width 1, depth $N$ and activation functions $h_{i}$ for $1\leqslant i\leqslant N$ mapping $\mathbb{R}$ into itself.


Let $\left\{ \left(w_{i},b_{i}\right)\right\} _{i=1}^{N}$ denote a sequence of weights and biases.

As mentioned above, for a given input $x_{1}$, our approximating function $\hat f$ evaluated
at $x_1$ equals the "output" $x_{N+1}$ from our network that  can be computed by iterating on $x_{i+1}=h_{i}\left(w_{i}x_{i}+b_{i}\right)$.

For a given "prediction" $\hat{y} (x) $ and target $y= f(x)$, consider the loss function $\mathcal{L} \left(\hat{y},y\right)(x)=\frac{1}{2}\left(\hat{y}-y\right)^{2}(x)$.

This criterion is a function of the parameters $\left\{ \left(w_{i},b_{i}\right)\right\} _{i=1}^{N}$
and the point $x$.

We're interested in solving the following problem:

$$
\min_{\left\{ \left(w_{i},b_{i}\right)\right\} _{i=1}^{N}} \int {\mathcal L}\left(x_{N+1},y\right)(x) d \mu(x)
$$

where $\mu(x)$ is some measure of  points $x \in R$ over which we want a good approximation $\hat f(x)$ to $f(x)$.

Stack the weights into a vector of parameters $p$ as follows:

$$ 
p = \begin{bmatrix}     
  w_1 \cr 
  b_1 \cr
  w_2 \cr
  b_2 \cr
  \vdots \cr
  w_N \cr
  b_N 
\end{bmatrix}
$$


Applying a "poor man's version" of a "stocastic "gradient descent" method for finding a zero of a function leads to the following update rule for parameters:

$$
p_{k+1}=p_k-\alpha\frac{d \mathcal{L}}{dx_{N+1}}\frac{dx_{N+1}}{dp_k}
$$

where $\frac{d {\mathcal L}}{dx_{N+1}}=-\left(x_{N+1}-y\right)$ and $\alpha > 0 $ is a step size.

(See [this](https://en.wikipedia.org/wiki/Gradient_descent#Description) and [this](https://en.wikipedia.org/wiki/Newton%27s_method)) to gather insights about how stochastic gradient descent
relates to Newton's method.)

To implement one step of this parameter update rule, we want  the vector of derivatives $\frac{dx_{N+1}}{dp_k}$.

In the neural network literature, this step is accomplished by what is known as **back propogation**

Here we'll show that thanks to some properties of

* the chain and product rules for differentiation, and

* lower triangular matrices
   
what is called back propogation is accomplished in one step by inversion of a lower triangular matrix and matrix multiplication.

(We got the idea to try this from the last 7 minutes of this great youtube video by MIT's Alan Edelman)

```{youtube} rZS2LGiurKY
```


Here goes.





Define the derivative of $h(z)$ with respect to $z$ evaluated at $z = z_i$  as $\delta_i$:

$$
\delta_i = \frac{d}{d z} h(z)|_{z=z_i}
$$

or 

$$
\delta_{i}=h'\left(w_{i}x_{i}+b_{i}\right). 
$$ 

Repeated application of the chain rule and product rule to our recursion {eq}`eq:recursion` allows us to obtain:

$$
dx_{i+1}=\delta_{i}\left(dw_{i}x_{i}+w_{i}dx_{i}+b_{i}\right)
$$

After imposing $dx_{1}=0$, we get the following system of equations:

$$
\left(\begin{array}{c}
dx_{2}\\
\vdots\\
dx_{N+1}
\end{array}\right)=\underbrace{\left(\begin{array}{ccccc}
\delta_{1}w_{1} & \delta_{1} & 0 & 0 & 0\\
0 & 0 & \ddots & 0 & 0\\
0 & 0 & 0 & \delta_{N}w_{N} & \delta_{N}
\end{array}\right)}_{D}\left(\begin{array}{c}
dw_{1}\\
db_{1}\\
\vdots\\
dw_{N}\\
db_{N}
\end{array}\right)+\underbrace{\left(\begin{array}{cccc}
0 & 0 & 0 & 0\\
w_{2} & 0 & 0 & 0\\
0 & \ddots & 0 & 0\\
0 & 0 & w_{N} & 0
\end{array}\right)}_{L}\left(\begin{array}{c}
dx_{2}\\
\vdots\\
dx_{N+1}
\end{array}\right)
$$ 

or

$$
d x = D dp + L dx
$$

which implies that

$$
dx = (I -L)^{-1} D dp
$$

which in turn  implies

$$
\left(\begin{array}{c}
dx_{N+1}/dw_{1}\\
dx_{N+1}/db_{1}\\
\vdots\\
dx_{N+1}/dw_{N}\\
dx_{N+1}/db_{N}
\end{array}\right)=e_{N}\left(I-L\right)^{-1}D.
$$

We can then solve the above problem by applying our update for $p$ multiple times for a collection of input-output pairs $\left\{ \left(x_{1}^{i},y^{i}\right)\right\} _{i=1}^{M}$ that we'll call our "training set".



## Quentin: please check that what I say here makes sense.

The training set amounts to a choice of measure $\mu$ in the above  formulation of our  function approximation problem as a minimization problem.

In this spirit, below  we  use a uniform grid of, say, 50 or 200 or $\ldots$,  points. 

There are many possible routes for solving the minimization  problem posed above:

* batch gradient descent in which you use an average gradient over the training set

* stochastic gradient descent in which you sample points randomly and use individual gradients

* or something in-between (so-called "mini-batch gradient descent")
 
The update rule described above  corresponds to a stochastic gradient descent algorithm

```{code-cell} ipython3
import jax.numpy as jnp
from jax import grad, jit, jacfwd, vmap
from jax import random
import jax
from jax.ops import index_update
import plotly.graph_objects as go
```

```{code-cell} ipython3
# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1.):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
```

```{code-cell} ipython3
def compute_xδw_seq(params, x):
    # Initialize arrays
    δ = jnp.zeros(len(params))
    xs = jnp.zeros(len(params) + 1)
    ws = jnp.zeros(len(params))
    bs = jnp.zeros(len(params))
    
    h = jax.nn.sigmoid
    
    xs = index_update(xs, 0, x)
    for i, (w, b) in enumerate(params[:-1]):
        output = w * xs[i] + b
        activation = h(output[0, 0])
        
        # Store elements
        δ = index_update(δ, i, grad(h)(output[0, 0]))
        ws = index_update(ws, i, w[0, 0])
        bs = index_update(bs, i, b[0])
        xs = index_update(xs, i+1, activation)

    final_w, final_b = params[-1]
    preds = final_w * xs[-2] + final_b
    
    # Store elements
    δ = index_update(δ, -1, 1.)
    ws = index_update(ws, -1, final_w[0, 0])
    bs = index_update(ws, -1, final_b[0])
    xs = index_update(xs, -1, preds[0, 0])
    
    return xs, δ, ws, bs
    

def loss(params, x, y):
    xs, δ, ws, bs = compute_xδw_seq(params, x)
    preds = xs[-1]
    
    return 1 / 2 * (y - preds) ** 2
    
```

```{code-cell} ipython3
# Parameters
N = 3  # Number of layers
layer_sizes = [1, ] * (N + 1)
param_scale = 0.1
step_size = 0.01
params = init_network_params(layer_sizes, random.PRNGKey(1))
```

```{code-cell} ipython3
x = 5
y = 3
xs, δ, ws, bs = compute_xδw_seq(params, x)
```

```{code-cell} ipython3
dxs_ad = jacfwd(lambda params, x: compute_xδw_seq(params, x)[0], argnums=0)(params, x)
dxs_ad_mat = jnp.block([dx.reshape((-1, 1)) for dx_tuple in dxs_ad for dx in dx_tuple ])[1:]
```

```{code-cell} ipython3
jnp.block([[δ * xs[:-1]], [δ]])
```

```{code-cell} ipython3
L = jnp.diag(δ * ws, k=-1)
L = L[1:, 1:]

D = jax.scipy.linalg.block_diag(*[row.reshape((1, 2)) for row in jnp.block([[δ * xs[:-1]], [δ]]).T])

dxs_la = jax.scipy.linalg.solve_triangular(jnp.eye(N) - L, D, lower=True)
```

```{code-cell} ipython3
# Check that the `dx` generated by the linear algebra method
# are the same as the ones generated using automatic differentiation
jnp.max(jnp.abs(dxs_ad_mat - dxs_la))
```

```{code-cell} ipython3
grad_loss_ad = jnp.block([dx.reshape((-1, 1)) for dx_tuple in grad(loss)(params, x, y) for dx in dx_tuple ])
```

```{code-cell} ipython3
# Check that the gradient of the loss is the same for both approaches
jnp.max(jnp.abs(-(y - xs[-1]) * dxs_la[-1] - grad_loss_ad))
```

```{code-cell} ipython3
@jit
def update_ad(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]

@jit
def update_la(params, x, y):
    xs, δ, ws, bs = compute_xδw_seq(params, x)
    N = len(params)
    L = jnp.diag(δ * ws, k=-1)
    L = L[1:, 1:]

    D = jax.scipy.linalg.block_diag(*[row.reshape((1, 2)) for row in jnp.block([[δ * xs[:-1]], [δ]]).T])
    
    dxs_la = jax.scipy.linalg.solve_triangular(jnp.eye(N) - L, D, lower=True)
    
    grads = -(y - xs[-1]) * dxs_la[-1]
    
    return [(w - step_size * dw, b - step_size * db) 
            for (w, b), (dw, db) in zip(params, grads.reshape((-1, 2)))]
    
```

```{code-cell} ipython3
# Check that both updates are the same
update_la(params, x, y)
```

```{code-cell} ipython3
update_ad(params, x, y)
```

### Example 1

Consider the following function 

$$
f\left(x\right)=-3x+2
$$

on $\left[0.5,3\right]$. 

We use a uniform grid of 200 points and update the parameters for each point on the grid 300 times. 

$h_{i}$ is the sigmoid activation function for all layers except the final one for which we use the identity function and $N=3$.

Weights are initialized randomly.

```{code-cell} ipython3
def f(x):
    return -3 * x + 2

M = 200
grid = jnp.linspace(0.5, 3, num=M)
f_val = f(grid)
```

```{code-cell} ipython3
indices = jnp.arange(M)
key = random.PRNGKey(0)

def train(params, grid, f_val, key, num_epochs=300):
    for epoch in range(num_epochs):
        key, _ = random.split(key)
        random_permutation = random.permutation(random.PRNGKey(1), indices)
        for x, y in zip(grid[random_permutation], f_val[random_permutation]):
            params = update_la(params, x, y)
            
    return params 
```

```{code-cell} ipython3
# Parameters
N = 3  # Number of layers
layer_sizes = [1, ] * (N + 1)
params_ex1 = init_network_params(layer_sizes, key)
```

```{code-cell} ipython3
%%time 
params_ex1 = train(params_ex1, grid, f_val, key, num_epochs=500)
```

```{code-cell} ipython3
predictions = vmap(compute_xδw_seq, in_axes=(None, 0))(params_ex1, grid)[0][:, -1]
```

```{code-cell} ipython3
fig = go.Figure()
fig.add_trace(go.Scatter(x=grid, y=f_val, name=r'$-3x+2$'))
fig.add_trace(go.Scatter(x=grid, y=predictions, name='Approximation'))

fig.show()
```

### Quentin: it would be fun to "deepen" the neural net for the above example and show how the approximation
### improves. Can you please do this?


<span style="color:red">There are several problems here. First, if the network is too deep, you'll run into the [vanishing gradient problem](http://neuralnetworksanddeeplearning.com/chap5.html). Besides, other parameters such as the step size and the number of epochs are probably more important than the number of layers in these examples. In fact, since $f$ is a linear function of $x$, a one-layer network with the identity map as an activation would probably work best. I've added a comparison for the example below. </span>


### Example 2

We use the same setup as for the previous example with

$$
f\left(x\right)=\log\left(x\right)
$$

```{code-cell} ipython3
def f(x):
    return jnp.log(x)

grid = jnp.linspace(0.5, 3, num=M)
f_val = f(grid)
```

```{code-cell} ipython3
# Parameters
N = 1  # Number of layers
layer_sizes = [1, ] * (N + 1)
params_ex2_1 = init_network_params(layer_sizes, key)
```

```{code-cell} ipython3
# Parameters
N = 2  # Number of layers
layer_sizes = [1, ] * (N + 1)
params_ex2_2 = init_network_params(layer_sizes, key)
```

```{code-cell} ipython3
# Parameters
N = 3  # Number of layers
layer_sizes = [1, ] * (N + 1)
params_ex2_3 = init_network_params(layer_sizes, key)
```

```{code-cell} ipython3
params_ex2_1 = train(params_ex2_1, grid, f_val, key, num_epochs=300)
```

```{code-cell} ipython3
params_ex2_2 = train(params_ex2_2, grid, f_val, key, num_epochs=300)
```

```{code-cell} ipython3
params_ex2_3 = train(params_ex2_3, grid, f_val, key, num_epochs=300)
```

```{code-cell} ipython3
predictions_1 = vmap(compute_xδw_seq, in_axes=(None, 0))(params_ex2_1, grid)[0][:, -1]
predictions_2 = vmap(compute_xδw_seq, in_axes=(None, 0))(params_ex2_2, grid)[0][:, -1]
predictions_3 = vmap(compute_xδw_seq, in_axes=(None, 0))(params_ex2_3, grid)[0][:, -1]
```

```{code-cell} ipython3
fig = go.Figure()
fig.add_trace(go.Scatter(x=grid, y=f_val, name=r'$\log{x}$'))
fig.add_trace(go.Scatter(x=grid, y=predictions_1, name='One-layer neural network'))
fig.add_trace(go.Scatter(x=grid, y=predictions_2, name='Two-layer neural network'))
fig.add_trace(go.Scatter(x=grid, y=predictions_3, name='Three-layer neural network'))

fig.show()
```

```{code-cell} ipython3
## to check that gpu is activated in environment

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
```