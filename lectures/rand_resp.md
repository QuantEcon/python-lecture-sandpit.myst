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

# Randomized Response Surveys

Social stigmas can make  people prefer to tell the truth when asked about potentially embarrassing activities or opinions. 

To illustrate how social scientists have thought about learning about such embarrassing activities and opinions,this lecture describes a classic approach  of S. L.  
Warner {cite}`warner1965randomized`.

Warner  put elementary  probability to work with the aim of constructing ways to protect the privacy  of **individual** respondents to surveys while still  accurately  estimating  the fraction of a **collection** of individuals   who think that they have a socially stichmatized characteristic, or who know that they engage in a a socially stimatized activity.  

The idea is to design a survey that assures the respondent that survey taker cannot observe his answer.

Warner's idea was to  add **noise** between the respondent's answer and the **signal** about that answer that the  survey taker ultimately receives.  

Statistical properties of the  noise injection procedure can  assure a respondent
 **plausible deniability**.   

This idea is the central to the design of modern **differential privacy** systems.

(See https://en.wikipedia.org/wiki/Differential_privacy)


#### Strategy


Thus, when people are reluctant to participate a sample survey about personally  sensitive issues,
they  might decline  to participate, and even if they do participate, they might choose to provide incorrect answers to  sensitive questions.

These problems induce  **selection**  biases that make it difficult to trust interpret the survey statistics.


To confront such problems, {cite}`warner1965randomized` recommended an  interviewing technique designed to preserve subjects'   privacy while also providing information sought by the interviewer.

It  uses  **randomized responses** to assure subjects **plausible deniability**.

The idea is to inject noise into respondents' answers from the point of view of the surveyer.

+++



As usual, let's bring in the Python modules we'll be using.


```{code-cell} ipython3
import numpy as np
import pandas as pd
```



## The Randomized Response Model

Suppose that every person in population either belongs to Group A or Group B. 

We want to estimate the proportion $\pi$ who belong to Group A while protecting individual respondents' privacy.


Warner {cite}`warner1965randomized` proposed and analyzed the following procedure.

- A  random sample of $n$ people is drawn with replacement from the population and  each person is interviewed.
- Draw $n$ random samples from the population with replacement and interview each person.
- Prepare a **random spinner** that with $p$ probability points to the Letter A and with $(1-p)$ probability points to the Letter B.
- In each interview, the interviewee spins a random spinner and sees an outcome (A or B)  that the interviewer  does **not observe**.
- The interviewee   answers whether he belongs to the group to which the spinner points.
- If the spinner points to  the group that the spinner  belongs, the interviewee  reports “yes”; otherwise he reports “no”.
- The  interviewee is assumed to  **report truthfully**.



Warner proceeds to construct a maximum  likelihood estimators of the proportion of the population in set A.


Let

+++

- $\pi$ : True probability of A in the population

+++

- $p$ : Probability that the spinner points to A

+++

- $X_{i}=\begin{cases}1,\text{ if the } i\text{th} \text{interviewee says yes}\\0,\text{ if the } i\text{th} \text{ interviewee says no}\end{cases}$

+++

Index the sample set so that  the first $n_1$ report "yes", while the second $n-n_1$ report "no".

The likelihood function of a sample set is 

+++

$$
\begin{equation}
L=\left[\pi p + (1-\pi)(1-p)\right]^{n_{1}}\left[(1-\pi) p +\pi (1-p)\right]^{n-n_{1}} 
 \tag{1}
\end{equation}
$$

+++

The log of the likelihood function is:

+++

$$
\begin{equation}
\log(L)= n_1 \log \left[\pi p + (1-\pi)(1-p)\right] + (n-n_{1}) \log \left[(1-\pi) p +\pi (1-p)\right] \tag{2}
\end{equation}
$$

+++

The first-order necessary condition for maximimizng the log likelihood function with respect to  $\pi$ is:

+++

$$
\frac{(n-n_1)(2p-1)}{(1-\pi) p +\pi (1-p)}=\frac{n_1 (2p-1)}{\pi p + (1-\pi)(1-p)} 
$$

+++

or

+++

$$
\begin{equation}
\pi p + (1-\pi)(1-p)=\frac{n_1}{n} \tag{3}
\end{equation}
$$

+++

If  $p \neq \frac{1}{2}$, then the maximum likelihood estimator (MLE) of $\pi$ is:

+++

$$
\begin{equation}
\hat{\pi}=\frac{p-1}{2p-1}+\frac{n_1}{(2p-1)n} \tag{4}
\end{equation}
$$

+++

We compute the mean and variance of the MLE estimator $\hat \pi$ to be:

+++

$$
\begin{align}
\mathbb{E}(\hat{\pi})&= \frac{1}{2 p-1}\left[p-1+\frac{1}{n} \sum_{i=1}^{n} \mathbb{E} X_i \right] \\
&=\frac{1}{2 p-1} \left[ p -1 + \pi p + (1-\pi)(1-p)\right] \\
&=\pi  \tag{5}
\end{align}
$$

+++

and

+++

$$
\begin{align}
Var(\hat{\pi})&=\frac{n Var(X_i)}{(2p - 1 )^2 n^2} \\
&= \frac{\left[\pi p + (1-\pi)(1-p)\right]\left[(1-\pi) p +\pi (1-p)\right]}{(2p - 1 )^2 n^2}\\
&=\frac{\frac{1}{4}+(2 p^2 - 2 p +\frac{1}{2})(- 2 \pi^2 + 2 \pi -\frac{1}{2})}{(2p - 1 )^2 n^2}\\
&=\frac{1}{n}\left[\frac{1}{16(p-\frac{1}{2})^2}-(\pi-\frac{1}{2})^2 \right] \tag{6}
\end{align}
$$

+++

Equation (5) indicates  that $\hat{\pi}$ is an **unbiased estimator** of $\pi$ while equation (6) tell us the variance of the estimator.

To compute a  confidence interval, first  rewrite (6) as:

+++

$$
\begin{equation}
Var(\hat{\pi})=\frac{\frac{1}{4}-(\pi-\frac{1}{2})^2}{n}+\frac{\frac{1}{16(p-\frac{1}{2})^2}-\frac{1}{4}}{n} \tag{7}
\end{equation}
$$

+++

This equation indicates that the variance of $\hat{\pi}$ can be represented as a sum of the variance due to sampling plus the variance due to the random device.



From the expressions above we can find that:

- When $p$ is $\frac{1}{2}$, expression (1) degenerates to a constant.

- When $p$ is $1$ or $0$, the randomized estimate degenerates to an estimator without randomized sampling.


+++

We shall analyze only discuss the situation in which $p \in (\frac{1}{2},1)$

(the situation in which $p \in (0,\frac{1}{2})$ is symmetric).

From expressions (5) and (7) we can deduce that: 

- The MSE of $\hat{\pi}$  decreases as $p$ increasing.

+++

## Comparing two survey designs 

Let's compare the preceding randomized-response method with a stylized non-randomized response method.

+++

In our non-randomized response method, we suppose that:

+++

- Members of Group A tells the truth with probability of $T_a$ while the members of Group B tells the truth with probability of $T_b$

+++

- $Y_i$ is $1$ or $0$ according to whether the sample's $i\text{th}$ member's report is in Group A or not.

+++

Then we can estimate $\pi$ as:

+++

$$
\begin{equation}
\hat{\pi}=\frac{\sum_{i=1}^{n}Y_i}{n} \tag{8}
\end{equation}
$$

+++

We calculate the expectation, bias, and variance of the estimator to be:

+++

$$
\begin{align}
\mathbb{E}(\hat{\pi})&=\pi T_a + \left[ (1-\pi)(1-T_b)\right] \tag{9}\\
\\
Bias(\hat{\pi})&=\mathbb{E}(\hat{\pi}-\pi)\\
&=\pi [T_a + T_b -2 ] + [1- T_b] \tag{10}\\
\\
Var(\hat{\pi})&=\frac{ \left[ \pi T_a + (1-\pi)(1-T_b)\right]  \left[1- \pi T_a -(1-\pi)(1-T_b)\right] }{n} \tag{11}
\end{align}
$$


It is useful to define a



$$
\text{MSE Ratio}=\frac{\text{Mean Square Error Randomized}}{\text{Mean Square Error Regular}}
$$

+++

We can compute  MSE Ratios for different surveys and survey designs associated with different parameter values.

+++

The following Python code computes the objects we want to stare at in order to make comparisons
under  different values of $\pi_A$ and $n$:

```{code-cell} ipython3
class Comparison:
    def __init__(self,A,n):
        self.A = A
        self.n = n
        TaTb = np.array([[0.95,1],[0.9,1],[0.7,1],[0.5,1],[1,0.95],[1,0.9],[1,0.7],[1,0.5],[0.95,0.95],[0.9,0.9],[0.7,0.7],[0.5,0.5]])
        self.p_arr = np.array([0.6,0.7,0.8,0.9])
        self.p_map = dict(zip(self.p_arr,["MSE Ratio: p=" + str(x) for x in self.p_arr]))
        self.template = pd.DataFrame(columns = self.p_arr)
        self.template[['T_a','T_b']] = TaTb
        self.template['Bias']=None
    
    def theoretical(self):
        df = self.template.copy()
        df['Bias']=self.A*(df['T_a']+df['T_b']-2)+(1-df['T_b'])
        for p in self.p_arr:
            df[p] = (1 / (16 * (p - 1/2)**2) - (self.A - 1/2)**2)/self.n / \
                    (df['Bias']**2 + ((self.A * df['T_a'] + (1 - self.A)*(1 - df['T_b']))*(1 - self.A*df['T_a'] - (1 - self.A)*(1 - df['T_b'])) / self.n))
            df[p] = df[p].round(2)
        df = df.set_index(["T_a", "T_b","Bias"]).rename(columns=self.p_map)
        return df
        
    def MCsimulation(self, size=1000, seed=123456):
        df = self.template.copy()
        np.random.seed(seed)
        sample = np.random.rand(size, self.n) <= self.A
        random_device = np.random.rand(size, self.n)
        mse_rd = {}
        for p in self.p_arr:
            spinner = random_device <= p
            rd_answer = sample*spinner + (1-sample)*(1-spinner)
            n1 = rd_answer.sum(axis=1)
            pi_hat = (p-1)/(2*p-1) + n1 / self.n / (2*p-1)
            mse_rd[p] = np.sum((pi_hat - self.A)**2)
        for inum, irow in df.iterrows():
            truth_a = np.random.rand(size, self.n) <= irow.T_a
            truth_b = np.random.rand(size, self.n) <= irow.T_b
            trad_answer = sample * truth_a + (1-sample) * (1-truth_b)
            pi_trad = trad_answer.sum(axis=1) / self.n
            df.loc[inum,'Bias'] = pi_trad.mean() - self.A
            mse_trad = np.sum((pi_trad - self.A)**2)
            for p in self.p_arr:
                df.loc[inum,p] = (mse_rd[p] / mse_trad).round(2)
        df = df.set_index(["T_a", "T_b","Bias"]).rename(columns=self.p_map)
        return df
```

Let's put the code to work for parameter values

+++

- $\pi_A=0.6$
- $n=1000$

+++

We can generate MSE Ratios theoretically using the above formulas.

We can also perform a  Monte-Carlo simulation  of the MSE Ratio.

```{code-cell} ipython3
cp1 = Comparison(0.6,1000)
df1_theoretical = cp1.theoretical()
df1_theoretical
```

```{code-cell} ipython3
df1_mc = cp1.MCsimulation()
df1_mc
```

The theoretical calculations  do a good job of predicting the Monte Carlo results.

We see that in many situations, especially when the bias is not small, the MSE of the randomized-samplijng  methods is smaller than that of the non-randomized sampling method. 

These differences become larger as  $p$ increases.

+++

By adjusting  parameters $\pi_A$ and $n$, we can study outcomes in different situations.

For example, for another situation described in Warner {cite}`warner1965randomized`:

+++

- $\pi_A=0.5$
- $n=1000$

+++

we can use the code

```{code-cell} ipython3
cp2=Comparison(0.5,1000)
df2_theoretical = cp2.theoretical()
df2_theoretical
```

```{code-cell} ipython3
df2_mc = cp2.MCsimulation()
df2_mc
```

We can also revisit a calculation in the  concluding section of Warner {cite}`warner1965randomized` in which 

+++

- $\pi_A=0.6$
- $n=2000$

+++

We use the code

```{code-cell} ipython3
cp3=Comparison(0.6,2000)
df3_theoretical = cp3.theoretical()
df3_theoretical
```

```{code-cell} ipython3
df3_mc = cp3.MCsimulation()
df3_mc
```

Evidently, as $n$ increases, the randomized response method does  better performance in more situations.

+++

## Concluding Remarks

In  quantecon lecture XXXX, we shall describe some alternative randomized response surveys.

That lecture presents the utilitarian analysis of those alternatives conducted by Lars Ljungqvist
{cite}`ljungqvist1993unified`.
