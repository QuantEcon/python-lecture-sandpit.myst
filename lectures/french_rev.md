---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region user_expressions=[] -->
# Inflation During French Revolution 


## Overview and Data Sources

This notebook uses data from three spreadsheets:

  * _static/fig_3.ods
  * _static/dette.xlsx
  * _static/assignat.xlsx
<!-- #endregion -->

<!-- #region user_expressions=[] -->
## To Do for Zejin

I want to tweak and consolidate the extra lines that Zejin drew on   **Figure 7**.  

I'd like to experiment in plotting **six** extra lines on the graph -- a pair of lines for each of our subsamples

  * one for the $y$ on $x$ regression line
  * another for the $x$ on $y$ regression line

I'd like the  $y$ on $x$ and $x$ on $y$ lines to be in separate colors.

Zejin, I can explain on zoom the lessons I want to convey with this.  

It will be fun. 

To compute the regression lines, Zejin wrote  a  function that use standard formulas
for a and b in a least squares regression y = a + b x + residual -- i.e., b is ratio of sample covariance of y,x to sample variance of x; while a is then computed from a =  sample mean of y - \hat b *sample mean of x

We could presumably tell students how to do this with a couple of numpy lines

 
<!-- #endregion -->

```python
!pip install odfpy
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
```

<!-- #region user_expressions=[] -->
## Figure 1
<!-- #endregion -->

```python
# Read the data from the Excel file
data1 = pd.read_excel('_static/dette.xlsx', sheet_name='Debt', usecols='R:S', skiprows=5, nrows=99, header=None)
data1a = pd.read_excel('_static/dette.xlsx', sheet_name='Debt', usecols='P', skiprows=89, nrows=15, header=None)

# Plot the data
plt.figure()
plt.plot(range(1690, 1789), 100 * data1.iloc[:, 1], linewidth=0.8)

date = np.arange(1690, 1789)
index = (date < 1774) & (data1.iloc[:, 0] > 0)
plt.plot(date[index], 100 * data1[index].iloc[:, 0], '*:', color='r', linewidth=0.8)

# Plot the additional data
plt.plot(range(1774, 1789), 100 * data1a, '*:', color='orange')

# Note about the data
# The French data before 1720 don't match up with the published version
# Set the plot properties
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().set_xlim([1688, 1788])
plt.ylabel('% of Taxes')

plt.tight_layout()
plt.show()

#plt.savefig('frfinfig1.pdf', dpi=600)
#plt.savefig('frfinfig1.jpg', dpi=600)
```

<!-- #region user_expressions=[] -->
## Figure 2
<!-- #endregion -->

```python
# Read the data from Excel file
data2 = pd.read_excel('_static/dette.xlsx', sheet_name='Militspe', usecols='M:X', skiprows=7, nrows=102, header=None)

# Plot the data
plt.figure()
plt.plot(range(1689, 1791), data2.iloc[:, 5], linewidth=0.8)
plt.plot(range(1689, 1791), data2.iloc[:, 11], linewidth=0.8, color='red')
plt.plot(range(1689, 1791), data2.iloc[:, 9], linewidth=0.8, color='orange')
plt.plot(range(1689, 1791), data2.iloc[:, 8], 'o-', markerfacecolor='none', linewidth=0.8, color='purple')

# Customize the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().tick_params(labelsize=12)
plt.xlim([1689, 1790])
plt.ylabel('millions of pounds', fontname='Times New Roman', fontsize=12)

# Add text annotations
plt.text(1765, 1.5, 'civil', fontname='Times New Roman', fontsize=10)
plt.text(1760, 4.2, 'civil plus debt service', fontname='Times New Roman', fontsize=10)
plt.text(1708, 15.5, 'total govt spending', fontname='Times New Roman', fontsize=10)
plt.text(1759, 7.3, 'revenues', fontname='Times New Roman', fontsize=10)

# Save the figure as a PDF
#plt.savefig('frfinfig2.pdf', dpi=600)
```

<!-- #region user_expressions=[] -->
## Figure 3 


<!-- #endregion -->

```python
# Read the data from the Excel file
data1 = pd.read_excel('_static/fig_3.ods', sheet_name='Sheet1', usecols='C:F', skiprows=5, nrows=30, header=None, engine="odf")

data1.replace(0, np.nan, inplace=True)
```

```python
# Plot the data
plt.figure()

plt.plot(range(1759, 1789, 1), data1.iloc[:, 0], '-x', linewidth=0.8)
plt.plot(range(1759, 1789, 1), data1.iloc[:, 1], '--*', linewidth=0.8)
plt.plot(range(1759, 1789, 1), data1.iloc[:, 2], '-o', linewidth=0.8, markerfacecolor='none')
plt.plot(range(1759, 1789, 1), data1.iloc[:, 3], '-*', linewidth=0.8)

plt.text(1775, 610, 'total spending', fontname='Times New Roman', fontsize=10)
plt.text(1773, 325, 'military', fontname='Times New Roman', fontsize=10)
plt.text(1773, 220, 'civil plus debt service', fontname='Times New Roman', fontsize=10)
plt.text(1773, 80, 'debt service', fontname='Times New Roman', fontsize=10)
plt.text(1785, 500, 'revenues', fontname='Times New Roman', fontsize=10)



plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim([0, 700])
plt.ylabel('millions of livres')

#plt.savefig('frfinfig3.jpg', dpi=600)
```

```python
# Plot the data
plt.figure()

plt.plot(np.arange(1759, 1789, 1)[~np.isnan(data1.iloc[:, 0])], data1.iloc[:, 0][~np.isnan(data1.iloc[:, 0])], '-x', linewidth=0.8)
plt.plot(np.arange(1759, 1789, 1)[~np.isnan(data1.iloc[:, 1])], data1.iloc[:, 1][~np.isnan(data1.iloc[:, 1])], '--*', linewidth=0.8)
plt.plot(np.arange(1759, 1789, 1)[~np.isnan(data1.iloc[:, 2])], data1.iloc[:, 2][~np.isnan(data1.iloc[:, 2])], '-o', linewidth=0.8, markerfacecolor='none')
plt.plot(np.arange(1759, 1789, 1)[~np.isnan(data1.iloc[:, 3])], data1.iloc[:, 3][~np.isnan(data1.iloc[:, 3])], '-*', linewidth=0.8)

plt.text(1775, 610, 'total spending', fontname='Times New Roman', fontsize=10)
plt.text(1773, 325, 'military', fontname='Times New Roman', fontsize=10)
plt.text(1773, 220, 'civil plus debt service', fontname='Times New Roman', fontsize=10)
plt.text(1773, 80, 'debt service', fontname='Times New Roman', fontsize=10)
plt.text(1785, 500, 'revenues', fontname='Times New Roman', fontsize=10)


plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim([0, 700])
plt.ylabel('millions of livres')

#plt.savefig('frfinfig3_ignore_nan.jpg', dpi=600)
```

<!-- #region user_expressions=[] -->
## Figure 4
<!-- #endregion -->

```python
# French military spending, 1685-1789, in 1726 livres
data4 = pd.read_excel('_static/dette.xlsx', sheet_name='Militspe', usecols='D', skiprows=3, nrows=105, header=None).squeeze()
years = range(1685, 1790)

plt.figure()
plt.plot(years, data4, '*-', linewidth=0.8)

plt.plot(range(1689, 1791), data2.iloc[:, 4], linewidth=0.8)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().tick_params(labelsize=12)
plt.xlim([1689, 1790])
plt.xlabel('*: France')
plt.ylabel('Millions of livres')
plt.ylim([0, 475])

#plt.savefig('frfinfig4.pdf', dpi=600)
```

<!-- #region user_expressions=[] -->
## Figure 5
<!-- #endregion -->

```python
# Read data from Excel file
data5 = pd.read_excel('_static/dette.xlsx', sheet_name='Debt', usecols='K', skiprows=41, nrows=120, header=None)

# Plot the data
plt.figure()
plt.plot(range(1726, 1846), data5.iloc[:, 0], linewidth=0.8)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().tick_params(labelsize=12)
plt.xlim([1726, 1845])
plt.ylabel('1726 = 1', fontsize=12, fontname='Times New Roman')

# Save the figure as a PDF
#plt.savefig('frfinfig5.pdf', dpi=600)
```

<!-- #region user_expressions=[] -->
## Figure 7

I'd like to create three additional versions of the following figure. 

The additional versions would have least squares regression lines of the y variable on the x data, in different colors
during three pertinent subperiods:


* subperiod 1: ("real bills period): January 1791 to July 1793

* subperiod 2: ("terror:):  August 1793 - July 1794

* subperiod 3: ("classic Cagan hyperinflation): August 1794 - March 1796

We could try several versions and figure out which is most informative and beautiful.  

One possibility would be to have all three regression lines on the same graph -- they we'd just have the original graph followed by this one.

I can explain what this is designed to show.


To compute the regression lines we could simply use python to compute the constant and slope in a couple of Python lines. I don't want the R^2 and other stuff produced by statsmodels at this point
<!-- #endregion -->

```python
def fit(x, y):

    b = np.cov(x, y)[0, 1] / np.var(x)
    a = y.mean() - b * x.mean()

    return a, b
```

```python
caron = np.array([
    [1791, 96.696],
    [1791 + 1/12, 96.426],
    [1791 + 2/12, 95.822],
    [1791 + 3/12, 95.074],
    [1791 + 4/12, 93.606],
    [1791 + 5/12, 92.921],
    [1791 + 6/12, 92.130],
    [1791 + 7/12, 89.927],
    [1791 + 8/12, 89.832],
    [1791 + 9/12, 89.620],
    [1791 + 10/12, 88.035],
    [1791 + 11/12, 85.605],
    [1792, 81.324],
    [1792 + 1/12, 76.178],
    [1792 + 2/12, 73.931],
    [1792 + 3/12, 75.893],
    [1792 + 4/12, 71.867],
    [1792 + 5/12, 71.914],
    [1792 + 6/12, 72.720],
    [1792 + 7/12, 72.311],
    [1792 + 8/12, 75.099],
    [1792 + 9/12, 75.861],
    [1792 + 10/12, 76.415],
    [1792 + 11/12, 74.994],
    [1793, 66.025],
    [1793 + 1/12, 64.809],
    [1793 + 2/12, 62.833],
    [1793 + 3/12, 57.618],
    [1793 + 4/12, 57.092],
    [1793 + 5/12, 48.559],
    [1793 + 6/12, 42.205],
    [1793 + 7/12, 40.263],
    [1793 + 8/12, 40.601],
    [1793 + 9/12, 42.283],
    [1793 + 10/12, 48.283],
    [1793 + 11/12, 54.751],
    [1794, 49.558],
    [1794 + 1/12, 48.932],
    [1794 + 2/12, 45.587],
    [1794 + 3/12, 44.501],
    [1794 + 4/12, 42.479],
    [1794 + 5/12, 39.822],
    [1794 + 6/12, 41.518],
    [1794 + 7/12, 39.119],
    [1794 + 8/12, 36.225],
    [1794 + 9/12, 34.409],
    [1794 + 10/12, 31.247],
    [1794 + 11/12, 27.725],
    [1795, 24.313],
    [1795 + 1/12, 21.974],
    [1795 + 2/12, 18.654],
    [1795 + 3/12, 14.326],
    [1795 + 4/12, 9.158],
    [1795 + 5/12, 5.654],
    [1795 + 6/12, 4.471],
    [1795 + 7/12, 3.716],
    [1795 + 8/12, 2.859],
    [1795 + 9/12, 2.149],
    [1795 + 10/12, 1.217],
    [1795 + 11/12, 0.820],
    [1796, 0.634],
    [1796 + 1/12, 0.547],
    [1796 + 3/12, 0.431]
])


nom_balances= np.array([[1789+10/12   ,  90.000   ],
[1789+11/12 ,   105.000   ],
[1790       ,   124.000   ],
[1790+1/12  ,   142.000   ],
[1790+2/12  ,   170.000   ],
[1790+3/12  ,   190.237   ],
[1790+4/12  ,   212.795   ],
[1790+5/12  ,   258.606   ],
[1790+6/12  ,   307.669   ],
[1790+7/12  ,   348.881   ],
[1790+8/12  ,   390.294   ],
[1790+9/12  ,   437.095   ],
[1790+10/12 ,   485.095   ],
[1790+11/12 ,   529.095   ],
[1791       ,   627.018   ],
[1791+1/12  ,   722.398   ],
[1791+2/12  ,   808.214   ],
[1791+3/12  ,   897.784   ],
[1791+4/12  ,   984.671   ],
[1791+5/12  ,  1041.232   ],
[1791+6/12  ,  1109.164   ],
[1791+7/12  ,  1162.808   ],
[1791+8/12  ,  1200.332   ],
[1791+9/12  ,  1299.622   ],
[1791+10/12 ,  1366.562   ],
[1791+11/12 ,  1394.241   ],
[1792       ,  1460.602   ],
[1792+1/12  ,  1529.391   ],
[1792+2/12  ,  1570.098   ],
[1792+3/12  ,  1621.548   ],
[1792+4/12  ,  1693.131   ],
[1792+5/12  ,  1722.802   ],
[1792+6/12  ,  1765.895   ],
[1792+7/12  ,  1836.786   ],
[1792+8/12  ,  1950.917   ],
[1792+9/12  ,  2062.232   ],
[1792+10/12 ,  2172.500   ],
[1792+11/12 ,  2291.201   ],
[1793       ,  2413.451   ],
[1793+1/12  ,  2583.358   ],
[1793+2/12  ,  2834.287   ],
[1793+3/12  ,  3114.676   ],
[1793+4/12  ,  3388.209   ],
[1793+5/12  ,  3527.772   ],
[1793+6/12  ,  3766.645   ],
[1793+7/12  ,  4084.883   ],
[1793+8/12  ,  4452.083   ],
[1793+9/12  ,  4603.083   ],
[1793+10/12 ,  4878.728   ],
[1793+11/12 ,  5114.951   ],
[1794       ,  5229.818   ],
[1794+1/12  ,  5382.818   ],
[1794+2/12  ,  5478.720   ],
[1794+3/12  ,  5705.913   ],
[1794+4/12  ,  5905.769],
[1794+5/12  ,  6054.298],
[1794+6/12  ,  6217.455],
[1794+7/12  ,  6397.486],
[1794+8/12  ,  6573.364],
[1794+9/12  ,  6721.252],
[1794+10/12 ,  6962.986],
[1794+11/12 ,  7154.619],
[1795       ,  7349.853],
[1795+1/12  ,  7702.848],
[1795+2/12  ,  8148.652],
[1795+3/12  ,  8903.508],
[1795+4/12  , 10055.347],
[1795+5/12  , 11374.560],
[1795+6/12  , 13822.633],
[1795+7/12  , 15469.625],
[1795+8/12  , 17271.144],
[1795+9/12  , 19462.168],
[1795+10/12 , 22356.131],
[1795+11/12 , 25457.189],
[1796       , 34611.005],
[1796+1/12  , 35620.096],
[1796+2/12  , 37540.933],
[1796+3/12  , 36758.033],
[1796+4/12  , 35800.495],
[1796+5/12  , 34682.422],
[1796+6/12  , 33555.590]])
```

```python
infl = np.concatenate(([np.nan], -np.log(caron[1:63, 1] / caron[0:62, 1])))
bal = nom_balances[14:77, 1] * caron[:, 1] / 1000

plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# first subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')

# second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')

# third subsample
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()
#plt.savefig('frfinfig7.pdf', dpi=600)
```

```python
infl = np.concatenate(([np.nan], -np.log(caron[1:63, 1] / caron[0:62, 1])))
bal = nom_balances[14:77, 1] * caron[:, 1] / 1000

plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# first subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')
a1, b1 = fit(bal[1:31], infl[1:31])
plt.plot(bal[1:31], a1 + bal[1:31] * b1, color='blue')

# second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')
# a2, b2 = fit(bal[31:44], infl[31:44])
# plt.plot(bal[31:44], a2 + bal[31:44] * b2, color='red')

# third subsample
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')
# a3, b3 = fit(bal[44:63], infl[44:63])
# plt.plot(bal[44:63], a3 + bal[44:63] * b3, color='orange')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()
#plt.savefig('frfinfig7_line1.pdf', dpi=600)
```

```python
infl = np.concatenate(([np.nan], -np.log(caron[1:63, 1] / caron[0:62, 1])))
bal = nom_balances[14:77, 1] * caron[:, 1] / 1000

plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# first subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')
a1_rev, b1_rev = fit(infl[1:31], bal[1:31])
plt.plot(a1_rev + b1_rev * infl[1:31], infl[1:31], color='blue')

# second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')
# a2, b2 = fit(bal[31:44], infl[31:44])
# plt.plot(bal[31:44], a2 + bal[31:44] * b2, color='red')

# third subsample
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')
# a3, b3 = fit(bal[44:63], infl[44:63])
# plt.plot(bal[44:63], a3 + bal[44:63] * b3, color='orange')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()
#plt.savefig('frfinfig7_line1_rev.pdf', dpi=600)
```

```python
infl = np.concatenate(([np.nan], -np.log(caron[1:63, 1] / caron[0:62, 1])))
bal = nom_balances[14:77, 1] * caron[:, 1] / 1000

plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# first subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')
# a1, b1 = fit(bal[1:31], infl[1:31])
# plt.plot(bal[1:31], a1 + bal[1:31] * b1, color='blue')

# second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')
a2, b2 = fit(bal[31:44], infl[31:44])
plt.plot(bal[31:44], a2 + bal[31:44] * b2, color='red')

# third subsample
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')
# a3, b3 = fit(bal[44:63], infl[44:63])
# plt.plot(bal[44:63], a3 + bal[44:63] * b3, color='orange')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()
#plt.savefig('frfinfig7_line2.pdf', dpi=600)
```

```python
infl = np.concatenate(([np.nan], -np.log(caron[1:63, 1] / caron[0:62, 1])))
bal = nom_balances[14:77, 1] * caron[:, 1] / 1000

plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# first subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')
# a1, b1 = fit(bal[1:31], infl[1:31])
# plt.plot(bal[1:31], a1 + bal[1:31] * b1, color='blue')

# second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')
a2_rev, b2_rev = fit(infl[31:44], bal[31:44])
plt.plot(a2_rev + b2_rev * infl[31:44], infl[31:44], color='red')

# third subsample
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')
# a3, b3 = fit(bal[44:63], infl[44:63])
# plt.plot(bal[44:63], a3 + bal[44:63] * b3, color='orange')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()
#plt.savefig('frfinfig7_line2_rev.pdf', dpi=600)
```

```python
infl = np.concatenate(([np.nan], -np.log(caron[1:63, 1] / caron[0:62, 1])))
bal = nom_balances[14:77, 1] * caron[:, 1] / 1000

plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# first subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')
# a1, b1 = fit(bal[1:31], infl[1:31])
# plt.plot(bal[1:31], a1 + bal[1:31] * b1, color='blue')

# second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')
# a2, b2 = fit(bal[31:44], infl[31:44])
# plt.plot(bal[31:44], a2 + bal[31:44] * b2, color='red')

# third subsample
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')
a3, b3 = fit(bal[44:63], infl[44:63])
plt.plot(bal[44:63], a3 + bal[44:63] * b3, color='orange')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()
#plt.savefig('frfinfig7_line3.pdf', dpi=600)
```

```python
infl = np.concatenate(([np.nan], -np.log(caron[1:63, 1] / caron[0:62, 1])))
bal = nom_balances[14:77, 1] * caron[:, 1] / 1000

plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# first subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')
# a1, b1 = fit(bal[1:31], infl[1:31])
# plt.plot(bal[1:31], a1 + bal[1:31] * b1, color='blue')

# second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')
# a2, b2 = fit(bal[31:44], infl[31:44])
# plt.plot(bal[31:44], a2 + bal[31:44] * b2, color='red')

# third subsample
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')
a3_rev, b3_rev = fit(infl[44:63], bal[44:63])
plt.plot(a3_rev + b3_rev * infl[44:63], infl[44:63], color='orange')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()
#plt.savefig('frfinfig7_line3_rev.pdf', dpi=600)
```

<!-- #region user_expressions=[] -->
## Figure 8
<!-- #endregion -->

```python
# Read the data from Excel file
data7 = pd.read_excel('_static/assignat.xlsx', sheet_name='Data', usecols='P:Q', skiprows=4, nrows=80, header=None)
data7a = pd.read_excel('_static/assignat.xlsx', sheet_name='Data', usecols='L', skiprows=4, nrows=80, header=None)

# Create the figure and plot
plt.figure()
h = plt.plot(pd.date_range(start='1789-11-01', periods=len(data7), freq='M'), (data7a.values * [1, 1]) * data7.values, linewidth=1.)
plt.setp(h[1], linestyle='--', color='red')

# Hold on equivalent in matplotlib is just plotting on the same figure
plt.vlines(['1793-07-15', '1793-07-15'], 0, 3000, linewidth=0.8, color='orange')
plt.vlines(['1794-07-15', '1794-07-15'], 0, 3000, linewidth=0.8, color='purple')

plt.ylim([0, 3000])

# Set properties of the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().tick_params(labelsize=12)
plt.xlim(pd.Timestamp('1789-11-01'), pd.Timestamp('1796-06-01'))
plt.ylabel('millions of livres', fontname='Times New Roman', fontsize=12)

# Add text annotations
plt.text(pd.Timestamp('1793-09-01'), 200, 'Terror', fontname='Times New Roman', fontsize=12)
plt.text(pd.Timestamp('1791-05-01'), 750, 'gold value', fontname='Times New Roman', fontsize=12)
plt.text(pd.Timestamp('1794-10-01'), 2500, 'real value', fontname='Times New Roman', fontsize=12)

# Save the figure as a PDF
#plt.savefig('frfinfig8.pdf', dpi=600)
```

<!-- #region user_expressions=[] -->
## Figure 9
<!-- #endregion -->

```python
# Create the figure and plot
plt.figure()
x = np.arange(1789 + 10/12, 1796 + 5/12, 1/12)
h, = plt.plot(x, 1. / data7.iloc[:, 0], linestyle='--')
h, = plt.plot(x, 1. / data7.iloc[:, 1], color='r')

# Set properties of the plot
plt.gca().tick_params(labelsize=12)
plt.yscale('log')
plt.xlim([1789 + 10/12, 1796 + 5/12])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add vertical lines
plt.axvline(x=1793 + 6.5/12, linestyle='-', linewidth=0.8, color='orange')
plt.axvline(x=1794 + 6.5/12, linestyle='-', linewidth=0.8, color='purple')

# Add text
plt.text(1793.75, 120, 'Terror', fontname='Times New Roman', fontsize=12)
plt.text(1795, 2.8, 'price level', fontname='Times New Roman', fontsize=12)
plt.text(1794.9, 40, 'gold', fontname='Times New Roman', fontsize=12)

#plt.savefig('frfinfig9.pdf', dpi=600)
```

<!-- #region user_expressions=[] -->
## Figure 11
<!-- #endregion -->

```python
# Read data from Excel file
data11 = pd.read_excel('_static/assignat.xlsx', sheet_name='Budgets', usecols='J:K', skiprows=22, nrows=52, header=None)

# Prepare the x-axis data
x_data = np.concatenate([
    np.arange(1791, 1794 + 8/12, 1/12),
    np.arange(1794 + 9/12, 1795 + 3/12, 1/12)
])

# Remove NaN values from the data
data11_clean = data11.dropna()

# Plot the data
plt.figure()
h = plt.plot(x_data, data11_clean.values[:, 0], linewidth=0.8)
h = plt.plot(x_data, data11_clean.values[:, 1], '--', linewidth=0.8)


# Set plot properties
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.xlim([1791, 1795 + 3/12])
plt.xticks(np.arange(1791, 1796))
plt.yticks(np.arange(0, 201, 20))

# Set the y-axis label
plt.ylabel('millions of livres', fontsize=12, fontname='Times New Roman')

#plt.savefig('frfinfig11.pdf', dpi=600)
```

<!-- #region user_expressions=[] -->
## Figure 12
<!-- #endregion -->

```python
# Read data from Excel file
data12 = pd.read_excel('_static/assignat.xlsx', sheet_name='seignor', usecols='F', skiprows=6, nrows=75, header=None).squeeze()


# Create a figure and plot the data
plt.figure()
plt.plot(pd.date_range(start='1790', periods=len(data12), freq='M'), data12, linewidth=0.8)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.axhline(y=472.42/12, color='r', linestyle=':')
plt.xticks(ticks=pd.date_range(start='1790', end='1796', freq='AS'), labels=range(1790, 1797))
plt.xlim(pd.Timestamp('1791'), pd.Timestamp('1796-02') + pd.DateOffset(months=2))
plt.ylabel('millions of livres', fontsize=12, fontname='Times New Roman')
plt.text(pd.Timestamp('1793-11'), 39.5, 'revenues in 1788', verticalalignment='top', fontsize=12, fontname='Times New Roman')

#plt.savefig('frfinfig12.pdf', dpi=600)
```

<!-- #region user_expressions=[] -->
## Figure 13
<!-- #endregion -->

```python
# Read data from Excel file
data13 = pd.read_excel('_static/assignat.xlsx', sheet_name='Exchge', usecols='P:T', skiprows=3, nrows=502, header=None)

# Plot the last column of the data
plt.figure()
plt.plot(data13.iloc[:, -1], linewidth=0.8)

# Set properties of the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_xlim([1, len(data13)])

# Set x-ticks and x-tick labels
ttt = np.arange(1, len(data13) + 1)
plt.xticks(ttt[~np.isnan(data13.iloc[:, 0])], 
           ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb',
           'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'])

# Add text to the plot
plt.text(1, 120, '1795', fontname='Times New Roman', fontsize=12, ha='center')
plt.text(262, 120, '1796', fontname='Times New Roman', fontsize=12, ha='center')

# Draw a horizontal line and add text
plt.axhline(y=186.7, color='red', linestyle='-', linewidth=0.8)
plt.text(150, 190, 'silver parity', fontname='Times New Roman', fontsize=12)

# Add an annotation with an arrow
plt.annotate('end of the assignat', xy=(340, 172), xytext=(380, 160),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontname='Times New Roman', fontsize=12)

#plt.savefig('frfinfig13.pdf', dpi=600)
```

<!-- #region user_expressions=[] -->
## Figure 14
<!-- #endregion -->

```python
# figure 14
data14 = pd.read_excel('_static/assignat.xlsx', sheet_name='Post-95', usecols='I', skiprows=9, nrows=91, header=None).squeeze()
data14a = pd.read_excel('_static/assignat.xlsx', sheet_name='Post-95', usecols='F', skiprows=100, nrows=151, header=None).squeeze()

plt.figure()
h = plt.plot(data14, '*-', markersize=2, linewidth=0.8)
plt.plot(np.concatenate([np.full(data14.shape, np.nan), data14a]), linewidth=0.8)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_xticks(range(20, 237, 36))
plt.gca().set_xticklabels(range(1796, 1803))
plt.xlabel('*: Before the 2/3 bankruptcy')
plt.ylabel('Francs')
#plt.savefig('frfinfig14.pdf', dpi=600)
```

<!-- #region user_expressions=[] -->
## Figure 15
<!-- #endregion -->

```python
# figure 15
data15 = pd.read_excel('_static/assignat.xlsx', sheet_name='Post-95', usecols='N', skiprows=4, nrows=88, header=None).squeeze()

plt.figure()
h = plt.plot(range(2, 90), data15, '*-', linewidth=0.8)
plt.setp(h, markersize=2)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.text(47.5, 11.4, '17 brumaire', horizontalalignment='left', fontname='Times New Roman', fontsize=12)
plt.text(49.5, 14.75, '19 brumaire', horizontalalignment='left', fontname='Times New Roman', fontsize=12)
plt.text(15, -1, 'Vendémiaire 8', fontname='Times New Roman', fontsize=12, horizontalalignment='center')
plt.text(45, -1, 'Brumaire', fontname='Times New Roman', fontsize=12, horizontalalignment='center')
plt.text(75, -1, 'Frimaire', fontname='Times New Roman', fontsize=12, horizontalalignment='center')
plt.ylim([0, 25])
plt.xticks([], [])
plt.ylabel('Francs')
#plt.savefig('frfinfig15.pdf', dpi=600)
```

```python

```
