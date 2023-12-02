# Applied Modern Porfolio Theory

To implement the backtesting using the EWMA method for computing $\text{VaR}_{10D,t}$, we need to follow several steps:

1. Calculate the EWMA variance $\sigma^{2}_{t+1|t}$ for the entire dataset.
2. Compute the $\text{VaR}_{10D,t}$ using the EWMA variance.

Let's start by computing the EWMA variance for the entire dataset using the given formula:

$$
\sigma_{t+1|t}^{2} = \lambda \sigma^{2}_{t|t-1} + (1-\lambda)r_{t}^{2}
$$

Where:

- $\lambda = 0.72$ is the smoothing parameter.
- $\sigma_{t+1|t}^{2}$ is the updated variance estimate at time $t+1$ given information up to time $t$.
- $\sigma^{2}_{t|t-1}$ is the previous variance estimate.
- $r_{t}^{2}$ is the squared return at time $t$.

Let's implement this calculation for the entire dataset:

```python
# Calculate EWMA variance
lambda_ewma = 0.72

# Calculate squared returns
df['rets_sq'] = df['rets']**2

# Calculate the EWMA variance
ewma_var = [df['rets_sq'].iloc[0]]  # Using the first return squared as initial variance
for i in range(1, len(df)):
    ewma_var.append(lambda_ewma * ewma_var[i - 1] + (1 - lambda_ewma) * df['rets_sq'].iloc[i])

df['EWMA_Variance'] = ewma_var
```

Now, with the EWMA variance calculated for the entire dataset, we can proceed to compute $\text{VaR}_{10D,t}$ using this variance estimate. We'll follow the same steps as before to calculate VaR:

```python
# Compute 10 days VaR using EWMA variance
window_ewma = 21
df['10D_VaR_EWMA'] = factor * np.sqrt(10) * df['rets'].rolling(window=window_ewma).std() * np.sqrt(df['EWMA_Variance'])

# Identify breaches
df['breach_EWMA'] = (df['10D_rets'] < df['10D_VaR_EWMA']).astype(int)

# Count number of breaches and percentage breaches
mask_ewma = df['breach_EWMA'] == 1
pct_var_breaches_ewma = df[mask_ewma]['breach_EWMA'].count() / len(df)
print('No. of Breaches (EWMA)', df[mask_ewma]['breach_EWMA'].count())
print('Percentage VaR Breaches (EWMA)', pct_var_breaches_ewma)
```

This code will compute the number of breaches and the percentage of VaR breaches using the EWMA method. The breaches are identified by comparing the 10D forward returns to the calculated VaR using the EWMA variance. The breaches are counted and the percentage of breaches is calculated based on the length of the dataset.

Finally, for visualizing the breaches, you can use the same plotting code as before, but this time, use the `breach_EWMA` column to identify breaches using the EWMA method:

```python
fig = px.line(df, x=df.index, y=['10D_VaR_EWMA', '10D_rets'], title='NASDAQ100 VaR Back Testing (EWMA) of 99%/10day')
fig.add_scatter(x=df[df['breach_EWMA'] == 1].index, y=df[df['breach_EWMA'] == 1]['10D_rets'], mode='markers', marker=dict(color='red', symbol='x'), name='Breach (EWMA)')
fig.update_layout(xaxis_title='Date', yaxis_title='Values')
fig.show()
```

This will generate a plot similar to the previous one but with breaches identified using the EWMA method for calculating VaR. This way, you can visually compare the breaches identified using the standard method and the EWMA method for VaR computation.