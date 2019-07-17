# %%
import numpy as np
import pandas as pd
# import seaborn as sns
from matplotlib import pyplot as plt

# %%
'''
5.1 For your chosen stock, calculate the mean daily return 
and daily standard deviation of returns, and then just annualise 
them to get mean expected annual return and volatility of that 
single stock. ( annual mean = daily mean * 252 , annual stdev 
 = daily stdev * sqrt(252) )
 
+

5.2 Now, we need to diversify our portfolio. Build your own 
portfolio by choosing any 5 stocks, preferably of different 
sectors and different caps. Assume that all 5 have the same 
weightage, i.e. 20% . Now calculate the annual returns and 
volatility of the entire portfolio 
( Hint : Don't forget to use the covariance )
'''

stocks = {'nifty_df': 'Nifty50.csv', 
          'adani_df':'ADANIPOWER.csv', 
          'cipla_df':'CIPLA.csv', 
          'hero_df':'HEROMOTOCO.csv', 
          'ashoka_df':'ASHOKA.csv'
}

dfs = {}
dailyValues = {}
annualValues = {}
for name, csv in stocks.items():
    # print(csv)
    # creating dataframe
    dfs[name] = pd.read_csv(csv)
    if csv != 'Nifty50.csv':
        dfs[name] = dfs[name][dfs[name]["Series"] == "EQ"]   
    dfs[name].Date = pd.to_datetime(dfs[name]['Date'])
    dfs[name] = dfs[name].set_index('Date')
    
    # calculating values i.e. mean and std
    dfs[name]["dailyChange"] = dfs[name]["Close Price"].pct_change()
    dfs[name].dropna(inplace = True)
    
    # (std, mean)
    dailyValues[name] = [dfs[name].dailyChange.std(), dfs[name].dailyChange.mean()]
    annualValues[name] = [dailyValues[name][0]*(252**0.5), dailyValues[name][1]*252]

print('(name: [std, mean])')
print(' --  --  -- Daily')
for i in dailyValues.items():
    print(i)
    
print('')
print(' --  --  -- Annual')    
for i in annualValues.items():
    print(i)

# %%
# calculating Returns and Volatility at equal weight for all 5 stocks
S = {}
M = []
for stock in dfs.keys():
    S[stock] = dfs[stock]["dailyChange"]
    M.append(dfs[stock]["dailyChange"].mean())
    
S = pd.DataFrame(data = S) 
C = np.cov(S.values.reshape((S.shape[1], S.shape[0]))) # covarience matrix
M = np.array(M).reshape((1, 5)) # mean

W = np.ones((5, 1))*(1/5) # waights
volatility = np.sqrt(np.matmul(np.matmul(W.T, C), W))
Return = np.matmul(M, W)

print('Returns = ', Return[0][0], "Volatility = ", volatility[0][0])

# %%
'''
5.3 Prepare a scatter plot for differing weights of the individual 
stocks in the portfolio , the axes being the returns and volatility. 
Colour the data points based on the Sharpe Ratio ( Returns/Volatility) 
of that particular portfolio.

+

5.4 Mark the 2 portfolios where -
Portfolio 1 - The Sharpe ratio is the highest
Portfolio 2 - The volatility is the lowest. 
'''
# generating weights
temp = np.linspace(0, 1, 21)
weights = []
for i in temp:
    for j in temp:
        if i+j<=1:
            for k in temp:
                if i+j+k<=1:
                    for l in temp:
                        if i+j+k+l<=1:
                            weights.append([i, j, k, l, 1-(i+j+k+l)])
                        else: break
                else: break        
        else: break  
                  
# calculating values of return and volatility
std = []
ret = []
for W in weights:
    W = np.array(W).reshape((5, 1))
    std.append(np.sqrt(np.matmul(np.matmul(W.T, C), W))[0][0])
    ret.append(np.matmul(M, W)[0][0])

ratio = [i/j for i,j in zip(ret, std)]

# plotting
fig, ax = plt.subplots(figsize=(10, 5))
cm = plt.cm.get_cmap('RdYlBu')
ax = plt.scatter(std, ret, c=ratio, cmap=cm)
plt.colorbar(ax)
plt.xlim(min(std)-0.001, max(std)+0.001)
plt.ylim(min(ret)-0.0005, max(ret)+0.0005)
plt.show()

#%%
def findMax(arr, max=float("inf"):
    maximum = max
    max_index = 0
    for index, element in enumerate(arr[:-1]):
        if element > maximum:
            maximum = element
            max_index = index-1
    return index, maximum
    
def findMax(arr, max=float("inf")):
    maximum = max
    max_index = 0
    for index, element in enumerate(arr[:-1]):
        if element > maximum:
            maximum = element
            max_index = index-1
    return index, maximum

port1 = 
