#%%
import sklearn

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
sns.set(style="darkgrid")


#%%
"""
2.1 Load the week2.csv file into a dataframe. 
What is the type of the Date column? Make sure it is of type datetime64. 
Convert the Date column to the index of the dataframe.
Plot the closing price of each of the days for the entire time frame to get an 
idea of what the general outlook of the stock is.
"""
df = pd.read_csv("week2.csv")
df.Date = pd.to_datetime(df['Date'])
df = df.set_index('Date')
print(df.index.dtype == "datetime64[ns]")

fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(data=df, x=df.index, y='Close Price', ax=ax)
plt.xticks(plt.xticks()[0], df.index.date, rotation=45)
plt.tight_layout()
plt.show()

#%%
"""
2.2 A stem plot is a discrete series plot, ideal for plotting daywise data. It can be plotted using the plt.stem() 
function. Display a stem plot of the daily change in of the stock price in percentage. This column was calculated 
in module 1 and should be already available in week2.csv. Observe whenever there's a large change.
"""
fig, ax = plt.subplots(figsize=(8, 4))
ax.stem(df.index, df.Day_Perc_Change, 'g', markerfmt='bo', label='Percente Change')
plt.xlabel('Date')
plt.ylabel('Day_Perc_Change')
plt.xticks(plt.xticks()[0], df.index.date, rotation=45)
plt.tight_layout()
plt.legend()
plt.show()

"""
The graph sugest that these variables are highly correlated as high volume make more Percentage change and vice versa.
"""
#%%
"""
2.3 Plot the daily volumes as well and compare the percentage stem plot to it. 
Document your analysis of the relationship between volume and daily percentage change. 
"""

scaledvolume =  df["No. of Trades"] - df["No. of Trades"].min()
scaledvolume = scaledvolume/scaledvolume.max() * df.Day_Perc_Change.max()

fig, ax = plt.subplots(figsize=(8, 4))

ax.stem(df.index, df.Day_Perc_Change , 'g', markerfmt='bo', label='Percente Change')
ax.plot(df.index, scaledvolume, 'k', label='Volume')

ax.set_xlabel('Date')
plt.legend(loc=2)

plt.tight_layout()
plt.xticks(plt.xticks()[0], df.index.date, rotation=45)
plt.show()

#%%
"""
2.4 We had created a Trend column in module.
1. We want to see how often each Trend type occurs.
This can be seen as a pie chart, with each sector representing the percentage of days each trend occurs. 
Plot a pie chart for all the 'Trend' to know about relative frequency of each trend. You can use the 
groupby function with the trend column to group all days with the same trend into a single group before 
plotting the pie chart. From the grouped data, create a BAR plot of average & median values of the 'Total 
Traded Quantity' by Trend type. 
"""
gridsize = (2, 6)
fig = plt.figure(figsize=(12, 8))
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=1)
ax2 = plt.subplot2grid(gridsize, (0, 3), colspan=3)
ax3 = plt.subplot2grid(gridsize, (1, 0), colspan=6)

df['ones'] = np.ones((df.shape[0]))
sums = df.ones.groupby(df.Trend).sum()
explod = [0.2, 0.2, 0.5, 0, 0, 0, 0 ,0]
ax1.pie(sums, labels=sums.index, autopct='%1.1f%%', explode=explod)
ax2.title.set_text('Trend')
df = df.drop(['ones'], axis=1)

bard1 = df[['Trend', 'Total Traded Quantity']].groupby(['Trend'], as_index=False).mean()
bar1 = sns.barplot("Trend", 'Total Traded Quantity', data=bard1, ci=None, ax=ax2)
for item in bar1.get_xticklabels():
    item.set_rotation(45)
ax2.set_ylabel('') 
ax2.title.set_text('Trend to mean of Total Traded Quantity')

bard2 = df[['Trend', 'Total Traded Quantity']].groupby(['Trend'], as_index=False).median()
bar2 = sns.barplot("Trend", 'Total Traded Quantity', data=bard2, ci=None, ax=ax3)
for item in bar2.get_xticklabels():
    item.set_rotation(45)
ax3.set_ylabel('') 
ax3.title.set_text('Trend to meadian of Total Traded Quantity')

plt.tight_layout()
plt.show()

#%%
"""
2.5 Plot the daily return (percentage) distribution as a histogram.
Histogram analysis is one of the most fundamental methods of exploratory data analysis. 
In this case, it'd return a frequency plot of various values of percentage changes .
"""
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df.Day_Perc_Change, bins=50)
ax.set_ylabel('Percent Change')
plt.show()

#%%
"""
2.6 We next want to analyse how the behaviour of different stocks are correlated. The correlation is performed on the 
percentage change of the stock price instead of the stock price.
Load any 5 stocks of your choice into 5 dataframes. Retain only rows for which ‘Series’ column has 
value ‘EQ’. Create a single dataframe which contains the ‘Closing Price’ of each stock. This dataframe should 
hence have five columns. Rename each column to the name of the stock that is contained in the column. Create 
a new dataframe which is a percentage change of the values in the previous dataframe. Drop Nan’s from this dataframe.
Using seaborn, analyse the correlation between the percentage changes in the five stocks. This is extremely useful for 
a fund manager to design a diversified portfolio. To know more, check out these resources on correlation and diversification. 
"""
stocks = ['RELIANCE.csv', 'TCS.csv', "HEROMOTOCO.csv", 'CIPLA.csv', 'AXISBANK.csv']
dfs={}

for i in stocks:
    stock = i.split('.')[0]
    temp_df = pd.read_csv(i)
    temp_df = temp_df[temp_df["Series"] == "EQ"]
    temp_df['Day_Perc_Change'] = temp_df['Close Price'].pct_change()*100
    temp_df = temp_df['Day_Perc_Change']
    temp_df = temp_df.drop(temp_df.index[0])
    dfs[stock] = temp_df

dfs = pd.DataFrame(dfs)
sns.pairplot(dfs)
plt.show()

#%%
"""
2.7 Volatility is the change in variance in the returns of a stock over a specific period of time.Do give the following 
documentation on volatility a read. You have already calculated the percentage changes in several stock prices. Calculate 
the 7 day rolling average of the percentage change of any of the stock prices, then compute the standard deviation 
(which is the square root of the variance) and plot the values.
Note: pandas provides a rolling() function for dataframes and a std() function also which you can use.
"""
for i in dfs.columns:
    dfs[f'{i}_rollingStd'] = dfs[f'{i}'].rolling(7).std()

dfs = dfs.dropna()

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(np.arange(len(dfs.AXISBANK_rollingStd)), dfs.AXISBANK_rollingStd, 'k')
plt.title('Axis Bank Volatility')
plt.show()

#%%
"""
2.8 Calculate the volatility for the Nifty index and compare the 2. This leads us to a useful indicator 
known as 'Beta' ( We'll be covering this in length in Module 3).
"""
ndf = pd.read_csv('Nifty50.csv')
ndf['Day_Perc_Change'] = ndf['Close'].pct_change()*100
ndf['rolling_std'] = ndf['Day_Perc_Change'].rolling(7).std()
ndf = ndf.dropna()

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(np.arange(len(dfs.AXISBANK_rollingStd)), dfs.AXISBANK_rollingStd, 'k', label='AXISBANK')
ax.plot(np.arange(len(ndf.rolling_std)), ndf.rolling_std, 'r', label='NIFTY')
plt.legend()
plt.title('Volatility')
plt.show()

#%%
"""
2.9 Trade Calls - Using Simple Moving Averages. Study about moving averages here. 
Plot the 21 day and 34 day Moving average with the average price and decide a Call ! 
Call should be buy whenever the smaller moving average (21) crosses over longer moving 
average (34) AND the call should be sell whenever smaller moving average crosses under 
longer moving average. One of the most widely used technical indicators.
"""
ndf.Date = pd.to_datetime(ndf['Date'])
fig, ax = plt.subplots(figsize=(15, 5))
ndf['roll21'] = ndf['Close'].rolling(21).mean()
ndf['roll34'] = ndf['Close'].rolling(34).mean()
ndf.dropna()

def whenCrosses(values):
    l=[]
    were = values[0]
    flag = True
    for i, ele in enumerate(values):
        if were==ele:
            l.append(0)
        else:
            l.append(1)
            were = ele
    return l

ndf['buy'] = ndf['roll34']<ndf['roll21']
ndf['sell'] = ndf['roll34']>ndf['roll21']

ndf['buy_change'] = np.array(whenCrosses(ndf.buy.values.reshape(1, len(ndf.buy)).flatten()))
ndf['sell_change'] = np.array(whenCrosses(ndf.sell.values.reshape(1, len(ndf.sell)).flatten()))

ndf['buy'] = ndf['buy_change'].where(ndf['buy']==True)
ndf['buy'] = ndf['roll21'].where(ndf['buy']==1)

ndf['sell'] = ndf['sell_change'].where(ndf['sell']==True)
ndf['sell'] = ndf['roll21'].where(ndf['sell']==1)

ax.plot(ndf.Date, ndf.Close, 'r')
ax.plot(ndf.Date, ndf.roll34, 'b', label='34_SMA')
ax.plot(ndf.Date, ndf.roll21, 'g', label='21_SMA')
ax.plot(ndf.Date, ndf.buy, "g^")
ax.plot(ndf.Date, ndf.sell, "kv")

ax.set_xlabel('Date')
plt.legend(loc=2)
plt.tight_layout()
plt.xticks(plt.xticks()[0], df.index.date, rotation=45)
plt.show()

#%%
"""
2.10 Trade Calls - Using Bollinger Bands 
Plot the bollinger bands for this stock - the duration of 14 days and 2 standard deviations away from the average 
The bollinger bands comprise the following data points- 

The 14 day rolling mean of the closing price (we call it the average) 
Upper band which is the rolling mean + 2 standard deviations away from the average. 
Lower band which is the rolling mean - 2 standard deviations away from the average. 
Average Daily stock price.

Bollinger bands are extremely reliable , with a 95% accuracy at 2 standard deviations , and especially useful in sideways moving market. 
Observe the bands yourself , and analyse the accuracy of all the trade signals provided by the bollinger bands. 
Save to a new csv file.
"""
ndf['bollinger'] = ndf['Close'].rolling(14).mean()
std = ndf['Close'].rolling(14).std()
ndf['upper'] = ndf['bollinger'] + 2 * std 
ndf['lower'] = ndf['bollinger'] - 2 * std

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(ndf.Date, ndf['Close'], 'k', lw=2)
ax.plot(ndf.Date, ndf['bollinger'], 'b', label='14 days Average')
# ax.plot(ndf.Date, ndf['upper'], 'g', label='Upper Limit')
# ax.plot(ndf.Date, ndf['lower'], 'r', label='Lower Limit')
ax.fill_between(ndf.Date, ndf['upper'], ndf['lower'], color='grey')

ax.set_xlabel('Date')
plt.legend(loc=2)
plt.tight_layout()
plt.xticks(plt.xticks()[0], df.index.date, rotation=45)
plt.show()

#%%