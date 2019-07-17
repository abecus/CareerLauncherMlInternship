#%%
import pandas as pd
import numpy as np

#%%
'''
Import the csv file of the stock you have been allotted using 'pd.read_csv()'
function into a dataframe. Shares of a company can be offered in more than one
category. The category of a stock is indicated in the ‘Series’ column. 
If the csv file has data on more than one category, the ‘Date’ column 
will have repeating values. To avoid repetitions in the date, remove all the 
rows where 'Series' column is NOT 'EQ'.
'''
df = pd.read_csv("HEROMOTOCO.csv")
df = df[df["Series"] == "EQ"]

#%%
"""
Change the date column from 'object' type to 'datetime64(ns)' for future convenience.
"""
df.Date = pd.to_datetime(df['Date'])

#%%
""" 
Add a column 'Day_Perc_Change' where the values are the daily change in percentages 
i.e. the percentage change between 2 consecutive day's closing prices.
"""
df['Day_Perc_Change'] = df['Close Price'].pct_change()*100
df = df.drop(df.index[0])
# VWAP = sum(price*volume)/sum(volume)

#%%
''' Add another column 'Trend' whose values are:
'Slight or No change' for 'Day_Perc_Change' in between -0.5 and 0.5
'Slight positive' for 'Day_Perc_Change' in between 0.5 and 1
'Slight negative' for 'Day_Perc_Change' in between -0.5 and -1
'Positive' for 'Day_Perc_Change' in between 1 and 3
'Negative' for 'Day_Perc_Change' in between -1 and -3
'Among top gainers' for 'Day_Perc_Change' in between 3 and 7
'Among top losers' for 'Day_Perc_Change' in between -3 and -7
'Bull run' for 'Day_Perc_Change' >7
'Bear drop' for 'Day_Perc_Change' <-7
'''
def cases(num):
    print(num)
    if num >= -0.5 and num <= 0.5 : return 'Slight or No change'
    elif num >= 0.5 and num <= 1 : return 'Slight positive'
    elif num >= -1 and num <= -0.5 : return 'Slight negative'
    elif num >= 1 and num <= 3 : return 'Positive'
    elif num >= -3  and num <= -1 : return 'Negative'
    elif num >= 3 and num <= 7 : return 'Among top gainers'
    elif num >= -7  and num <= -3 : return 'Among top losers'
    elif num > 7 : return 'Bull run'
    elif num < -7 : return 'Bear drop'

df["Trend"] = df.Day_Perc_Change.apply(cases)
#%%
"""
Find the average and median values of the column 'Total Traded Quantity' 
for each of the types of 'Trend'. {Hint : use 'groupby()' on the 'Trend' 
column and then calculate the average and median values of the column 
'Total Traded Quantity'}
"""
print(df[['Trend', 'Total Traded Quantity']].groupby(['Trend'], as_index=False).mean().sort_values(by='Total Traded Quantity', ascending=False))

#%%
"""
SAVE the dataframe with the additional columns computed as a csv file week2.csv.
"""
export_csv = df.to_csv(r'week2.csv', index=False)
#%%
