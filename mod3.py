#%%
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
sns.set(style="darkgrid")

#%%
'''
3.1 Import the file 'gold.csv' (you will find this in the intro section 
to download or in '/Data/gold.csv' if you are using the jupyter notebook), 
which contains the data of the last 2 years price action of Indian (MCX) 
gold standard. Explore the dataframe. You'd see 2 unique columns - 'Pred' 
and 'new'. One of the 2 columns is a linear combination of the OHLC prices 
with varying coefficients while the other is a polynomial function of the 
same inputs. Also, one of the 2 columns is partially filled.

Using linear regression, find the coefficients of the inputs and using
the same trained model, complete the entire column.

Also, try to fit the other column as well using a new linear regression model. 
Check if the predictions are accurate. Mention which column is a linear function 
and which is polynomial.
(Hint: Plotting a histogram & distplot helps in recognizing the discrepencies in prediction, if any.)
'''

df = pd.read_csv("GOLD.csv")
df.Date = pd.to_datetime(df['Date'])
df = df.set_index('Date')
print(df.index.dtype == "datetime64[ns]")
# print(df.columns)
print(df.info())

# for furthure exploration
df_droped = df.dropna()
df_to_fill = df.iloc[411 :, :]

# checking for empty column
empties = np.where(pd.isnull(df))
print('(starting_row, col) = ', empties[0][0], empties[1][0])


#%%
from sklearn.linear_model import LinearRegression
X = df_droped[['Open', 'High', 'Low']].values
pred = df_droped.Pred.values
new =  df_droped.new.values

#_______________________________linear Regression______________________
reg = LinearRegression().fit(X, pred)
print('pred\'s score =', reg.score(X, pred))
print('pred\'s coeff. = ', reg.coef_, 'bias  = ', reg.intercept_)

# pridicting the emty column and filling the rows of Pred
pred_to_fill = reg.predict(df_to_fill[['Open', 'High', 'Low']])
new_pridicted_pred = np.hstack((df_droped['Pred'].values, pred_to_fill))
df.Pred = new_pridicted_pred
print('')

reg2 = LinearRegression().fit(X, new)
print('new\'s score =', reg2.score(X, new))
print('new\'s coeff. = ', reg2.coef_, 'bias = ', reg2.intercept_)


ax = sns.kdeplot(pred_to_fill, label='Pridicted Entries', shade=True, color="r")
ax = sns.kdeplot(df_droped['Pred'], label='Pred Entries', shade=True, color="b")
#%%
#_______________________________Polynomial Regression__________________________
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())])

model = model.fit(X, pred)
print(model.score(X, pred))
print(model.named_steps['linear'].coef_)
print('')

model = model.fit(X, new)
print('new\'s score =', model.score(X, new))
print('new\'s coeff. = ', model.named_steps['linear'].coef_)
print('')

print(f"""Since the new column is giving the Score for Simple Linear Regression\
higher than the pred's column Score hence i think  new is linear function of\
OHL columns. Also the pred is polynomial function of OHL columns because it gives\
more +ve change in Score than the linear one while comparing to the new column""")

#%%
"""
3.2 Import the stock of your choosing AND the Nifty index. 
Using linear regression (OLS), calculate -

The daily Beta value for the past 3 months. (Daily= Daily returns)

The monthly Beta value. (Monthly= Monthly returns)

Refrain from using the (covariance(x,y)/variance(x)) formula. 
Attempt the question using regression.(Regression Reference) 
Were the Beta values more or less than 1 ? What if it was negative ? 
Discuss. Include a brief writeup in the bottom of your jupyter notebook
with your inferences from the Beta values and regression results
"""
# calling nifty as index
ndf = pd.read_csv('Nifty50.csv')
ndf.Date = pd.to_datetime(ndf['Date'])
ndf = ndf.set_index('Date')

# calling heromoto as stock
hdf = pd.read_csv('HEROMOTOCO.csv')
dhf = hdf[hdf["Series"] == "EQ"]
hdf.Date = pd.to_datetime(hdf['Date'])
hdf = hdf.set_index('Date')

# ______________________ Daily Beta ______________________________

index = ndf["Open"].where(hdf.index == ndf.index)
inedexReturn = (index.pct_change()*100).dropna().values.reshape(-1, 1)

stock = hdf["Open Price"].where(hdf.index == ndf.index)
stockReturn = (stock.pct_change()*100).dropna().values.reshape(-1, 1)

reg_for_beta_d = LinearRegression().fit(inedexReturn[len(inedexReturn)-60:].reshape(-1, 1), stockReturn[len(stockReturn)-60:].reshape(-1, 1))
print('3 Month\'s Daily Beta = ', reg_for_beta_d.coef_.tolist())

#%%
# Using covarience
daily_covMat = np.cov(inedexReturn[len(inedexReturn)-60:].reshape(1, -1), stockReturn[len(stockReturn)-60:].reshape(1, -1))
print('3 Month\'s Daily Beta = ', daily_covMat[0][1] / daily_covMat[0][0])


#%%
# _______________________ Monthly Beta ___________________________

index_m_last = index.loc[index.groupby(index.index.to_period('M')).apply(lambda x: x.index.max())]
inedexReturn_m_lats = (index_m_last.pct_change()*100).dropna().values.reshape(-1, 1)

stock_m_last = stock.loc[stock.groupby(stock.index.to_period('M')).apply(lambda x: x.index.max())]
stockReturn_m_last = (stock_m_last.pct_change()*100).dropna().values.reshape(-1, 1)

reg_for_beta_m = LinearRegression().fit(inedexReturn_m_lats.reshape(-1, 1), stockReturn_m_last.reshape(-1, 1))
print('Monthly Beta = ', reg_for_beta_m.coef_.tolist())

# Using covarianece
monthly_covMat = np.cov(inedexReturn_m_lats.reshape(1, -1), stockReturn_m_last.reshape(1, -1))
print('Monthly Beta = ', monthly_covMat[0][1] / monthly_covMat[0][0])

#%%
print("_"*20, "Write Up", "_"*20, end="\n")
print("""
\tIn both type of calculations (Monthly and daily) the Beta values are less than 1\
if the beta value happens to be negative then it is indication of that the stock as\
a whole loosing or in loss while index or market is in profit or gaining.
\n\tThe beta vaule of last Heromoto for daily last 3 months is ~0.82 which sugget that stock in less\
voletile than the market. while the monthly beta value (~0.52) suggest that the stock is even less voletile\
than the market or index. The daily beta value matches to the original beta value for Heromoto corp.\
with less than 0.02 error.
""")



#%%
