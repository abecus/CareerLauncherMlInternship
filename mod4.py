#%%
import pandas as pd
import numpy as np
# import seaborn as sns
from matplotlib import pyplot as plt

#%%
"""
4.1 Import the csv file of the stock which contained the Bollinger columns as well.
Create a new column 'Call' , whose entries are - 

'Buy' if the stock price is below the lower Bollinger band 

'Hold Buy/ Liquidate Short' if the stock price is between the lower and middle Bollinger band 

'Hold Short/ Liquidate Buy' if the stock price is between the middle and upper Bollinger band 

'Short' if the stock price is above the upper Bollinger band

Now train a classification model with the 3 bollinger columns and the stock price as inputs and 'Calls' 
as output. Check the accuracy on a test set. (There are many classifier models to choose from, try each one 
out and compare the accuracy for each)

Import another stock data and create the bollinger columns. Using the already defined model, predict the 
daily calls for this new stock.
"""

ndf = pd.read_csv('Nifty50.csv')
ndf.Date = pd.to_datetime(ndf['Date'])

ndf['bollinger'] = ndf['Close'].rolling(14).mean()
std = ndf['Close'].rolling(14).std()
ndf['upper'] = ndf['bollinger'] + 2 * std 
ndf['lower'] = ndf['bollinger'] - 2 * std
ndf = ndf.dropna()

conditions = [
    (ndf.Open < ndf['lower']),
    (ndf.Open > ndf['lower']) & (ndf.Open < ndf["bollinger"]),
    (ndf.Open < ndf['upper']) & (ndf.Open > ndf["bollinger"]),
    (ndf.Open > ndf['upper'])
    ]

choices = ['Buy', 'Hold Buy', 'Hold Short', "Short"]
ndf["Call"] = np.select(conditions, choices) 
# print(ndf.head(10))

#%%
from sklearn.model_selection import train_test_split

X = ndf[['lower','bollinger', 'upper']].values
Y = ndf["Call"].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# __________________________ KNN ___________________________________
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(4, n_jobs=-1)
knn.fit(X_train, Y_train)

print("KNN\'s Score = ", knn.score(X_test, Y_test)) 

#%%
# __________________________ SVM ___________________________________
from sklearn import svm

clf = svm.SVC(gamma='auto', decision_function_shape='ovo')
clf.fit(X_train, Y_train)
print("SVM\'s Score = ", clf.score(X_test, Y_test)) 

#%%
# ________________________ Decision Tree ____________________________
from sklearn import tree

tree = tree.DecisionTreeClassifier().fit(X_train, Y_train)
print("Tree\'s Score", tree.score(X_test, Y_test))

#%%
# _________________________ On Other Data ____________________________
hdf = pd.read_csv('HEROMOTOCO.csv')
hdf = hdf[hdf["Series"] == "EQ"]
hdf.Date = pd.to_datetime(hdf['Date'])

hdf['bollinger'] = hdf['Close Price'].rolling(14).mean()
std = hdf['Close Price'].rolling(14).std()
hdf['upper'] = hdf['bollinger'] + 2 * std 
hdf['lower'] = hdf['bollinger'] - 2 * std
hdf = hdf.dropna()

# Using SVM Model
X = hdf[['lower','bollinger', 'upper']].values
hdf["Call Pridicted"] = clf.predict(X)
print(hdf[["Open Price", 'bollinger', 'upper', 'lower', 'Call Pridicted']].head(5))

#%%
"""
4.2 Now, we'll again utilize classification to make a trade call, and measure the 
efficiency of our trading algorithm over the past two years. For this assignment , 
we will use RandomForest classifier.
Import the stock data file of your choice

Define 4 new columns , whose values are: 
% change between Open and Close price for the day 

% change between Low and High price for the day 

5 day rolling mean of the day to day % change in Close Price 

5 day rolling std of the day to day % change in Close Price

Create a new column 'Action' whose values are: 

1 if next day's price(Close) is greater than present day's. 

(-1) if next day's price(Close) is less than present day's. 

i.e. Action [ i ] = 1 if Close[ i+1 ] > Close[ i ] 
i.e. Action [ i ] = (-1) if Close[ i+1 ] < Close[ i ]

Construct a classification model with the 4 new inputs and 'Action' as target
Check the accuracy of this model , also , plot the net cumulative returns (in %) 
if we were to follow this algorithmic model
"""

ndf = pd.read_csv('Nifty50.csv')
ndf.Date = pd.to_datetime(ndf['Date'])
ndf = ndf.set_index('Date')

ndf['pctch_open_close'] = (ndf.Close - ndf.Open)/ndf.Open *100
ndf['pctch_low_high'] = (ndf.High - ndf.Low)/ndf.Low *100
ndf['rolling_mean5'] = ndf['Close'].rolling(5).mean()
ndf["std5"] = ndf['Close'].rolling(5).std()

l=[]
for i in range(ndf.Close[:-1].shape[0]):
    if ndf.Close[i] < ndf.Close[i+1]:
        l.append(1)
    elif ndf.Close[i] > ndf.Close[i+1]:
        l.append(-1)
    else: 
        l.append("NaN")
l.append('NaN')

ndf["Action"] = np.array([l]).reshape((len(l),1))
ndf = ndf.dropna()

X = ndf[['pctch_open_close', 'pctch_low_high', 'rolling_mean5', "std5"]].values
Y = ndf.Action.values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#%%
# _______________________________ Random Forest _________________________________
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

random_forest = RandomForestClassifier(n_estimators=50, random_state=42)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)

print("Accuracy:",metrics.accuracy_score(Y_pred, Y_test))

#%%
""" 
plot the net cumulative returns (in %)
if we were to follow this algorithmic model 
"""
# pridicting the entire column
pridicted_Y = random_forest.predict(X)[:-1].astype('int')
pridicted_Y = np.hstack((1, pridicted_Y))

raw = np.array([i*j for i, j in zip(pridicted_Y, ndf.Close)])

# starting with no loss i.e. 1 and if their is jump from loss to profit
# we buy the stock and make the profit when it gets rise(thats why their 
# is abs for privious stock price) else their is loss(buying at higher 
# and getting price dropped). 

temp = [1, ]

for i in range(1, len(pridicted_Y)):
    if pridicted_Y[i] < 1:
        continue    
    else:
        profit = (ndf.Close[i] - abs(ndf.Close[i-1])) / abs(ndf.Close[i-1])
        temp.append(profit)


returns_perc = []
for i in range(1, len(temp)):
    returns_perc.append(np.sum(temp[:i]))

fig, ax  = plt.subplots(figsize=(15, 6))
ax.plot(range(len(returns_perc)), returns_perc)
plt.show()

#%%
