#%%
import os
import fnmatch

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#%%
"""
6.1 Create a table/data frame with the closing prices of 30 different 
stocks, with 10 from each of the caps

6.2 Calculate average annual percentage return and volatility of all 
30 stocks over a theoretical one year period.
"""
caps = ["Large_Cap/Large_Cap",
        "Mid_Cap",
        "Small_Cap"
        ]

dfs = {}
dailyValues = {}
annualValues = {}
for cap in caps:
    i=1
    
    for file_name in os.listdir('Data/'+f"{cap}"):        
        if fnmatch.fnmatch(file_name, '*.csv') and i<11:
            name = file_name.split('.')[0]  # getting the name without extension
            dfs[name] = pd.read_csv(f'Data/{cap}/{file_name}')  # making df out of csv file

            # preprocessing
            dfs[name] = dfs[name][dfs[name]["Series"] == "EQ"]   
            dfs[name].Date = pd.to_datetime(dfs[name]['Date'])
            dfs[name] = dfs[name].set_index('Date')
            
            # calculating values i.e. mean and std
            dfs[name]["dailyChange"] = dfs[name]["Close Price"].pct_change()
            dfs[name].dropna(inplace = True)
            
            # (std, mean)
            dailyValues[name] = [dfs[name].dailyChange.std(), dfs[name].dailyChange.mean()]
            annualValues[name] = [dailyValues[name][0]*(252**0.5), dailyValues[name][1]*252]
            
            i +=1
print('(name: [std, mean])')
print(' --  --  -- Daily')
for i in dailyValues.items():
    print(i)
    
print('')
print(' --  --  -- Annual')    
for i in annualValues.items():
    print(i)
    
#%%
"""
6.3 Cluster the 30 stocks according to their mean annual Volatilities 
and Returns using K-means clustering. Identify the optimum number of 
clusters using the Elbow curve method
"""
data = pd.DataFrame.from_dict(annualValues, orient='index', columns=["std", "mean"])

#%%
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Transforming data
mms = MinMaxScaler()
mms.fit(data)
transformedData = mms.transform(data)

# calculating Inertia avrage dist. b/w the cluster center and its points
Sum_of_squared_distances = []
K = range(1, 31)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(transformedData)
    Sum_of_squared_distances.append(km.inertia_)

# Plotting the Elbow Method Graph
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(K, Sum_of_squared_distances, 'bx-')

plt.xticks(K)
plt.xlabel('k')

# plt.yticks(Sum_of_squared_distances)
plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')
plt.show()

#%%
"""
6.4 Prepare a separate Data frame to show which  stocks belong to the 
same cluster 
"""
# Using k=6
k=6
km = KMeans(n_clusters=k)
km = km.fit(transformedData)

#%%
data['labels'] = km.labels_
centers = km.cluster_centers_
y_kmeans = km.predict(transformedData)

#%%
plt.style.use('bmh')

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(transformedData[:, 0], 
            transformedData[:, 1], 
            c=y_kmeans, 
            s=50, 
            cmap='viridis'
            )

ax.scatter(centers[:, 0], 
           centers[:, 1], 
           c='black', 
           s=8000, 
           alpha=0.05
           )

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 6,
        }

for x, y, name in zip(transformedData[:, 0], transformedData[:, 1], data.index):
    ax.text(x, 
            y, 
            name, 
            fontdict=font,
            # verticalalignment='center', 
            # horizontalalignment='center',
            rotation=0
            )


font = {'family': 'serif',
        'color':  'gray',
        'weight': 'normal',
        'size': 28,
        }

for i, cord in enumerate(centers):
    ax.text(cord[0], 
            cord[1], 
            f"C{i}",
            horizontalalignment='center',
            verticalalignment='center', 
            fontdict=font,             
            rotation=0, 
            alpha=0.3
            )

plt.title('K-Means Clustering')
plt.show()
#%%
