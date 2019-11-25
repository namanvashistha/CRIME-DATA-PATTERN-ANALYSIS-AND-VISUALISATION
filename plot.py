from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
#import plotly.graph_objects as go

#%matplotlib inline

df = pd.read_csv("crime.csv")
#df.head()

#plt.scatter(df.Latitude,df['Longitude'])
#plt.xlabel('Latitude')
#plt.ylabel('Longitude')

#Elbow Plot

sse = []
K = []
k_rng = range(1,50)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Latitude','Longitude']])
    sse.append(km.inertia_)
    K.append(k)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
axes= plt.axes()

#axes.set_xlim([0,25])

axes.set_xticks(K)
plt.grid()
plt.plot(k_rng,sse)
#plt.scatter(k_rng,sse,color='green')
