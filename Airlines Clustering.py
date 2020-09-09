import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist 


# K-Means Clustering

Airln = pd.read_csv("D:\\ExcelR Data\\Assignments\\clustering\\EastWestAirlines.csv")

Airln.columns
#Index['ID', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles',
#       'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12',
#       'Days_since_enroll', 'Award'],
#       dtype='object')
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    #x = (i-i.mean())/i.std()
    return (x)


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Airln.iloc[:,1:])

df_norm.describe()
#Now I'm going for the visualizations
plt.plot(Airln.ID,Airln.Balance,"ro");plt.xlabel("ID");plt.ylabel("Balance")
plt.plot(Airln.Qual_miles,Airln.cc1_miles,"ro");plt.xlabel("Qual_miles");plt.ylabel("cc1_miles")
plt.plot(Airln.cc2_miles,Airln.cc3_miles,"bo");plt.xlabel("cc2_miles");plt.ylabel("cc3_miles")
plt.plot(Airln.Bonus_trans,Airln.Bonus_miles,"go");plt.xlabel("Bonus_trans");plt.ylabel("Bonus_miles")
plt.plot(Airln.Flight_miles_12mo,Airln.Flight_trans_12,"go");plt.xlabel("Flight_miles_12mo");plt.ylabel("Flight_trans_12")
plt.plot(Airln.Days_since_enroll,Airln.Award,"go");plt.xlabel("Days_since_enroll");plt.ylabel("Award")

#first five obsevation of my normalized dataframe
df_norm.head()
#Now I'm going to perform KMEANS clustering

k=list(range(2,15))#here i'm defining my clusters range randomly from 2 to 15

#Next I need to identify the total sum of square using TWSS[] 
TWSS=[]
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    WSS=[]#With in sum of squares
    for j in range(i):
         WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
# Scree plot 
plt.plot(k,TWSS,'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
#So I'm going to choose my cluster number as 10

model=KMeans(n_clusters=10) 
model.fit(df_norm)

model.labels_     
#array([4, 4, 4, ..., 7, 5, 5])
md=pd.Series(model.labels_)
md.head()#first five label of cluster

md.tail()#last five label of cluster
#adding my labels of clusters to my original dataset
Airln["clusters"]=md
Airln=Airln.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
Airln.head()

Airln.tail()

#getting the aggregate mean of each cluster
Airln.groupby(Airln.clusters).mean()
