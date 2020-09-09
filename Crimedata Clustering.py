import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.spatial.distance import cdist 

crimedata= pd.read_csv("D:\\ExcelR Data\\Assignments\\clustering\\crimedata.csv")
crimedata.columns
#Index(['Unnamed: 0', 'Murder', 'Assault', 'UrbanPop', 'Rape'], dtype='object')
#first five observations
crimedata.head()

#last five observation from my dataset
crimedata.tail()

# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    #x = (i-i.mean())/i.std()
    return (x)



# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crimedata.iloc[:,1:])

#Next i'm going for the EDA parts
# visualizations
plt.hist(df_norm["Murder"])
plt.hist(df_norm["Assault"])
plt.hist(df_norm["UrbanPop"])
plt.hist(df_norm["Rape"])
plt.plot(df_norm.Murder,crimedata.Assault,"ro");plt.xlabel("Muder");plt.ylabel("Assault")
plt.plot(df_norm.UrbanPop,df_norm.Rape,"ro");plt.xlabel("UrbanPop");plt.ylabel("Rape")

#calculating the mean,median,mode,sd,variance,max value and min value using describe function
df_norm.describe()
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 
#Now i'm calculating the Euclidean distance using single linkage function
L=linkage(df_norm,method="single",metric="Euclidean")
plt.figure(figsize=(15,5));plt.title("Hierarchical clustering dendogram");plt.xlabel("index");plt.ylabel("distance")
sch.dendrogram(
    L,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

#here we have 50  obervations,so root of (50/2)=3..So consider the cluster as 3 
# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from sklearn.cluster import AgglomerativeClustering
L_single= AgglomerativeClustering(n_clusters=3,linkage="single",affinity="euclidean").fit(df_norm)
#labels
L_single.labels_
#from the labels I can see most of the datapoints are belongs to cluster'0'

cluster_labels=pd.Series(L_single.labels_)
cluster_labels.head()

cluster_labels.tail()

#next we have to add  cluster_variable to the original dataset
crimedata['cluster']=cluster_labels
crimedata=crimedata.iloc[:,[5,0,1,2,3,4]]
crimedata.head()

crimedata.tail()

#getting the aggregate mean of each cluster
crimedata.groupby(crimedata.cluster).mean()
