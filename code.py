# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
df = pd.read_csv(path)
# check the null values
df.isnull().sum()
# drop null values
df.dropna(subset=['Description','CustomerID'],inplace=True)
# check the null values
df.isnull().sum()
# only take one country
df = df[df.Country== 'United Kingdom']

# create new colums returns
df['Return']=df.InvoiceNo.str.contains('C')
# store the result in purchase 
df['Purchase'] = np.where(df["Return"]==True,0,1)


# --------------
# create new dataframe customer
customers = pd.DataFrame({'CustomerID': df['CustomerID'].unique()},dtype=int)

# calculate the recency
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Recency'] = pd.to_datetime("2011-12-10") - (df['InvoiceDate'])

# remove the time factor
df.Recency = df.Recency.dt.days

# purchase equal to one 
temp = df[df['Purchase']==1]

# customers latest purchase day
recency=temp.groupby(by='CustomerID',as_index=False).min()
customers=customers.merge(recency[['CustomerID','Recency']],on='CustomerID')




# --------------
# Removing invoice number duplicates

temp_1=df[['CustomerID','InvoiceNo','Purchase']]
temp_1.drop_duplicates(subset=['InvoiceNo'],inplace=True)

# calculte the frequency of the purchases
annual_invoice=temp_1.groupby(by='CustomerID',as_index=False).sum()
annual_invoice.rename(columns={'Purchase':'Frequency'},inplace=True)

# merge in the Customer
customers=customers.merge(annual_invoice,on='CustomerID')
print(customers.shape)


# --------------
# code starts here
df['Amount'] = df['Quantity'] * df['UnitPrice']
annual_sales = df.groupby(by='CustomerID',as_index=False).sum()
annual_sales.rename(columns={'Amount':'monetary'},inplace=True)
customers=customers.merge(annual_sales[['CustomerID','monetary']],on='CustomerID')
# code ends here


# --------------
# code ends here
# negative monetory removed because they returned the object 
customers['monetary']=np.where(customers['monetary']<0,0,customers['monetary'])    

# log transform
customers['Recency_log']=np.log(customers['Recency']+0.1) # there values equals to zero to avoid log zero increase by +0.1
customers['Frequency_log']=np.log(customers['Frequency'])
customers['Monetary_log']=np.log(customers['monetary']+0.1)

# code ends here


# --------------
#Elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
# Empty list for storing WCSS across all values of k
dist = []
for i in range(1,10):
    km = KMeans(n_clusters=i,init='k-means++',max_iter = 300, n_init = 10,random_state = 0)
    km.fit(customers.iloc[:,1:7])
    dist.append(km.inertia_)
    
#intialize figure
plt.figure(figsize=(10,10))
#Line Plot with Clusters on X_axis and WCSS on y-axis
plt.plot(range(1,10),dist)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()


# --------------

# Code starts here

# initialize KMeans object
cluster = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)

# create 'cluster' column
customers['cluster'] = cluster.fit_predict(customers.iloc[:,1:7])

# plot the cluster

customers.plot.scatter(x= 'Frequency_log', y= 'Monetary_log', c='cluster', colormap='viridis')
plt.show()

# Code ends here


