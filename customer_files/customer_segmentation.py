# coding: utf-8

# # RMF Customer Clustering

# In this project, I will cluster customers based on three dimensions - recency, monetary value, and frequency. Then I will assign them an overall score, and classify customers to low-value, medium-value, and high-value segments.

# In[3]:


# import the packages
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import seaborn as sns

# import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go
import plotly.express as px

# pyoff.init_notebook_mode()

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
#
# get_ipython().magic(u'matplotlib inline')
sns.set()
import warnings

warnings.filterwarnings('ignore')

SEED = 42

# In[4]:


# load the data

# In[5]:
def cust_seg_rmf():
    customer = pd.read_csv(r'./input_original_datasets/olist_customers_dataset.csv')
    order = pd.read_csv(r'./input_original_datasets/olist_orders_dataset.csv')
    item = pd.read_csv(r'./input_original_datasets/olist_order_items_dataset.csv')
    payment = pd.read_csv(r'./input_original_datasets/olist_order_payments_dataset.csv')

    # merge the data
    df = customer.merge(order, how='inner', on='customer_id')
    df = df.merge(item, how='inner', on='order_id')
    df = df.merge(payment, how='inner', on='order_id')

    #########################################################################


    df.shape

    # In[5]:


    df.head()

    # In[6]:


    df.info()

    # In[24]:


    # select features needed for this project
    df = df[['customer_unique_id', 'order_id', 'order_status',
             'order_purchase_timestamp', 'payment_value']]


    # In[25]:


    # check missing values
    def check_missing(df):
        missing = df.isnull().sum()
        missing_percentage = (df.isnull().sum() / len(df) * 100).round(2)
        missing_val = pd.concat([missing, missing_percentage], axis=1)
        missing_val.columns = ['Missing Values', '% Missing']
        total_columns = df.shape[1]
        missing_columns = (df.isnull().sum() > 0).sum()
        print('Out of {} columns, {} columns have missing values'.format(total_columns, missing_columns))
        return missing_val


    check_missing(df)

    # In[26]:


    # check order status
    df.order_status.value_counts()

    # In[27]:


    # only keep delivered order for customer clustering
    df = df.loc[df.order_status == 'delivered']

    # In[28]:


    # convert order_purchase_timestamp to datetime datatype
    df['order_datetime'] = pd.to_datetime(df.order_purchase_timestamp)
    df.order_datetime.describe()

    # ## Feature Engineering

    # In[87]:


    rmf = df[['customer_unique_id']]

    # ### Recency

    # In[88]:


    # get the latest purchase date
    cutoff_date = df.order_datetime.max()
    cutoff_date

    # In[89]:


    # calculate the latest purchase date for each customer
    recency = df.groupby('customer_unique_id').order_datetime.max().reset_index()
    recency.columns = ['customer_unique_id', 'most_recent_purchase']

    # In[90]:


    # calculate the days between each customer's most recent purchase and the cutoff date
    recency['recency'] = (cutoff_date - recency.most_recent_purchase).dt.days

    # In[91]:


    # append recency to rmf df
    rmf = rmf.merge(recency, how='left', on='customer_unique_id')

    # In[92]:


    # visualize recency
    plot_data = [go.Histogram(x=rmf['recency'])]
    plot_layout = go.Layout(title='Recency')

    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.iplot(fig)

    # ### Monetary Value (Revenue)

    # In[93]:


    # calculate revenue for each customer
    revenue = df.groupby('customer_unique_id').payment_value.sum().reset_index()
    revenue.columns = ['customer_unique_id', 'revenue']

    # In[94]:


    # append revenue to rmf df
    rmf = rmf.merge(revenue, how='left', on='customer_unique_id')

    # The revenue distribution is extremely right skewed.

    # In[95]:


    # visualize revenue
    plot_data = [go.Histogram(x=rmf['revenue'])]
    plot_layout = go.Layout(title='Revenue')

    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.iplot(fig)

    # ### Frequency

    # In[96]:


    # count each customers' order numbers
    frequency = df.groupby('customer_unique_id').order_id.count().reset_index()
    frequency.columns = ['customer_unique_id', 'frequency']

    # In[97]:


    # append frequency to rmf df
    rmf = rmf.merge(frequency, how='left', on='customer_unique_id')

    # Frequency distribution is also right skewed. Most customers only had 1 purchase at Olist.

    # In[98]:


    # visualize frequency
    plot_data = [go.Histogram(x=rmf['frequency'])]
    plot_layout = go.Layout(title='Frequency')

    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.iplot(fig)

    # ## Clustering

    # ### Recency

    # In[105]:


    # select the optimal number of clusters: 4
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=400, n_init=10, random_state=SEED)
        kmeans.fit(rmf[['recency']])
        wcss.append(kmeans.inertia_)

    # plot a line graph to observe 'The elbow'
    sns.set()
    sns.lineplot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Association')
    plt.ylabel('WCSS')
    plt.show()

    # In[104]:


    # cluster by recency
    kmeans = KMeans(n_clusters=4, random_state=SEED)
    kmeans.fit(rmf[['recency']])
    rmf['recency_cluster'] = kmeans.predict(rmf[['recency']])


    # In[110]:


    # in order to make cluster ordinal, we need to reorder the cluster number based on each cluster's average recency
    def order_cluster(col, cluster_col, ascending):
        df = rmf.groupby(cluster_col)[col].mean().reset_index()
        df = df.sort_values(by=col, ascending=ascending).reset_index(drop=True)
        df['new_cluster'] = df.index
        reorder_dict = dict(zip(df[cluster_col], df.new_cluster))
        return reorder_dict


    reordered_recency = order_cluster('recency', 'recency_cluster', False)
    rmf['recency_rank'] = rmf.recency_cluster.map(reordered_recency)

    # In[112]:


    # check the clustering
    rmf.groupby('recency_rank').recency.describe()

    # ### Revenue

    # In[82]:


    # select the optimal number of clusters: 4
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=400, n_init=10, random_state=SEED)
        kmeans.fit(rmf[['revenue']])
        wcss.append(kmeans.inertia_)

    # plot a line graph to observe 'The elbow'
    sns.lineplot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Association')
    plt.ylabel('WCSS')
    plt.show()

    # In[113]:


    # cluster by revenue
    kmeans = KMeans(n_clusters=4, random_state=SEED)
    kmeans.fit(rmf[['revenue']])
    rmf['revenue_cluster'] = kmeans.predict(rmf[['revenue']])

    # In[116]:


    reordered_revenue = order_cluster('revenue', 'revenue_cluster', True)
    rmf['revenue_rank'] = rmf.revenue_cluster.map(reordered_revenue)

    # In[117]:


    # check the clustering
    rmf.groupby('revenue_rank').revenue.describe()

    # ### Frequency

    # In[83]:


    # select the optimal number of clusters: 4
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=400, n_init=10, random_state=SEED)
        kmeans.fit(rmf[['frequency']])
        wcss.append(kmeans.inertia_)

    # plot a line graph to observe 'The elbow'
    sns.lineplot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Association')
    plt.ylabel('WCSS')
    plt.show()

    # In[118]:


    # cluster by frequency
    kmeans = KMeans(n_clusters=4, random_state=SEED)
    kmeans.fit(rmf[['frequency']])
    rmf['frequency_cluster'] = kmeans.predict(rmf[['frequency']])

    # In[119]:


    reordered_frequency = order_cluster('frequency', 'frequency_cluster', True)
    rmf['frequency_rank'] = rmf.frequency_cluster.map(reordered_frequency)

    # In[121]:


    # check the clustering
    rmf.groupby('frequency_rank').frequency.describe()

    # ### Customer Value

    # In[122]:


    # create an overall score
    rmf['customer_value'] = rmf.recency_rank + rmf.revenue_rank + rmf.frequency_rank

    # In[126]:


    rmf.groupby('customer_value')['recency', 'revenue', 'frequency'].agg(['count', 'mean'])

    # In[127]:


    # create the value rank: 0-2 -> low, 3-4 -> medium, 5-7 -> high
    rmf['value_rank'] = rmf.customer_value.apply(lambda x: 0 if x in range(3) else (1 if x in range(3, 5) else 2))

    # In[136]:


    # visualize the customer segments: plotly
    fig = px.scatter_3d(rmf, x='recency', y='revenue', z='frequency', color='value_rank', opacity=0.5, size_max=5)
    fig.show()

    # In[138]:


    # save to csv
    rmf.to_csv('rmf.csv', index=False)


