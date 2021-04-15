
# coding: utf-8

# ## What is Customer Lifetime Value(CLV)?

# In marketing, **customer lifetime value(CLV or CLTV),lifetime customer value(LCV), or life-time value(LTV)** is a prognostication of the net profit contributed to the whole future relationship with a customer. The prediction model can have verying levels of sophistication and accuracy, ranging from a crude heuristic to the use of complex predictive analytics techniques.
#
# Customer lifetime value can also be defined as the monetary value of a customer relationship, based on the present value of the projected future cash flows from the customer relationship. Customer lifetime value is an important concept in that it encourages firms to shift their focus from quarterly profits to the long-term health of their customer relationships. Customer lifetime value is an important metric because it represents an upper limit on spending to acquire new customers. For this reason it is an important element in calculating payback of advertising spent in marketing mix modeling.

# ## You can use CLV models to answer these types of questions about customers :
#
# * **Number of purchases**: How many purchases will the customer make in a given future time range?
# * **Lifetime**: How much time will pass before the customer becomes permanently inactive?
# * **Monetary**: How much monetary value will the customer generate in a given future time range?

# ### CLV concepts: RFM (Customer Segmentation)
# Three important inputs into CLV models are recency, frequency, and monetary value:
#
# * **Recency**: When was the customer's last order?
# * **Frequency**: How often do they buy?
# * **Monetary**: What amount do they spend?
# We will use RFM framework to build our customer segmentation model.

# ### Importing Libraries:

# In[29]:

def cust_lifetime():
    import pandas as pd
    import numpy as np
    from datetime import timedelta
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')

    from sklearn.cluster import KMeans


    # Loading data
    customer = pd.read_csv(r'input_original_datasets/olist_customers_dataset.csv')
    order = pd.read_csv(r'input_original_datasets/olist_orders_dataset.csv')
    item = pd.read_csv(r'input_original_datasets/olist_order_items_dataset.csv')
    product = pd.read_csv(r'input_original_datasets/olist_products_dataset.csv')
    product_category = pd.read_csv(r'input_original_datasets/product_category_name_translation.csv')
    payment = pd.read_csv(r'input_original_datasets/olist_order_payments_dataset.csv')
    review = pd.read_csv(r'input_original_datasets/olist_order_reviews_dataset.csv')

    # Merging data
    merged_df = customer.merge(order, how='inner', on='customer_id')
    merged_df = merged_df.merge(item, how='inner', on='order_id')
    merged_df = merged_df.merge(payment, how='inner', on='order_id')
    merged_df = merged_df.merge(product, how='inner', on='product_id')
    merged_df = merged_df.merge(review, how='inner', on='order_id')
    merged_df = merged_df.merge(product_category, how='inner', on='product_category_name')


    # In[4]:


    merged_df


    # In[5]:


    merged_df.columns


    # In[6]:


    #checking the data type
    merged_df.info()


    # In[7]:


    #checking  for missing values
    merged_df.isnull().sum()


    # In[8]:


    merged_df.columns


    # In[51]:


    df = merged_df[['customer_unique_id', 'order_id', 'order_status','order_item_id',
             'order_purchase_timestamp','price', 'payment_value','customer_state']]


    # In[52]:


    df.isnull().sum()


    # In[53]:


    # convert order_purchase_timestamp to datetime datatype
    df['order_datetime'] = pd.to_datetime(df.order_purchase_timestamp)


    # In[54]:


    df.info()


    # In[55]:


    df=df.drop_duplicates()


    # ### RFM Modelling:
    # Todo the RFM analysis, we need to create 3 freatures from the data:
    # * Recency: Latest date- last purchase date
    # * Frequency: Total no. of transactions made by a single customer (count of order id)
    # * Monetary: Total value of transacted sales by each customer.

    # ### Recency

    # In[105]:


    # Create a dataframe to store customers uuid and recency scores
    recency = df[['customer_unique_id', 'order_datetime']].copy()


    # In[106]:


    # Since a customer may have more than one order, we will obtain his/her last purchase timestamp
    recency = recency.groupby('customer_unique_id')['order_datetime'].max().reset_index()
    recency.columns = ['customer_unique_id', 'last_purchase_timestamp']

    # Calculate the number of days since customers' last purchase
    recency['inactive_days'] = (recency['last_purchase_timestamp'].max() - recency['last_purchase_timestamp']).dt.days
    recency.drop(columns='last_purchase_timestamp', inplace=True)


    # In[107]:


    sns.distplot(recency['inactive_days'], bins=50)
    plt.xlabel('Inactive days')
    plt.ylabel('Density')
    plt.title("Distribution of customers' inactive days")


    # We will use K-means clustering to assign each customer with a recency score. But first, we will use the elbow method to identify the optimal number of clusters here.

    # In[108]:


    inertia = {}

    for k in range(1,10):
        km = KMeans(n_clusters=k)
        km = km.fit(recency[['inactive_days']])
        inertia[k] = km.inertia_


    # In[109]:


    plt.figure(figsize=(8,5))
    plt.plot(list(inertia.keys()), list(inertia.values()))
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')


    # In[111]:


    km = KMeans(n_clusters=4, random_state=42)
    km.fit(recency[['inactive_days']])
    recency['cluster'] = km.labels_
    recency.head()


    # In[112]:


    recency.groupby('cluster')['inactive_days'].describe().sort_values(by='mean')


    # from the above table we can say that the cluster labels do not intuitively represent the inactive days. Let's assign cluster 0 to the highly inactive customers and cluster 4 to the most active customers.

    # In[113]:


    # Renaming the clusters according to mean number of inactive_days
    recency_cluster = recency.groupby('cluster')['inactive_days'].mean().reset_index()
    recency_cluster = recency_cluster.sort_values(by='inactive_days', ascending=False).reset_index(drop=True)
    recency_cluster['index'] = np.arange(0,4)
    recency_cluster.set_index('cluster', inplace=True)
    cluster_dict = recency_cluster['index'].to_dict()
    recency['cluster'].replace(cluster_dict, inplace=True)


    # In[114]:


    # Check that the clusters have been renamed correctly
    recency.head()


    # In[115]:


    recency.groupby('cluster')['inactive_days'].describe().sort_values(by='mean')


    # the cluster labels now make more sense. **Cluster 3** are our most valued customers as they have completed a more recent transaction compared to customers from the other clusters.

    # ### Frequency

    # **We will count the number of unique orders made by each customer to obtain their purchase frequencies.**

    # In[116]:


    # Create a dataframe to store customers uuid and frequency scores
    frequency = df[['customer_unique_id', 'order_id']].copy()


    # In[117]:


    # Count the number of orders for each customer
    frequency = frequency.groupby('customer_unique_id')['order_id'].count().reset_index()
    frequency.columns = ['customer_unique_id', 'number_of_orders']


    # In[118]:


    frequency['number_of_orders'].plot.hist(bins=100)
    plt.xlim(0,10)
    plt.xlabel('Number of orders')
    plt.ylabel('Denisty')
    plt.title('Distribution of the number of orders per customer')


    # In[119]:


    inertia = {}

    for k in range(1,10):
        km = KMeans(n_clusters=k)
        km = km.fit(frequency[['number_of_orders']])
        inertia[k] = km.inertia_


    # In[120]:


    plt.figure(figsize=(8,5))
    plt.plot(list(inertia.keys()), list(inertia.values()))
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia');


    # In[121]:


    km = KMeans(n_clusters=4, random_state=42)
    km.fit(frequency[['number_of_orders']])
    frequency['cluster'] = km.labels_


    # In[122]:


    frequency.groupby('cluster')['number_of_orders'].describe().sort_values(by='mean')


    # In[123]:


    # Renaming the clusters according the mean number_of_orders
    frequency_cluster = frequency.groupby('cluster')['number_of_orders'].mean().reset_index()
    frequency_cluster = frequency_cluster.sort_values(by='number_of_orders').reset_index(drop=True)
    frequency_cluster['index'] = np.arange(0,4)
    frequency_cluster.set_index('cluster', inplace=True)
    cluster_dict = frequency_cluster['index'].to_dict()
    frequency['cluster'].replace(cluster_dict, inplace=True)


    # In[124]:


    frequency.groupby('cluster')['number_of_orders'].describe().sort_values(by='mean')


    # Cluster 3 are highest frequency customer

    # ### Monetary

    # In[125]:


    # Create a dataframe to store customer uuid and monetary scores
    monetary = df[['customer_unique_id', 'payment_value']].copy()


    # In[126]:


    # Total payment value per customer
    monetary = monetary.groupby('customer_unique_id')['payment_value'].sum().reset_index()


    # In[127]:


    sns.distplot(monetary['payment_value'], hist=False)
    plt.xlabel('Payment value')
    plt.ylabel('Density')
    plt.title('Distribution of payment value per customer');


    # In[128]:


    inertia = {}

    for k in range(1,10):
        km = KMeans(n_clusters=k)
        km = km.fit(monetary[['payment_value']])
        inertia[k] = km.inertia_


    # In[129]:


    plt.figure(figsize=(8,5))
    plt.plot(list(inertia.keys()), list(inertia.values()))
    plt.xlabel('k')
    plt.ylabel('Inertia');


    # In[130]:


    km = KMeans(n_clusters=4, random_state=42)
    km.fit(monetary[['payment_value']])
    monetary['cluster'] = km.labels_


    # In[131]:


    monetary.head()


    # In[132]:


    monetary.groupby('cluster')['payment_value'].describe().sort_values(by='mean')


    # In[133]:


    # Renaming the clusters according to mean number of payment_value
    monetary_cluster = monetary.groupby('cluster')['payment_value'].mean().reset_index()
    monetary_cluster = monetary_cluster.sort_values(by='payment_value').reset_index(drop=True)
    monetary_cluster['index'] = np.arange(0,4)
    monetary_cluster.set_index('cluster', inplace=True)
    cluster_dict = monetary_cluster['index'].to_dict()
    monetary['cluster'].replace(cluster_dict, inplace=True)


    # In[134]:


    monetary.groupby('cluster')['payment_value'].describe().sort_values(by='mean')


    # **Cluster 2 and 3** are biggest spending customers.

    # ### Overall score

    # In[153]:


    # Merge recency, frequency and monetary dataframes together on customer uuid
    overall = recency.merge(frequency, on='customer_unique_id')
    overall = overall.merge(monetary, on='customer_unique_id')

    # Rename cluster columns
    overall.rename(columns={'cluster_x': 'recency_cluster',
                           'cluster_y': 'frequency_cluster',
                           'cluster': 'monetary_cluster'},
                  inplace=True)

    # Sum up the clusters to obtain the overall score
    overall['overall_score'] = overall['recency_cluster'] + overall['frequency_cluster'] + overall['monetary_cluster']


    # In[154]:


    overall.head()


    # In[155]:


    overall.groupby('overall_score')[['inactive_days', 'number_of_orders', 'payment_value']].mean()


    # We have now assigned customers a score that ranges from 0 to 8, with 8 being Olist's most valuable customers. For simplicity, we will re-group these customers into 3 segments:
    #
    # * Scores 0 to 2: Low value
    # * Scores 3 to 4: Mid value
    # * Scores 5+: High value

    # In[156]:


    overall['segment'] = overall['overall_score'].map(lambda x: 'low' if x < 3 else ('mid' if x < 5 else 'high'))


    # In[157]:


    overall['segment'].value_counts()


    # In[158]:


    overall.head(20)

    overall.to_csv(r'result_datasets/data_overall.csv')


    # In[159]:


    overall.groupby('segment')[['inactive_days', 'number_of_orders', 'payment_value']].mean().sort_values(by='payment_value')


    # In[160]:


    fig, ax = plt.subplots(3, 1, figsize=(15,20))
    sns.scatterplot(x='inactive_days', y='number_of_orders', ax=ax[0], hue='segment', data=overall)
    sns.scatterplot(x='inactive_days', y='payment_value', ax=ax[1], hue='segment', data=overall)
    sns.scatterplot(x='number_of_orders', y='payment_value', ax=ax[2], hue='segment', data=overall);


    # ### The RFM framework helps us to clearly differentiate the low, mid and high segments.

