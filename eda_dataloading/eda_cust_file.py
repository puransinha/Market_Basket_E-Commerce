# coding: utf-8

# # Brazilian E-Commerce Public Dataset by Olist

# # Problem statement

# Most customers do not post a review rating or any comment after purchasing a product which is a challenge for any ecommerce platform to perform If a company predicts whether a customer liked/disliked a product so that they can recommend more similar and related products as well as they can decide whether or not a product should be sold at their end.
# This is crucial for ecommerce based company because they need to keep track of each product of each seller , so that none of products discourage their customers to come shop with them again. Moreover, if a specific product has very few rating and that too negetive, a company must not drop the product straight away, may be many customers who found the product to be useful haven't actually rated it.
#
# Some reasons could possibly be comparing your product review with those of your competitors beforehand,gaining lots of insight about the product and saving a lot of manual data pre-processin,maintain good customer relationship with company,lend gifts, offers and deals if the company feels the customer is going to break the relation.
#
# Objective of this case study is centered around predicting customer satisfaction with a product which can be deduced after predicting the product rating a user would rate after he makes a purchase.

# # Constraints

# High Accuracy
#
# Low latency (Rating should be known within the completion of the order)
#
# Prone to outliers
#

# ## Loading packages and dataset

# Imprting the Datasets

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "data/" directory.

import time, warnings
import datetime as dt
import re
import datetime
from datetime import datetime as dt
from sklearn.metrics.pairwise import cosine_similarity
from prettytable import PrettyTable

# visualizations
import matplotlib.pyplot as plt

# get_ipython().magic(u'matplotlib inline')
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
import os
#from mpl_toolkits.basemap import Basemap
import cufflinks as cf
#
# py.offline.init_notebook_mode(connected=True)
# cf.go_offline()

# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans

# Importing libraries for building the neural network
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
# from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import xgboost as xgb
from eli5.sklearn import PermutationImportance
import eli5

# Reading the Files from csv

def reading_files():
    ##reading and checking dataset
    df_cust = pd.read_csv('./input_original_datasets/olist_customers_dataset.csv')
    df_loc = pd.read_csv('./input_original_datasets/olist_geolocation_dataset.csv')
    df_items = pd.read_csv('./input_original_datasets/olist_order_items_dataset.csv')
    df_pmt = pd.read_csv('./input_original_datasets/olist_order_payments_dataset.csv')
    df_rvw = pd.read_csv('./input_original_datasets/olist_order_reviews_dataset.csv')
    df_products = pd.read_csv('./input_original_datasets/olist_products_dataset.csv')
    df_orders = pd.read_csv('./input_original_datasets/olist_orders_dataset.csv')
    df_sellers = pd.read_csv('./input_original_datasets/olist_sellers_dataset.csv')
    df_cat_name = pd.read_csv('./input_original_datasets/product_category_name_translation.csv')

    print(df_cust.head())
    df_cust.head()
    df_loc.head()
    df_items.head()
    df_pmt.head()
    df_rvw.head()
    df_products.head()
    df_orders.head()
    df_sellers.head()
    df_cat_name.head().T
    df_orders.describe().T

    print('Printing Customers Table')
    df_cust.head()
    df_cust.isnull().sum()



    df_cust.customer_state.value_counts().plot(kind='pie', figsize=(6, 8), autopct='%.1f%%', radius=2)
    plt.legend()
    plt.show()
    # Top 10 cities with their value counts
    df_cust.customer_city.value_counts().sort_values(ascending=False)[:10]
    return
    df_cust.info()
    print(
        'Total Nos of Customers: {} \n Total generated IDs: {}'.format(len(df_cust['customer_id'].unique()),
                                                                       len(df_cust)))
    print('Total Nos of Customers: {} \n Total generated unique IDs: {}'.format(
        len(df_cust['customer_unique_id'].unique()),
        len(df_cust)))
    df_cust['customer_unique_id'].duplicated().sum()

    # dropping ALL duplicte values

    df_cust.sort_values('customer_unique_id', inplace=True)
    df_cust.drop_duplicates(subset='customer_unique_id', keep=False, inplace=True)

    # displaying data
    df_cust['customer_unique_id'].duplicated().sum()
    return


