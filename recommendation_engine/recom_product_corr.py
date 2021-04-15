
# coding: utf-8

# # Creating a Product based Recommender System with the help of Correlation.

import pandas as pd
import numpy as np
import logging.config

def recom_product_corr():
    logging.info('Successfully reaches product based recommendation system.......')
    combined=pd.read_csv("./clean_datasets/Combined.csv")
    # print(combined.head())
    # print(combined.shape)
    logging.info('Successfully Loaded Data from Combined CSV files......')
    df1=pd.read_csv("./clean_datasets/Combined.csv", usecols=['product_id', 'customer_unique_id', 'product_category_name_english', 'review_score'])
    # print(df1)
    logging.info('Successfully Loaded Data use columns from CSV files ')

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('white')
    # get_ipython().magic(u'matplotlib inline')
    df1.groupby('product_category_name_english')['review_score'].mean().sort_values(ascending=False).head()

    df1.groupby('product_category_name_english')['review_score'].count().sort_values(ascending=False).head()
    reviews = pd.DataFrame(df1.groupby('product_category_name_english')['review_score'].mean())
    print(reviews.head())
    reviews.to_csv('result_datasets/reviews_data.csv')

    reviews['num_of_reviews'] = pd.DataFrame(df1.groupby('product_category_name_english')['review_score'].count())
    reviews.head()

    # Let's explore the data a bit and get a look at some of the best review product.

    plt.figure(figsize=(10,4))
    reviews['num_of_reviews'].hist(bins=70)

    plt.figure(figsize=(10,4))
    reviews['review_score'].hist(bins=70)
    logging.info('Successfully visualised the best review product......')

    # Now that we have a general idea of what the data looks like,

    # # Recommending Similar Product

    # Create a matrix that has the customer unique id on one access and the product_category_name english on another axis. Each cell will then consist of the review the customer gave to that product there will be a lot of NaN values, because most people have not review most of the product

    rating_pivot = df1.pivot_table(index='customer_unique_id',columns='product_category_name_english',values='review_score')
    rating_pivot.head()
    logging.info('Successfully create a Pivot matrix.....')

    # Most reviewed product:

    reviews.sort_values('num_of_reviews',ascending=False).head(10)
    # print(reviews)

    # health_beauty
    health_beauty_rating = rating_pivot['health_beauty']
    similar_to_health_beauty = rating_pivot.corrwith(health_beauty_rating)
    print(similar_to_health_beauty)
    corr_health_beauty = pd.DataFrame(similar_to_health_beauty, columns=['pearsonR'])
    print(corr_health_beauty)
    corr_health_beauty.dropna(inplace=True)
    corr_health_beauty_summary = corr_health_beauty.join(reviews['num_of_reviews'])
    corr_health_beauty_summary = corr_health_beauty_summary[corr_health_beauty_summary['num_of_reviews'] > 100].sort_values('pearsonR', ascending=False).head()
    print("recommmendation for healthy_beauty",corr_health_beauty_summary)

    # furniture_decore
    furniture_decor_rating = rating_pivot['furniture_decor']
    similar_to_furniture_decor = rating_pivot.corrwith(furniture_decor_rating)
    similar_to_furniture_decor
    corr_furniture_decor = pd.DataFrame(similar_to_furniture_decor, columns=['pearsonR'])
    corr_furniture_decor.dropna(inplace=True)
    corr_furniture_decor_summary = corr_furniture_decor.join(reviews['num_of_reviews'])
    corr_furniture_decor_summary = corr_furniture_decor_summary[corr_furniture_decor_summary['num_of_reviews'] > 100].sort_values('pearsonR', ascending=False).head()
    print("recommendation for furniture_decor",corr_furniture_decor_summary)


    logging.info('Successfully filtering out products that have less than 100 reviews....')
    logging.info('Successfully Recommend based on product correlated with Telephony product....')

    # # Methodology:
    # Users are separated into repeat customers and first time customers and the recommendation system works as follows.
    #
    # Repeat Customers
    # Collaborative filtering recommendation
    # Hot Products
    # Popular in your area
    # New Customers
    # Hot products
    # Popular in your area
    logging.info('Successfully complete recommendation based on product correlation...')

