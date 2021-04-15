# coding: utf-8

# ### Importing Libraries

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')

# NLTK tools for text processing
import re, nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS

# In[2]:


get_ipython().system(u'pip install wordcloud')

# For language translation we need to install Google Translate API.

# In[5]:


# installing the trranslate API
get_ipython().system(u'pip install google_trans_new')

# In[6]:


import google_trans_new

print(google_trans_new.LANGUAGES)  # available languages

# In[9]:


from google_trans_new import google_translator

translator = google_translator()
def recommendation_data():
    translate_text = translator.translate("I live in India, My name is Ishita Maity", lang_tgt='bn',
                                          lang_src='en')  # englist to bengali
    # translate_text2=translator.translate('I live in India, My name is Ishita Maity',lang_tgt='hi',lang_src='en') #english to hindi
    translate_text = translator.translate("I live in India, My name is Ishita Maity", lang_tgt='te',
                                          lang_src='en')  # englist to bengali
    translate_text3 = translator.translate('I live in India, My name is Ishita Maity', lang_tgt='zh-cn',
                                           lang_src='en')  # english to chinese
    print(translate_text)
    print(translate_text2)
    print(translate_text)

    # ### Loading Datasets

    # In[92]:

    #
    # df_cust = pd.read_csv(r'C:\Users\ishita\Downloads\olist\olist_customers_dataset.csv')
    # df_loc = pd.read_csv(r'C:\Users\ishita\Downloads\olist\olist_geolocation_dataset.csv')
    # df_items = pd.read_csv(r'C:\Users\ishita\Downloads\olist\olist_order_items_dataset.csv')
    # df_pmt = pd.read_csv(r'C:\Users\ishita\Downloads\olist\olist_order_payments_dataset.csv')
    # df_rvw = pd.read_csv(r'C:\Users\ishita\Downloads\olist\olist_order_reviews_dataset.csv')
    # df_products = pd.read_csv(r'C:\Users\ishita\Downloads\olist\olist_products_dataset.csv')
    # df_orders = pd.read_csv(r'C:\Users\ishita\Downloads\olist\olist_orders_dataset.csv')
    # df_sellers = pd.read_csv(r'C:\Users\ishita\Downloads\olist\olist_sellers_dataset.csv')
    # df_cat_name = pd.read_csv(r'C:\Users\ishita\Downloads\olist\product_category_name_translation.csv')

    # ##### review table

    # In[7]:


    df_rvw.head()

    # In[8]:


    df_rvw.shape

    # In[9]:


    df_rvw.isnull().sum()

    # review dataset has missing values in review_comment_title and review_comment_message

    # To deal with these missing values, we would seperate the reviews and the titles and drop the missing rows seperately so that we don't have unequal shapes of rows

    # In[10]:


    review_title = df_rvw['review_comment_title']

    review_data = df_rvw.drop(['review_comment_title'], axis=1)

    # In[11]:


    # removing nan values
    review_title = review_title.dropna()
    review_data = review_data.dropna()

    # In[12]:


    review_data

    # In[13]:


    review_title

    # In[14]:


    print(review_data.shape)
    print(review_title.shape)

    # In[15]:


    review_data = review_data.reset_index(drop=True)

    # ##### Now I would transform the reviews data by removing stopwords, using regular expressions module to accept only letters, tokenize those words and then make all the words lower case for consistency

    # In[16]:


    # Transforming the reviews data by removing stopwords, using regular expression
    comments = []
    stop_words = set(stopwords.words('portuguese'))

    for words in review_data['review_comment_message']:
        only_letters = re.sub("[^a-zA-Z]", " ", words)
        tokens = nltk.word_tokenize(only_letters)  # tokenize the sentences
        lower_case = [word.lower() for word in tokens]  # convert all letters to lower case
        filtered_result = list(
            filter(lambda word: word not in stop_words, lower_case))  # remove stopwords from the comments
        comments.append(' '.join(filtered_result))

    # In[17]:


    comments

    # In[147]:


    # using wordcloud to visualize the comments
    unique_string = (" ").join(comments)
    wordcloud = WordCloud(width=3000, height=1500, background_color='red').generate(unique_string)
    plt.figure(figsize=(20, 15))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # These words are in Portuguese. we need to translate these words to English.

    # In[19]:


    # using CountVectorizer to get the most important trigrams
    from sklearn.feature_extraction.text import CountVectorizer

    co = CountVectorizer(ngram_range=(3, 3))
    counts = co.fit_transform(comments)
    important_trigrams = pd.DataFrame(counts.sum(axis=0), columns=co.get_feature_names()).T.sort_values(0,
                                                                                                        ascending=False).head(
        50)
    # we reset the index, rename the columns and apply the translate module to get the english translations
    important_trigrams = important_trigrams.reset_index()
    important_trigrams.rename(columns={'index': 'trigrams', 0: 'frequency'}, inplace=True)

    important_trigrams['english_translation'] = important_trigrams['trigrams'].apply(translator.translate)
    important_trigrams

    # In[51]:


    comments_eng = []
    for words in important_trigrams['english_translation']:
        only_letters = re.sub("[^a-zA-Z]", " ", words)
        tokens = nltk.word_tokenize(only_letters)  # tokenize the sentences
        lower_case = [word.lower() for word in tokens]  # convert all letters to lower case
        comments_eng.append(' '.join(lower_case))

    # In[52]:


    comments_eng

    # In[53]:


    unique_string = (" ").join(comments_eng)
    wordcloud = WordCloud(width=3000, height=1500, background_color='pink').generate(unique_string)
    plt.figure(figsize=(20, 15))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # In[20]:


    # using countvecrtorizer to get the most important unigrams
    co = CountVectorizer(ngram_range=(1, 1))
    counts = co.fit_transform(comments)
    important_unigrams = pd.DataFrame(counts.sum(axis=0), columns=co.get_feature_names()).T.sort_values(0,
                                                                                                        ascending=False).head(
        50)
    # we reset the index, rename the columns and apply the translate module to get the english translations
    important_unigrams = important_unigrams.reset_index()
    important_unigrams.rename(columns={'index': 'unigrams', 0: 'frequency'}, inplace=True)

    important_unigrams['english_translation'] = important_unigrams['unigrams'].apply(translator.translate)
    important_unigrams

    # In[21]:


    # using countvecrtorizer to get the most important bigrams
    co = CountVectorizer(ngram_range=(2, 2))
    counts = co.fit_transform(comments)
    important_bigrams = pd.DataFrame(counts.sum(axis=0), columns=co.get_feature_names()).T.sort_values(0,
                                                                                                       ascending=False).head(
        50)
    # we reset the index, rename the columns and apply the translate module to get the english translations
    important_bigrams = important_bigrams.reset_index()
    important_bigrams.rename(columns={'index': 'bigrams', 0: 'frequency'}, inplace=True)

    important_bigrams['english_translation'] = important_bigrams['bigrams'].apply(translator.translate)
    important_bigrams

    # from the unigrams,bigrams and trigrams we can say that most of the customers were satisfied with the delivery service and also product quality

    # In[22]:


    comment_titles = []
    stop_words = set(stopwords.words('portuguese'))

    for words in review_title:
        only_letters = re.sub("[^a-zA-Z]", " ", words)
        tokens = nltk.word_tokenize(only_letters)
        lower_case = [l.lower() for l in tokens]
        filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
        comment_titles.append(' '.join(filtered_result))

    # In[23]:


    # visualize the comment_titles
    unique_string = (" ").join(comment_titles)
    wordcloud = WordCloud(width=4000, height=2500, background_color='pink').generate(unique_string)
    plt.figure(figsize=(20, 15))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # In[24]:


    # using countvecrtorizer to get the most important unigrams
    co = CountVectorizer(ngram_range=(1, 1))
    counts = co.fit_transform(comment_titles)
    important_unigrams = pd.DataFrame(counts.sum(axis=0), columns=co.get_feature_names()).T.sort_values(0,
                                                                                                        ascending=False).head(
        50)
    # we reset the index, rename the columns and apply the translate module to get the english translations
    important_unigrams = important_unigrams.reset_index()
    important_unigrams.rename(columns={'index': 'unigrams', 0: 'frequency'}, inplace=True)

    important_unigrams['english_translation'] = important_unigrams['unigrams'].apply(translator.translate)
    important_unigrams

    # In[41]:


    # Transforming the reviews data by removing stopwords, using regular expression
    comment_titles_eng = []

    for words in important_unigrams['english_translation']:
        only_letters = re.sub("[^a-zA-Z]", " ", words)
        tokens = nltk.word_tokenize(only_letters)  # tokenize the sentences
        lower_case = [word.lower() for word in tokens]  # convert all letters to lower case
        comment_titles_eng.append(' '.join(lower_case))

    # In[58]:


    unique_string = (" ").join(comment_titles_eng)
    wordcloud = WordCloud(width=4000, height=2500, background_color='yellow').generate(unique_string)
    plt.figure(figsize=(20, 15))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # From this above wordcloud we can see most frequent comments in english.

    # In[25]:


    # using countvecrtorizer to get the most important bigrams
    co = CountVectorizer(ngram_range=(2, 2))
    counts = co.fit_transform(comment_titles)
    important_bigrams = pd.DataFrame(counts.sum(axis=0), columns=co.get_feature_names()).T.sort_values(0,
                                                                                                       ascending=False).head(
        50)
    # we reset the index, rename the columns and apply the translate module to get the english translations
    important_bigrams = important_bigrams.reset_index()
    important_bigrams.rename(columns={'index': 'bigrams', 0: 'frequency'}, inplace=True)

    important_bigrams['english_translation'] = important_bigrams['bigrams'].apply(translator.translate)
    important_bigrams

    # In[26]:


    # using CountVectorizer to get the most important trigrams
    from sklearn.feature_extraction.text import CountVectorizer

    co = CountVectorizer(ngram_range=(3, 3))
    counts = co.fit_transform(comment_titles)
    important_trigrams = pd.DataFrame(counts.sum(axis=0), columns=co.get_feature_names()).T.sort_values(0,
                                                                                                        ascending=False).head(
        50)
    # we reset the index, rename the columns and apply the translate module to get the english translations
    important_trigrams = important_trigrams.reset_index()
    important_trigrams.rename(columns={'index': 'trigrams', 0: 'frequency'}, inplace=True)

    important_trigrams['english_translation'] = important_trigrams['trigrams'].apply(translator.translate)
    important_trigrams

    # The unigrams, bigrams and trigrams of the review titles data have revealed the unhappy customers. These comments include: poorly packaged product,fale relay,I didn't receive prod, not delivered etc. We have also seen satisfaction among other customer.

    # In[55]:


    comment_titles_eng2 = []

    for words in important_trigrams['english_translation']:
        only_letters = re.sub("[^a-zA-Z]", " ", words)
        tokens = nltk.word_tokenize(only_letters)  # tokenize the sentences
        lower_case = [word.lower() for word in tokens]  # convert all letters to lower case
        comment_titles_eng2.append(' '.join(lower_case))

    # In[57]:


    unique_string = (" ").join(comment_titles_eng2)
    wordcloud = WordCloud(width=4000, height=2500, background_color='yellow').generate(unique_string)
    plt.figure(figsize=(20, 15))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # In[27]:


    # plotting review score
    plt.figure(figsize=(20, 20))
    sns.countplot(df_rvw['review_score'], color='yellow')  # before removing nan values
    sns.countplot(review_data['review_score'], color='blue')  # after removing nan values

    # * close to 60,000 people gave 5 star ratings and little above 10,000 people gave 1 star ratings(before removing nan values)
    #
    # * little above 20,000 people gave 5 star ratings and close to 10,000 people gave 1 star rating (after removing nan values)

    # In[30]:


    review_data

    # In[60]:


    review_data['review_creation_date'] = pd.to_datetime(review_data['review_creation_date'])
    review_data

    # In[64]:


    review_data['review_creation_year'] = review_data['review_creation_date'].dt.year
    review_data['review_creation_year']

    # In[65]:


    sns.countplot(x='review_creation_year', data=review_data)

    # Most of the reviews are created in 2018

    # In[66]:


    review_data['review_creation_month'] = review_data['review_creation_date'].dt.month
    review_data['review_creation_month']

    # In[67]:


    sns.countplot(x='review_creation_month', data=review_data)

    # In[68]:


    review_data['review_creation_day'] = review_data['review_creation_date'].dt.day_name()
    review_data['review_creation_day']

    # In[69]:


    sns.countplot(x='review_creation_day', data=review_data)

    # from the above graph we can say less people visited the site on monday.

    # In[73]:


    df_cust

    # ### Merging

    # In[106]:


    merged_df = df_orders.copy()
    merged_df = merged_df.merge(df_cust, on='customer_id', indicator=True)
    merged_df = merged_df.merge(review_data, on='order_id')
    merged_df = merged_df.merge(df_items, on='order_id')
    merged_df = merged_df.merge(df_products, on='product_id')
    merged_df = merged_df.merge(df_sellers, on='seller_id')

    # In[109]:


    merged_df.columns

    # In[128]:


    plt.figure(figsize=(20, 30))
    sns.barplot(x='customer_state', y='review_score', data=merged_df, errcolor='red')
    plt.xlabel('customer_state', fontsize=50)
    plt.ylabel('review_score', fontsize=50)

    plt.show()

    # ### Connect Python and MySQL

    # In[131]:


    get_ipython().system(u'pip install mysql-connector-python')

    # In[132]:


    import mysql.connector
    from mysql.connector import Error

    try:
        connection = mysql.connector.connect(host='localhost', database='ecommerce', user='root', password='ishita')

        if connection.is_connected():
            db_Info = connection.get_server_info()
            print("Connected to Mysql", db_Info)
            cursor = connection.cursor()
            cursor.execute('select database();')
            record = cursor.fetchone()
            print("you're connected to database: ", record)

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

    # In[145]:


    import mysql.connector
    from mysql.connector import Error
    import csv

    connection = mysql.connector.connect(host='localhost', database='ecommerce', user='root', password='ishita')
    cursor = connection.cursor()
    mysql_Create_Table_Query = """CREATE TABLE review(
                                    review_id VARCHAR(255),
                                    order_id VARCHAR(255),
                                    review_score VARCHAR(255) ,
                                    review_comment_message VARCHAR(255),
                                    review_creation_date VARCHAR(255),
                                    review_answer_timestamp VARCHAR(255),
                                    review_creation_year VARCHAR(255),
                                    review_creation_month VARCHAR(255),
                                    review_creation_day VARCHAR(255));"""
    # Execute a command: this creates a new table
    cursor.execute(mysql_Create_Table_Query)
    connection.commit()
    print("Table created successfully in MySQL ")

    # In[146]:


    for i, row in review_data.iterrows():
        sql = "INSERT INTO ecommerce.review VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        cursor.execute(sql, tuple(row))
    connection.commit()
    print('record inserted')

    # In[138]:


    review_data.isnull().sum()


