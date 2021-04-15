
def recommend():

    # Import modules.
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mlxtend.preprocessing import TransactionEncoder


    # Load orders dataset.
    orders = pd.read_csv(r'./input_original_datasets/olist_order_items_dataset.csv')
    # orders = pd.read_csv(r'./input_original_datasets/olist_order_items_dataset.csv')

    products = pd.read_csv(r'./input_original_datasets/olist_products_dataset.csv')

    # Load translations dataset.
    translations = pd.read_csv(r'./input_original_datasets/product_category_name_translation.csv')

    # Print orders header.
    orders.head()

    # Print orders info.
    orders.info()


    # Print products header.
    products.head()


    # Print products info.
    products.info()


    # Print translations header.
    translations.head()


    # Print translations info.
    translations.info()


    # Translate product names to English.
    products = products.merge(translations, on='product_category_name', how="left")

    # Print English names.
    products['product_category_name_english']


    # # Convert product IDs to product category names.**


    # Define product category name in orders DataFrame.
    orders = orders.merge(products[['product_id','product_category_name_english']], on='product_id', how='left')

    # Print orders header.
    orders.head()


     # Drop products without a defined category.
    orders.dropna(inplace=True, subset=['product_category_name_english'])
    # Print number of unique items.
    len(orders['product_id'].unique())

    # Print number of unique categories.
    len(orders['product_category_name_english'].unique())


    # # Construct transactions from order and product data**

    # Identify transactions associated with example order.
    example1 = orders[orders['order_id'] == 'fe64170e936bc5f6a6a41def260984b9']['product_category_name_english']

    # Print example.
    example1


    # Identify transactions associated with example order.
    example2 = orders[orders['order_id'] == 'fffb9224b6fc7c43ebb0904318b10b5f']['product_category_name_english']

    # Print example.
    example2


    # # Map orders to transactions.
    #
    #

    # Recover transaction itemsets from orders DataFrame.
    transactions = orders.groupby("order_id").product_category_name_english.unique()

    # Print transactions header.
    transactions.head()


    # Plot 50 largest categories of transactions.
    transactions.value_counts()[:50].plot(kind='bar', figsize=(15,5))


    # Convert the pandas series to list of lists.
    transactions = transactions.tolist()

    # Print length of transactions.
    len(transactions)


    # Count number of unique item categories for each transaction.
    counts = [len(transaction) for transaction in transactions]
    # Print median number of items in a transaction.
    np.median(counts)


    # Print maximum number of items in a transaction.
    np.max(counts)


    # # Association Rules and Metrics


    from mlxtend.preprocessing import TransactionEncoder

    # Instantiate an encoder.
    encoder = TransactionEncoder()

    # Fit encoder to list of lists.
    encoder.fit(transactions)

    # Transform lists into one-hot encoded array.
    onehot = encoder.transform(transactions)

    # Convert array to pandas DataFrame.
    onehot = pd.DataFrame(onehot, columns = encoder.columns_)
    # Print header.
    onehot.head()


    # # Compute the support metric
    #

    # Print support metric over all rows for each column.
    onehot.mean(axis=0)


    # # Compute the item count distribution over transactions


    # Print distribution of item counts.
    onehot.sum(axis=1).value_counts()


    # # Create a column for an itemset with multiple items
    #


    # Add sports_leisure and health_beauty to DataFrame.
    onehot['sports_leisure_health_beauty'] = onehot['sports_leisure'] & onehot['health_beauty']

    # Print support value.
    onehot['sports_leisure_health_beauty'].mean(axis = 0)


    # # **Aggregate the dataset further by combining product sub-categories**
    # We can use the inclusive OR operation to combine multiple categories.
    # * True | True = True
    # * True | False = True
    # * False | True = True
    # * False | False = False

    # Merge books_imported and books_technical.
    onehot['books'] = onehot['books_imported'] | onehot['books_technical']

    # Print support values for books, books_imported, and books_technical.
    onehot[['books','books_imported','books_technical']].mean(axis=0)


    # # Compute the confidence metric
    #

    # Compute joint support for sports_leisure and health_beauty.
    joint_support = (onehot['sports_leisure'] & onehot['health_beauty']).mean()

    # Print confidence metric for sports_leisure -> health_beauty.
    joint_support / onehot['sports_leisure'].mean()


    # Print confidence for health_beauty -> sports_leisure.
    joint_support / onehot['sports_leisure'].mean()


    # # The Apriori Algorithm and Pruning

    from mlxtend.frequent_patterns import apriori

    # Apply apriori algorithm to data with min support threshold of 0.01.
    frequent_itemsets = apriori(onehot, min_support = 0.01)

    # Print frequent itemsets.
    frequent_itemsets


    # Apply apriori algorithm to data with min support threshold of 0.001.
    frequent_itemsets = apriori(onehot, min_support = 0.001, use_colnames = True)

    # Print frequent itemsets.
    frequent_itemsets


    # Apply apriori algorithm to data with min support threshold of 0.00005.
    frequent_itemsets = apriori(onehot, min_support = 0.00005, use_colnames = True)

    # Print frequent itemsets.
    frequent_itemsets

    # Apply apriori algorithm to data with a two-item limit.
    frequent_itemsets = apriori(onehot, min_support = 0.00005, max_len = 2, use_colnames = True)


    # # Computing association rules from Apriori output**

    from mlxtend.frequent_patterns import association_rules

    # Recover association rules using support and a minimum threshold of 0.0001.
    rules = association_rules(frequent_itemsets, metric = 'support', min_threshold = 0.0001)

    # Print rules header.
    rules.head()
    rules.to_csv('result_datasets/result_apriori.csv')


    # # Pruning association rules

    # Recover association rules using confidence threshold of 0.01.
    rules = association_rules(frequent_itemsets, metric = 'confidence', min_threshold = 0.01)

    # Print rules.
    rules
    rules.to_csv('result_datasets/result_Pruning.csv')


    # Select rules with a consequent support above 0.095.
    rules = rules[rules['consequent support'] > 0.095]

    # Print rules.
    rules


    # # The leverage metric
    #

    # Select rules with leverage higher than 0.0.
    rules = rules[rules['leverage'] > 0.0]

    # Print rules.
    rules


    # # Visualizing patterns in metrics

    # Recover association rules with a minimum support greater than 0.000001.
    rules = association_rules(frequent_itemsets, metric = 'support', min_threshold = 0.000001)



    # # Plot leverage against confidence.
    # plt.figure(figsize=(15,5))
    # sns.scatterplot(x="leverage", y="confidence", data=rules)






