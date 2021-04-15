
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.graphics.api import qqplot
# get_ipython().magic(u'matplotlib inline')

def timeseries():

    customers_data = pd.read_csv("../input_original_datasets/olist_customers_dataset.csv")
    items_data = pd.read_csv("../input_original_datasets/olist_order_items_dataset.csv")
    payments_data = pd.read_csv("../input_original_datasets/olist_order_payments_dataset.csv")
    orders_data = pd.read_csv("../input_original_datasets/olist_orders_dataset.csv")
    products_data = pd.read_csv("../input_original_datasets/olist_products_dataset.csv")
    category_translation_data = pd.read_csv("../input_original_datasets/product_category_name_translation.csv")

    customers_data.head()

    items_data.head()


    payments_data.head()

    orders_data.head()

    products_data.head()

    category_translation_data.head()

    items_data.dropna(inplace=True)
    items_data.info()
    items_data.isnull().sum().sort_values()

    del items_data['order_id']

    del items_data['product_id']

    del items_data['seller_id']

    del items_data['freight_value']


    del items_data['order_item_id']


    items_data.head()


    items_data.set_index('shipping_limit_date',inplace=True)


    items_data.head()


    items_data.plot() #Data Is Seasonal


    # get_ipython().magic(u'matplotlib inline')
    items_data.plot(figsize=(11,8))


    from pandas import read_csv
    from statsmodels.tsa.stattools import adfuller
    test_result=adfuller(items_data['price']) # testing stationarity



    def adfuller_test(price):
        result=adfuller(price)
        labels=['ADF Test Statistic','P-value','#lags used','number of observations used']
        for value,label in zip(result,labels):
            print(labels.append(value))
        if result[1]<=0.05:
            print("strong evidence against the null hypothesis(Ho)")
        else:
            print("weak evidence against null hypothesis")

    adfuller_test(items_data['price'])


    sm.stats.durbin_watson(items_data) #The value of Durbin-Watson statistic is close to 2 if the errors are uncorrelated.

    # get_ipython().magic(u'matplotlib inline')
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(items_data.values.squeeze(), lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(items_data, lags=40, ax=ax2)


    from pandas.plotting import autocorrelation_plot
    # show plots in the notebook
    # get_ipython().magic(u'matplotlib inline')
    items_data['price_2'] = items_data['price']
    items_data['price_2'] = (items_data['price_2'] - items_data['price_2'].mean()) / (items_data['price_2'].std())
    plt.acorr(items_data['price_2'],maxlags = len(items_data['price_2']) -1, linestyle = "solid", usevlines = False, marker='')
    plt.show()
    autocorrelation_plot(items_data['price'])
    plt.show()


    arma_mod20 = sm.tsa.ARMA(items_data['price'], (1,0)).fit()
    print(arma_mod20.params)


    print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)


    sm.stats.durbin_watson(arma_mod20.resid.values)

    # get_ipython().magic(u'matplotlib inline')
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax = arma_mod20.resid.plot(ax=ax);


    resid20 = arma_mod20.resid
    stats.normaltest(resid20)


    # get_ipython().magic(u'matplotlib inline')
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    fig = qqplot(resid20, line='q', ax=ax, fit=True)

    r,q,p = sm.tsa.acf(resid20.values.squeeze(), qstat=True)
    data = np.c_[range(1,41), r[1:], q, p]
    table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
    print(table.set_index('lag'))


    arma_mod20.predict(2017, 2018)

    predict_price20 = arma_mod20.predict(1990, 2018, dynamic=True)
    print(predict_price20)


    ax = items_data.loc[:].plot(figsize=(12,8))
    ax = predict_price20.plot(ax=ax, style='r--', label='Dynamic Prediction');
    ax.legend();
    ax.axis((-20.0, 38.0, -4.0, 200.0));


    payments_data1 = pd.read_csv("olist_order_payments_dataset.csv")
    payments_data1


    payments_data1.dropna(inplace=True)
    payments_data1.info()
    payments_data1.isnull().sum().sort_values()

    payments_data1.plot()


    payments_data1.head()

    del payments_data1['order_id']

    del payments_data1['payment_sequential']


    del payments_data1['payment_type']

    payments_data1.head()


    payments_data1.set_index('payment_installments',inplace=True)


    payments_data1.head()


    payments_data1.plot()


    # get_ipython().magic(u'matplotlib inline')
    payments_data1.plot(figsize=(11,8))


    from pandas import read_csv
    from statsmodels.tsa.stattools import adfuller
    test_result=adfuller(items_data['price']) # testing stationarity
