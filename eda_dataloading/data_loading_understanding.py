import pandas as pd

import logging

class dataload():


    def data_loading():
        logging.info('You have reached at Data Sections... Loading Function start...')
        # try:
        datatype = input("Choose to Load the Datasets -- \n 1 - csv\n 2 - json (*not Implemented)\n 3 - pdf(* not Implemented)\n 4 - txt(* not Implemented)\n 5 - Load from Database (*not Yet Provided)\n\n")
        try:
            if datatype == '1':
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

                logging.info('Successfully Loaded Data from CSV files')
                print('Successfully Loaded Data from CSV files')

                from main import module_start
                return module_start()
        except Exception as error:
            logging.error('Read error : ' + str(error))
            print('Database root not found ... Kindly check and restart .... Quitting the program ....')
