# import schedule
import logging.config
import time
import datetime
from eda_dataloading.eda_cust_file import reading_files
# from visualize import visualize_eda
from nlp_reviews.nlp_review import nlp_review_func
from Models.model_building import model_build
from eda_dataloading.data_loading_understanding import dataload
from recommendation_engine.recommendation import recommend
from customer_files.customer_segmentation import cust_seg_rmf
from customer_files.customer_lifetime_value import cust_lifetime
from time_series.Time_series import time_series_analysis
from result_datasets.data_conversion_json import csv_to_json
from recommendation_engine.recom_product_corr import recom_product_corr

logging.basicConfig(filename="Logs/file_{}.log".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d')),
                    level=logging.INFO, format='%(asctime)s %(message)s %(lineno)s %(filename)s', datefmt='%m/%d/%Y %I:%M:%S %p', )

def module_start():

    options = input("Choose the Options from below to run the Module : \n '1' - * Data Loading & Selections \n '2' - * EDA \n '3' - * Recommendation Engine \n '4' - * Reviews \n '5' - * Customer Segmentation\n '6' - * Customer LifeTime Values \n '7' - * Model Building \n '8' - * Time_Series \n\n '***' - * Press ( * ) Show Dashboard \n\n \nChoise is :")
    # selection = input("Enter the Selection : \n '1' - EDA \n '2' Visualize the Data" )
    if '1' in options:
        dataload.data_loading()
        main()
        time.sleep(15)
        module_start()
    elif '2' in options:
        opt=input("choose the options : \n '1') - ** Customer \n '2' - ** Seller : ")
        if '1' in opt:
            cust=pd.read_csv(r'result_datasets/data_overall.csv')
            cust.head()
        else:
            pass
        reading_files()
        time.sleep(15)
        module_start()
    elif '3' in options:
        print('Start Recommendations using Apriori & Pruning Method')
        recommend()
        recom_product_corr()
        print(' Restarting the Modules...')
        time.sleep(15)
        module_start()
    elif '4' in options:
        nlp_review_func()
        time.sleep(15)
        module_start()
    elif '5' in options:
        cust_seg_rmf()
        time.sleep(15)
        module_start()
    elif '6' in options:
        cust_lifetime()
        time.sleep(15)
        module_start()
    elif '7' in options:
        model_build()
        time.sleep(15)
        module_start()
    elif '8' in options:
        time_series_analysis()
        time.sleep(15)
        module_start()
    elif '9' in options:
        print('conversion in JSON')
        csv_to_json()
        print('Successfully Converted')
        time.sleep(15)
        module_start()

    elif '*' in options:
        pass

    else:
        print('Thank You for visiting the Page.. Hope you enjoyed the Project!!!')

if __name__ == '__main__':
    try:
        logging.info('Program Starts in Main Function ')
        module_start()

    except Exception as ex:
        print('Thank you')
        # logging.error('Error Occurred ' + str(ex))


