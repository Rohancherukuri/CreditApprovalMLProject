# Implementing eda as class based structure in python file
import logging
import warnings 
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")
PATH = "/home/rohanoxob/MachineLearning/CreditApprovalProject/logs/eda.log"
#logging.basicConfig(filename = PATH,level = logging.INFO,format = LOG_FORMAT,filemode = "w")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
file_handler = logging.FileHandler(PATH)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
class EDA:
    # This is a class which implements EDA process
    def __init__(self): 
        logger.info("Inside the EDA class constructor")

    def read_file(self):
        # Reading Files:
        """
            Method Name: read_file
            Description: Reads the csv files 
            Output: Returns the csv files
            On Failure: Raise Exception
        """
        logger.info("Inside the read_file function of EDA class")
        try:
            self.application = pd.read_csv('Datasets/application_record.csv',index_col='ID')
            self.credit = pd.read_csv('Datasets/credit_record.csv',index_col='ID')
            logger.info("Successfully read the csv files from the read_file function of EDA class")
            return self.application, self.credit
        except Exception as e:
            logger.warning("Failed to read the csv files,exitting from the read_csv function of EDA class")
            logger.warning("Exception message: "+str(e))
            print("Error occured inside read_file function of EDA class"+str(e))
            raise Exception()
        
    def get_distinct_values(self):
        # Getting Distinct values to work with:
        """
            Method Name: get_distinct_values
            Description: Gets the distinct value of particular columns of the dataset 
            Output: Makes distinct columns
            On Failure: Raise Exception 
        """
        logger.info("Inside the get_distinct_values function of EDA class")
        try:
            uniqueID = (list(set(self.application.index).intersection(set(self.credit.index))))
            self.application = self.application.loc[uniqueID]
            self.credit = self.credit.loc[uniqueID]
            self.application_clean = self.application.sort_values(by = self.application.columns.to_list())
            self.application_clean['cust_id'] = self.application.sum(axis=1).map(hash)
            
            grouped_cust = self.application.sum(axis=1).map(hash).reset_index().rename(columns={0:'customer_id'})
            grouped_cust = grouped_cust.set_index('ID')
            credit_trsf = self.credit.merge(grouped_cust, how = 'inner', on = 'ID').reset_index()[['customer_id','ID', 'MONTHS_BALANCE', 'STATUS']]
            self.cred_df_g = credit_trsf.sort_values(by=['customer_id', 'ID', 'MONTHS_BALANCE'], ascending = [True, True, False]).reset_index(drop=True)
            self.cred_df_g['link_ID'] = self.cred_df_g.groupby(['customer_id','ID'], sort = False).ngroup().add(1)
            self.cred_df_g.drop(columns = ['ID'], inplace=True)
            self.cred_df_g = self.cred_df_g[['customer_id', 'link_ID', 'MONTHS_BALANCE', 'STATUS']]
            logger.info("Successfully excuted the get_distinct function of EDA class")
        except Exception as e:
            logger.warning("Failed to execute the get_distinct_values function of EDA class,exitting from the get_distinct_values function of EDA class")
            logger.warning("Exception message: "+str(e))
            print("Error occured inside get_distinct_values function of EDA class "+str(e))
            raise Exception()

    def label_data(self):
        # Labelling customer data:
        """
             Method Name: label_data
            Description: Labels the data 
            Output: Makes the labels on particular dataset
            On Failure: Raise Exception
        """
        logger.info("Inside the label_data function of EDA class")
        try:
            self.cred_df_g['monthly_behaviour'] = np.where( self.cred_df_g.STATUS.isin(['2','3','4','5']), 'b', 'g' )
            self.cred_df_g.groupby(['customer_id', 'monthly_behaviour']).size()
            cust_behaviour = pd.DataFrame( round( self.cred_df_g.groupby(['customer_id', 'monthly_behaviour']).size() / self.cred_df_g.groupby(['customer_id']).size() * 100, 2), columns = ['behaviour_score']).reset_index().set_index('customer_id')
            bad_cust = \
            cust_behaviour[ ( (cust_behaviour.monthly_behaviour=='g') & (cust_behaviour.behaviour_score <= 50) ) | \
                ( (cust_behaviour.monthly_behaviour=='b') & (cust_behaviour.groupby('customer_id').size()==1) )        ]
            bad_cust['customer_type'] = 'bad'
            bad_cust.drop(columns=['monthly_behaviour', 'behaviour_score'], inplace=True)
            good_cust = \
                cust_behaviour[( (cust_behaviour.monthly_behaviour=='g') & (cust_behaviour.behaviour_score > 50) ) |
                    ( (cust_behaviour.monthly_behaviour=='g') & (cust_behaviour.groupby('customer_id').size()==1) )]
            good_cust['customer_type'] = 'good'
            good_cust.drop(columns=['monthly_behaviour', 'behaviour_score'], inplace=True)
            self.credit_clean = pd.concat([bad_cust, good_cust])
            self.credit_clean['months_in_book'] = self.cred_df_g.groupby('customer_id').size()
            self.credit_clean['contracts_nr'] = self.cred_df_g.groupby(['customer_id'])['link_ID'].nunique()
            logger.info("Successfully excuted the label_data function of EDA class")
        except Exception as e:
            logger.warning("Failed to excute label_data,exitting from the label_data function of the EDA class")
            logger.warning("Exception message: "+str(e))
            print("Error occured inside the label_data function of EDA class"+str(e))
            raise Exception()

    def fill_values(self):
        # Fill values (temporary)
        """
            Method Name: fill_values
            Description: Fills the missing data points of the dataset 
            Output: Returns the filled datasets
            On Failure: Raise Exception
        """
        logger.info("Inside the fill_values function of EDA class")
        try:
            self.application_clean['OCCUPATION_TYPE'] = self.application_clean['OCCUPATION_TYPE'].fillna('Not Available')
            self.application_clean['FLAG_OWN_CAR'] = self.application_clean['FLAG_OWN_CAR'].replace({'Y':1,'N':0})
            self.application_clean['FLAG_OWN_REALTY'] = self.application_clean['FLAG_OWN_REALTY'].replace({'Y':1,'N':0})
            logger.info("Successfully excuted the fill_values function of EDA class")
            return self.application_clean
        except Exception as e:
            logger.warning("Failed to excute the fill_values function of EDA class,exitting from fill_values function of EDA class")
            logger.warning("Exception message: "+str(e))
            print("Error occured inside the fill_values function of EDA class")
            raise Exception()


# This is for testing purposes
if __name__ == "__main__":
    try:
        e = EDA()
        df1,df2 = e.read_file()
        print(df1.head())
        print(df2.head())
        e.get_distinct_values()
        e.label_data()
        df3 = e.fill_values()
        print(df3)
    except Exception as e:
        print("Sorry error occured: "+str(e))
