import warnings
import logging
import pandas as pd
warnings.simplefilter("ignore")
PATH = "/home/rohanoxob/MachineLearning/CreditApprovalProject/logs/data_loader.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
file_handler = logging.FileHandler(PATH)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class DataGetter:
    # This is a class for obtainig the data from the source for training
    def __init__(self):
        DATA_FILE_PATH = "/home/rohanoxob/MachineLearning/CreditApprovalProject/data.csv"
        self.data_file = DATA_FILE_PATH
        logger.info("Inside the DataGetter class constructor")
        
    def get_data(self):
        """
            Method Name: get_data
            Description: This is method reads the data from the source.
            Output: Returns a pandas dataframe
            On Failure: Raise Exception
        """
        logger.info("Inside the get_data function of DataGetter class")
        try:
            self.data = pd.read_csv(self.data_file) # reading the data file
            logger.info("The data is loaded successfully.Exited the get_data function of DataGetter class")
            return self.data
        except Exception as e:
            logger.warning("An Exception has occured in get_data function in DataGetter class, Exception message: "+str(e))
            logger.warning("Data loading is unsuccessfull.Exitted the get_data function of the DataGetter class")
            raise Exception()
        
        
# This is for testing purposes
if __name__ == "__main__":
    try:
        d = DataGetter()
        print(d.get_data())
    except Exception as e:
        print("Sorry there was an error in your code: "+str(e))
