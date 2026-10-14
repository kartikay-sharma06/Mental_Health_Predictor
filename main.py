
# from src.components.data_ingestion import data_ingestion

# if __name__ == "__main__":
    
        # data_ingestion = data_ingestion()
        # train_data_path, test_data_path = (
        #     data_ingestion.initiate_data_ingestion()
        # )

        # print("Data ingestion completed successfully.")
        # print(f"Training data saved at: {train_data_path}")
        # print(f"Testing data saved at: {test_data_path}")

    

import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import data_ingestion
from src.components.data_transform import DataTransformation

if __name__=="__main__":
    obj=data_ingestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_trans=DataTransformation()
    train_array,test_array,preprocesssor_path=(data_trans.initiate_data_trans(train_data,test_data))