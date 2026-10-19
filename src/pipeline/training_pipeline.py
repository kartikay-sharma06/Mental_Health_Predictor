
import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import data_ingestion
from src.components.data_transform import DataTransformation
from src.components.model_training import ModelTraining


class TrainPipeline:

    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")

            # Step 1: Data Ingestion
            logging.info("Starting data ingestion")

            ingestion_obj = data_ingestion()

            train_path, test_path = (
                ingestion_obj.initiate_data_ingestion()
            )

            logging.info("Data ingestion completed successfully")

            # Step 2: Data Transformation
            logging.info("Starting data transformation")

            data_transformation = DataTransformation()

            train_array, test_array, preprocessor_path = (
                data_transformation.initiate_data_trans(
                    train_path,
                    test_path
                )
            )

            logging.info("Data transformation completed successfully")
            logging.info(
                f"Preprocessor saved at: {preprocessor_path}"
            )

            # Step 3: Model Training
            logging.info("Starting model training")

            model_trainer = ModelTraining()

            final_r2_score = model_trainer.Initiate_Model_training(
                train_array,
                test_array
            )

            logging.info("Training pipeline completed successfully")
            logging.info(
                f"Final R2 score: {final_r2_score:.4f}"
            )

            return final_r2_score

        except Exception as e:
            logging.exception("Training pipeline failed")
            raise CustomException(e, sys.exc_info()) from e
