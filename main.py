
import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import data_ingestion
from src.components.data_transform import DataTransformation
from src.components.model_training import ModelTraining


if __name__ == "__main__":
    try:
        logging.info("Machine Learning Pipeline Started")

        # Step 1: Data Ingestion
        logging.info("Starting Data Ingestion")

        obj = data_ingestion()

        train_data, test_data = obj.initiate_data_ingestion()

        logging.info("Data Ingestion Completed Successfully")
        logging.info(f"Training data path: {train_data}")
        logging.info(f"Testing data path: {test_data}")

        # Step 2: Data Transformation
        logging.info("Starting Data Transformation")

        data_trans = DataTransformation()

        train_array, test_array, preprocessor_path = (
            data_trans.initiate_data_trans(
                train_data,
                test_data
            )
        )

        logging.info("Data Transformation Completed Successfully")
        logging.info(f"Preprocessor saved at: {preprocessor_path}")

        # Step 3: Model Training and Hyperparameter Tuning
        logging.info("Starting Model Training")

        modeltrainer = ModelTraining()

        final_r2_score = modeltrainer.Initiate_Model_training(
            train_array,
            test_array
        )

        logging.info("Model Training Completed Successfully")
        logging.info(f"Final Test R2 Score: {final_r2_score:.4f}")

        print("\nPipeline completed successfully.")
        print(f"Final Test R2 Score: {final_r2_score:.4f}")
        print(f"Preprocessor path: {preprocessor_path}")
        print("Trained model path: artifacts/model.pkl")

    except Exception as e:
        logging.exception("Machine Learning Pipeline Failed")
        raise CustomException(e, sys.exc_info()) from e
