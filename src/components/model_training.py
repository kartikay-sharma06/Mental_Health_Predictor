
import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass

from src.utils import save_object, evaluate_model
from src.components.hyperparameter_tunning import tune_models
from src.logger import logging
from src.exception import CustomException

from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score as r2_score_metric


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTraining:

    def __init__(self):
        self.model_training_config = ModelTrainerConfig()

    def Initiate_Model_training(self, train_array, test_array):

        try:
            logging.info(
                "Splitting dependent and independent variables "
                "from train and test data"
            )

            # Split features and target
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define models
            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "ElasticNet": ElasticNet(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "BaggingRegressor": BaggingRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor()
            }

            # Initial model evaluation
            logging.info("Starting initial model evaluation")

            initial_model_report = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            logging.info(
                f"Initial model evaluation completed: "
                f"{initial_model_report}"
            )

            # Hyperparameter tuning
            logging.info("Starting hyperparameter tuning")

            tuned_models, cv_model_report = tune_models(
                models=models,
                X_train=X_train,
                y_train=y_train
            )

            logging.info("Hyperparameter tuning completed")

            # Evaluate tuned models on test data
            model_report = {}

            logging.info(
                "Evaluating tuned models on test dataset"
            )

            for model_name, model in tuned_models.items():

                predicted = model.predict(X_test)

                score = r2_score_metric(y_test, predicted)

                model_report[model_name] = score

                logging.info(
                    f"{model_name} test R2 score: {score:.4f}"
                )

            # Check whether any tuned models were returned
            if not model_report:
                raise ValueError(
                    "Hyperparameter tuning did not return "
                    "any trained models."
                )

            # Select the best model using test R2 score
            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model = tuned_models[best_model_name]
            best_model_score = model_report[best_model_name]

            # Log best model details
            logging.info("=" * 60)
            logging.info("BEST MODEL SELECTED")
            logging.info(f"Best Model Name: {best_model_name}")

            if best_model_name in cv_model_report:
                logging.info(
                    f"Best CV R2 Score: "
                    f"{cv_model_report[best_model_name]:.4f}"
                )

            logging.info(
                f"Best Test R2 Score: {best_model_score:.4f}"
            )
            logging.info("=" * 60)

            # Validate model performance
            if best_model_score < 0.6:
                raise ValueError(
                    "No acceptable model found with an R2 score "
                    "of at least 0.6. "
                    f"Best model: {best_model_name}, "
                    f"R2 score: {best_model_score:.4f}"
                )

            # Create artifacts directory
            artifact_directory = os.path.dirname(
                self.model_training_config.trained_model_file_path
            )

            os.makedirs(
                artifact_directory or ".",
                exist_ok=True
            )

            # Save the best model
            save_object(
                file_path=(
                    self.model_training_config.trained_model_file_path
                ),
                obj=best_model
            )

            logging.info(
                f"Best Model Name: {best_model_name}"
            )

            logging.info(
                f"Best tuned model saved at "
                f"{self.model_training_config.trained_model_file_path}"
            )

            # Calculate final R2 score
            predicted = best_model.predict(X_test)

            final_r2_score = r2_score_metric(
                y_test,
                predicted
            )

            logging.info(
                f"Final R2 score: {final_r2_score:.4f}"
            )

            return final_r2_score

        except Exception as e:
            raise CustomException(e, sys.exc_info()) from e
