
import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    OrdinalEncoder,
    FunctionTransformer
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numeric_feature = [
                "Age",
                "Avg_Daily_Usage_Hours",
                "Daily_Unlocks",
                "Study_Hours",
                "Sleep_Hours_Per_Night"
            ]

            skewed_feature = [
                "Physical_Activity_Hours"
            ]

            ordinal_feature = [
                "Stress_Level"
            ]

            nominal_feature = [
                "Gender",
                "Country",
                "Academic_Level",
                "Most_Used_Platform",
                "Purpose_Of_Use"
            ]

            # Skewed numerical feature pipeline
            skew_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "log_transform",
                        FunctionTransformer(
                            np.log1p,
                            validate=False
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            # Regular numerical feature pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="mean")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            # Ordinal categorical feature pipeline
            ordinal_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "ordinal_encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1
                        )
                    )
                ]
            )

            # Nominal categorical feature pipeline
            nominal_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            )

            # Combine all preprocessing pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "skew",
                        skew_pipeline,
                        skewed_feature
                    ),
                    (
                        "num",
                        numerical_pipeline,
                        numeric_feature
                    ),
                    (
                        "ord",
                        ordinal_pipeline,
                        ordinal_feature
                    ),
                    (
                        "nom",
                        nominal_pipeline,
                        nominal_feature
                    )
                ],
                remainder="drop"
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys.exc_info())

    def initiate_data_trans(self, train_path, test_path):
        try:
            # Read train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Remove leading/trailing spaces from column names
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            logging.info(
                "Train and test data reading completed"
            )

            # Define target column
            target_feature = "Mental_Health_Score"

            # Validate target column
            if target_feature not in train_df.columns:
                raise ValueError(
                    f"Target '{target_feature}' not found in "
                    f"training data. Available columns: "
                    f"{train_df.columns.tolist()}"
                )

            if target_feature not in test_df.columns:
                raise ValueError(
                    f"Target '{target_feature}' not found in "
                    f"testing data. Available columns: "
                    f"{test_df.columns.tolist()}"
                )

            # Separate input and target features
            input_feature_train_df = train_df.drop(
                columns=[target_feature]
            )

            target_feature_train_df = train_df[
                target_feature
            ]

            input_feature_test_df = test_df.drop(
                columns=[target_feature]
            )

            target_feature_test_df = test_df[
                target_feature
            ]

            # Expected input columns
            expected_features = [
                "Age",
                "Gender",
                "Country",
                "Academic_Level",
                "Most_Used_Platform",
                "Purpose_Of_Use",
                "Avg_Daily_Usage_Hours",
                "Daily_Unlocks",
                "Study_Hours",
                "Physical_Activity_Hours",
                "Sleep_Hours_Per_Night",
                "Stress_Level"
            ]

            # Validate required input columns
            for column in expected_features:
                if column not in input_feature_train_df.columns:
                    raise ValueError(
                        f"Missing column '{column}' in training data"
                    )

                if column not in input_feature_test_df.columns:
                    raise ValueError(
                        f"Missing column '{column}' in testing data"
                    )

            # Ensure the same feature order in both datasets
            input_feature_train_df = (
                input_feature_train_df[expected_features]
            )

            input_feature_test_df = (
                input_feature_test_df[expected_features]
            )

            logging.info(
                f"Training input shape: "
                f"{input_feature_train_df.shape}"
            )

            logging.info(
                f"Training target shape: "
                f"{target_feature_train_df.shape}"
            )

            # Obtain preprocessing object
            preprocessor_obj = self.get_data_transformer_obj()

            logging.info(
                "Applying preprocessing to training data"
            )

            # Fit only on training data
            input_feature_train_arr = (
                preprocessor_obj.fit_transform(
                    input_feature_train_df
                )
            )

            logging.info(
                "Applying preprocessing to testing data"
            )

            # Transform test data using the fitted preprocessor
            input_feature_test_arr = (
                preprocessor_obj.transform(
                    input_feature_test_df
                )
            )

            # Convert sparse matrices to dense arrays if needed
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = (
                    input_feature_train_arr.toarray()
                )

            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = (
                    input_feature_test_arr.toarray()
                )

            # Convert targets to one-dimensional arrays
            y_train = target_feature_train_df.to_numpy().ravel()
            y_test = target_feature_test_df.to_numpy().ravel()

            # Validate row counts before combining
            if input_feature_train_arr.shape[0] != len(y_train):
                raise ValueError(
                    "Training row mismatch: transformed inputs have "
                    f"{input_feature_train_arr.shape[0]} rows, "
                    f"but target has {len(y_train)} rows."
                )

            if input_feature_test_arr.shape[0] != len(y_test):
                raise ValueError(
                    "Testing row mismatch: transformed inputs have "
                    f"{input_feature_test_arr.shape[0]} rows, "
                    f"but target has {len(y_test)} rows."
                )

            # Combine transformed features and target
            train_arr = np.column_stack(
                (input_feature_train_arr, y_train)
            )

            test_arr = np.column_stack(
                (input_feature_test_arr, y_test)
            )

            # Create artifacts directory
            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config
                    .preprocessor_obj_file_path
                ),
                exist_ok=True
            )

            # Save fitted preprocessor
            save_object(
                file_path=(
                    self.data_transformation_config
                    .preprocessor_obj_file_path
                ),
                obj=preprocessor_obj
            )

            logging.info(
                f"Transformed training array shape: {train_arr.shape}"
            )

            logging.info(
                f"Transformed testing array shape: {test_arr.shape}"
            )

            logging.info(
                "Data transformation completed successfully"
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config
                .preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys.exc_info())
