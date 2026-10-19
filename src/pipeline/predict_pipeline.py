
import os
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Model and preprocessor paths
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            # Load trained model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Transform input features
            data_scaled = preprocessor.transform(features)

            # Generate predictions
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys.exc_info()) from e


class CustomData:

    def __init__(
        self,
        Age,
        Gender,
        Country,
        Academic_Level,
        Most_Used_Platform,
        Purpose_Of_Use,
        Avg_Daily_Usage_Hours,
        Daily_Unlocks,
        Study_Hours,
        Physical_Activity_Hours,
        Sleep_Hours_Per_Night,
        Stress_Level
    ):
        self.Age = Age
        self.Gender = Gender
        self.Country = Country
        self.Academic_Level = Academic_Level
        self.Most_Used_Platform = Most_Used_Platform
        self.Purpose_Of_Use = Purpose_Of_Use
        self.Avg_Daily_Usage_Hours = Avg_Daily_Usage_Hours
        self.Daily_Unlocks = Daily_Unlocks
        self.Study_Hours = Study_Hours
        self.Physical_Activity_Hours = Physical_Activity_Hours
        self.Sleep_Hours_Per_Night = Sleep_Hours_Per_Night
        self.Stress_Level = Stress_Level

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Gender": [self.Gender],
                "Country": [self.Country],
                "Academic_Level": [self.Academic_Level],
                "Most_Used_Platform": [self.Most_Used_Platform],
                "Purpose_Of_Use": [self.Purpose_Of_Use],
                "Avg_Daily_Usage_Hours": [
                    self.Avg_Daily_Usage_Hours
                ],
                "Daily_Unlocks": [self.Daily_Unlocks],
                "Study_Hours": [self.Study_Hours],
                "Physical_Activity_Hours": [
                    self.Physical_Activity_Hours
                ],
                "Sleep_Hours_Per_Night": [
                    self.Sleep_Hours_Per_Night
                ],
                "Stress_Level": [self.Stress_Level]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys.exc_info()) from e
