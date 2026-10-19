
import sys

from sklearn.model_selection import GridSearchCV, KFold

from src.exception import CustomException
from src.logger import logging


def get_hyperparameter_grids():
    params = {
        "LinearRegression": {},

        "Lasso": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "max_iter": [1000, 5000, 10000]
        },

        "Ridge": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        },

        "KNeighborsRegressor": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "p": [1, 2]
        },

        "ElasticNet": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.5, 0.7, 0.9],
            "max_iter": [1000, 5000, 10000]
        },

        "DecisionTreeRegressor": {
            "criterion": ["squared_error", "absolute_error"],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },

        "RandomForestRegressor": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        },

        "XGBRegressor": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        },

        "CatBoostRegressor": {
            "iterations": [100, 200],
            "learning_rate": [0.03, 0.1],
            "depth": [4, 6]
        },

        "AdaBoostRegressor": {
            "n_estimators": [50, 100],
            "learning_rate": [0.05, 0.1, 1.0],
            "loss": ["linear", "square", "exponential"]
        },

        "BaggingRegressor": {
            "n_estimators": [10, 50, 100],
            "max_samples": [0.5, 0.8, 1.0],
            "max_features": [0.5, 0.8, 1.0]
        },

        "GradientBoostingRegressor": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
            "subsample": [0.8, 1.0]
        }
    }

    return params


def tune_models(models, X_train, y_train):
    try:
        params = get_hyperparameter_grids()

        tuned_models = {}
        model_report = {}

        logging.info("Starting hyperparameter tuning")

        # Use a consistent cross-validation strategy
        cv_strategy = KFold(
            n_splits=3,
            shuffle=True,
            random_state=42
        )

        for model_name, model in models.items():

            if model_name not in params:
                logging.warning(
                    f"No hyperparameter grid found for {model_name}. "
                    "Skipping model."
                )
                continue

            logging.info(
                f"Hyperparameter tuning started for {model_name}"
            )

            try:
                # LinearRegression has no hyperparameters to tune
                if not params[model_name]:
                    model.fit(X_train, y_train)

                    tuned_models[model_name] = model
                    model_report[model_name] = None

                    logging.info(
                        f"{model_name} fitted without hyperparameter tuning"
                    )

                    continue

                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=params[model_name],
                    scoring="r2",
                    cv=cv_strategy,
                    n_jobs=-1,
                    verbose=0,
                    error_score="raise",
                    refit=True
                )

                grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_

                tuned_models[model_name] = best_model
                model_report[model_name] = grid_search.best_score_

                logging.info(
                    f"{model_name} best parameters: "
                    f"{grid_search.best_params_}"
                )

                logging.info(
                    f"{model_name} best CV R2 score: "
                    f"{grid_search.best_score_:.4f}"
                )

            except Exception as model_error:
                logging.exception(
                    f"Hyperparameter tuning failed for {model_name}: "
                    f"{model_error}"
                )
                continue

        if not tuned_models:
            raise ValueError(
                "No models were successfully trained during "
                "hyperparameter tuning."
            )

        logging.info("Hyperparameter tuning completed")

        return tuned_models, model_report

    except Exception as e:
        raise CustomException(e, sys.exc_info()) from e
