
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
import argparse

def main(alpha):
    mlflow.set_experiment("BostonHousingRegression")

    with mlflow.start_run():
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        y = pd.Series(boston.target)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")

        print(f"Model logged with MSE: {mse}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    main(args.alpha)
