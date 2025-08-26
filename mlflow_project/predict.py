
import mlflow
import pandas as pd
from sklearn.datasets import load_boston

model = mlflow.sklearn.load_model("runs:/<RUN_ID>/model")

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)

predictions = model.predict(X[:5])
print("Predictions:", predictions)
