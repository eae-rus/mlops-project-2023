import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import boto3
import os
import mlflow
from mlflow.models.signature import infer_signature

os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.0:9000'
os.environ['MLflow_TRACKING_USERNAME'] = 'MLflow-user'

mlflow.set_tracking_uri("http://0.0.0.0:5000")  # need to change IP adress
mlflow.set_experiment('lgbm')
mlflow.lightgbm.autolog()

data = pd.read_csv('/home/apollo/projects/mlops-project-2023/data/interim/data.csv')
useful_data = data[data['is_use_for_ml'] == 1].copy()

X_train = useful_data[useful_data['bus'] == 1][['current_phase_a',	'current_phase_c',	'voltage_phase_a',	'voltage_phase_b',	'voltage_phase_c',	'voltage_3U0']].copy()
y_train = useful_data[useful_data['bus'] == 1]['ml_anomaly'].values
X_test = useful_data[useful_data['bus'] == 2][['current_phase_a',	'current_phase_c',	'voltage_phase_a',	'voltage_phase_b',	'voltage_phase_c',	'voltage_3U0']].copy()
y_test = useful_data[useful_data['bus'] == 2]['ml_anomaly'].values

model = LGBMClassifier()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
signature = infer_signature(X_test, prediction)
# mlflow.lightgbm.log_model(model, "signals", signature=signature)
autolog_run = mlflow.last_active_run()
