import mlflow
import time
# from azure.ai.ml import MLClient
# from azure.identity import DefaultAzureCredential
# ml_client = MLClient.from_config(credential=DefaultAzureCredential())
# mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
# print(mlflow_tracking_uri)

# uri = mlflow_tracking_uri + "/Transformers_test/versions/1"
# uri = "models:/Transformers_test/1"
uri = "runs:/e82e8be3-5a82-4d22-9bc9-43680e2a429c/src"
loaded = mlflow.pyfunc.load_model(uri)
predicted = loaded.predict("The fox told Setu that Azure ML is")
print(predicted)
time.sleep(1)