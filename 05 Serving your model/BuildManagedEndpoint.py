from mlflow.tracking import MlflowClient
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
import mlflow
import json
import os

tenant_id = os.environ.get("TENANT_ID")
client_id = os.environ.get("CLIENT_ID")
client_secret = os.environ.get("CLIENT_SECRET")
subscription_id = os.environ.get("SUBSCRIPTION_ID")

credential = ClientSecretCredential(
    tenant_id=tenant_id,
    client_id=client_id,
    client_secret=client_secret,
)

ml_client = MLClient.from_config(path="./config.json", credential=credential)

azureml_tracking_uri = ml_client.workspaces.get(
    ml_client.workspace_name
).mlflow_tracking_uri
mlflow.set_tracking_uri(azureml_tracking_uri)

print(azureml_tracking_uri)

import random
import string

# Creating a unique endpoint name by including a random suffix
allowed_chars = string.ascii_lowercase + string.digits
endpoint_suffix = "".join(random.choice(allowed_chars) for x in range(5))
endpoint_name = "heart-classifier-" + endpoint_suffix

print(f"Endpoint name: {endpoint_name}")

from mlflow.deployments import get_deploy_client

deployment_client = get_deploy_client(mlflow.get_tracking_uri())

endpoint = deployment_client.create_endpoint(endpoint_name)

scoring_uri = deployment_client.get_endpoint(endpoint=endpoint_name)["properties"][
    "scoringUri"
]
print(scoring_uri)

deployment_name = "default"

deploy_config = {
    "instance_type": "Standard_NC24ads_A100_v4",
    "instance_count": 1,
}

# uri = "runs:/e82e8be3-5a82-4d22-9bc9-43680e2a429c/src"
uri ="runs:/9bba4ef0-8129-45fa-b408-974e3544373f/src"

deployment_config_path = "deployment_config.json"
with open(deployment_config_path, "w") as outfile:
    outfile.write(json.dumps(deploy_config))

deployment = deployment_client.create_deployment(
    name=deployment_name,
    endpoint=endpoint_name,
    model_uri=uri,
    config={"deploy-config-file": deployment_config_path},
)

traffic_config = {"traffic": {deployment_name: 100}}
traffic_config_path = "traffic_config.json"
with open(traffic_config_path, "w") as outfile:
    outfile.write(json.dumps(traffic_config))

deployment_client.update_endpoint(
    endpoint=endpoint_name,
    config={"endpoint-config-file": traffic_config_path},
)