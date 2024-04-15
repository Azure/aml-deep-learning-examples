import argparse
import yaml
import glob
import mlflow
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
import glob
import os


tenant_id = os.environ.get("TENANT_ID")
client_id = os.environ.get("CLIENT_ID")
client_secret = os.environ.get("CLIENT_SECRET")

credential = ClientSecretCredential(
    tenant_id=tenant_id,
    client_id=client_id,
    client_secret=client_secret,
)


parser = argparse.ArgumentParser("prep")
parser.add_argument("--yml_folder", type=str, help="Path to raw data")
parser.add_argument("--updated_yml", type=str, help="Path of prepped data")

filename = parser.parse_args().yml_folder
output_fn = parser.parse_args().updated_yml
aml_config_fn = filename + "/config.json"

yml_files = glob.glob(f"{filename}/*.yml")
if len(yml_files) != 1:
    raise ValueError("There should be exactly one YML file in the folder.")
filename = yml_files[0]

ml_client = MLClient.from_config(path = aml_config_fn ,credential=credential)
# ml_client = MLClient.from_config()
mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
print(mlflow_tracking_uri)

with mlflow.start_run():
    current_experiment_id = mlflow.active_run().info.experiment_id
    current_experiment = mlflow.get_experiment(current_experiment_id)
    current_experiment_name = current_experiment.name


with open(filename, 'r') as f:
    phi_ft_config = yaml.safe_load(f)

phi_ft_config["mlflow_tracking_uri"] = mlflow_tracking_uri
phi_ft_config["hf_mlflow_log_artifacts"] = False
phi_ft_config["mlflow_experiment_name"] = current_experiment_name
phi_ft_config["output_dir"] = "./outputs"

with open(output_fn, 'w') as f:
    yaml.dump(phi_ft_config, f)