# Logging your trained model to the AzureML MLFlow registry

In this section, you will build your first AzureML pipeline using components. Components allow you to effectively collaborate with others, by creating a flow of information composed of components, where each component can be developed/optimized/troubleshooted by different parties. The pipeline you will produce will perform three steps:

1. You will modify the Axolotl YML file in a Pipelinestep, by adding your MLFlow URI to the YML configuration. This way, Axolotl will report metrics to the pipeline.
2. It will finetune a model, and save the model to the `./outputs` folder.
3. A second component will then load the model to the AzureML MLFlow registry.

Decoupling operations this way will help you build complicated pipelines composed of multiple steps. You can find more information anout [pipelines](https://learn.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines?view=azureml-api-2) on the documentations website.
Saving your models in the registry will ease their deployment for inference at a later stage.


## Goal:

Build an AzureML pipeline that finetunes a model, then saves the model to MLFlow.


### Running the pipeline

To run the pipeline, rely on the `az ml job create --file pipeline.yml`. Read the documentation to learn how to install the `ml` extension on your machine, and make sure you are properly logged in using `az login` if required.
Before starting the pipeline, make sure to review:

1. That you have the compute required and referred to in the YML files. The compute must be created.
2. You must retrieve the version of your curated environment (axolotl_acpt) and use it in the appropriate YML files.


### Contents

The pipeline is composed of two steps, each defined in its own YML file. the pipeline is collectively defined in `pipeline.yml`.