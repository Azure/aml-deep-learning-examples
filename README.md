# Azure ML Deep Learning Examples

The Azure ML Deep Learning Examples repository is a comprehensive collection of tutorials and examples that provide a complete view of training operations and operationalization of components. It offers practical guidance on Finetuning a model progressively, by demonstrating the individual configurations and features along the way. By following the sequential order of the notebooks, users can gradually enhance their AML workspace with OSS finetuning capabilities. The repository assumes familiarity with Azure ML and provides instructions on how to run the notebooks and scripts. Contributions and suggestions are welcome, subject to the Contributor License Agreement. The project adheres to the Microsoft Open Source Code of Conduct and respects trademark and brand guidelines.

The notebooks represent a journey of configuration, each one progressively building on the skills you gain from previous ones:

1. Build a custom Docker environment to run our finetuning exercise. We will leverage a library called [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) to finetune the models. We will install Axolotl on [Azure Container for PyTorch (ACPT)](https://learn.microsoft.com/en-us/azure/machine-learning/resource-azure-container-for-pytorch?view=azureml-api-2).
2. Test the newly created environment, by running the Docker container on a newly created compute cluster, and launching a Python script to run basic diagnostics. The goal of this notebook is to demonstrate how to run your custom code on the curated environment you created in the previous step.
3. Axolotl allows you to finetune a wide variety of models on a broad range of datasets (on HuggingFace, or your own). In this example, you will learn how to submit your first finetune job. You will use the curated environment you've created above, to launch Axolotl, and integrate it with AzureML MLFlow to track your experiment's performance.


When added to a pipeline, it will be possible for customers to use the above to experiment with different configurations / models / datasets, to determine which of the many models is the best performing on their datasets.
These notebooks are built to be run in sequential order. In each notebook, you will gradually add more features until ultimately you will have an OSS finetuning capability in your AML workspace. 


## How to run these notebooks and scripts

These notebooks and scripts will modify your Azure ML environment. Therefore, we assume you have already created an Azure ML workspace, and are familiar with the different methods to modify your Azure ML environment, such as the SDK or the CLI. You are better off creating a compute instance on Azure to perform these steps in a Jupyter notebook. You do not need a GPU for this developer workstation.
Notebooks 3 forward will require a compute cluster (composed of 1 node) with a GPU.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
