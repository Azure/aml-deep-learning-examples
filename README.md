# Azure ML Deep Learning Examples

The internet is full of tutorials that typically fail at building a complete view of a  training operation, specially with regards to operationalization of components, and usage of real, non-fictional data. This repo includes examples for:

- Building a custom Docker environment and compute cluster to support finetuning on A100 and using a training library (in this case, Axolotl)
- Fine tuning a model stored ono HuggingFace, and saving the finetuned model in the Model repository
- Creating a PromptFlow eval, and evaluate the model against public benchmarks

When added to a pipeline, it will be possible for customers to use the above to experiment with different configurations / models / datasets, to determine which of the many models is the best performing on their datasets.
These notebooks are built to be run in sequential order. In each notebook, you will gradually add more features until ultimately you will have an OSS finetuning capability in your AML workspace. 

## How to run these notebooks and scripts

These notebooks and scripts will modify your Azure ML environment. Therefore, we assume you have already created an Azure ML workspace, and are familiar with the different methods to modify your Azure ML environment, such as the SDK or the CLI.


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
