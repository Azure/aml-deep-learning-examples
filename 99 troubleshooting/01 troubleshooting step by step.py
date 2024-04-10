import yaml
import axolotl.cli.train as train_main
import fire

config = '99 troubleshooting/src/phi-ft-modified.yml'

# with open('99 troubleshooting/src/phi-ft-modified.yml', 'r') as file:
#     config = yaml.safe_load(file)

fire.Fire(train_main.do_cli(config))