"""
Trains a model and logs the transformers model and tokenizer to MLFlow
"""
import logging
logging.basicConfig(level=logging.DEBUG)
from pathlib import Path
from typing import Tuple, Union
import argparse
import yaml



import fire
from transformers.hf_argparser import HfArgumentParser
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from axolotl.cli import (
    check_accelerate_default_config,
    check_user_token,
    load_cfg,
    load_datasets,
    load_rl_datasets,
    print_axolotl_text_art,
)

from axolotl.prompt_strategies.sharegpt import register_chatml_template
from axolotl.common.cli import TrainerCliArgs, load_model_and_tokenizer

LOG = logging.getLogger("axolotl.cli.train")

parser = argparse.ArgumentParser("prep")
parser.add_argument("--model_weights", type=str, help="Path to Model Weights of finetuned model")
parser.add_argument("--config_yml", type=str, help="Path to config file")
parser.add_argument("--run_id", type=str, help="Name of the file on which the runID is written")

weights = parser.parse_args().model_weights
config_fn = parser.parse_args().config_yml
run_id = parser.parse_args().run_id

with open(config_fn, 'r') as f:
    phi_ft_config = yaml.safe_load(f)

phi_ft_config["output_dir"] = weights

temp_fn = "./temp_config.yml"

with open(temp_fn, 'w') as f:
    yaml.dump(phi_ft_config, f)

def do_cli(config= temp_fn, **kwargs):
    # pylint: disable=duplicate-code
    parsed_cfg = load_cfg(config, **kwargs)
    parser = HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    return do_train(parsed_cfg, parsed_cli_args)


def do_train(cfg, cli_args) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    print_axolotl_text_art()
    check_accelerate_default_config()
    check_user_token()
    if cfg.chat_template == "chatml" and cfg.default_system_message:
        LOG.info(
            f"ChatML set. Adding default system message: {cfg.default_system_message}"
        )
        register_chatml_template(cfg.default_system_message)
    else:
        register_chatml_template()

    if cfg.rl and cfg.rl != "orpo":
        dataset_meta = load_rl_datasets(cfg=cfg, cli_args=cli_args)
    else:
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

    return register(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)


def register( *, cfg, cli_args: TrainerCliArgs, dataset_meta):
    import mlflow
    from mlflow.utils.environment import _mlflow_conda_env
    model, tokenizer = load_model_and_tokenizer(cfg=cfg, cli_args=cli_args)
    extra_config = _mlflow_conda_env(
        additional_pip_deps=["packaging"]
    )
    components = {
        "model": model,
        "tokenizer": tokenizer,
    }
    with mlflow.start_run():
        out = mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path="src",
            registered_model_name="Transformers_test",
            task="text-generation",
            input_example="The cat sat on the mat",
            #extra_pip_requirements=["packaging==23.0"]
        )
    
    loaded = mlflow.pyfunc.load_model(out.model_uri)
    val = loaded.predict("The cat sat on the mat")
    print(val)
    print(f"Model logged to {out.model_uri}")
    with open(run_id, 'w') as f:
        f.write(out.model_uri)
    print(f"Done - configuration saved on {run_id}")


if __name__ == "__main__":
    fire.Fire(do_cli)