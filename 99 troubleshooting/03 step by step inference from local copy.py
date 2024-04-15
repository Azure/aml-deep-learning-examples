import importlib
from axolotl.cli import (
    do_inference,
    do_inference_gradio,
    load_cfg,
    print_axolotl_text_art,
)
import torch
import transformers
from axolotl.common.cli import TrainerCliArgs
from axolotl.common.cli import TrainerCliArgs, load_model_and_tokenizer
from transformers import GenerationConfig, TextIteratorStreamer, TextStreamer

config = '99 troubleshooting/src/phi-ft-modified.yml'

def do_cli(config, gradio=False, **kwargs):
    # pylint: disable=duplicate-code
    parsed_cfg = load_cfg(config, **kwargs)
    parsed_cfg.sample_packing = False
    parser = transformers.HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    parsed_cli_args.inference = True
    do_inference(cfg=parsed_cfg, cli_args=parsed_cli_args, prompt="The cat is")


def do_inference(*, cfg, cli_args, prompt):
    model, tokenizer = load_model_and_tokenizer(cfg=cfg, cli_args=cli_args)
    prompter = cli_args.prompter
    default_tokens = {"unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"}

    for token, symbol in default_tokens.items():
        # If the token isn't already specified in the config, add it
        if not (cfg.special_tokens and token in cfg.special_tokens):
            tokenizer.add_special_tokens({token: symbol})

    prompter_module = None
    if prompter:
        prompter_module = getattr(
            importlib.import_module("axolotl.prompters"), prompter
        )

    model = model.to(cfg.device, dtype=cfg.torch_dtype)


    print("=" * 80)
    # support for multiline inputs
    instruction = prompt
    if not instruction:
        return
    if prompter_module:
        prompt: str = next(
            prompter_module().build_prompt(instruction=instruction.strip("\n"))
        )
    else:
        prompt = instruction.strip()
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    print("=" * 40)
    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=1.1,
            max_new_tokens=1024,
            temperature=0.9,
            top_p=0.95,
            top_k=40,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            use_cache=False,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )
        streamer = TextStreamer(tokenizer)
        generated = model.generate(
            inputs=batch["input_ids"].to(cfg.device),
            generation_config=generation_config,
            streamer=streamer,
        )
    print("=" * 40)
    print(tokenizer.decode(generated["sequences"].cpu().tolist()[0]))


do_cli(config)