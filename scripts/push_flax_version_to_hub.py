from dataclasses import dataclass
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer
import json
import jax
import transformers
import shutil

from tokenkit.models import sharding

DEFAULT_TMP_PATH = "/mnt/disks/persist/tmp/model"

@dataclass
class Args:
    model_name_or_path: str = "Qwen/Qwen2-0.5B"
    hub_user: str = "benjamin"
    model_class: str = "Llama"
    extra_args: str | None = None # for Qwen2: "{\"attention_bias\": true, \"max_length\": 8192}", for Llama3: "{\"max_length\": 8192}"
    use_cpu: bool = False
    output_dir: str | None = None # if set, save the Flax model locally here instead of pushing to the hub
    tmp_path: str = DEFAULT_TMP_PATH # staging dir for the intermediate PyTorch checkpoint


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    print(args)

    if args.use_cpu:
        jax.config.update('jax_default_device', jax.devices('cpu')[0])
        mesh = sharding.get_mesh(devices=jax.devices("cpu"))
    else:
        mesh = sharding.get_mesh()

    shutil.rmtree(args.tmp_path, ignore_errors=True)
    AutoModelForCausalLM.from_pretrained(args.model_name_or_path).save_pretrained(
        args.tmp_path, max_shard_size="100GB"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config_class = getattr(transformers, args.model_class + "Config")
    if hasattr(transformers, "Flax" + args.model_class + "ForCausalLM"):
        model_class = getattr(transformers, "Flax" + args.model_class + "ForCausalLM")
    elif hasattr(transformers, "Flax" + args.model_class +  "LMHeadModel"):
        model_class = getattr(transformers, "Flax" + args.model_class +  "LMHeadModel")
    else:
        raise ValueError(f"Model class '{args.model_class}' not found")

    config = config_class.from_pretrained(args.tmp_path, args.model_name_or_path)
    for key, value in json.loads(args.extra_args or "{}").items():
        setattr(config, key, value)

    config.mesh = mesh

    flax_model = model_class.from_pretrained(args.tmp_path, config=config)

    del config.mesh

    if args.output_dir is not None:
        flax_model.save_pretrained(args.output_dir, safe_serialization=False)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved Flax model locally to: {args.output_dir}")
    else:
        model_name = (
            args.hub_user + "/" + args.model_name_or_path.split("/")[-1] + "-flax"
        )
        flax_model.push_to_hub(model_name, private=True, safe_serialization=False)
        tokenizer.push_to_hub(model_name, private=True)