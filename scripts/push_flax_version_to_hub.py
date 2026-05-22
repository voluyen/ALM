from dataclasses import dataclass
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer, FlaxAutoModelForCausalLM
import json
import jax
import shutil

from tokenkit.models import sharding
from tokenkit.hf import get_config


@dataclass
class Args:
    model_name_or_path: str = "Qwen/Qwen2-0.5B"
    hub_user: str = "benjamin"
    extra_args: str | None = None  # e.g. '{"attention_bias": true, "max_length": 8192}'
    use_cpu: bool = False
    tmp_path: str = "/tmp/push_flax_model"


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

    config = get_config(args.tmp_path)
    for key, value in json.loads(args.extra_args or "{}").items():
        setattr(config, key, value)
    config.mesh = mesh

    flax_model = FlaxAutoModelForCausalLM.from_pretrained(args.tmp_path, config=config)
    model_name = args.hub_user + "/" + args.model_name_or_path.split("/")[-1] + "-flax"

    del config.mesh

    flax_model.push_to_hub(model_name, private=True, safe_serialization=False)
    tokenizer.push_to_hub(model_name, private=True)
