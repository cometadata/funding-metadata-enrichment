# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "tinker",
#     "tinker-cookbook",
#     "chz",
# ]
# ///

import sys
import json
import asyncio
import logging
import random

import chz
import datasets

from tinker_cookbook import cli_utils, hyperparam_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import renderers as renderers_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from funding_reward import SYSTEM_PROMPT, USER_TEMPLATE


def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _example_to_messages(example: dict) -> list[dict]:
    completion = json.dumps(example.get("funders", []), indent=2, ensure_ascii=False)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(funding_statement=example["funding_statement"])},
        {"role": "assistant", "content": completion},
    ]


@chz.chz
class FundingSFTDatasetBuilder(SupervisedDatasetBuilder):
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    renderer_name: str = ""
    train_path: str = "train.jsonl"
    synthetic_path: str = "synthetic.jsonl"
    synthetic_upsample: int = 2
    max_length: int = 4096
    batch_size: int = 128
    eval_fraction: float = 0.1
    seed: int = 42

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        tokenizer = get_tokenizer(self.model_name)
        renderer = renderers_module.get_renderer(renderer_name, tokenizer)

        train_data = _load_jsonl(self.train_path)
        synthetic_data = _load_jsonl(self.synthetic_path)
        logger.info(
            "Loaded %d train + %d synthetic examples (upsample %dx → %d synthetic)",
            len(train_data),
            len(synthetic_data),
            self.synthetic_upsample,
            len(synthetic_data) * self.synthetic_upsample,
        )
        data = train_data + synthetic_data * self.synthetic_upsample

        rng = random.Random(self.seed)
        rng.shuffle(data)

        conversations = [{"messages": _example_to_messages(ex)} for ex in data]

        split = int(len(conversations) * (1 - self.eval_fraction))
        train_convos = conversations[:split]
        eval_convos = conversations[split:]

        logger.info("SFT dataset: %d train, %d eval examples", len(train_convos), len(eval_convos))

        train_hf = datasets.Dataset.from_list(train_convos)
        eval_hf = datasets.Dataset.from_list(eval_convos)

        def map_fn(row):
            return conversation_to_datum(
                row["messages"],
                renderer,
                self.max_length,
                TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )

        train_ds = SupervisedDatasetFromHFDataset(train_hf, self.batch_size, map_fn=map_fn)
        eval_ds = SupervisedDatasetFromHFDataset(eval_hf, self.batch_size, map_fn=map_fn)

        return train_ds, eval_ds


@chz.chz
class CLIConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    renderer_name: str = ""
    lora_rank: int = 128
    learning_rate: float = 0.0
    num_epochs: int = 2
    batch_size: int = 128
    max_length: int = 4096
    train_path: str = "train.jsonl"
    synthetic_path: str = "synthetic.jsonl"
    synthetic_upsample: int = 2
    eval_fraction: float = 0.1
    seed: int = 42
    log_path: str = ""
    eval_every: int = 50
    save_every: int = 50
    wandb_project: str | None = None
    wandb_name: str | None = None
    load_checkpoint_path: str | None = None
    behavior_if_log_dir_exists: str = "ask"


def build_train_config(cli: CLIConfig) -> train.Config:
    from datetime import datetime
    log_path = cli.log_path or f"/tmp/tinker-funding-sft-llama/{datetime.now():%Y%m%d_%H%M%S}"
    renderer_name = cli.renderer_name or model_info.get_recommended_renderer_name(cli.model_name)
    learning_rate = cli.learning_rate or hyperparam_utils.get_lr(cli.model_name, is_lora=True)

    dataset_builder = FundingSFTDatasetBuilder(
        model_name=cli.model_name,
        renderer_name=renderer_name,
        train_path=cli.train_path,
        synthetic_path=cli.synthetic_path,
        synthetic_upsample=cli.synthetic_upsample,
        max_length=cli.max_length,
        batch_size=cli.batch_size,
        eval_fraction=cli.eval_fraction,
        seed=cli.seed,
    )

    logger.info("Learning rate: %s (auto=%s)", learning_rate, cli.learning_rate == 0.0)

    return train.Config(
        model_name=cli.model_name,
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=learning_rate,
        num_epochs=cli.num_epochs,
        lr_schedule="linear",
        lora_rank=cli.lora_rank,
        eval_every=cli.eval_every,
        save_every=cli.save_every,
        rolling_save_every=20,
        rolling_ttl_seconds=7200,
        load_checkpoint_path=cli.load_checkpoint_path,
        wandb_project=cli.wandb_project,
        wandb_name=cli.wandb_name,
    )


async def main(cli_config: CLIConfig):
    config = build_train_config(cli_config)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    logger.info("Starting funding SFT (Llama 3.1 8B) → %s", config.log_path)
    await train.main(config)


if __name__ == "__main__":
    blueprint = chz.Blueprint(CLIConfig)
    cli_config = blueprint.make_from_argv(sys.argv[1:], allow_hyphens=True)
    asyncio.run(main(cli_config))
