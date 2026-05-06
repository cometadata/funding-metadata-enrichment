# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "tinker",
#     "tinker-cookbook",
#     "chz",
# ]
# ///
"""RL training recipe for funding extraction using GRPO."""

import asyncio
import logging
import sys
from datetime import datetime

try:
    import chz

    from tinker_cookbook import checkpoint_utils, cli_utils, model_info
    from tinker_cookbook.rl import train

    from funding_reward import FundingDatasetBuilder

    _HAS_TINKER = True
except ImportError:
    _HAS_TINKER = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if _HAS_TINKER:

    @chz.chz
    class CLIConfig:
        # Model
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
        renderer_name: str = ""  # auto-detected from model_name
        lora_rank: int = 128

        # Training
        learning_rate: float = 3e-5
        max_tokens: int = 4096
        temperature: float = 0.8
        batch_size: int = 16
        group_size: int = 8
        max_steps: int | None = None

        # KL penalty against SFT reference
        kl_penalty_coef: float = 0.03
        kl_reference_model: str = ""  # defaults to model_name
        kl_reference_checkpoint: str | None = None  # defaults to load_checkpoint_path

        # Data
        train_path: str = "rl_train.jsonl"
        train_synthetic_path: str = "rl_synthetic.jsonl"
        eval_path: str = "rl_eval_train.jsonl"
        eval_synthetic_path: str = "rl_eval_synthetic.jsonl"
        seed: int = 42

        # Logging & checkpointing
        log_path: str = ""  # auto-generated if empty
        eval_every: int = 10
        save_every: int = 10
        wandb_project: str | None = None
        wandb_name: str | None = None

        # Resume
        load_checkpoint_path: str | None = None
        behavior_if_log_dir_exists: str = "ask"

        # Reward shaping
        w_funder: float = 0.50
        w_id_hier: float = 0.40
        flat_id_weight: float = 0.10
        soft_id_matching: bool = True

    def build_train_config(cli: CLIConfig) -> train.Config:
        log_path = cli.log_path or f"/tmp/tinker-funding-rl/{datetime.now():%Y%m%d_%H%M%S}"
        renderer_name = cli.renderer_name or model_info.get_recommended_renderer_name(cli.model_name)

        dataset_builder = FundingDatasetBuilder(
            batch_size=cli.batch_size,
            group_size=cli.group_size,
            model_name_for_tokenizer=cli.model_name,
            renderer_name=renderer_name,
            train_path=cli.train_path,
            train_synthetic_path=cli.train_synthetic_path,
            eval_path=cli.eval_path,
            eval_synthetic_path=cli.eval_synthetic_path,
            seed=cli.seed,
            w_funder=cli.w_funder,
            w_id_hier=cli.w_id_hier,
            flat_id_weight=cli.flat_id_weight,
            soft_id_matching=cli.soft_id_matching,
        )

        # KL reference: use the SFT checkpoint as anchor
        # Note: KL reference needs a sampler_weights path, not a weights path
        kl_reference_config = None
        if cli.kl_penalty_coef > 0:
            kl_ref_path = cli.kl_reference_checkpoint
            if not kl_ref_path and cli.load_checkpoint_path:
                # Convert weights/ path to sampler_weights/ path
                kl_ref_path = cli.load_checkpoint_path.replace("/weights/", "/sampler_weights/")
            kl_reference_config = train.KLReferenceConfig(
                base_model=cli.kl_reference_model or cli.model_name,
                load_checkpoint_path=kl_ref_path,
            )
            logger.info("KL penalty: coef=%s, reference=%s", cli.kl_penalty_coef,
                        kl_reference_config.load_checkpoint_path or kl_reference_config.base_model)

        return train.Config(
            model_name=cli.model_name,
            renderer_name=renderer_name,
            log_path=log_path,
            dataset_builder=dataset_builder,
            learning_rate=cli.learning_rate,
            max_tokens=cli.max_tokens,
            temperature=cli.temperature,
            lora_rank=cli.lora_rank,
            loss_fn="importance_sampling",
            eval_every=cli.eval_every,
            save_every=cli.save_every,
            max_steps=cli.max_steps,
            load_checkpoint_path=cli.load_checkpoint_path,
            kl_penalty_coef=cli.kl_penalty_coef,
            kl_reference_config=kl_reference_config,
            wandb_project=cli.wandb_project,
            wandb_name=cli.wandb_name,
        )

    async def main(cli_config: CLIConfig):
        config = build_train_config(cli_config)
        cli_utils.check_log_dir(config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
        logger.info("Starting funding RL training → %s", config.log_path)
        await train.main(config)

    if __name__ == "__main__":
        blueprint = chz.Blueprint(CLIConfig)
        cli_config = blueprint.make_from_argv(sys.argv[1:], allow_hyphens=True)
        asyncio.run(main(cli_config))
