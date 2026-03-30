# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "tinker",
#     "tinker-cookbook",
#     "rapidfuzz>=3.0.0",
#     "ftfy>=6.0",
#     "scipy>=1.11.0",
#     "numpy>=1.24.0",
#     "chz",
# ]
# ///

from __future__ import annotations

import re
import json
import logging
import random
from dataclasses import dataclass
from functools import partial
from typing import List, Sequence

try:
    import chz

    from tinker_cookbook import model_info, renderers, tokenizer_utils
    from tinker_cookbook.renderers import Renderer, get_renderer
    from tinker_cookbook.rl.types import (
        Action,
        ActionExtra,
        Env,
        EnvGroupBuilder,
        Metrics,
        Observation,
        RLDataset,
        RLDatasetBuilder,
        StepResult,
        StopCondition,
        Trajectory,
    )
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    _HAS_TINKER = True
except ImportError:
    _HAS_TINKER = False
    Env = object  # type: ignore[misc,assignment]
    EnvGroupBuilder = object  # type: ignore[misc,assignment]
    RLDataset = object  # type: ignore[misc,assignment]
    RLDatasetBuilder = object  # type: ignore[misc,assignment]

from evaluate_predictions import (
    Funder,
    _ensure_funder_list,
    compute_hierarchical_reward,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert at extracting structured funding metadata from academic papers. "
    "Given a funding statement, extract all funders and their associated awards. "
    "Return a JSON array of funder objects. Each funder has:\n"
    '- "funder_name": string or null\n'
    '- "awards": array of objects with "award_ids" (array of strings), '
    '"funding_scheme" (array of strings), and "award_title" (array of strings)\n'
    "Return ONLY the JSON array, no other text."
)

USER_TEMPLATE = "Extract funding information from the following statement:\n\n{funding_statement}"


_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def extract_funders_from_text(text: str) -> List[Funder]:
    if not text or not text.strip():
        return []

    m = _CODE_BLOCK_RE.search(text)
    candidate = m.group(1).strip() if m else text.strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict) and "funders" in parsed:
        return _ensure_funder_list(parsed["funders"])

    return _ensure_funder_list(parsed)


class FundingExtractionEnv(Env):
    def __init__(
        self,
        renderer: Renderer,
        funding_statement: str,
        gold_funders: List[Funder],
        funder_threshold: float = 0.8,
        threshold: float = 0.8,
    ):
        self.renderer = renderer
        self.funding_statement = funding_statement
        self.gold_funders = gold_funders
        self.funder_threshold = funder_threshold
        self.threshold = threshold

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(funding_statement=self.funding_statement)},
        ]
        model_input = self.renderer.build_generation_prompt(messages)
        stop_condition = self.renderer.get_stop_sequences()
        return model_input, stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)

        pred_funders = extract_funders_from_text(content)
        format_valid = float(len(pred_funders) > 0 or not content.strip())

        if not pred_funders and self.gold_funders:
            reward = 0.0
            metrics: Metrics = {
                "format_valid": format_valid,
                "funder_f0_5": 0.0,
                "award_id_f0_5": 0.0,
                "scheme_f0_5": 0.0,
                "title_f0_5": 0.0,
                "flat_award_id_f0_5": 0.0,
                "association_gap": 0.0,
            }
        else:
            reward, metrics = compute_hierarchical_reward(
                self.gold_funders,
                pred_funders,
                funder_threshold=self.funder_threshold,
                threshold=self.threshold,
            )
            metrics["format_valid"] = format_valid

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=self.renderer.build_generation_prompt(
                [{"role": "system", "content": ""}]
            ),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics=metrics,
        )


@dataclass(frozen=True)
class FundingGroupBuilder(EnvGroupBuilder):
    env_thunk: partial
    num_envs: int
    dataset_name: str = "funding"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]


class FundingRLDataset(RLDataset):
    def __init__(
        self,
        data: list[dict],
        renderer: Renderer,
        batch_size: int,
        group_size: int,
        funder_threshold: float = 0.8,
        threshold: float = 0.8,
    ):
        self.data = data
        self.renderer = renderer
        self.batch_size = batch_size
        self.group_size = group_size
        self.funder_threshold = funder_threshold
        self.threshold = threshold

    def _make_group_builder(self, example: dict) -> FundingGroupBuilder:
        funding_statement = example["funding_statement"]
        gold_funders = _ensure_funder_list(example.get("funders", []))

        thunk = partial(
            FundingExtractionEnv,
            renderer=self.renderer,
            funding_statement=funding_statement,
            gold_funders=gold_funders,
            funder_threshold=self.funder_threshold,
            threshold=self.threshold,
        )
        return FundingGroupBuilder(env_thunk=thunk, num_envs=self.group_size)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.data))
        return [self._make_group_builder(self.data[i]) for i in range(start, end)]

    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size


def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if _HAS_TINKER:

    @chz.chz
    class FundingDatasetBuilder(RLDatasetBuilder):
        batch_size: int = 16
        group_size: int = 8
        model_name_for_tokenizer: str = "meta-llama/Llama-3.1-8B-Instruct"
        renderer_name: str = ""
        train_path: str = "train.jsonl"
        synthetic_path: str = "synthetic.jsonl"
        eval_fraction: float = 0.1
        seed: int = 42
        funder_threshold: float = 0.8
        threshold: float = 0.8

        async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
            data = _load_jsonl(self.train_path) + _load_jsonl(self.synthetic_path)
            rng = random.Random(self.seed)
            rng.shuffle(data)

            split = int(len(data) * (1 - self.eval_fraction))
            train_data = data[:split]
            eval_data = data[split:]

            tokenizer = get_tokenizer(self.model_name_for_tokenizer)
            renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name_for_tokenizer)
            renderer = get_renderer(renderer_name, tokenizer)

            train_ds = FundingRLDataset(
                data=train_data,
                renderer=renderer,
                batch_size=self.batch_size,
                group_size=self.group_size,
                funder_threshold=self.funder_threshold,
                threshold=self.threshold,
            )
            eval_ds = FundingRLDataset(
                data=eval_data,
                renderer=renderer,
                batch_size=self.batch_size,
                group_size=1,
                funder_threshold=self.funder_threshold,
                threshold=self.threshold,
            )

            logger.info("Funding RL dataset: %d train, %d eval examples", len(train_data), len(eval_data))
            return train_ds, eval_ds
