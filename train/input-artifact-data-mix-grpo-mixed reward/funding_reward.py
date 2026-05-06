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
"""RL environment and dataset for funding extraction with hierarchical reward."""

from __future__ import annotations

import json
import logging
import random
import re
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
    # Provide base classes so the module can be imported without tinker
    Env = object  # type: ignore[misc,assignment]
    EnvGroupBuilder = object  # type: ignore[misc,assignment]
    RLDataset = object  # type: ignore[misc,assignment]
    RLDatasetBuilder = object  # type: ignore[misc,assignment]

from rapidfuzz.distance import Levenshtein

from evaluate_predictions import (
    Award,
    Funder,
    LevelMetrics,
    _build_level_metrics,
    _collect_awards,
    _ensure_funder_list,
    _merge_funders,
    award_ids_match,
    normalize_award_id,
    optimal_match,
    similarity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RL-specific reward shaping
# ---------------------------------------------------------------------------
# The canonical evaluation in evaluate_predictions.py uses strictly binary
# award-id matching with hierarchical-only credit. The RL reward adds:
#   - soft award-id matching (partial credit for edit-distance-1 near-misses)
#   - a small additive flat-id term (credit when right ID under wrong funder)
# These live here, not in evaluate_predictions.py, so the eval script stays
# a pristine apples-to-apples scorer across experiments.
# ---------------------------------------------------------------------------


def _soft_award_id_score(
    pred_id: str,
    gold_id: str,
    id_match_mode: str = "normalized",
    max_partial: float = 0.5,
    max_edit_distance: int = 1,
    min_length: int = 6,
) -> float:
    """Continuous [0, 1] award-id score for RL reward shaping.

    Exact (per-mode-normalized) match → 1.0. Otherwise award up to
    ``max_partial`` for near-misses satisfying a length floor and edit
    distance cap — gives the policy a gradient on truncation / one-char
    errors without rewarding wildly wrong guesses. Capped strictly below
    1.0 so exact matches always dominate during pairing.
    """
    if award_ids_match(pred_id, gold_id, mode=id_match_mode):
        return 1.0
    if id_match_mode == "normalized":
        np_, ng_ = normalize_award_id(pred_id), normalize_award_id(gold_id)
    else:
        np_, ng_ = pred_id.strip().upper(), gold_id.strip().upper()
    if min(len(np_), len(ng_)) < min_length:
        return 0.0
    d = Levenshtein.distance(np_, ng_)
    if d == 0 or d > max_edit_distance:
        return 0.0
    if max_edit_distance <= 1:
        return max_partial
    return max_partial * (1.0 - (d - 1) / max_edit_distance)


def _optimal_match_ids(
    gold_items: list,
    pred_items: list,
    id_match_mode: str = "normalized",
    soft: bool = False,
):
    """Hungarian-optimal 1:1 pairing of award IDs.

    With ``soft=False`` the matching is strictly binary (1.0 or dropped).
    With ``soft=True`` we run two phases: binary Hungarian first (so exact
    matches are never displaced by near-misses), then a second Hungarian
    over leftover gold/pred using ``_soft_award_id_score`` with a floor
    of 0.5. Matched pair scores are continuous in [0.5, 1.0].
    """
    def binary_score_fn(g, p):
        return 1.0 if award_ids_match(p, g, mode=id_match_mode) else 0.0

    binary_matches = optimal_match(gold_items, pred_items, binary_score_fn, threshold=1.0)
    if not soft:
        return binary_matches

    used_gold = {gi for gi, _, _ in binary_matches}
    used_pred = {pi for _, pi, _ in binary_matches}
    leftover_gold_idx = [i for i in range(len(gold_items)) if i not in used_gold]
    leftover_pred_idx = [i for i in range(len(pred_items)) if i not in used_pred]
    if not leftover_gold_idx or not leftover_pred_idx:
        return binary_matches

    leftover_gold = [gold_items[i] for i in leftover_gold_idx]
    leftover_pred = [pred_items[i] for i in leftover_pred_idx]

    def soft_score_fn(g, p):
        return _soft_award_id_score(p, g, id_match_mode=id_match_mode)

    soft_local = optimal_match(leftover_gold, leftover_pred, soft_score_fn, threshold=0.5)
    return binary_matches + [
        (leftover_gold_idx[lgi], leftover_pred_idx[lpi], score)
        for lgi, lpi, score in soft_local
    ]


def _evaluate_with_soft_ids(
    gold_funders: List[Funder],
    pred_funders: List[Funder],
    funder_threshold: float,
    threshold: float,
    id_match_mode: str,
    soft_id_matching: bool,
):
    """Replicates evaluate_predictions._evaluate_per_funder with the sole
    difference that award-id matching inside the per-funder-pair loop
    uses the soft two-phase Hungarian when ``soft_id_matching`` is True.
    Funder, scheme, and title matching are unchanged from the canonical
    evaluation path.
    """
    gold_merged = _merge_funders(gold_funders, funder_threshold, similarity)
    pred_merged = _merge_funders(pred_funders, funder_threshold, similarity)

    gold_named_indices = [i for i, f in enumerate(gold_merged) if f.funder_name]
    pred_named_indices = [i for i, f in enumerate(pred_merged) if f.funder_name]
    gold_named = [gold_merged[i].funder_name for i in gold_named_indices]
    pred_named = [pred_merged[i].funder_name for i in pred_named_indices]

    funder_matches = optimal_match(gold_named, pred_named, similarity, funder_threshold)
    funder_metrics = _build_level_metrics(
        len(gold_named), len(pred_named), len(funder_matches)
    )

    matched_gold_set: set = set()
    matched_pred_set: set = set()
    paired = []
    for gm_idx, pm_idx, _score in funder_matches:
        gi = gold_named_indices[gm_idx]
        pi = pred_named_indices[pm_idx]
        paired.append((gi, pi))
        matched_gold_set.add(gi)
        matched_pred_set.add(pi)

    unnamed_gold_idx = next((i for i, f in enumerate(gold_merged) if not f.funder_name), None)
    unnamed_pred_idx = next((i for i, f in enumerate(pred_merged) if not f.funder_name), None)
    if unnamed_gold_idx is not None and unnamed_pred_idx is not None:
        paired.append((unnamed_gold_idx, unnamed_pred_idx))
        matched_gold_set.add(unnamed_gold_idx)
        matched_pred_set.add(unnamed_pred_idx)

    total_id_gold = total_id_pred = 0
    total_id_matched: float = 0.0
    total_scheme_gold = total_scheme_pred = total_scheme_matched = 0
    total_title_gold = total_title_pred = total_title_matched = 0

    for gi, pi in paired:
        g_ids, g_schemes, g_titles = _collect_awards(gold_merged[gi])
        p_ids, p_schemes, p_titles = _collect_awards(pred_merged[pi])

        id_matches = _optimal_match_ids(g_ids, p_ids, id_match_mode, soft=soft_id_matching)
        total_id_gold += len(g_ids)
        total_id_pred += len(p_ids)
        total_id_matched += sum(score for _, _, score in id_matches)

        scheme_matches = optimal_match(g_schemes, p_schemes, similarity, threshold)
        total_scheme_gold += len(g_schemes)
        total_scheme_pred += len(p_schemes)
        total_scheme_matched += len(scheme_matches)

        title_matches = optimal_match(g_titles, p_titles, similarity, threshold)
        total_title_gold += len(g_titles)
        total_title_pred += len(p_titles)
        total_title_matched += len(title_matches)

    for gi in range(len(gold_merged)):
        if gi not in matched_gold_set:
            g_ids, g_schemes, g_titles = _collect_awards(gold_merged[gi])
            total_id_gold += len(g_ids)
            total_scheme_gold += len(g_schemes)
            total_title_gold += len(g_titles)

    for pi in range(len(pred_merged)):
        if pi not in matched_pred_set:
            p_ids, p_schemes, p_titles = _collect_awards(pred_merged[pi])
            total_id_pred += len(p_ids)
            total_scheme_pred += len(p_schemes)
            total_title_pred += len(p_titles)

    id_metrics = _build_level_metrics(total_id_gold, total_id_pred, total_id_matched)
    scheme_metrics = _build_level_metrics(total_scheme_gold, total_scheme_pred, total_scheme_matched)
    title_metrics = _build_level_metrics(total_title_gold, total_title_pred, total_title_matched)

    return funder_metrics, id_metrics, scheme_metrics, title_metrics


def compute_shaped_reward(
    gold_funders: List[Funder],
    pred_funders: List[Funder],
    funder_threshold: float = 0.8,
    threshold: float = 0.8,
    id_match_mode: str = "normalized",
    weights: tuple = (0.50, 0.40, 0.0, 0.0),
    flat_id_weight: float = 0.10,
    soft_id_matching: bool = True,
):
    """RL reward with soft award-id matching + flat-id term.

    ``weights`` = (w_funder, w_id_hier, w_scheme, w_title) applied to F0.5
    per field. ``flat_id_weight`` adds an additive term from the flat
    (no-funder-gating) award-id F0.5, giving partial credit when the right
    ID is extracted under the wrong funder. ``soft_id_matching`` enables
    partial credit for edit-distance-1 near-miss IDs (see
    ``_soft_award_id_score``). Reward range: [0, sum(weights) + flat_id_weight].
    """
    w_funder, w_id, w_scheme, w_title = weights

    funder_m, id_m, scheme_m, title_m = _evaluate_with_soft_ids(
        gold_funders, pred_funders,
        funder_threshold, threshold, id_match_mode, soft_id_matching,
    )

    all_gold_ids: list = []
    all_pred_ids: list = []
    for f in gold_funders:
        for a in f.awards:
            all_gold_ids.extend(a.award_ids)
    for f in pred_funders:
        for a in f.awards:
            all_pred_ids.extend(a.award_ids)
    flat_id_matches = _optimal_match_ids(
        all_gold_ids, all_pred_ids, id_match_mode, soft=soft_id_matching
    )
    flat_id_matched = sum(score for _, _, score in flat_id_matches)
    flat_id_m = _build_level_metrics(len(all_gold_ids), len(all_pred_ids), flat_id_matched)

    def _f0_5_or_empty(m):
        if m.gold_count == 0 and m.pred_count == 0:
            return 1.0
        return m.f0_5

    funder_f0_5 = _f0_5_or_empty(funder_m)
    id_f0_5 = _f0_5_or_empty(id_m)
    scheme_f0_5 = _f0_5_or_empty(scheme_m)
    title_f0_5 = _f0_5_or_empty(title_m)
    flat_id_f0_5 = _f0_5_or_empty(flat_id_m)

    if len(gold_funders) > 0 or len(pred_funders) > 0:
        active_score = funder_f0_5 * id_f0_5
        if scheme_m.gold_count == 0 and scheme_m.pred_count == 0:
            scheme_f0_5 = active_score
        if title_m.gold_count == 0 and title_m.pred_count == 0:
            title_f0_5 = active_score

    reward = (
        w_funder * funder_f0_5
        + w_id * id_f0_5
        + flat_id_weight * flat_id_f0_5
        + w_scheme * scheme_f0_5
        + w_title * title_f0_5
    )

    metrics = {
        "funder_f0_5": funder_f0_5,
        "funder_precision": funder_m.precision,
        "funder_recall": funder_m.recall,
        "award_id_f0_5": id_f0_5,
        "award_id_precision": id_m.precision,
        "award_id_recall": id_m.recall,
        "scheme_f0_5": scheme_f0_5,
        "title_f0_5": title_f0_5,
        "flat_award_id_f0_5": flat_id_f0_5,
        "association_gap": flat_id_f0_5 - id_f0_5,
    }
    return reward, metrics

# ---------------------------------------------------------------------------
# Prompt template — must match what the SFT LoRA was trained with.
# Update SYSTEM_PROMPT / USER_TEMPLATE if the SFT used different wording.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def extract_funders_from_text(text: str) -> List[Funder]:
    """Parse model output text into a list of Funder objects.

    Handles: raw JSON list, {"funders": [...]}, markdown code blocks,
    single funder dict. Returns [] on any parse failure.
    """
    if not text or not text.strip():
        return []

    # Strip markdown code blocks if present
    m = _CODE_BLOCK_RE.search(text)
    candidate = m.group(1).strip() if m else text.strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return []

    # {"funders": [...]} wrapper
    if isinstance(parsed, dict) and "funders" in parsed:
        return _ensure_funder_list(parsed["funders"])

    # List of funders or single funder dict
    return _ensure_funder_list(parsed)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class FundingExtractionEnv(Env):
    """Single-turn env: present funding statement → parse JSON → hierarchical reward."""

    def __init__(
        self,
        renderer: Renderer,
        funding_statement: str,
        gold_funders: List[Funder],
        funder_threshold: float = 0.8,
        threshold: float = 0.8,
        weights: tuple = (0.50, 0.40, 0.0, 0.0),
        flat_id_weight: float = 0.10,
        soft_id_matching: bool = True,
    ):
        self.renderer = renderer
        self.funding_statement = funding_statement
        self.gold_funders = gold_funders
        self.funder_threshold = funder_threshold
        self.threshold = threshold
        self.weights = weights
        self.flat_id_weight = flat_id_weight
        self.soft_id_matching = soft_id_matching

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
            # Model produced nothing (or garbage) but should have extracted something
            reward = 0.0
            n_gold_ids = sum(len(a.award_ids) for f in self.gold_funders for a in f.awards)
            metrics: Metrics = {
                "format_valid": format_valid,
                "funder_f0_5": 0.0,
                "award_id_f0_5": 0.0,
                "scheme_f0_5": 0.0,
                "title_f0_5": 0.0,
                "flat_award_id_f0_5": 0.0,
                "association_gap": 0.0,
                "pred_id_count": 0.0,
                "gold_id_count": float(n_gold_ids),
                "pred_id_count_ratio": 0.0 if n_gold_ids > 0 else 1.0,
            }
        else:
            reward, metrics = compute_shaped_reward(
                self.gold_funders,
                pred_funders,
                funder_threshold=self.funder_threshold,
                threshold=self.threshold,
                weights=self.weights,
                flat_id_weight=self.flat_id_weight,
                soft_id_matching=self.soft_id_matching,
            )
            metrics["format_valid"] = format_valid
            # Hedging diagnostic: pred id count / gold id count per rollout.
            n_gold_ids = sum(len(a.award_ids) for f in self.gold_funders for a in f.awards)
            n_pred_ids = sum(len(a.award_ids) for f in pred_funders for a in f.awards)
            metrics["pred_id_count"] = float(n_pred_ids)
            metrics["gold_id_count"] = float(n_gold_ids)
            metrics["pred_id_count_ratio"] = (
                n_pred_ids / n_gold_ids if n_gold_ids > 0 else (1.0 if n_pred_ids == 0 else 0.0)
            )

        # next_observation is unused (episode ends), but the protocol requires it
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=self.renderer.build_generation_prompt(
                [{"role": "system", "content": ""}]
            ),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics=metrics,
        )


# ---------------------------------------------------------------------------
# Group builder
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FundingGroupBuilder(EnvGroupBuilder):
    """Creates group_size copies of FundingExtractionEnv for GRPO advantage centering."""

    env_thunk: partial  # partial(FundingExtractionEnv, renderer=..., ...)
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FundingRLDataset(RLDataset):
    """Wraps a list of (funding_statement, gold_funders) into batches of EnvGroupBuilders."""

    def __init__(
        self,
        data: list[dict],
        renderer: Renderer,
        batch_size: int,
        group_size: int,
        funder_threshold: float = 0.8,
        threshold: float = 0.8,
        weights: tuple = (0.50, 0.40, 0.0, 0.0),
        flat_id_weight: float = 0.10,
        soft_id_matching: bool = True,
    ):
        self.data = data
        self.renderer = renderer
        self.batch_size = batch_size
        self.group_size = group_size
        self.funder_threshold = funder_threshold
        self.threshold = threshold
        self.weights = weights
        self.flat_id_weight = flat_id_weight
        self.soft_id_matching = soft_id_matching

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
            weights=self.weights,
            flat_id_weight=self.flat_id_weight,
            soft_id_matching=self.soft_id_matching,
        )
        return FundingGroupBuilder(env_thunk=thunk, num_envs=self.group_size)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.data))
        return [self._make_group_builder(self.data[i]) for i in range(start, end)]

    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size


# ---------------------------------------------------------------------------
# Dataset builder (chz-compatible for train.Config)
# ---------------------------------------------------------------------------
def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if _HAS_TINKER:

    @chz.chz
    class FundingDatasetBuilder(RLDatasetBuilder):
        batch_size: int = 16
        group_size: int = 8
        model_name_for_tokenizer: str = "meta-llama/Llama-3.1-8B-Instruct"
        renderer_name: str = ""  # auto-detected from model_name_for_tokenizer
        train_path: str = "rl_train.jsonl"
        train_synthetic_path: str = "rl_synthetic.jsonl"
        eval_path: str = "rl_eval_train.jsonl"
        eval_synthetic_path: str = "rl_eval_synthetic.jsonl"
        seed: int = 42
        funder_threshold: float = 0.8
        threshold: float = 0.8
        # Reward shaping — train path (delivered to the policy)
        w_funder: float = 0.50
        w_id_hier: float = 0.40
        flat_id_weight: float = 0.10
        soft_id_matching: bool = True
        # Eval path keeps the historical binary definition so F0.5
        # numbers remain apples-to-apples with prior runs.
        eval_flat_id_weight: float = 0.0
        eval_soft_id_matching: bool = False

        async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
            train_data = _load_jsonl(self.train_path) + _load_jsonl(self.train_synthetic_path)
            eval_data = _load_jsonl(self.eval_path) + _load_jsonl(self.eval_synthetic_path)
            rng = random.Random(self.seed)
            rng.shuffle(train_data)
            rng.shuffle(eval_data)

            tokenizer = get_tokenizer(self.model_name_for_tokenizer)
            renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name_for_tokenizer)
            renderer = get_renderer(renderer_name, tokenizer)

            train_weights = (self.w_funder, self.w_id_hier, 0.0, 0.0)
            eval_weights = (0.50, 0.50, 0.0, 0.0)  # historical default

            train_ds = FundingRLDataset(
                data=train_data,
                renderer=renderer,
                batch_size=self.batch_size,
                group_size=self.group_size,
                funder_threshold=self.funder_threshold,
                threshold=self.threshold,
                weights=train_weights,
                flat_id_weight=self.flat_id_weight,
                soft_id_matching=self.soft_id_matching,
            )
            eval_ds = FundingRLDataset(
                data=eval_data,
                renderer=renderer,
                batch_size=self.batch_size,
                group_size=1,  # deterministic eval
                funder_threshold=self.funder_threshold,
                threshold=self.threshold,
                weights=eval_weights,
                flat_id_weight=self.eval_flat_id_weight,
                soft_id_matching=self.eval_soft_id_matching,
            )

            logger.info("Funding RL dataset: %d train, %d eval examples", len(train_data), len(eval_data))
            return train_ds, eval_ds
