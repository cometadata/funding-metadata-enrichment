import os
import json
import random
import argparse
import dspy
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class FunderMetadata(BaseModel):
    funder_name: Optional[str] = Field(description="The name of the organization providing funding, in-kind support, or access to facilities or similar resources.")
    funding_scheme: Optional[str] = Field(description="The name of the specific funding program or scheme mentioned.")
    award_ids: List[str] = Field(default_factory=list, description="A list of grant or award numbers associated with this funder.")
    award_title: Optional[str] = Field(description="The title of the award, if explicitly mentioned.")

    @field_validator('award_ids', mode='before')
    def clean_award_ids(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v


class FundingData(BaseModel):
    funders: List[FunderMetadata]

class FundingExtractionSignature(dspy.Signature):
    funding_statement: str = dspy.InputField(desc="The unstructured text of the funding statement.")
    funding_data: FundingData = dspy.OutputField(desc="A JSON object containing a list of all extracted funder objects.")


class FundingExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(FundingExtractionSignature)

    def forward(self, funding_statement: str):
        return self.extractor(funding_statement=funding_statement)

def create_funders_set(funders_list):
    funders_set = set()
    if not funders_list:
        return funders_set

    for funder in funders_list:
        funder_dict = funder if isinstance(
            funder, dict) else funder.model_dump()
        funder_name = funder_dict.get('funder_name') or ''
        funding_scheme = funder_dict.get('funding_scheme') or ''
        award_ids = frozenset(sorted(funder_dict.get('award_ids') or []))
        funder_tuple = (
            funder_name.strip().lower(),
            funding_scheme.strip().lower(),
            award_ids
        )
        funders_set.add(funder_tuple)
    return funders_set


def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    try:
        gold_funders = gold.funders
        pred_funders = pred.funding_data.funders
    except (AttributeError, TypeError, KeyError):
        score = 0.0
        feedback = "Feedback: The prediction was malformed or did not follow the specified JSON structure."
        return dspy.Prediction(score=score, feedback=feedback) if pred_name else score

    gold_set = create_funders_set(gold_funders)
    pred_set = create_funders_set(pred_funders)

    true_positives = len(gold_set.intersection(pred_set))
    false_positives = len(pred_set.difference(gold_set))
    false_negatives = len(gold_set.difference(pred_set))

    precision = true_positives / \
        (true_positives + false_positives) if (true_positives +
                                               false_positives) > 0 else 0
    recall = true_positives / \
        (true_positives + false_negatives) if (true_positives +
                                               false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision +
                                           recall) if (precision + recall) > 0 else 0

    if pred_name is None:
        return f1_score

    feedback_messages = []
    if f1_score == 1.0 and false_positives == 0 and false_negatives == 0:
        feedback = "Perfect extraction. All funders and awards were correctly identified."
    else:
        if false_negatives > 0:
            missed = gold_set.difference(pred_set)
            feedback_messages.append(f"You missed {len(missed)} funders. These included: {list(missed)[:3]}")

        if false_positives > 0:
            extra = pred_set.difference(gold_set)
            feedback_messages.append(f"You incorrectly extracted {len(extra)} extra funders. These included: {list(extra)[:3]}")

        feedback_messages.append(
            "Review the statement carefully to find all funders and ensure all details are captured accurately.")
        feedback = "Feedback: " + " ".join(feedback_messages)

    return dspy.Prediction(score=f1_score, feedback=feedback)


def setup_lms(args):
    print(f"Setting up LMs: student='{args.student_model}', teacher='{args.teacher_model}'")

    teacher_lm = dspy.LM(model=args.teacher_model,
                         api_key=args.api_key, temperature=0.7)

    student_lm = dspy.LM(model=args.student_model, api_key=args.api_key)

    dspy.configure(lm=student_lm, reflection_lm=teacher_lm)
    print("LMs configured successfully.")


def extract_score_from_result(result: Union[float, 'dspy.evaluate.evaluate.EvaluationResult']) -> float:
    if isinstance(result, (int, float)):
        return float(result)
    elif hasattr(result, 'score'):
        return float(result.score)
    else:
        try:
            return float(result)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Cannot extract score from result of type {type(result)}: {e}")


def load_and_prepare_data(args):
    print(f"Loading data from: {args.data_path}")
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)

    dspy_dataset = [
        dspy.Example(
            funding_statement=item['funding_statement'],
            funders=item['funders']
        ).with_inputs('funding_statement')
        for item in dataset
    ]

    random.seed(42)
    random.shuffle(dspy_dataset)

    total_size = len(dspy_dataset)
    train_end = int(total_size * args.train_fraction)
    val_end = train_end + int(total_size * args.val_fraction)

    trainset = dspy_dataset[:train_end]
    valset = dspy_dataset[train_end:val_end]
    testset = dspy_dataset[val_end:]

    print(f"Data split complete:")
    print(f"  - Training examples: {len(trainset)}")
    print(f"  - Validation examples: {len(valset)}")
    print(f"  - Test examples: {len(testset)}")

    return trainset, valset, testset


def main(args):
    setup_lms(args)
    trainset, valset, testset = load_and_prepare_data(args)
    unoptimized_program = FundingExtractor()
    print("\n--- Evaluating unoptimized program (baseline) ---")
    evaluator = dspy.Evaluate(
        devset=testset, num_threads=8, display_progress=True)
    baseline_result = evaluator(
        unoptimized_program, metric=metric_with_feedback)
    baseline_score = extract_score_from_result(baseline_result)
    print(f"Baseline F1 Score on Test Set: {baseline_score:.2f}")
    print("\n--- Starting GEPA optimization ---")
    reflection_lm = dspy.LM(model=args.teacher_model,
                            api_key=args.api_key, temperature=1.0, max_tokens=8000)

    optimizer = dspy.GEPA(
        metric=metric_with_feedback,
        auto="light",
        track_stats=True,
        reflection_lm=reflection_lm
    )

    optimized_program = optimizer.compile(
        student=unoptimized_program,
        trainset=trainset,
        valset=valset
    )

    print("\n--- Evaluating optimized program ---")
    optimized_result = evaluator(
        optimized_program, metric=metric_with_feedback)
    optimized_score = extract_score_from_result(optimized_result)
    print(f"Optimized F1 Score on Test Set: {optimized_score:.2f}")

    print(f"\nSaving optimized program to: {args.output_path}")
    optimized_program.save(args.output_path)
    print("Script finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GEPA optimization for funding metadata extraction.")

    parser.add_argument("-d", "--data_path", type=str,
                        required=True, help="Path to the input JSON data file.")
    parser.add_argument("-o", "--output_path", type=str, required=True,
                        help="Path to save the final optimized DSPy program.")

    parser.add_argument("--train_fraction", type=float, default=0.5,
                        help="Fraction of data to use for training.")
    parser.add_argument("--val_fraction", type=float, default=0.25,
                        help="Fraction of data to use for validation.")

    parser.add_argument("--student_model", type=str, default="gemini/gemini-2.5-flash",
                        help="Name of the student/task language model.")
    parser.add_argument("--teacher_model", type=str, default="gemini/gemini-2.5-pro",
                        help="Name of the teacher/reflection language model.")

    parser.add_argument("--api_key", type=str, default=None,
                        help="Gemini API key. If not provided, it will check the GOOGLE_API_KEY environment variable.")

    args = parser.parse_args()

    if not args.api_key:
        args.api_key = os.getenv("GEMINI_API_KEY")
        if not args.api_key:
            raise ValueError(
                "Gemini API key must be provided via --api_key argument or GOOGLE_API_KEY environment variable.")

    main(args)
