# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.0",
#     "transformers>=4.45",
#     "peft>=0.13",
#     "huggingface-hub>=0.25",
# ]
# ///
"""Merge the SFT LoRA adapter into the model and push the merged model."""

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model", required=True,
    )
    parser.add_argument(
        "--adapter",
       required=True,
    )
    parser.add_argument(
        "--output", required=True,
    )
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype="auto", device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging and unloading LoRA weights...")
    model = model.merge_and_unload()

    if args.push:
        print(f"Pushing to {args.output}")
        model.push_to_hub(args.output)
        tokenizer.push_to_hub(args.output)
        print("Done.")
    else:
        print(f"Saving locally to {args.output}")
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print(f"Saved. Run with --push to upload to HuggingFace Hub.")


if __name__ == "__main__":
    main()
