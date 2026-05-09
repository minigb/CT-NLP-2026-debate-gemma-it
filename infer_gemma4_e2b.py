#!/usr/bin/env python3
"""Run simple local inference with the base google/gemma-4-E2B model."""

import argparse

import torch
from transformers import AutoProcessor

try:
    from transformers import Gemma4ForConditionalGeneration as Gemma4Model
except ImportError:
    from transformers import AutoModelForImageTextToText as Gemma4Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with Gemma 4 E2B.")
    parser.add_argument(
        "--model-path",
        default="models/google-gemma-4-E2B",
        help="Local model directory or Hugging Face repo id.",
    )
    parser.add_argument(
        "--prompt",
        help="Prompt to continue. This uses the base model, so write a completion-style prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--device-map",
        default="single",
        help='Device placement: "single", "auto", "cpu", or a device such as "cuda:0".',
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Transformers to download missing files instead of requiring the local model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_files_only = not args.allow_download

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if args.device_map == "single":
        device_map = {"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"}
    elif args.device_map in {"cpu", "cuda", "cuda:0", "cuda:1"}:
        device_map = {"": args.device_map}
    else:
        device_map = args.device_map

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=local_files_only,
    )
    model = Gemma4Model.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=device_map,
        local_files_only=local_files_only,
    )
    model.eval()

    inputs = processor(text=args.prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[-1]

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    new_tokens = output_ids[:, input_length:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    print(text.strip())


if __name__ == "__main__":
    main()
