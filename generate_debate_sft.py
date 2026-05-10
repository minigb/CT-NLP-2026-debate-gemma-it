#!/usr/bin/env python3
"""Generate Korean natural-chat debate SFT rows with Qwen3.

Input:
    JSONL or CSV file with at least a `topic` field.
    Optional field: seed_user_utterance

Output:
    JSONL file where each line is a trainer-ready SFT row:
    {
      "id": "...",
      "messages": [
        {"role": "system", "content": ""},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "metadata": {...}
    }

The default model is the locally downloaded Qwen3-8B checkpoint at
models/Qwen-Qwen3-8B. This script loads the checkpoint directly and does not
apply BitsAndBytesConfig.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_legacy_device_args(argv: list[str]) -> list[str]:
    return ["--devices" if arg == "--device" else arg for arg in argv]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Korean natural-chat debate SFT rows with Qwen3."
    )

    parser.add_argument(
        "--model-id",
        default="models/Qwen-Qwen3-8B",
        help="Qwen3 teacher model id or local path. Default: models/Qwen-Qwen3-8B.",
    )
    parser.add_argument(
        "--input",
        default="seeds/debate_topics.jsonl",
        help="Input JSONL/CSV file. Expected fields: topic, optional seed_user_utterance.",
    )
    parser.add_argument(
        "--output",
        default="generated/qwen3_8b_debate_sft.jsonl",
        help="Base output JSONL path. A timestamp is inserted before the suffix.",
    )
    parser.add_argument(
        "--failed-output",
        default="",
        help="Optional failed-generation base JSONL path. A timestamp is inserted before the suffix.",
    )
    parser.add_argument(
        "--persona-prompt-file",
        required=True,
        help="Text file containing the fixed persona prompt.",
    )
    parser.add_argument(
        "--system-prompt-file",
        required=True,
        help="System prompt template file used for generation; its path is stored in metadata.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Number of input seed rows to use. Use -1 for all rows.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start row offset.",
    )
    parser.add_argument(
        "--num-generations-per-seed",
        type=int,
        default=3,
        help="How many synthetic rows to generate per input seed row.",
    )
    parser.add_argument(
        "--num-turns",
        type=int,
        default=1,
        help="Number of user-assistant pairs to generate. 1 means single-turn, 2+ means multi-turn.",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=4,
        help="Number of next-message prompts to decode together per worker. Use 1 to disable batching.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=180,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.08,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--devices",
        default="cuda:0",
        help='Comma-separated CUDA devices for generation, e.g. "cuda:0" or "cuda:0,cuda:1".',
    )
    return parser.parse_args(normalize_legacy_device_args(sys.argv[1:]))


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        import pandas as pd

        return pd.read_csv(path).fillna("").to_dict(orient="records")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_rows(rows: list[dict[str, Any]], start: int, limit: int) -> list[dict[str, Any]]:
    if start < 0:
        raise ValueError("--start must be >= 0.")
    if limit < 0:
        return rows[start:]
    return rows[start : start + limit]


def read_prompt_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def prompt_file_ref(path: str) -> str:
    return path


def require_file(path: str, arg_name: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{arg_name} does not exist or is not a file: {path}")


def clean_row(row: dict[str, Any]) -> dict[str, str]:
    return {k: "" if v is None else str(v) for k, v in row.items()}


def require_topic(row: dict[str, str], row_index: int) -> None:
    if not row.get("topic", "").strip():
        raise ValueError(f"Seed row {row_index} is missing required field: topic")


def timestamped_path(path: Path, timestamp: str) -> Path:
    suffix = path.suffix or ".jsonl"
    return path.with_name(f"{path.stem}_{timestamp}{suffix}")


def shard_path(path: Path, worker_id: int) -> Path:
    suffix = path.suffix or ".jsonl"
    return path.with_name(f"{path.stem}.part{worker_id:02d}{suffix}")


def config_path_for_output(output_path: Path) -> Path:
    return output_path.with_suffix(".config.json")


def build_run_config(
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
    fail_path: Path,
    config_path: Path,
    persona_ref: str,
    timestamp: str,
    num_seed_rows: int,
    num_success: int | None = None,
    num_failed: int | None = None,
) -> dict[str, Any]:
    config = {
        "run_timestamp": timestamp,
        "model": {
            "model_id": args.model_id,
            "devices": parse_devices(args),
        },
        "input": {
            "seed_file": str(input_path),
            "start": args.start,
            "limit": args.limit,
            "num_selected_seed_rows": num_seed_rows,
        },
        "output": {
            "output_file": str(output_path),
            "failed_output_file": str(fail_path),
            "config_file": str(config_path),
        },
        "persona": {
            "persona_prompt_file": persona_ref,
        },
        "system": {
            "system_prompt_file": args.system_prompt_file,
        },
        "dataset": {
            "num_turns": args.num_turns,
            "num_generations_per_seed": args.num_generations_per_seed,
            "format": format_name_for_turns(args.num_turns),
        },
        "generation": {
            "seed": args.seed,
            "per_message_seed_formula_when_batch_size_1": "seed + source_seed_index * 1000 + generation_index * 100 + step_index",
            "batch_seed_formula": "min(seed + source_seed_index * 1000 + generation_index * 100 + step_index for prompts in batch)",
            "generation_batch_size": args.generation_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
        },
    }
    if num_success is not None and num_failed is not None:
        config["result"] = {
            "num_success": num_success,
            "num_failed": num_failed,
        }
    return config


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_generation_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_devices(args: argparse.Namespace) -> list[str]:
    return [device.strip() for device in args.devices.split(",") if device.strip()]


def build_teacher_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a careful synthetic data generator. Return only valid JSON.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def apply_qwen3_chat_template(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def load_model(model_id: str, device: str):
    if not device.startswith("cuda:"):
        raise ValueError('This script expects CUDA devices like "cuda:0".')

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but this script requires a CUDA device.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
        trust_remote_code=True,
    )

    # Load the local Qwen3-8B checkpoint directly.
    # Do NOT pass BitsAndBytesConfig here.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return tokenizer, model


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()

    # Remove common markdown fences if the model adds them.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # First try direct JSON parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the first {...} block.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def is_valid_message(message: Any, expected_role: str) -> bool:
    if not isinstance(message, dict):
        return False
    if message.get("role") != expected_role:
        return False
    content = message.get("content", "")
    if not isinstance(content, str):
        return False
    return bool(content.strip())


def format_name_for_turns(num_turns: int) -> str:
    return "single_turn_natural_debate" if num_turns == 1 else "multi_turn_natural_debate"


def is_valid_generated_message(obj: dict[str, Any] | None, expected_role: str) -> bool:
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != {"role", "content"}:
        return False
    if not is_valid_message(obj, expected_role):
        return False

    content = obj["content"].strip()
    if expected_role == "user" and len(content) < 5:
        return False
    if expected_role == "assistant":
        if len(content) < 50 or len(content) > 2500:
            return False
        if "저는 AI라서" in content or "인공지능이라서" in content:
            return False

    return True


def conversation_context(messages: list[dict[str, str]]) -> str:
    if not messages:
        return "(empty)"
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def format_generation_prompt(
    system_prompt_template: str,
    persona_prompt: str,
    row_clean: dict[str, str],
    generation_index: int,
    conversation_messages: list[dict[str, str]],
    next_role: str,
) -> str:
    try:
        return system_prompt_template.format(
            persona_spec=persona_prompt,
            topic=row_clean.get("topic", row_clean.get("instruction", "")),
            seed_user_utterance=row_clean.get("seed_user_utterance", ""),
            generation_index=generation_index,
            conversation_so_far=conversation_context(conversation_messages),
            next_role=next_role,
        )
    except KeyError as exc:
        raise ValueError(
            f"Unknown system prompt template field {exc!s}. "
            "Escape literal JSON braces as '{{' and '}}'."
        ) from exc


def generate_one(
    tokenizer,
    model,
    prompt: str,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any] | None]:
    return generate_many(tokenizer, model, [prompt], args)[0]


def generate_many(
    tokenizer,
    model,
    prompts: list[str],
    args: argparse.Namespace,
) -> list[tuple[str, dict[str, Any] | None]]:
    texts = [
        apply_qwen3_chat_template(tokenizer, build_teacher_messages(prompt))
        for prompt in prompts
    ]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
    ).to(args.device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    new_tokens = output_ids[:, inputs["input_ids"].shape[-1] :]
    raw_outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [
        (raw.strip(), extract_json_object(raw))
        for raw in raw_outputs
    ]


def chunked(items: list[Any], chunk_size: int) -> list[list[Any]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def make_sft_record(
    record_id: str,
    generated_messages: list[dict[str, str]],
    source_row: dict[str, Any],
    system_prompt_file: str,
    persona_prompt_file: str,
    model_id: str,
    seed_index: int,
    generation_index: int,
    generation_seed: int,
    num_turns: int,
) -> dict[str, Any]:
    # Keep long prompt text out of rows; prompt file paths are tracked in metadata.
    messages = [{"role": "system", "content": ""}]
    messages.extend(
        {
            "role": message["role"],
            "content": message["content"].strip(),
        }
        for message in generated_messages
    )

    return {
        "id": record_id,
        "messages": messages,
        "metadata": {
            "source": "qwen3_synthetic",
            "teacher_model": model_id,
            "system_prompt_file": system_prompt_file,
            "persona_prompt_file": persona_prompt_file,
            "seed_file": str(source_row.get("__seed_file", "")),
            "source_seed_index": seed_index,
            "generation_index": generation_index,
            "generation_seed": generation_seed,
            "topic": source_row.get("topic", ""),
            "seed_user_utterance": source_row.get("seed_user_utterance", ""),
            "format": format_name_for_turns(num_turns),
            "num_turns": num_turns,
        },
    }


def generate_shard(
    args: argparse.Namespace,
    indexed_rows: list[tuple[int, dict[str, Any]]],
    device: str,
    output_path: Path,
    fail_path: Path,
    input_path: Path,
    system_ref: str,
    system_prompt_template: str,
    persona_prompt: str,
    persona_ref: str,
    worker_id: int = 0,
    show_progress: bool = True,
) -> dict[str, int | str]:
    worker_args = argparse.Namespace(**vars(args))
    worker_args.device = device
    worker_args.devices = device

    tokenizer, model = load_model(worker_args.model_id, device)

    num_success = 0
    num_failed = 0
    conversations: list[dict[str, Any]] = []

    for seed_index, row in indexed_rows:
        row_clean = clean_row(row)
        require_topic(row_clean, seed_index)
        row_clean["__seed_file"] = str(input_path)

        for gen_idx in range(worker_args.num_generations_per_seed):
            base_seed = worker_args.seed + seed_index * 1000 + gen_idx * 100
            generated_messages: list[dict[str, str]] = []
            seed_user_utterance = row_clean.get("seed_user_utterance", "").strip()
            if seed_user_utterance:
                generated_messages.append({"role": "user", "content": seed_user_utterance})

            conversations.append(
                {
                    "record_id": f"seed_{seed_index:06d}_gen_{gen_idx:02d}",
                    "seed_index": seed_index,
                    "row_clean": row_clean,
                    "gen_idx": gen_idx,
                    "base_seed": base_seed,
                    "messages": generated_messages,
                    "failed_reason": "",
                    "failed_step": None,
                    "written": False,
                }
            )

    total_steps = worker_args.num_turns * 2
    total_generations = sum(
        max(0, total_steps - len(conversation["messages"]))
        for conversation in conversations
    )
    progress = tqdm(
        total=total_generations,
        desc=f"worker {worker_id} {device}",
        disable=not show_progress,
        position=worker_id,
    )

    with output_path.open("w", encoding="utf-8") as out, fail_path.open("w", encoding="utf-8") as fail_out:
        def write_failure(conversation: dict[str, Any], reason: str, failed_step: int | None) -> None:
            nonlocal num_failed
            row_clean = conversation["row_clean"]
            fail_out.write(
                json.dumps(
                    {
                        "id": conversation["record_id"],
                        "model_id": worker_args.model_id,
                        "seed_file": str(input_path),
                        "source_seed_index": conversation["seed_index"],
                        "generation_index": conversation["gen_idx"],
                        "generation_seed": conversation["base_seed"],
                        "failed_step": failed_step,
                        "topic": row_clean.get("topic", ""),
                        "seed_user_utterance": row_clean.get("seed_user_utterance", ""),
                        "reason": reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            conversation["written"] = True
            num_failed += 1

        def write_success(conversation: dict[str, Any]) -> None:
            nonlocal num_success
            row_clean = conversation["row_clean"]
            record = make_sft_record(
                record_id=conversation["record_id"],
                generated_messages=conversation["messages"],
                source_row=row_clean,
                system_prompt_file=system_ref,
                persona_prompt_file=persona_ref,
                model_id=worker_args.model_id,
                seed_index=conversation["seed_index"],
                generation_index=conversation["gen_idx"],
                generation_seed=conversation["base_seed"],
                num_turns=worker_args.num_turns,
            )
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            conversation["written"] = True
            num_success += 1

        for step_index in range(total_steps):
            step_conversations = [
                conversation
                for conversation in conversations
                if not conversation["written"]
                and not conversation["failed_reason"]
                and len(conversation["messages"]) == step_index
            ]
            if not step_conversations:
                continue

            next_role = "user" if step_index % 2 == 0 else "assistant"
            for batch in chunked(step_conversations, worker_args.generation_batch_size):
                batch_seed = min(conversation["base_seed"] + step_index for conversation in batch)
                set_generation_seed(batch_seed)
                prompts = [
                    format_generation_prompt(
                        system_prompt_template,
                        persona_prompt,
                        conversation["row_clean"],
                        conversation["gen_idx"],
                        conversation["messages"],
                        next_role,
                    )
                    for conversation in batch
                ]

                generated = generate_many(tokenizer, model, prompts, worker_args)
                for conversation, (_raw, parsed) in zip(batch, generated, strict=True):
                    if not is_valid_generated_message(parsed, next_role):
                        conversation["failed_reason"] = "invalid_json_or_failed_message_validation"
                        conversation["failed_step"] = step_index
                        write_failure(conversation, conversation["failed_reason"], step_index)
                    else:
                        conversation["messages"].append(
                            {
                                "role": next_role,
                                "content": parsed["content"].strip(),
                            }
                        )
                        if len(conversation["messages"]) == total_steps:
                            write_success(conversation)
                progress.update(len(batch))
                out.flush()
                fail_out.flush()
        progress.close()

        for conversation in conversations:
            if conversation["written"]:
                continue

            if len(conversation["messages"]) != total_steps:
                write_failure(conversation, "incomplete_generation", len(conversation["messages"]))
                continue

            write_success(conversation)

        if num_failed:
            fail_out.flush()
        if num_success:
            out.flush()

    return {
        "worker_id": worker_id,
        "device": device,
        "num_success": num_success,
        "num_failed": num_failed,
        "output_path": str(output_path),
        "fail_path": str(fail_path),
    }


def merge_jsonl_files(part_paths: list[Path], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as out:
        for part_path in part_paths:
            if not part_path.exists():
                continue
            with part_path.open("r", encoding="utf-8") as part:
                for line in part:
                    out.write(line)


def split_round_robin(items: list[tuple[int, dict[str, Any]]], num_shards: int) -> list[list[tuple[int, dict[str, Any]]]]:
    shards = [[] for _ in range(num_shards)]
    for item_index, item in enumerate(items):
        shards[item_index % num_shards].append(item)
    return shards


def main() -> None:
    args = parse_args()

    if args.num_turns < 1 or args.num_turns > 4:
        raise ValueError("--num-turns should be between 1 and 4.")
    if args.generation_batch_size < 1:
        raise ValueError("--generation-batch-size must be >= 1.")

    devices = parse_devices(args)
    if not devices:
        raise ValueError("No CUDA devices were provided.")
    torch.manual_seed(args.seed)

    input_path = Path(args.input)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_base_path = Path(args.output)
    output_path = timestamped_path(output_base_path, timestamp)
    config_path = config_path_for_output(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    require_file(args.system_prompt_file, "--system-prompt-file")
    require_file(args.persona_prompt_file, "--persona-prompt-file")
    system_prompt_template = read_prompt_file(args.system_prompt_file)
    system_ref = prompt_file_ref(args.system_prompt_file)
    persona_prompt = read_prompt_file(args.persona_prompt_file)
    persona_ref = prompt_file_ref(args.persona_prompt_file)

    rows = read_rows(input_path)
    selected_rows = select_rows(rows, args.start, args.limit)
    indexed_rows = list(enumerate(selected_rows, start=args.start))

    num_success = 0
    num_failed = 0

    failed_base_path = Path(args.failed_output) if args.failed_output else output_base_path.with_suffix(".failed.jsonl")
    fail_path = timestamped_path(failed_base_path, timestamp)
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    write_json(
        config_path,
        build_run_config(
            args=args,
            input_path=input_path,
            output_path=output_path,
            fail_path=fail_path,
            config_path=config_path,
            persona_ref=persona_ref,
            timestamp=timestamp,
            num_seed_rows=len(indexed_rows),
        ),
    )

    if len(devices) == 1:
        result = generate_shard(
            args=args,
            indexed_rows=indexed_rows,
            device=devices[0],
            output_path=output_path,
            fail_path=fail_path,
            input_path=input_path,
            system_ref=system_ref,
            system_prompt_template=system_prompt_template,
            persona_prompt=persona_prompt,
            persona_ref=persona_ref,
            worker_id=0,
            show_progress=True,
        )
        num_success = int(result["num_success"])
        num_failed = int(result["num_failed"])
    else:
        shards = split_round_robin(indexed_rows, len(devices))
        output_parts = [shard_path(output_path, worker_id) for worker_id in range(len(devices))]
        fail_parts = [shard_path(fail_path, worker_id) for worker_id in range(len(devices))]

        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as executor:
            futures = []
            for worker_id, (device, shard_rows) in enumerate(zip(devices, shards, strict=True)):
                futures.append(
                    executor.submit(
                        generate_shard,
                        args,
                        shard_rows,
                        device,
                        output_parts[worker_id],
                        fail_parts[worker_id],
                        input_path,
                        system_ref,
                        system_prompt_template,
                        persona_prompt,
                        persona_ref,
                        worker_id,
                        True,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                num_success += int(result["num_success"])
                num_failed += int(result["num_failed"])

        merge_jsonl_files(output_parts, output_path)
        merge_jsonl_files(fail_parts, fail_path)

    write_json(
        config_path,
        build_run_config(
            args=args,
            input_path=input_path,
            output_path=output_path,
            fail_path=fail_path,
            config_path=config_path,
            persona_ref=persona_ref,
            timestamp=timestamp,
            num_seed_rows=len(indexed_rows),
            num_success=num_success,
            num_failed=num_failed,
        ),
    )

    print(f"Done. Success: {num_success}, Failed: {num_failed}")
    print(f"Output: {output_path}")
    print(f"Failed generations: {fail_path}")
    print(f"Run config: {config_path}")


if __name__ == "__main__":
    main()
