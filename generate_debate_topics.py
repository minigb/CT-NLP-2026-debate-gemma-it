#!/usr/bin/env python3
"""Generate Korean debate topic seeds in JSONL format with Qwen3.

Each output line matches seeds/debate_topics.jsonl:
    {"topic": "...", "seed_user_utterance": "..."}
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DOMAIN_FOCUS_CHOICES = [
    "education",
    "jobs",
    "housing",
    "economy",
    "AI",
    "law",
    "healthcare",
    "environment",
    "platform labor",
    "entertainment",
    "internet culture",
    "welfare",
    "transport",
    "consumer rights",
]


SIDE_FOCI = [
    "supporting or accepting the topic",
    "opposing or questioning the topic",
]


DEFAULT_PROMPT_TEMPLATE = """Generate Korean debate seed data.

Return only valid JSON. No markdown, no explanations.

Task:
You are writing ONE natural Korean first-user utterance for a debate chatbot.

Topic: {topic}
Side: {side}

Topic rules:
- Korean only.
- Use the provided topic exactly as the "topic" value in every output item.
- Do not invent, rename, translate, broaden, or narrow the topic.

User utterance rules:
- Korean only.
- Write exactly one natural first message, like a real chat.
- Include the topic or clear keywords from the topic.
- Make the utterance match the provided side.
- Include a clear stance or doubt and a short reason.
- 1 to 2 sentences.
- Keep it concise: usually 20 to 90 Korean characters.
- Do not write structured labels like "주제:", "입장:", "요청:".
- Do not write "반박해줘" or "토론해줘".
- Do not make neutral info questions like "정시 확대가 뭐야?"
- Do not make vague utterances like "이건 좀 문제 아닌가요?"

Diversity rules:
- Make the provided side sound plausible.
- Vary tone and endings.
- Mix casual statements, hesitant opinions, blunt complaints, and confident claims.
- Do not make all utterances use the same pattern.
- Avoid duplicates and near-duplicates.

Good examples:
- "정시 확대는 별로인 것 같아. 결국 사교육 많이 받은 애들이 더 유리해지잖아."
- "정시는 좀 늘려야 한다고 봐. 그래도 같은 시험으로 평가하는 게 제일 깔끔하니까."
- "AI 그림 저작권은 인정하면 안 될 것 같아. 원작자 작업물을 학습한 결과물이 너무 많잖아."
- "AI 그림도 어느 정도 저작권을 줘야 하지 않나? 사람이 프롬프트랑 방향을 잡는 것도 창작이니까."
- "노키즈존은 업주 자유라고 봐. 가게 분위기랑 손님 불편도 현실적인 문제잖아."
- "노키즈존은 좀 과한 것 같아. 아이 있다는 이유만으로 출입을 막는 건 차별처럼 느껴져."

Output schema:
{{
  "topic": "...",
  "seed_user_utterance": "..."
}}

Final constraints:
- Return exactly one JSON object.
- The object must have exactly two keys: "topic" and "seed_user_utterance".
- Use this exact topic: {topic}
- Do not include the side label in the output.
"""


def normalize_legacy_device_args(argv: list[str]) -> list[str]:
    return ["--devices" if arg == "--device" else arg for arg in argv]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Korean debate topic seed JSONL with Qwen3.")
    parser.add_argument(
        "--model-id",
        default="models/Qwen-Qwen3-8B",
        help="Qwen3 model id or local path. Default: models/Qwen-Qwen3-8B.",
    )
    parser.add_argument(
        "--output",
        default="generated/debate_topics.jsonl",
        help="Base output JSONL path. A timestamp is inserted before the suffix.",
    )
    parser.add_argument(
        "--failed-output",
        default="",
        help="Optional failed-generation base JSONL path. A timestamp is inserted before the suffix.",
    )
    parser.add_argument(
        "--input",
        default="datasets/debate_topics_base.jsonl",
        help="Input topic JSONL file. Each line must have a topic field; domain is optional.",
    )
    parser.add_argument(
        "--utterances-per-side",
        type=int,
        default=3,
        help="Number of first-user utterances to generate for each side of each topic.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--repetition-penalty",
        "--repetition_penalty",
        dest="repetition_penalty",
        type=float,
        default=1.08,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--devices",
        default="cuda:0",
        help='Comma-separated CUDA devices for generation, e.g. "cuda:0" or "cuda:0,cuda:1".',
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Python format string for topic generation prompt.",
    )
    return parser.parse_args(normalize_legacy_device_args(sys.argv[1:]))


def timestamped_path(path: Path, timestamp: str) -> Path:
    suffix = path.suffix or ".jsonl"
    return path.with_name(f"{path.stem}_{timestamp}{suffix}")


def shard_path(path: Path, worker_id: int) -> Path:
    suffix = path.suffix or ".jsonl"
    return path.with_name(f"{path.stem}.part{worker_id:02d}{suffix}")


def config_path_for_output(output_path: Path) -> Path:
    return output_path.with_suffix(".config.json")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_generation_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_devices(args: argparse.Namespace) -> list[str]:
    return [device.strip() for device in args.devices.split(",") if device.strip()]


def read_topic_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            topic = str(row.get("topic", "")).strip()
            if not topic:
                raise ValueError(f"{path}:{line_index} is missing required field: topic")
            rows.append(
                {
                    "topic": topic,
                    "domain": str(row.get("domain", "")).strip(),
                }
            )
    if not rows:
        raise ValueError(f"No topic rows found in {path}")
    return rows


def build_run_config(
    args: argparse.Namespace,
    input_path: Path,
    num_topics: int,
    output_path: Path,
    fail_path: Path,
    config_path: Path,
    timestamp: str,
    num_success: int | None = None,
    num_failed: int | None = None,
    num_success_topics: int | None = None,
) -> dict[str, Any]:
    config = {
        "run_timestamp": timestamp,
        "model": {
            "model_id": args.model_id,
            "devices": parse_devices(args),
        },
        "input": {
            "topic_file": str(input_path),
            "num_topic_rows": num_topics,
        },
        "output": {
            "output_file": str(output_path),
            "failed_output_file": str(fail_path),
            "config_file": str(config_path),
        },
        "topic_generation": {
            "num_topics": num_topics,
            "utterances_per_model_call": 1,
            "num_model_calls": num_topics * 2 * args.utterances_per_side,
            "sides_per_topic": 2,
            "utterances_per_side": args.utterances_per_side,
            "target_seed_cases": num_topics * 2 * args.utterances_per_side,
        },
        "generation": {
            "seed": args.seed,
            "per_utterance_seed_formula": "seed + topic_index * 1000 + side_index * 100 + utterance_index",
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
        },
    }
    if num_success is not None and num_failed is not None:
        config["result"] = {
            "num_success_seed_cases": num_success,
            "num_success_topics": num_success_topics,
            "num_failed_or_shortfall_seed_cases": num_failed,
        }
    return config


def build_teacher_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a careful synthetic data generator. Return only valid JSON."},
        {"role": "user", "content": prompt},
    ]


def apply_qwen3_chat_template(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(model_id: str, device: str):
    if not device.startswith("cuda:"):
        raise ValueError('This script expects CUDA devices like "cuda:0".')
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but this script requires a CUDA device.")

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", trust_remote_code=True)
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
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def is_valid_utterance(utterance: Any) -> bool:
    if not isinstance(utterance, str):
        return False
    utterance = utterance.strip()
    if len(utterance) < 10 or len(utterance) > 220:
        return False
    forbidden = ("주제:", "입장:", "요청:", "반박해줘")
    return not any(token in utterance for token in forbidden)


def is_valid_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if set(item.keys()) != {"topic", "seed_user_utterance"}:
        return False
    topic = item.get("topic")
    utterance = item.get("seed_user_utterance")
    if not isinstance(topic, str) or not isinstance(utterance, str):
        return False
    topic = topic.strip()
    if len(topic) < 2 or len(topic) > 40:
        return False
    return is_valid_utterance(utterance)


def valid_item(obj: dict[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(obj, dict):
        return None
    if not is_valid_item(obj):
        return None
    return {
        "topic": obj["topic"].strip(),
        "seed_user_utterance": obj["seed_user_utterance"].strip(),
    }


def generate_batch(tokenizer, model, prompt: str, args: argparse.Namespace) -> tuple[str, dict[str, Any] | None]:
    messages = build_teacher_messages(prompt)
    text = apply_qwen3_chat_template(tokenizer, messages)
    inputs = tokenizer([text], return_tensors="pt", padding=True).to(args.device)

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
    raw = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return raw, extract_json_object(raw)


def format_generation_prompt(
    args: argparse.Namespace,
    topic_index: int,
    topic_row: dict[str, str],
    side: str,
) -> tuple[str, str]:
    domain_focus = topic_row.get("domain") or random.choice(DOMAIN_FOCUS_CHOICES)
    try:
        prompt = args.prompt_template.format(
            topic_index=topic_index,
            topic=topic_row["topic"],
            side=side,
            domain_focus=domain_focus,
        )
    except KeyError as exc:
        raise ValueError(
            f"Unknown prompt-template field {exc!s}. "
            "Escape literal JSON braces as '{{' and '}}'."
        ) from exc

    prompt = prompt.rstrip() + f"\n\nThis call should focus on this domain: {domain_focus}\n"
    return prompt, domain_focus


def generate_batch_shard(
    args: argparse.Namespace,
    generation_tasks: list[tuple[int, dict[str, str], int, str, int]],
    device: str,
    output_path: Path,
    fail_path: Path,
    worker_id: int = 0,
    show_progress: bool = True,
) -> dict[str, int | str]:
    worker_args = argparse.Namespace(**vars(args))
    worker_args.device = device
    worker_args.devices = device

    tokenizer, model = load_model(worker_args.model_id, device)

    num_success = 0
    num_success_topics = 0
    num_failed = 0
    seen_rows = set()
    successful_topic_indices = set()

    progress = tqdm(
        generation_tasks,
        desc=f"worker {worker_id} {device}",
        total=len(generation_tasks),
        disable=not show_progress,
        position=worker_id,
    )

    with output_path.open("w", encoding="utf-8") as out, fail_path.open("w", encoding="utf-8") as fail_out:
        for topic_index, topic_row, side_index, side, utterance_index in progress:
            current_seed = worker_args.seed + topic_index * 1000 + side_index * 100 + utterance_index
            set_generation_seed(current_seed)

            prompt, domain_focus = format_generation_prompt(worker_args, topic_index, topic_row, side)
            raw, parsed = generate_batch(tokenizer, model, prompt, worker_args)
            row = valid_item(parsed)

            if row is None:
                fail_out.write(
                    json.dumps(
                        {
                            "topic_index": topic_index,
                            "side_index": side_index,
                            "side": side,
                            "utterance_index": utterance_index,
                            "generation_seed": current_seed,
                            "domain_focus": domain_focus,
                            "topic": topic_row["topic"],
                            "reason": "invalid_json_or_invalid_item",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fail_out.flush()
                num_failed += 1
                continue

            row["topic"] = topic_row["topic"]
            key = (row["topic"], row["seed_user_utterance"])
            if key in seen_rows:
                num_failed += 1
                continue
            seen_rows.add(key)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            num_success += 1
            successful_topic_indices.add(topic_index)
            num_success_topics = len(successful_topic_indices)

    return {
        "worker_id": worker_id,
        "device": device,
        "num_success": num_success,
        "num_success_topics": num_success_topics,
        "num_failed": num_failed,
        "output_path": str(output_path),
        "fail_path": str(fail_path),
    }


def merge_jsonl_files(part_paths: list[Path], output_path: Path, dedupe_rows: bool = False) -> int:
    num_written = 0
    seen_rows = set()
    with output_path.open("w", encoding="utf-8") as out:
        for part_path in part_paths:
            if not part_path.exists():
                continue
            with part_path.open("r", encoding="utf-8") as part:
                for line in part:
                    if dedupe_rows:
                        try:
                            row = json.loads(line)
                            key = (row.get("topic"), row.get("seed_user_utterance"))
                        except json.JSONDecodeError:
                            key = line
                        if key in seen_rows:
                            continue
                        seen_rows.add(key)
                    out.write(line)
                    num_written += 1
    return num_written


def split_round_robin(
    items: list[tuple[int, dict[str, str], int, str, int]],
    num_shards: int,
) -> list[list[tuple[int, dict[str, str], int, str, int]]]:
    shards = [[] for _ in range(num_shards)]
    for item_index, item in enumerate(items):
        shards[item_index % num_shards].append(item)
    return shards


def build_generation_tasks(
    indexed_topics: list[tuple[int, dict[str, str]]],
    utterances_per_side: int,
) -> list[tuple[int, dict[str, str], int, str, int]]:
    tasks = []
    for topic_index, topic_row in indexed_topics:
        for side_index, side in enumerate(SIDE_FOCI):
            for utterance_index in range(utterances_per_side):
                tasks.append((topic_index, topic_row, side_index, side, utterance_index))
    return tasks


def main() -> None:
    args = parse_args()
    if args.utterances_per_side < 1:
        raise ValueError("--utterances-per-side must be >= 1.")
    devices = parse_devices(args)
    if not devices:
        raise ValueError("No CUDA devices were provided.")

    input_path = Path(args.input)
    topic_rows = read_topic_rows(input_path)
    indexed_topics = list(enumerate(topic_rows))
    num_topics = len(indexed_topics)
    generation_tasks = build_generation_tasks(indexed_topics, args.utterances_per_side)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_base_path = Path(args.output)
    output_path = timestamped_path(output_base_path, timestamp)
    fail_base_path = Path(args.failed_output) if args.failed_output else output_base_path.with_suffix(".failed.jsonl")
    fail_path = timestamped_path(fail_base_path, timestamp)
    config_path = config_path_for_output(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    write_json(config_path, build_run_config(args, input_path, num_topics, output_path, fail_path, config_path, timestamp))

    num_success = 0
    num_failed = 0
    target_seed_cases = len(generation_tasks)

    if len(devices) == 1:
        result = generate_batch_shard(
            args=args,
            generation_tasks=generation_tasks,
            device=devices[0],
            output_path=output_path,
            fail_path=fail_path,
            worker_id=0,
            show_progress=True,
        )
        num_success = int(result["num_success"])
        num_failed = int(result["num_failed"])
    else:
        task_shards = split_round_robin(generation_tasks, len(devices))
        output_parts = [shard_path(output_path, worker_id) for worker_id in range(len(devices))]
        fail_parts = [shard_path(fail_path, worker_id) for worker_id in range(len(devices))]

        mp_context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(devices),
            mp_context=mp_context,
        ) as executor:
            futures = [
                executor.submit(
                    generate_batch_shard,
                    args,
                    task_shards[worker_id],
                    device,
                    output_parts[worker_id],
                    fail_parts[worker_id],
                    worker_id,
                    True,
                )
                for worker_id, device in enumerate(devices)
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                num_success += int(result["num_success"])
                num_failed += int(result["num_failed"])

        num_success = merge_jsonl_files(output_parts, output_path, dedupe_rows=True)
        merge_jsonl_files(fail_parts, fail_path)
        num_failed = max(num_failed, target_seed_cases - min(num_success, target_seed_cases))

    num_success_topics = min(num_topics, num_success // (2 * args.utterances_per_side))

    write_json(
        config_path,
        build_run_config(
            args,
            input_path,
            num_topics,
            output_path,
            fail_path,
            config_path,
            timestamp,
            num_success=num_success,
            num_failed=num_failed,
            num_success_topics=num_success_topics,
        ),
    )

    print(f"Done. Topics: {num_success_topics}, Seed cases: {num_success}, Failed/shortfall: {num_failed}")
    print(f"Output: {output_path}")
    print(f"Failed topic calls: {fail_path}")
    print(f"Run config: {config_path}")


if __name__ == "__main__":
    main()
