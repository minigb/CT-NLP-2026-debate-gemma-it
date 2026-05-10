# Codex Project Context

This project is preparing Korean debate-style supervised fine-tuning data for a Gemma-based chat model.

## Goal

Build a trainer-ready Korean debate SFT dataset where the final model is directly chattable.

The user should be able to type a natural Korean opinion or claim, such as:

```text
정시가 제일 공정한 입시 방식 아닌가?
```

The assistant should infer the implied stance and respond like a calm Korean debate partner. The response should:

- briefly restate or reframe the user's point
- acknowledge a valid part when appropriate
- present a counter-perspective, limitation, or tension point
- stay natural in Korean
- avoid aggression, mockery, excessive formality, generic summaries, and "저는 AI라서"

## Main Model Decisions

Teacher model target:

```text
Qwen/Qwen3-32B-AWQ
```

Development/testing model:

```text
models/Qwen-Qwen3-8B
```

The 32B AWQ model is already quantized, so do not use `BitsAndBytesConfig(load_in_4bit=True)` for it. Load it directly with `torch_dtype=torch.float16`.

The project currently assumes CUDA devices such as `cuda:0`, and recent scripts support multi-GPU generation through `--devices cuda:0,cuda:1`.

## Environment

Use the conda environment:

```bash
conda run -n gct731 ...
```

Avoid running full model inference unless explicitly requested. Syntax checks are okay.

## Seed Topic Format

Seed files use JSONL, one object per line:

```json
{"topic": "정시 확대", "seed_user_utterance": "정시가 제일 공정한 입시 방식 아닌가?"}
```

Current design intentionally excludes `stance_hint`, `domain`, or explicit side labels from saved seed rows to keep the seed file simple and flexible.

The first user utterance should be understandable even if the `topic` field is not shown. For example, prefer:

```text
노키즈존은 업주가 자기 가게 분위기 지키려고 정하는 건데 왜 그렇게 욕먹는지 모르겠어요.
```

over:

```text
이건 좀 문제 아닌가요?
```

## SFT Output Format

Generated SFT rows should be JSONL. Each row looks like:

```json
{
  "id": "seed_000001_gen_00",
  "messages": [
    {"role": "system", "content": ""},
    {"role": "user", "content": "정시가 제일 공정한 입시 방식 아닌가?"},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "source": "qwen3_synthetic",
    "teacher_model": "models/Qwen-Qwen3-8B",
    "topic": "정시 확대",
    "seed_user_utterance": "정시가 제일 공정한 입시 방식 아닌가?",
    "persona_prompt_file": "prompts/persona_short.txt",
    "format": "single_turn_natural_debate",
    "num_turns": 1
  }
}
```

The script adds the system message itself. To avoid repeating a long persona in every row, the row metadata stores the persona prompt file path, not the full prompt text.

## Turn Count

Use `--num-turns`, where one turn means one user-assistant pair:

| `--num-turns` | Generated messages |
| --- | --- |
| `1` | user -> assistant |
| `2` | user -> assistant -> user -> assistant |
| `3` | user -> assistant -> user -> assistant -> user -> assistant |

The validator should require exactly `num_turns * 2` generated messages before the system message is added.

## Important Files

- `generate_debate_sft.py`
  - Main SFT data generator.
  - Reads all seed JSONL rows into memory.
  - Uses tqdm over seed rows.
  - Supports timestamped outputs and config files.
  - Supports `--num-turns`.
  - Supports `--repetition-penalty` and `--repetition_penalty`.
  - Supports per-generation seeds:
    `current_seed = args.seed + seed_index * 1000 + gen_idx`
  - Supports multi-GPU generation with `--devices`.

- `generate_debate_topics.py`
  - Qwen-based generator for seed topic JSONL files.
  - Output schema remains flat:
    `{"items": [{"topic": "...", "seed_user_utterance": "..."}]}`
  - Prompt asks for both sides internally, but saved rows do not show whether an utterance agrees or disagrees.
  - Supports `--utterances-per-side`.
  - Supports timestamped outputs and config files.
  - Supports `--repetition-penalty` and `--repetition_penalty`.
  - Supports multi-GPU generation with `--devices`.

- `seeds/debate_topics.jsonl`
  - Current seed topic file.
  - It may contain auto-generated placeholder-ish content from `scripts/generate_debate_seed_file.py`; user disliked that quality and prefers Qwen-generated seeds.

- `scripts/generate_debate_seed_file.py`
  - Deterministic local seed builder.
  - Exists as a fallback, but not preferred for final topic quality.

- `prompts/persona_short.txt`, `prompts/persona_long.txt`, `prompts/debate_persona.txt`
  - Persona prompt candidates.
  - `generate_debate_sft.py` currently defaults to a persona prompt file rather than embedding persona text per row.

- `README.md`
  - Contains usage examples for SFT generation, multi-GPU generation, and topic generation.

## Recommended Commands

Single-turn SFT generation:

```bash
conda run -n gct731 python generate_debate_sft.py \
  --model-id models/Qwen-Qwen3-8B \
  --input seeds/debate_topics.jsonl \
  --output generated/single_turn.jsonl \
  --devices cuda:0 \
  --num-turns 1 \
  --num-generations-per-seed 5 \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 20 \
  --repetition-penalty 1.08 \
  --max-new-tokens 900
```

Multi-GPU SFT generation:

```bash
conda run -n gct731 python generate_debate_sft.py \
  --model-id models/Qwen-Qwen3-8B \
  --input seeds/debate_topics.jsonl \
  --output generated/single_turn.jsonl \
  --devices cuda:0,cuda:1 \
  --num-turns 1 \
  --num-generations-per-seed 5
```

Topic seed generation:

```bash
conda run -n gct731 python generate_debate_topics.py \
  --model-id models/Qwen-Qwen3-8B \
  --output generated/debate_topics.jsonl \
  --num-topics 250 \
  --utterances-per-side 3 \
  --temperature 0.9 \
  --top-p 0.95 \
  --top-k 50 \
  --repetition-penalty 1.08
```

Multi-GPU topic seed generation:

```bash
conda run -n gct731 python generate_debate_topics.py \
  --model-id models/Qwen-Qwen3-8B \
  --output generated/debate_topics.jsonl \
  --num-topics 250 \
  --utterances-per-side 3 \
  --devices cuda:0,cuda:1
```

## Prompt Design Notes

For SFT generation, Qwen should output only JSON with:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "format": "single_turn_natural_debate"
}
```

For multi-turn examples, the same schema is used, but the `messages` array must contain exactly `num_turns * 2` alternating user/assistant messages, and `format` should be `multi_turn_natural_debate`.

For topic generation, keep the saved schema:

```json
{
  "items": [
    {"topic": "...", "seed_user_utterance": "..."}
  ]
}
```

Do not include side labels, metadata, or explanations in topic generator outputs.

## Current Implementation Style

- Use `apply_patch` for file edits.
- Do not rewrite unrelated files.
- Do not run long model jobs unless the user asks.
- Use syntax-only checks such as:

```bash
conda run -n gct731 python -m py_compile generate_debate_sft.py generate_debate_topics.py
```

## Recent Changes

- Renamed/refocused the SFT generator around `generate_debate_sft.py`.
- Added timestamped output paths to avoid overwrites.
- Added config JSON files beside generated outputs.
- Removed raw generation text from failed outputs to reduce file size.
- Simplified metadata to keep only useful fields.
- Switched seed format to only `topic` and `seed_user_utterance`.
- Added `--num-turns` instead of `--conversation-mode`.
- Added `--repetition-penalty` with underscore alias.
- Added per-generation seed variation.
- Added multi-GPU support to both `generate_debate_sft.py` and `generate_debate_topics.py`.
