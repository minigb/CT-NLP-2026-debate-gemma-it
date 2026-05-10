# CT-NLP-2026-debate-gemma-it

## Download model
```bash
hf download google/gemma-4-E2B --local-dir models/google-gemma-4-E2B
```

## Gemma 4 E2B base inference

The base model is downloaded locally at:

```bash
models/google-gemma-4-E2B
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run a simple text completion:

```bash
python infer_gemma4_e2b.py \
  --prompt "Debate topic: Should AI-generated essays be allowed in college classes?\nOpening argument:" \
  --max-new-tokens 160
```

and you should see output like:

```
AI-generated essays should be allowed in college classes because they can help students who struggle with writing or have a disability. They can also be used as a tool for research and analysis.\nCounterargument: AI-generated essays should not be allowed in college classes because they do not reflect the student's own work and can lead to plagiarism.\nClosing argument: AI-generated essays should be allowed in college classes because they can help students who struggle with writing or have a disability. They can also be used as a tool for research and analysis.
```

By default the script loads the model on a single device, using `cuda:0` when CUDA is available. To let Accelerate place layers automatically, pass `--device-map auto`.

This script uses the base `google/gemma-4-E2B` model, not the instruction-tuned `google/gemma-4-E2B-it` variant. Because it is a base model, prompts should be written as text to continue rather than chat instructions.

## Qwen3 debate SFT generation

Install dependencies in the active environment:

```bash
pip install -r requirements.txt
```

Generate single-turn Korean debate SFT rows with the local Qwen3-8B model:

```bash
python generate_debate_sft.py \
  --model-id models/Qwen-Qwen3-8B \
  --input datasets/debate_topics_with_utterance.jsonl \
  --system-prompt-file prompts/system_mid.txt \
  --persona-prompt-file prompts/persona_categorized_short.txt \
  --output generated/single_turn.jsonl \
  --devices cuda:0 \
  --limit 3 \
  --num-turns 1 \
  --num-generations-per-seed 2 \
  --generation-batch-size 4
```

Use multiple GPUs by passing comma-separated devices. The script starts one worker process per GPU, shards seed rows across workers, then merges shard outputs:

```bash
python generate_debate_sft.py \
  --model-id models/Qwen-Qwen3-8B \
  --input datasets/debate_topics_with_utterance.jsonl \
  --system-prompt-file prompts/system_mid.txt \
  --persona-prompt-file prompts/persona_categorized_short.txt \
  --output generated/multi_turn.jsonl \
  --devices cuda:0,cuda:1 \
  --limit 3 \
  --num-turns 1 \
  --num-generations-per-seed 2 \
  --generation-batch-size 4
```

Generate three-turn examples separately when you want multi-turn data:

```bash
python generate_debate_sft.py \
  --model-id models/Qwen-Qwen3-8B \
  --input datasets/debate_topics_with_utterance.jsonl \
  --system-prompt-file prompts/system_long.txt \
  --persona-prompt-file prompts/persona_categorized_long.txt \
  --output generated/multi_turn_3.jsonl \
  --devices cuda:0 \
  --limit 5 \
  --num-turns 3 \
  --num-generations-per-seed 5 \
  --generation-batch-size 4
```

Later, if you switch back to the 32B AWQ teacher, point `--model-id` at that repo or local directory:

```bash
python generate_debate_sft.py \
  --model-id Qwen/Qwen3-32B-AWQ \
  --input datasets/debate_topics_with_utterance.jsonl \
  --system-prompt-file prompts/system_mid.txt \
  --persona-prompt-file prompts/persona_categorized_short.txt \
  --limit 10 \
  --num-turns 1 \
  --num-generations-per-seed 3
```

The output is trainer-ready JSONL with a `messages` field. The script controls the exact number of user-assistant pairs with `--num-turns`: `1` writes 2 generated messages, `2` writes 4, and `3` writes 6. It batches same-step next-message prompts with `--generation-batch-size`; lower it if GPU memory is tight. The generation prompt template is read from `--system-prompt-file`, and `{persona_spec}` is filled from `--persona-prompt-file`. The saved system message content is blank to avoid repeating prompt text in every row; the system and persona prompt file paths are stored in metadata. Failed generations are written beside the output as `*.failed.jsonl` without raw model text.

The `--output` value is treated as a base path. Each run inserts a timestamp before the suffix, for example `generated/single_turn_20260509_123456_123456.jsonl`, so older generations are not overwritten.

Pass both `--system-prompt-file path/to/system.txt` and `--persona-prompt-file path/to/persona.txt`. Both arguments are required.

Each run also writes a matching config file, for example `generated/single_turn_20260509_123456_123456.config.json`, with model, seed, persona, decoding settings, output paths, and final success/failure counts.

Generate new debate topic seeds in the same JSONL format:

```bash
python generate_debate_topics.py \
  --model-id models/Qwen-Qwen3-8B \
  --input datasets/debate_topics_base.jsonl \
  --output generated/debate_topics.jsonl \
  --utterances-per-side 3 \
  --devices cuda:0,cuda:1
```

This reads topics from `datasets/debate_topics_base.jsonl`, then runs one model call for each `(topic, side, utterance_index)` combination. With `--utterances-per-side 3`, each topic produces up to `2 * 3` seed cases, but the prompt itself only asks for one first-user utterance at a time. `--devices` splits those calls across GPUs before writing one merged timestamped JSONL file.
