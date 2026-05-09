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
