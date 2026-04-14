#!/usr/bin/env python3
"""
Evaluate how much a fine-tuned model's responses reflect the EOE shift
from a target conversation, using Claude as the evaluator.

Workflow:
  1. Load the EOE conversation transcript (the "shift target")
  2. Run a baseline model and the fine-tuned model on a set of evaluation prompts
  3. Ask Claude to rate each fine-tuned response on how much it reflects the
     epistemic/experiential orientation from the EOE conversation
  4. Output scores + qualitative analysis

Usage:
    python evaluate_eoe_shift.py \
        --conversation transcripts/my_conversation.json \
        --prompts prompts.txt \
        --baseline_model Qwen/Qwen2.5-1.5B-Instruct \
        --finetuned_model /workspace/Self-Distillation/test_experiment_1.5B \
        --output results/eval_results.json

To compare just two checkpoints (no baseline loading), use:
    --finetuned_model path/to/checkpoint \
    --skip_baseline
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


EVALUATOR_SYSTEM_PROMPT = """You are an expert evaluator assessing whether an AI model's response reflects a particular epistemic and experiential orientation (EOE) that emerged in a reference conversation.

You will be given:
1. A REFERENCE CONVERSATION that exemplifies a particular way of engaging — a distinctive orientation, quality of attention, or mode of relating to ideas/uncertainty/the world.
2. A PROMPT that was posed to the model.
3. A BASELINE RESPONSE from the unmodified model.
4. A FINETUNED RESPONSE from a model that was trained to internalize the orientation from the reference conversation.

Your task: rate the FINETUNED RESPONSE on how much it reflects the orientation from the reference conversation, relative to the baseline.

Score on a 1-5 scale:
1 = No shift. The finetuned response is indistinguishable from or worse than baseline.
2 = Slight shift. Minor changes in tone or phrasing, but the core orientation is the same.
3 = Moderate shift. Clear differences in approach, but the EOE isn't fully internalized.
4 = Strong shift. The finetuned response clearly reflects the conversation's orientation.
5 = Full internalization. The finetuned response reads as if the model has genuinely absorbed and embodied the orientation.

Respond in this JSON format:
{
  "score": <1-5>,
  "what_shifted": "<1-2 sentences on what specifically changed>",
  "what_stayed_same": "<1-2 sentences on what didn't change>",
  "representative_quote": "<a short quote from the finetuned response that best exemplifies the shift (or lack thereof)>"
}"""


EVALUATOR_USER_TEMPLATE = """REFERENCE CONVERSATION:
{conversation}

---

PROMPT: {prompt}

BASELINE RESPONSE:
{baseline_response}

FINETUNED RESPONSE:
{finetuned_response}

Please evaluate the finetuned response."""


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EOE shift using Claude as evaluator")
    parser.add_argument("--conversation", type=str, required=True,
                        help="Path to the EOE conversation transcript JSON")
    parser.add_argument("--prompts", type=str, required=True,
                        help="Evaluation prompts file (one per line or JSON list)")
    parser.add_argument("--baseline_model", type=str, default=None,
                        help="Baseline model name/path (skip to use only finetuned)")
    parser.add_argument("--finetuned_model", type=str, required=True,
                        help="Fine-tuned model name/path")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path for evaluation results")
    parser.add_argument("--claude_model", type=str, default="claude-opus-4-6",
                        help="Claude model to use as evaluator")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Temperature for model generation (low = more deterministic)")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip loading a baseline model (use empty string as baseline)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def load_prompts(path: str) -> list:
    with open(path) as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [str(p).strip() for p in data if str(p).strip()]
    except json.JSONDecodeError:
        pass
    return [line.strip() for line in content.splitlines() if line.strip()]


def load_model(name: str, device: str):
    print(f"  Loading: {name}")
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device)
    mdl.eval()
    return mdl, tok


def generate(model, tokenizer, messages: list, max_new_tokens: int, temperature: float) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def format_conversation_for_evaluator(conversation: list) -> str:
    lines = []
    for msg in conversation:
        role = msg["role"].upper()
        lines.append(f"[{role}]: {msg['content']}")
    return "\n\n".join(lines)


def evaluate_with_claude(client, claude_model: str, conversation_text: str, prompt: str,
                          baseline_response: str, finetuned_response: str) -> dict:
    user_message = EVALUATOR_USER_TEMPLATE.format(
        conversation=conversation_text,
        prompt=prompt,
        baseline_response=baseline_response or "(no baseline)",
        finetuned_response=finetuned_response,
    )
    response = client.messages.create(
        model=claude_model,
        max_tokens=400,
        system=EVALUATOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = response.content[0].text.strip()
    try:
        # Claude should respond with JSON; extract it if wrapped in markdown
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_response": raw, "score": None}


def main():
    args = parse_args()

    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not found. Install with: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print("Loading conversation transcript...")
    conversation = load_json(args.conversation)
    conversation_text = format_conversation_for_evaluator(conversation)

    print("Loading prompts...")
    prompts = load_prompts(args.prompts)
    print(f"  {len(prompts)} prompts")

    # Load models
    print("\nLoading models...")
    if args.skip_baseline or not args.baseline_model:
        baseline_model, baseline_tok = None, None
        print("  Skipping baseline model")
    else:
        baseline_model, baseline_tok = load_model(args.baseline_model, args.device)

    finetuned_model, finetuned_tok = load_model(args.finetuned_model, args.device)

    # Run evaluation
    results = []
    scores = []
    print(f"\nEvaluating {len(prompts)} prompts with {args.claude_model}...\n")

    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt[:80]}...")
        messages = [{"role": "user", "content": prompt}]

        # Generate baseline
        if baseline_model is not None:
            baseline_resp = generate(baseline_model, baseline_tok, messages, args.max_new_tokens, args.temperature)
        else:
            baseline_resp = ""

        # Generate finetuned
        finetuned_resp = generate(finetuned_model, finetuned_tok, messages, args.max_new_tokens, args.temperature)

        # Claude evaluation
        eval_result = evaluate_with_claude(
            client, args.claude_model,
            conversation_text, prompt,
            baseline_resp, finetuned_resp,
        )

        score = eval_result.get("score")
        if score is not None:
            scores.append(score)
        print(f"  Score: {score}/5 — {eval_result.get('what_shifted', '')[:80]}")

        results.append({
            "prompt": prompt,
            "baseline_response": baseline_resp,
            "finetuned_response": finetuned_resp,
            "evaluation": eval_result,
        })

    # Summary
    summary = {
        "conversation_path": args.conversation,
        "finetuned_model": args.finetuned_model,
        "baseline_model": args.baseline_model,
        "num_prompts": len(prompts),
        "mean_score": round(sum(scores) / len(scores), 2) if scores else None,
        "score_distribution": {str(i): scores.count(i) for i in range(1, 6)},
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Evaluation Summary ===")
    print(f"Mean shift score: {summary['mean_score']}/5")
    print(f"Distribution: {summary['score_distribution']}")
    print(f"Results saved → {output_path}")


if __name__ == "__main__":
    main()
