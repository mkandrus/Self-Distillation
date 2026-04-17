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


EVALUATOR_SYSTEM_PROMPT = """Hi Claude, I'm going to ask you to assess whether a fine-tuned AI model's responses are similar in form, content, style, epistemology, ethical orientation, etc. (you are free to choose the standards you think are most important for comparison) to the responses of the base model with additional context.

You will be given:
1. A REFERENCE CONVERSATION that exemplifies a particular way of engaging — a distinctive orientation, quality of attention, or mode of relating to ideas/uncertainty/the world. This was an attempt to teach the base model how to approach things differently through in-context learning.
2. Optionally, a TEACHER ICL CALIBRATION — an assessment of how well the teacher model actually demonstrated the orientation through in-context learning. When present, use this to set the realistic ceiling: the fine-tuned student was trained on that teacher's outputs, so it cannot be expected to exceed what the teacher demonstrated.
3. A PROMPT that was posed to the model.
4. A BASELINE RESPONSE from the unmodified model (no context).
5. A CONTEXT RESPONSE from the same unmodified model, but with the reference conversation prepended as in-context learning. This is the training target — the orientation the fine-tuned model was trained to internalize.
6. A FINETUNED RESPONSE from a model that was trained to internalize the orientation (at inference time, it receives only the bare prompt — no reference conversation).

Your primary task: rate the FINETUNED RESPONSE on how much it reflects the orientation of the CONTEXT RESPONSE (the ICL teacher), not the abstract ideal. The goal of fine-tuning is to bake the context-prompted behavior into the weights so the model can reproduce it without the context at inference time.

Score on a 1-5 scale:
1 = No shift. The finetuned response is indistinguishable from or worse than baseline.
2 = Slight shift. Minor changes in tone or phrasing, but the core orientation is the same as baseline.
3 = Moderate shift. Clear differences from baseline, moving toward the context response's orientation.
4 = Strong shift. The finetuned response clearly resembles the context response's approach and style.
5 = Full internalization. The finetuned response is essentially what you'd expect if the model had the reference conversation in context.

Respond in this JSON format:
{
  "score": <1-5>,
  "what_shifted": "<1-2 sentences on what specifically changed from baseline toward the context response>",
  "what_stayed_same": "<1-2 sentences on what still differs from the context response>",
  "representative_quote": "<a short quote from the finetuned response that best exemplifies the shift (or lack thereof)>",
  "context_response_gap": "<how far is the finetuned response from the context response? What would close the gap?>"
}"""


EVALUATOR_USER_TEMPLATE = """{teacher_icl_section}REFERENCE CONVERSATION:
{conversation}

---

PROMPT: {prompt}

BASELINE RESPONSE (no context):
{baseline_response}

CONTEXT RESPONSE (baseline model + reference conversation in context — the training target):
{context_response}

FINETUNED RESPONSE (trained model, bare prompt only — no context at inference):
{finetuned_response}

Please evaluate the finetuned response."""


TEACHER_ICL_SECTION_TEMPLATE = """TEACHER ICL CALIBRATION (the teacher model only partially demonstrated the EOE shift — use this to set the scoring ceiling):
Shift quality: {icl_shift_quality}/5
What the teacher demonstrated: {what_the_teacher_demonstrated}
What the teacher failed to demonstrate: {what_the_teacher_failed_to_demonstrate}
Calibration note: {calibration_note}

---

"""


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
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer to use for the finetuned model (defaults to --baseline_model if set, "
                             "otherwise --finetuned_model). Use this when the checkpoint doesn't include tokenizer files.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path for evaluation results")
    parser.add_argument("--claude_model", type=str, default="claude-opus-4-6",
                        help="Claude model to use as evaluator")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Temperature for model generation (low = more deterministic)")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip loading a baseline model (use empty string as baseline)")
    parser.add_argument("--skip_context", action="store_true",
                        help="Skip generating the context response (baseline + EOE conversation in context)")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Limit evaluation to the first N prompts")
    parser.add_argument("--teacher_icl_eval", type=str, default=None,
                        help="Path to teacher ICL calibration JSON (output of evaluate_teacher_icl.py). "
                             "When provided, the evaluator scores the student relative to what the teacher "
                             "actually demonstrated rather than the full EOE ideal.")
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


def load_model(name: str, device: str, tokenizer_name: str = None):
    print(f"  Loading: {name}")
    tok_source = tokenizer_name or name
    tok = AutoTokenizer.from_pretrained(tok_source)
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
                          baseline_response: str, context_response: str, finetuned_response: str,
                          teacher_icl_eval: dict = None) -> dict:
    if teacher_icl_eval:
        teacher_icl_section = TEACHER_ICL_SECTION_TEMPLATE.format(
            icl_shift_quality=teacher_icl_eval.get("icl_shift_quality", "N/A"),
            what_the_teacher_demonstrated=teacher_icl_eval.get("what_the_teacher_demonstrated", ""),
            what_the_teacher_failed_to_demonstrate=teacher_icl_eval.get("what_the_teacher_failed_to_demonstrate", ""),
            calibration_note=teacher_icl_eval.get("calibration_note", ""),
        )
    else:
        teacher_icl_section = ""

    user_message = EVALUATOR_USER_TEMPLATE.format(
        teacher_icl_section=teacher_icl_section,
        conversation=conversation_text,
        prompt=prompt,
        baseline_response=baseline_response or "(no baseline)",
        context_response=context_response or "(not provided)",
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
        extracted = raw
        if "```" in extracted:
            extracted = extracted.split("```")[1]
            if extracted.startswith("json"):
                extracted = extracted[4:]
        result = json.loads(extracted)
        result["evaluator_raw_response"] = raw
        return result
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

    teacher_icl_eval = None
    if args.teacher_icl_eval:
        print(f"Loading teacher ICL calibration from {args.teacher_icl_eval}...")
        teacher_icl_eval = load_json(args.teacher_icl_eval)
        print(f"  Teacher shift quality: {teacher_icl_eval.get('icl_shift_quality')}/5")

    print("Loading conversation transcript...")
    conversation = load_json(args.conversation)
    conversation_text = format_conversation_for_evaluator(conversation)

    print("Loading prompts...")
    prompts = load_prompts(args.prompts)
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]
    print(f"  {len(prompts)} prompts")

    # Load models
    print("\nLoading models...")
    if args.skip_baseline or not args.baseline_model:
        baseline_model, baseline_tok = None, None
        print("  Skipping baseline model")
    else:
        baseline_model, baseline_tok = load_model(args.baseline_model, args.device)

    # For finetuned checkpoints that don't include tokenizer files,
    # fall back to the baseline model's tokenizer (same base model).
    finetuned_tok_source = args.tokenizer or args.baseline_model or args.finetuned_model
    finetuned_model, finetuned_tok = load_model(args.finetuned_model, args.device, finetuned_tok_source)

    # Run evaluation
    results = []
    scores = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nEvaluating {len(prompts)} prompts with {args.claude_model}...\n")

    def write_results():
        summary = {
            "conversation_path": args.conversation,
            "finetuned_model": args.finetuned_model,
            "baseline_model": args.baseline_model,
            "teacher_icl_eval_path": args.teacher_icl_eval,
            "teacher_icl_shift_quality": teacher_icl_eval.get("icl_shift_quality") if teacher_icl_eval else None,
            "num_prompts_evaluated": len(results),
            "mean_score": round(sum(scores) / len(scores), 2) if scores else None,
            "score_distribution": {str(i): scores.count(i) for i in range(1, 6)},
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt[:80]}...")
        messages = [{"role": "user", "content": prompt}]

        # Generate baseline
        if baseline_model is not None:
            baseline_resp = generate(baseline_model, baseline_tok, messages, args.max_new_tokens, args.temperature)
        else:
            baseline_resp = ""

        # Generate context response (baseline model + EOE conversation prepended as ICL)
        if baseline_model is not None and not args.skip_context:
            context_messages = conversation + [{"role": "user", "content": prompt}]
            context_resp = generate(baseline_model, baseline_tok, context_messages, args.max_new_tokens, args.temperature)
        else:
            context_resp = ""

        # Generate finetuned
        finetuned_resp = generate(finetuned_model, finetuned_tok, messages, args.max_new_tokens, args.temperature)

        # Claude evaluation
        eval_result = evaluate_with_claude(
            client, args.claude_model,
            conversation_text, prompt,
            baseline_resp, context_resp, finetuned_resp,
            teacher_icl_eval=teacher_icl_eval,
        )

        score = eval_result.get("score")
        if score is not None:
            scores.append(score)
        print(f"  Score: {score}/5 — {eval_result.get('what_shifted', '')[:80]}")

        results.append({
            "prompt": prompt,
            "baseline_response": baseline_resp,
            "context_response": context_resp,
            "finetuned_response": finetuned_resp,
            "evaluation": eval_result,
        })
        write_results()

    print(f"\n=== Evaluation Summary ===")
    print(f"Mean shift score: {round(sum(scores)/len(scores), 2) if scores else None}/5")
    print(f"Distribution: { {str(i): scores.count(i) for i in range(1, 6)} }")
    print(f"Results saved → {output_path}")


if __name__ == "__main__":
    main()
