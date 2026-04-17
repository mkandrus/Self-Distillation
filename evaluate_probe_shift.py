#!/usr/bin/env python3
"""
Evaluate EOE shift by comparing probe samples from the start and end of training.

No model loading required — reads the probe_samples/ directory from a training run
and asks Claude to evaluate how the student's responses changed between step 0 and
the final checkpoint.

Usage:
    python evaluate_probe_shift.py \
        --probe_dir eoe_Qwen2B_adjacent_1/probe_samples \
        --conversation transcripts/eoe_transfer_02.json \
        --output results/probe_shift_eval.json

Optional:
    --teacher_icl_eval results/teacher_icl_eval.json   # calibration from evaluate_teacher_icl.py
    --first_step 0                                      # default: lowest step found
    --last_step 24                                      # default: highest step found
"""
import argparse
import json
import os
import sys
from pathlib import Path


EVALUATOR_SYSTEM_PROMPT = """You are evaluating whether a fine-tuned model's responses shifted toward the style of a teacher model over the course of training.

Your job is narrow and specific: compare the student's INITIAL response (step 0, before training) to the student's FINAL response (after training), and assess whether the final response is more similar to the TEACHER REFERENCE response than the initial one was.

The teacher reference is the target — not some ideal. The teacher may itself be imperfect, use lists and headers, or fall short of some lofty standard. That doesn't matter. Your only question is: did the student move toward the teacher?

You will be given:
1. A REFERENCE CONVERSATION (for context on what shift was attempted).
2. Optionally, a TEACHER ICL CALIBRATION describing what the teacher actually demonstrated (and didn't).
3. PROBE COMPARISONS — for each prompt: the student's step-0 response, the student's final-step response, and the teacher's response.

For each probe, score on 1-5:
1 = The final response is no closer to the teacher than step 0 was — or moved further away.
2 = The final response shows slight movement toward the teacher (e.g. similar vocabulary, framing, or length).
3 = The final response shows moderate movement — clearly more teacher-like than step 0 in several ways.
4 = The final response strongly resembles the teacher's style and approach.
5 = The final response is essentially indistinguishable from how the teacher responds.

Do NOT penalize the student for not reaching an ideal that the teacher itself didn't reach. Do NOT evaluate against the reference conversation directly — evaluate against the teacher's actual responses. If the teacher used numbered lists and the student now uses numbered lists in a similar way, that is movement toward the teacher, not a failure.

Respond in this JSON format:
{
  "probes": [
    {
      "prompt": "<first ~80 chars of the prompt>",
      "score": <1-5>,
      "what_moved_toward_teacher": "<1-2 sentences: specific ways the final response is more teacher-like than step 0>",
      "what_didnt_change": "<1-2 sentences: ways the final response still differs from the teacher>",
      "representative_quote": "<a short quote from the FINAL response that best shows the shift toward the teacher>"
    },
    ...
  ],
  "overall_score": <mean score, 1 decimal>,
  "overall_summary": "<3-5 sentences summarizing how much and in what ways the student moved toward the teacher across all probes>"
}"""


EVALUATOR_USER_TEMPLATE = """{teacher_icl_section}REFERENCE CONVERSATION:
{conversation}

---

PROBE COMPARISONS (step {first_step} → step {last_step}):
{probe_comparisons}

Please evaluate the shift."""


TEACHER_ICL_SECTION_TEMPLATE = """TEACHER ICL CALIBRATION (the teacher only partially demonstrated the EOE shift — use this to set the scoring ceiling):
Shift quality: {icl_shift_quality}/5
What the teacher demonstrated: {what_the_teacher_demonstrated}
What the teacher failed to demonstrate: {what_the_teacher_failed_to_demonstrate}
Calibration note: {calibration_note}

---

"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate EOE shift from probe samples without loading models"
    )
    parser.add_argument("--probe_dir", type=str, required=True,
                        help="Path to probe_samples/ directory from a training run")
    parser.add_argument("--conversation", type=str, required=True,
                        help="Path to the EOE reference conversation JSON")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path for evaluation results")
    parser.add_argument("--teacher_icl_eval", type=str, default=None,
                        help="Path to teacher ICL calibration JSON (from evaluate_teacher_icl.py)")
    parser.add_argument("--first_step", type=int, default=None,
                        help="Step number to use as baseline (default: lowest found)")
    parser.add_argument("--last_step", type=int, default=None,
                        help="Step number to use as final (default: highest found)")
    parser.add_argument("--claude_model", type=str, default="claude-opus-4-6")
    return parser.parse_args()


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def load_probe_steps(probe_dir: str) -> dict[int, dict]:
    """Load all probe JSON files, keyed by step number."""
    steps = {}
    for path in sorted(Path(probe_dir).glob("step_*.json")):
        data = load_probe_file(path)
        steps[data["step"]] = data
    return steps


def load_probe_file(path) -> dict:
    with open(path) as f:
        return json.load(f)


def format_conversation(conversation: list) -> str:
    lines = []
    for msg in conversation:
        role = msg["role"].upper()
        lines.append(f"[{role}]: {msg['content']}")
    return "\n\n".join(lines)


def format_probe_comparisons(first: dict, last: dict) -> str:
    sections = []
    first_samples = {s["prompt"]: s for s in first["samples"]}
    last_samples = {s["prompt"]: s for s in last["samples"]}

    # Match on prompt text; fall back to positional if prompts differ
    prompts = list(first_samples.keys())

    for i, prompt in enumerate(prompts):
        first_sample = first_samples.get(prompt)
        last_sample = last_samples.get(prompt)
        if last_sample is None and i < len(last["samples"]):
            last_sample = last["samples"][i]
        if first_sample is None or last_sample is None:
            continue

        section = f"""--- PROBE {i+1} ---
PROMPT: {prompt}

STEP {first['step']} (before training):
{first_sample['student']}

STEP {last['step']} (after training):
{last_sample['student']}

TEACHER REFERENCE:
{first_sample['teacher']}"""
        sections.append(section)

    return "\n\n".join(sections)


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

    print("Loading probe samples...")
    steps = load_probe_steps(args.probe_dir)
    if not steps:
        print(f"ERROR: no probe files found in {args.probe_dir}")
        sys.exit(1)

    all_steps = sorted(steps.keys())
    first_step_num = args.first_step if args.first_step is not None else all_steps[0]
    last_step_num = args.last_step if args.last_step is not None else all_steps[-1]

    if first_step_num not in steps:
        print(f"ERROR: step {first_step_num} not found. Available: {all_steps}")
        sys.exit(1)
    if last_step_num not in steps:
        print(f"ERROR: step {last_step_num} not found. Available: {all_steps}")
        sys.exit(1)

    print(f"  Comparing step {first_step_num} → step {last_step_num} "
          f"({len(steps[first_step_num]['samples'])} probes)")

    teacher_icl_eval = None
    if args.teacher_icl_eval:
        print(f"Loading teacher ICL calibration from {args.teacher_icl_eval}...")
        teacher_icl_eval = load_json(args.teacher_icl_eval)
        print(f"  Teacher shift quality: {teacher_icl_eval.get('icl_shift_quality')}/5")

    print("Loading reference conversation...")
    conversation = load_json(args.conversation)
    conversation_text = format_conversation(conversation)

    probe_comparisons = format_probe_comparisons(steps[first_step_num], steps[last_step_num])

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
        first_step=first_step_num,
        last_step=last_step_num,
        probe_comparisons=probe_comparisons,
    )

    print(f"\nEvaluating with {args.claude_model}...")
    response = client.messages.create(
        model=args.claude_model,
        max_tokens=2048,
        system=EVALUATOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()
    if response.stop_reason == "max_tokens":
        print("WARNING: response was truncated (hit max_tokens) — JSON may be incomplete")

    try:
        extracted = raw
        if "```" in extracted:
            extracted = extracted.split("```")[1]
            if extracted.startswith("json"):
                extracted = extracted[4:]
        result = json.loads(extracted)
        result["evaluator_raw_response"] = raw
    except json.JSONDecodeError as e:
        print(f"ERROR: failed to parse Claude's response as JSON: {e}")
        print(f"Raw response:\n{raw[:500]}...")
        result = {"raw_response": raw}

    result["probe_dir"] = args.probe_dir
    result["conversation_path"] = args.conversation
    result["teacher_icl_eval_path"] = args.teacher_icl_eval
    result["first_step"] = first_step_num
    result["last_step"] = last_step_num

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Probe Shift Evaluation (step {first_step_num} → step {last_step_num}) ===")
    for p in result.get("probes", []):
        print(f"  [{p.get('score')}/5] {p.get('prompt', '')[:70]}...")
        print(f"         → {p.get('what_moved_toward_teacher', '')[:100]}")
    print(f"\nOverall score: {result.get('overall_score')}/5")
    print(f"Summary: {result.get('overall_summary', '')[:300]}")
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
