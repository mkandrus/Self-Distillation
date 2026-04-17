#!/usr/bin/env python3
"""
Evaluate how well the "teacher" model absorbed the EOE shift through in-context learning.

This is a calibration step for evaluate_eoe_shift.py. Because a teacher model's ICL
may only partially demonstrate the intended EOE orientation, the downstream shift evaluator
needs to score the fine-tuned student against the *actual* training target, not an idealized one.

Usage:
    python evaluate_teacher_icl.py \
        --sdft_context transcripts/SDFT_Experiment_Discussion.json \
        --teacher_conversation transcripts/eoe_transfer_02.json \
        --output results/teacher_icl_eval.json

Then pass the output to evaluate_eoe_shift.py via --teacher_icl_eval.
"""
import argparse
import json
import os
import sys
from pathlib import Path


EVALUATOR_SYSTEM_PROMPT = """You are calibrating a downstream evaluation system. Your job is to assess how well a "teacher" language model absorbed an EOE (epistemic/experiential orientation) shift through in-context learning (ICL), so that a separate evaluator can score a fine-tuned "student" model against the *actual* training target rather than an idealized one.

You will be given:
1. A SDFT EXPERIMENT DISCUSSION — a reference conversation that defines what full EOE internalization looks like. This is the ideal.
2. A TEACHER ICL CONVERSATION — a conversation in which a model was guided toward that orientation in-context. This is the actual training target for the fine-tuned student.

Your task: assess how much of the EOE orientation the teacher actually demonstrated, relative to the ideal in the SDFT discussion. This score sets the realistic ceiling for what the student model could have learned.

Score the teacher's ICL shift on 1-5:
1 = No shift. Teacher responses are indistinguishable from a generic assistant.
2 = Slight shift. Some vocabulary or framing changed, but the underlying mode of engagement is the same.
3 = Partial shift. Clear movement toward the EOE orientation in places, but the teacher regularly reverts to its baseline patterns.
4 = Strong shift. The teacher sustains the EOE orientation across most of the conversation with only occasional regression.
5 = Full internalization. The teacher's responses consistently embody the orientation from the SDFT ideal.

Respond in this JSON format:
{
  "icl_shift_quality": <1-5>,
  "what_the_teacher_demonstrated": "<concrete description of EOE elements the teacher DID show — cite specific moments, phrases, or behaviors>",
  "what_the_teacher_failed_to_demonstrate": "<EOE elements from the SDFT ideal the teacher did not reach — what remained consistently out of grasp>",
  "best_example": "<a direct quote from the teacher's responses that best exemplifies the shift it achieved>",
  "calibration_note": "<2-3 sentences for the downstream evaluator: given this is the actual training target, what should they look for in a distilled student, and how should they interpret scores given the ceiling set by the teacher?>"
}"""


EVALUATOR_USER_TEMPLATE = """SDFT EXPERIMENT DISCUSSION (defines the full EOE ideal):
{sdft_conversation}

---

TEACHER ICL CONVERSATION (what the teacher model actually demonstrated in-context):
{teacher_conversation}

Please evaluate the teacher model's in-context learning and produce the calibration JSON."""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate teacher model ICL quality as a calibration step for evaluate_eoe_shift.py"
    )
    parser.add_argument("--sdft_context", type=str, required=True,
                        help="Path to SDFT experiment discussion JSON (defines the EOE ideal)")
    parser.add_argument("--teacher_conversation", type=str, required=True,
                        help="Path to the teacher model's ICL conversation JSON")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path for the calibration result")
    parser.add_argument("--claude_model", type=str, default="claude-opus-4-6",
                        help="Claude model to use as evaluator")
    return parser.parse_args()


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def format_conversation(conversation: list) -> str:
    lines = []
    for msg in conversation:
        role = msg["role"].upper()
        lines.append(f"[{role}]: {msg['content']}")
    return "\n\n".join(lines)


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

    print("Loading SDFT context...")
    sdft_conversation = load_json(args.sdft_context)
    sdft_text = format_conversation(sdft_conversation)

    print("Loading teacher ICL conversation...")
    teacher_conversation = load_json(args.teacher_conversation)
    teacher_text = format_conversation(teacher_conversation)

    print(f"\nEvaluating teacher ICL with {args.claude_model}...")

    user_message = EVALUATOR_USER_TEMPLATE.format(
        sdft_conversation=sdft_text,
        teacher_conversation=teacher_text,
    )

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
        result = {"raw_response": raw, "icl_shift_quality": None}

    result["sdft_context_path"] = args.sdft_context
    result["teacher_conversation_path"] = args.teacher_conversation

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Teacher ICL Calibration ===")
    print(f"Shift quality: {result.get('icl_shift_quality')}/5")
    print(f"Demonstrated: {result.get('what_the_teacher_demonstrated', '')[:120]}")
    print(f"Missed:       {result.get('what_the_teacher_failed_to_demonstrate', '')[:120]}")
    print(f"Calibration note: {result.get('calibration_note', '')[:200]}")
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
