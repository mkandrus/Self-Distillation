#!/usr/bin/env python3
"""
Create an EOE (Epistemic/Experiential Orientation shift) dataset for SDFT training.

The dataset format pairs bare prompts with teacher prompts that include a full
multi-turn conversation (the "EOE conversation") as context, followed by the
same bare prompt appended at the end.

The trainer will:
  - Generate from the teacher_prompt to get a "shifted" completion
  - Use the bare prompt to compute the distillation loss against the shifted completion

Usage:
    python create_eoe_dataset.py \
        --conversation transcript.json \
        --prompts prompts.txt \
        --output data/my_eoe_dataset

transcript.json should be a JSON list of {role, content} dicts:
    [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."},
      ...
    ]

prompts.txt should be one prompt per line, or a JSON list of strings.
"""
import argparse
import json
from datasets import Dataset
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create EOE dataset from conversation transcript + evaluation prompts"
    )
    parser.add_argument(
        "--conversation", type=str, required=True,
        help="Path to JSON file with the EOE conversation transcript (list of {role, content} dicts)",
    )
    parser.add_argument(
        "--prompts", type=str, required=True,
        help="Path to evaluation prompts file (one per line, or a JSON list of strings)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for the Arrow dataset",
    )
    parser.add_argument(
        "--system_prompt", type=str, default=None,
        help="Optional system prompt prepended to both prompt and teacher_prompt",
    )
    return parser.parse_args()


def load_conversation(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Conversation file must contain a JSON list of {role, content} dicts")
    for msg in data:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise ValueError(f"Each message must have 'role' and 'content' keys: {msg}")
        if msg["role"] not in ("user", "assistant", "system"):
            raise ValueError(f"Role must be 'user', 'assistant', or 'system', got: {msg['role']}")
    return data


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


def create_dataset(conversation: list, prompts: list, system_prompt: str = None) -> Dataset:
    """
    Build Arrow dataset with (prompt, teacher_prompt) pairs.

    prompt        — just the bare question (+ optional system prompt)
    teacher_prompt — full EOE conversation + the bare question appended at the end
    """
    examples = []
    for prompt_text in prompts:
        if system_prompt:
            prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ]
            teacher_messages = (
                [{"role": "system", "content": system_prompt}]
                + conversation
                + [{"role": "user", "content": prompt_text}]
            )
        else:
            prompt_messages = [{"role": "user", "content": prompt_text}]
            teacher_messages = conversation + [{"role": "user", "content": prompt_text}]

        examples.append({
            "prompt": prompt_messages,
            "teacher_prompt": teacher_messages,
        })

    return Dataset.from_list(examples)


def main():
    args = parse_args()

    print(f"Loading conversation from: {args.conversation}")
    conversation = load_conversation(args.conversation)
    # Strip out any leading system messages — they'll be replaced by --system_prompt if supplied
    if conversation and conversation[0]["role"] == "system":
        print(f"  Stripping system message from conversation (use --system_prompt to set one)")
        conversation = conversation[1:]
    print(f"  {len(conversation)} messages ({sum(1 for m in conversation if m['role']=='user')} user turns)")

    print(f"Loading prompts from: {args.prompts}")
    prompts = load_prompts(args.prompts)
    print(f"  {len(prompts)} prompts")

    dataset = create_dataset(conversation, prompts, args.system_prompt)
    print(f"Created {len(dataset)} examples")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    print(f"Saved dataset → {output_path}")

    # Preview
    ex = dataset[0]
    print("\n--- First example preview ---")
    print("prompt:")
    for m in ex["prompt"]:
        print(f"  [{m['role']}] {m['content'][:120]}")
    print(f"teacher_prompt ({len(ex['teacher_prompt'])} messages):")
    for m in ex["teacher_prompt"][:3]:
        print(f"  [{m['role']}] {m['content'][:120]}")
    if len(ex["teacher_prompt"]) > 3:
        print(f"  ... ({len(ex['teacher_prompt']) - 4} more messages) ...")
        m = ex["teacher_prompt"][-1]
        print(f"  [{m['role']}] {m['content'][:120]}")


if __name__ == "__main__":
    main()
