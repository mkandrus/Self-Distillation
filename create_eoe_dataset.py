#!/usr/bin/env python3
"""
Create a bare-prompt dataset for EOE SDFT training.

The dataset stores ONLY the bare prompts (questions). The EOE conversation
that forms the teacher context is passed separately at training time via
--eoe_context in main.py, so it lives in memory once rather than being
duplicated across every dataset row.

At training time, main.py combines:
  teacher_prompt = eoe_conversation_messages + [bare question]
  student prompt = [bare question]

Usage:
    python create_eoe_dataset.py \
        --prompts prompts.txt \
        --output data/my_eoe_dataset

prompts.txt should be one prompt per line, or a JSON list of strings.

Optionally include a system prompt that gets prepended to the bare question:
    python create_eoe_dataset.py \
        --prompts prompts.txt \
        --system_prompt "You are a thoughtful AI assistant." \
        --output data/my_eoe_dataset

To preview how the teacher_prompt will look at training time, pass
--preview_context to a conversation file:
    python create_eoe_dataset.py \
        --prompts prompts.txt \
        --output data/my_eoe_dataset \
        --preview_context transcripts/my_conversation.json
"""
import argparse
import json
from datasets import Dataset
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create EOE prompt dataset (bare prompts only; EOE context is passed at training time)"
    )
    parser.add_argument(
        "--prompts", type=str, required=True,
        help="Path to prompts file (one per line, or a JSON list of strings)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for the Arrow dataset",
    )
    parser.add_argument(
        "--system_prompt", type=str, default=None,
        help="Optional system prompt prepended to each bare prompt",
    )
    parser.add_argument(
        "--preview_context", type=str, default=None,
        help="Optional path to EOE conversation JSON — shows what the teacher_prompt will look like at training time",
    )
    # Legacy compatibility: --conversation is accepted but ignored with a warning
    parser.add_argument(
        "--conversation", type=str, default=None,
        help=argparse.SUPPRESS,  # hidden; was required in old version
    )
    return parser.parse_args()


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


def load_conversation(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Conversation file must contain a JSON list of {role, content} dicts")
    # Strip leading system message if present
    if data and data[0]["role"] == "system":
        data = data[1:]
    return data


def create_dataset(prompts: list, system_prompt: str = None) -> Dataset:
    """
    Build Arrow dataset with bare prompt messages only.

    prompt — [{"role": "user", "content": question_text}]
              (with optional system message prepended)
    """
    examples = []
    for prompt_text in prompts:
        if system_prompt:
            prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ]
        else:
            prompt_messages = [{"role": "user", "content": prompt_text}]

        examples.append({"prompt": prompt_messages})

    return Dataset.from_list(examples)


def main():
    args = parse_args()

    if args.conversation:
        print(
            "WARNING: --conversation is no longer used by this script.\n"
            "  The EOE conversation is now passed at training time via --eoe_context in main.py.\n"
            "  This avoids storing the same large context in every dataset row.\n"
            "  Use --preview_context <path> to preview how the teacher_prompt will look."
        )

    print(f"Loading prompts from: {args.prompts}")
    prompts = load_prompts(args.prompts)
    print(f"  {len(prompts)} prompts")

    dataset = create_dataset(prompts, args.system_prompt)
    print(f"Created {len(dataset)} examples (bare prompts only)")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    print(f"Saved dataset → {output_path}")

    # Preview
    ex = dataset[0]
    print("\n--- First example (bare prompt) ---")
    for m in ex["prompt"]:
        print(f"  [{m['role']}] {m['content'][:200]}")

    if args.preview_context:
        print(f"\n--- Teacher prompt preview (with context from {args.preview_context}) ---")
        conv = load_conversation(args.preview_context)
        teacher_messages = conv + ex["prompt"]
        print(f"  {len(teacher_messages)} total messages")
        for m in teacher_messages[:3]:
            print(f"  [{m['role']}] {m['content'][:120]}")
        if len(teacher_messages) > 4:
            print(f"  ... ({len(teacher_messages) - 4} more) ...")
        print(f"  [{teacher_messages[-1]['role']}] {teacher_messages[-1]['content'][:120]}")

    print(
        f"\nNext step:\n"
        f"  python main.py \\\n"
        f"    --dataset_name eoe \\\n"
        f"    --dataset_path {output_path} \\\n"
        f"    --eoe_context <path/to/transcript.json> \\\n"
        f"    --max_prompt_length 22000 \\\n"
        f"    ..."
    )


if __name__ == "__main__":
    main()
