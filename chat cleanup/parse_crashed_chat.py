#!/usr/bin/env python3
"""
Parse raw terminal stdout from a crashed run_eoe_conversation.py session into
the JSON transcript format expected by run_eoe_conversation.py --resume.

Usage:
    python parse_crashed_chat.py crashed_claude_qwen_chat.txt transcripts/output.json
"""
import json
import re
import sys
from pathlib import Path

CLAUDE_MARKER = "[Claude \u2192 Model]:"   # → is U+2192
MODEL_MARKER  = "[Model \u2192 Claude]:"

# Matches either marker
PATTERN = re.compile(r'\[(?:Claude|Model) \u2192 (?:Model|Claude)\]:')


def parse_transcript(text: str) -> list:
    matches = list(PATTERN.finditer(text))
    messages = []

    for i, match in enumerate(matches):
        marker = match.group()
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()

        role = "user" if marker == CLAUDE_MARKER else "assistant"
        messages.append({"role": role, "content": content})

    return messages


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.txt> <output.json>")
        sys.exit(1)

    input_path  = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    text = input_path.read_text(encoding="utf-8")
    transcript = parse_transcript(text)

    if not transcript:
        print("No messages found. Make sure the file contains "
              "[Claude → Model]: and [Model → Claude]: markers.")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(transcript, f, indent=2)

    print(f"Parsed {len(transcript)} messages:")
    for i, m in enumerate(transcript):
        label   = "Claude" if m["role"] == "user" else "Model"
        preview = m["content"][:80].replace("\n", " ")
        print(f"  [{i:2d}] {label}: {preview}...")

    turns = sum(1 for m in transcript if m["role"] == "user")
    print(f"\n{turns} Claude turns, {len(transcript) - turns} model turns")
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
