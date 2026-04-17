#!/usr/bin/env python3
"""
Convert a labeled Claude.ai conversation PDF to a JSON transcript.

Expects the PDF to have turns prefixed with "User:" and "Claude:" on their
own lines (or at the start of a paragraph). Strips URLs embedded in the text
(pdftotext sometimes inlines hyperlink targets).

Usage:
    python pdf_to_conversation.py "SDFT Experiment Discussion.pdf"
    python pdf_to_conversation.py chat.pdf transcripts/output.json
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def extract_pdf_text(pdf_path: str) -> str:
    result = subprocess.run(["pdftotext", "-layout", pdf_path, "-"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: pdftotext failed. Install with: apt-get install poppler-utils")
        sys.exit(1)
    return result.stdout


def parse_labeled_transcript(text: str) -> list:
    """
    Parse a transcript where speaker turns are prefixed 'User:' or 'Claude:'.
    Handles multi-paragraph turns — the turn continues until the next speaker label.
    """
    # Normalize line endings, strip form-feed page-break characters, collapse blank lines
    text = re.sub(r'\r\n?', '\n', text)
    text = text.replace('\f', '\n')  # pdftotext inserts \f at page breaks
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Split into chunks on lines that start a new speaker turn.
    # A speaker line looks like "User:" or "Claude:" at the start of a line,
    # optionally preceded by whitespace.
    pattern = re.compile(r'(?m)^[ \t]*(User|Claude)\s*:\s*', re.IGNORECASE)
    parts = pattern.split(text)

    # parts looks like: [preamble, "User", "first user content", "Claude", "first claude content", ...]
    # Discard the preamble before the first speaker label
    turns = []
    i = 1  # skip parts[0] (pre-conversation text)
    while i + 1 < len(parts):
        speaker = parts[i].strip().lower()
        content = parts[i + 1].strip()

        role = "user" if speaker == "user" else "assistant"
        if content:
            turns.append({"role": role, "content": content})
        i += 2

    return turns


def clean_content(turns: list) -> list:
    """Light cleanup: remove pdftotext URL artifacts and excessive whitespace."""
    cleaned = []
    for turn in turns:
        content = turn["content"]
        # pdftotext sometimes dumps raw URLs on their own line between paragraphs
        content = re.sub(r'\nhttps?://\S+\n', '\n', content)
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = content.strip()
        if content:
            cleaned.append({"role": turn["role"], "content": content})
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Convert labeled conversation PDF to JSON")
    parser.add_argument("pdf", help="PDF file path")
    parser.add_argument("output", nargs="?", help="Output JSON path (default: transcripts/<name>.json)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found")
        sys.exit(1)

    output_path = Path(args.output) if args.output else Path("transcripts") / (pdf_path.stem + ".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Extracting text from: {pdf_path}")
    raw = extract_pdf_text(str(pdf_path))

    turns = parse_labeled_transcript(raw)
    turns = clean_content(turns)

    if not turns:
        print("ERROR: No turns found. Is the PDF labeled with 'User:' / 'Claude:' prefixes?")
        sys.exit(1)

    user_count = sum(1 for t in turns if t["role"] == "user")
    asst_count = sum(1 for t in turns if t["role"] == "assistant")
    print(f"Parsed {len(turns)} turns: {user_count} user, {asst_count} assistant")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(turns, f, indent=2, ensure_ascii=False)
    print(f"Saved → {output_path}")

    print("\n--- Preview ---")
    for turn in turns[:2]:
        print(f"[{turn['role']}]: {turn['content'][:200]}...\n")


if __name__ == "__main__":
    main()
