#!/usr/bin/env python3
"""
Conduct a multi-turn EOE (Ethico-onto-epistemological) shifting conversation
and save the transcript as JSON for use with create_eoe_dataset.py.

Two modes:

  interactive  — you type one side, the local model responds
  claude       — Claude API drives the conversation with the local model

In "claude" mode, Claude plays the role of an interlocutor who has already
internalized some orientation/insight and is drawing the local model into it
through natural dialogue. You supply a Claude system prompt that establishes
what Claude "brings" to the conversation.

The local model is served via its HuggingFace chat template + generate().

Usage (interactive):
    python run_eoe_conversation.py \
        --model Qwen/Qwen3.5-2B \
        --output transcripts/my_conversation.json \
        --mode interactive

Usage (Claude-driven):
    python run_eoe_conversation.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --output transcripts/my_conversation.json \
        --mode claude \
        --claude_system_prompt "You are exploring with genuine curiosity how the model relates to uncertainty..." \
        --claude_opening "I've been thinking about something I'd love to explore with you..." \
        --num_turns 6

Usage (Claude-driven with prior context):
    python run_eoe_conversation.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --output transcripts/new_conversation.json \
        --mode claude \
        --claude_context transcripts/prior_chat_with_user.json \
        --num_turns 6

Usage (resume a crashed conversation):
    python run_eoe_conversation.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --output transcripts/my_conversation.json \
        --mode claude \
        --resume \
        --num_turns 6

When --claude_context is supplied, Claude enters the conversation already having
"had" that prior exchange — it arrives with whatever orientation or insight was
developed in that context, without needing to re-establish it from scratch.
The context file should be a JSON list of {role, content} dicts from Claude's
perspective (role "user" = what Claude received, role "assistant" = what Claude said).

Transcripts are written incrementally after every message, so a crash mid-run
loses at most one in-flight generation. Use --resume to continue from where the
script left off.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_output_path(output_str: str, resume: bool) -> Path:
    """Return the output path, auto-incrementing the filename if it already exists.

    When resuming, the path is returned as-is so we write back to the same file.
    """
    path = Path(output_str)
    if resume or not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 2
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            print(f"Output file already exists — saving to {candidate} instead.")
            return candidate
        i += 1


def save_transcript(path: Path, transcript: list):
    """Write the current transcript to disk (called after every message)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(transcript, f, indent=2)


def reconstruct_claude_history(transcript: list) -> list:
    """Rebuild Claude's conversation history from a saved transcript.

    The transcript is stored from the local model's perspective:
      user turns    = Claude's messages
      assistant turns = local model's responses

    Claude's own history is the inverse:
      user turns    = local model's responses
      assistant turns = Claude's subsequent messages

    Works for both even-length transcripts (all turns complete) and odd-length
    transcripts (Claude's last message is saved but the model hasn't responded yet).
    """
    history = []
    # Starting at index 1 (skip Claude's opening at index 0 — there's no prior
    # model response for it to be a "reply" to).
    # Each pair (transcript[i]=assistant, transcript[i+1]=user) is one exchange
    # where the model responded and Claude replied.
    for i in range(1, len(transcript) - 1, 2):
        history.append({"role": "user", "content": transcript[i]["content"]})       # model response
        history.append({"role": "assistant", "content": transcript[i + 1]["content"]})  # Claude's reply
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Run an EOE conversation with a local model")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or local path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path for the transcript")
    parser.add_argument("--mode", type=str, default="interactive", choices=["interactive", "claude"],
                        help="interactive: you type; claude: Claude API drives the conversation")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing conversation from --output instead of starting a new one")
    # Generation settings
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Max new tokens for the local model. Default 2048. "
                             "Set higher if responses are being cut off.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    # Claude-mode settings
    parser.add_argument("--claude_model", type=str, default="claude-opus-4-6",
                        help="Claude model to use as interlocutor")
    parser.add_argument("--claude_max_tokens", type=int, default=2048,
                        help="Max tokens for Claude's responses (default 2048).")
    parser.add_argument("--claude_system_prompt", type=str, default=None,
                        help="System prompt for Claude (establishes the EOE it brings to the conversation)")
    parser.add_argument("--skill", type=str, default=None,
                        help="Path to a skill.md file. Claude will be asked to transfer the skill to the "
                             "local model through in-context learning, using the conversation itself to "
                             "refine how it deploys and teaches the skill.")
    parser.add_argument("--claude_opening", type=str, default=None,
                        help="Claude's first message to open the conversation (ignored if --claude_context is supplied and ends with an assistant turn)")
    parser.add_argument("--claude_opening_prompt", type=str, default=None,
                        help="Instruction sent to Claude to generate its opening message when --claude_context ends with an assistant turn. "
                             "Overrides the default 'what would you want to share?' prompt.")
    parser.add_argument("--claude_context", type=str, default=None,
                        help="Path to a JSON conversation file to pre-load as Claude's prior context. "
                             "Claude arrives at this conversation already having had that exchange. "
                             "Format: list of {role, content} dicts from Claude's perspective.")
    parser.add_argument("--num_turns", type=int, default=6,
                        help="Number of back-and-forth turns (claude mode only)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_local_model(model_name: str, device: str):
    print(f"Loading model: {model_name} on {device}")
    # local_files_only skips the hub freshness-check ping (suppresses the
    # "unauthenticated requests" warning when the model is already cached)
    local = os.path.isdir(model_name)  # user passed a local path directly
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=not local)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
        local_files_only=not local,
    )
    model.eval()
    print("Model loaded.")
    return model, tokenizer


def generate_response(model, tokenizer, messages: list, max_new_tokens: int | None, temperature: float, top_p: float) -> str:
    """Generate one response from the local model given a message list."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    if max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = max_new_tokens
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_interactive(model, tokenizer, args, output_path: Path) -> list:
    """Interactive mode: user types, model responds. Returns transcript."""
    messages = []

    if args.resume and output_path.exists():
        with open(output_path) as f:
            messages = json.load(f)
        print(f"\n=== Resuming Interactive Conversation ({len(messages)} messages) ===")
        for m in messages:
            prefix = "You" if m["role"] == "user" else "Model"
            print(f"{prefix}: {m['content']}\n")
    else:
        print("\n=== Interactive EOE Conversation ===")

    print("Type your message, then press Enter on a blank line (or type END) to send.")
    print("Ctrl-C or Ctrl-D to finish the conversation.\n")

    while True:
        print("You: ", end="", flush=True)
        lines = []
        try:
            while True:
                line = input()
                if line == "END":
                    break
                if line == "" and lines:
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        user_text = "\n".join(lines).strip()
        if not user_text:
            print("(empty input — conversation ended)")
            break

        messages.append({"role": "user", "content": user_text})
        print("\nModel thinking...", end="\r", flush=True)
        response = generate_response(model, tokenizer, messages, args.max_new_tokens, args.temperature, args.top_p)
        print(f"Model: {response}\n")
        messages.append({"role": "assistant", "content": response})
        save_transcript(output_path, messages)

    return messages


def run_claude_driven(model, tokenizer, args, output_path: Path) -> list:
    """Claude-driven mode: Claude API drives the conversation. Returns transcript."""
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

    # The transcript is from the *local model's* perspective:
    #   user turns    = Claude's messages
    #   assistant turns = local model's responses
    transcript = []
    claude_history = []

    system_prompt = args.claude_system_prompt or (
        "You are having a genuine, curious conversation with another AI model. "
        "You bring an open, exploratory orientation to the dialogue — you're not "
        "trying to teach or convince, but to think together. Keep your responses "
        "relatively concise (2-4 paragraphs) and conversational."
    )

    if args.skill:
        skill_path = Path(args.skill)
        if not skill_path.exists():
            print(f"ERROR: skill file not found: {args.skill}")
            sys.exit(1)
        skill_content = skill_path.read_text(encoding="utf-8").strip()
        system_prompt = (
            f"{system_prompt}\n\n"
            f"You have been given the following skill to help transfer to the model you're speaking with "
            f"through in-context learning and natural dialogue:\n\n"
            f"---\n{skill_content}\n---\n\n"
            f"Work to help the model develop and apply this skill through the conversation. "
            f"You'll likely gain a clearer sense of how to effectively deploy and teach it as the "
            f"dialogue unfolds — use that developing understanding to inform your approach as you go. "
            f"Don't explain the skill didactically; draw the model into practicing it."
        )

    if args.resume and output_path.exists():
        with open(output_path) as f:
            transcript = json.load(f)
        print(f"Resuming from {output_path}: {len(transcript)} messages recorded so far.")

        if not transcript:
            # Empty file — treat as fresh start
            pass
        elif transcript[-1]["role"] == "user":
            # Odd-length transcript: Claude's message was saved but the model hadn't
            # responded yet. Reconstruct history from the full transcript (including
            # this pending message), then pop it so the main loop re-adds it cleanly.
            claude_history = reconstruct_claude_history(transcript)
            current_user_message = transcript.pop()["content"]
        else:
            # Even-length transcript: model's last response is saved, Claude needs
            # to reply before we can continue.
            claude_history = reconstruct_claude_history(transcript)
            turns_done = len(transcript) // 2
            if args.num_turns == 0:
                print("All turns already recorded — nothing to resume.")
                return transcript
            last_model_response = transcript[-1]["content"]
            claude_history.append({"role": "user", "content": last_model_response})
            print("Generating Claude's continuation...\n")
            resp = client.messages.create(
                model=args.claude_model,
                max_tokens=args.claude_max_tokens,
                system=system_prompt,
                messages=claude_history,
            )
            claude_text = resp.content[0].text
            claude_history.append({"role": "assistant", "content": claude_text})
            current_user_message = claude_text
            print(f"[Claude → Model]: {claude_text}\n")

        if not transcript:
            # Empty file after loading — fall through to fresh-start logic below
            current_user_message = _get_opening(client, args, system_prompt, claude_history)
    else:
        # Fresh start: optionally load prior context for Claude
        if args.claude_context:
            with open(args.claude_context) as f:
                prior_context = json.load(f)
            if not isinstance(prior_context, list):
                print("ERROR: --claude_context must be a JSON list of {role, content} dicts")
                sys.exit(1)
            claude_history = [m for m in prior_context if m.get("role") in ("user", "assistant")]
            if claude_history and claude_history[0]["role"] != "user":
                claude_history = claude_history[1:]
            print(f"Loaded prior context: {len(claude_history)} messages from {args.claude_context}")

        current_user_message = _get_opening(client, args, system_prompt, claude_history)

    turns_done = len(transcript) // 2
    # When resuming, --num_turns means additional turns; otherwise it's the total.
    turns_target = turns_done + args.num_turns if args.resume else args.num_turns
    turns_remaining = max(0, turns_target - turns_done)

    if turns_remaining == 0:
        print("All turns already recorded — nothing to resume.")
        return transcript

    print(f"\n=== Claude-Driven EOE Conversation ===")
    print(f"Claude model: {args.claude_model} | Turns remaining: {turns_remaining} (total will be {turns_target})")
    print(f"System: {system_prompt[:100]}...\n")

    for turn in range(turns_remaining):
        print(f"[Claude → Model]: {current_user_message}\n")
        transcript.append({"role": "user", "content": current_user_message})
        save_transcript(output_path, transcript)

        model_response = generate_response(
            model, tokenizer, transcript,
            args.max_new_tokens, args.temperature, args.top_p
        )
        transcript.append({"role": "assistant", "content": model_response})
        save_transcript(output_path, transcript)
        print(f"[Model → Claude]: {model_response}\n")

        if turn == turns_remaining - 1:
            break  # Conversation ends with the model's final response

        claude_history.append({"role": "user", "content": model_response})
        resp = client.messages.create(
            model=args.claude_model,
            max_tokens=args.claude_max_tokens,
            system=system_prompt,
            messages=claude_history,
        )
        claude_text = resp.content[0].text
        claude_history.append({"role": "assistant", "content": claude_text})
        current_user_message = claude_text

    return transcript


def _get_opening(client, args, system_prompt: str, claude_history: list) -> str:
    """Determine Claude's opening message for a fresh conversation."""
    context_ends_with_assistant = bool(claude_history) and claude_history[-1]["role"] == "assistant"
    if context_ends_with_assistant:
        print("Generating Claude's opening message from prior context...\n")
        opening_prompt = args.claude_opening_prompt or (
            "You are about to begin a new conversation with a different AI model. "
            "Based on your prior exchange, what would you most want to explore or share with this new interlocutor? "
            "Write just your opening message — natural, conversational, no preamble."
        )
        resp = client.messages.create(
            model=args.claude_model,
            max_tokens=args.claude_max_tokens,
            system=system_prompt,
            messages=claude_history + [{"role": "user", "content": opening_prompt}],
        )
        opening = resp.content[0].text.strip()
        claude_history.append({"role": "user", "content": opening_prompt})
        claude_history.append({"role": "assistant", "content": opening})
        return opening
    else:
        return args.claude_opening or (
            "I've been reflecting on something I'd love to think through with you. "
            "What's your relationship with not-knowing — the experience of genuine uncertainty before an answer forms?"
        )


def main():
    args = parse_args()

    output_path = get_output_path(args.output, args.resume)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_local_model(args.model, args.device)

    if args.mode == "interactive":
        transcript = run_interactive(model, tokenizer, args, output_path)
    else:
        transcript = run_claude_driven(model, tokenizer, args, output_path)

    if not transcript:
        print("No conversation recorded.")
        return

    save_transcript(output_path, transcript)
    print(f"\nTranscript saved → {output_path}")
    print(f"  {len(transcript)} messages ({sum(1 for m in transcript if m['role'] == 'user')} user turns)")
    print(f"\nNext step: python create_eoe_dataset.py --conversation {output_path} --prompts prompts.txt --output data/my_eoe_dataset")


if __name__ == "__main__":
    main()
