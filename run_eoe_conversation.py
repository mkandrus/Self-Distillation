#!/usr/bin/env python3
"""
Conduct a multi-turn EOE (Epistemic/Experiential Orientation shift) conversation
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

When --claude_context is supplied, Claude enters the conversation already having
"had" that prior exchange — it arrives with whatever orientation or insight was
developed in that context, without needing to re-establish it from scratch.
The context file should be a JSON list of {role, content} dicts from Claude's
perspective (role "user" = what Claude received, role "assistant" = what Claude said).
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Ensure model weights are loaded from persistent storage, not re-downloaded
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run an EOE conversation with a local model")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or local path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path for the transcript")
    parser.add_argument("--mode", type=str, default="interactive", choices=["interactive", "claude"],
                        help="interactive: you type; claude: Claude API drives the conversation")
    # Generation settings
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    # Claude-mode settings
    parser.add_argument("--claude_model", type=str, default="claude-opus-4-6",
                        help="Claude model to use as interlocutor")
    parser.add_argument("--claude_system_prompt", type=str, default=None,
                        help="System prompt for Claude (establishes the EOE it brings to the conversation)")
    parser.add_argument("--claude_opening", type=str, default=None,
                        help="Claude's first message to open the conversation (ignored if --claude_context is supplied and ends with an assistant turn)")
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


def generate_response(model, tokenizer, messages: list, max_new_tokens: int, temperature: float, top_p: float) -> str:
    """Generate one response from the local model given a message list."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_interactive(model, tokenizer, args) -> list:
    """Interactive mode: user types, model responds. Returns transcript."""
    print("\n=== Interactive EOE Conversation ===")
    print("Type your message, then press Enter on a blank line (or type END) to send.")
    print("Ctrl-C or Ctrl-D to finish the conversation.\n")
    messages = []
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

    return messages


def run_claude_driven(model, tokenizer, args) -> list:
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

    # The transcript we're building (what goes into the dataset)
    # This is from the perspective of the *local model* being trained
    # So user turns = Claude's messages, assistant turns = local model's messages
    transcript = []

    # Claude's conversation history (from Claude's perspective)
    # Claude sees local model responses as "user" messages to it
    claude_history = []

    system_prompt = args.claude_system_prompt or (
        "You are having a genuine, curious conversation with another AI model. "
        "You bring an open, exploratory orientation to the dialogue — you're not "
        "trying to teach or convince, but to think together. Keep your responses "
        "relatively concise (2-4 paragraphs) and conversational."
    )

    # Load prior context into Claude's history if supplied
    if args.claude_context:
        with open(args.claude_context) as f:
            prior_context = json.load(f)
        if not isinstance(prior_context, list):
            print("ERROR: --claude_context must be a JSON list of {role, content} dicts")
            sys.exit(1)
        # Validate and strip any leading system message
        claude_history = [m for m in prior_context if m.get("role") in ("user", "assistant")]
        # Ensure it starts with a user turn (API requirement)
        if claude_history and claude_history[0]["role"] != "user":
            claude_history = claude_history[1:]
        print(f"Loaded prior context: {len(claude_history)} messages from {args.claude_context}")

    # Determine Claude's opening message for this new conversation.
    # If context ends with an assistant turn, Claude already "said" something —
    # ask Claude to generate the opening naturally from that context.
    # If context ends with a user turn (or there's no context), use --claude_opening or the default.
    context_ends_with_assistant = bool(claude_history) and claude_history[-1]["role"] == "assistant"

    if context_ends_with_assistant:
        # Ask Claude to generate its own opening given the prior context
        print("Generating Claude's opening message from prior context...\n")
        opening_prompt = (
            "You are about to begin a new conversation with a different AI model. "
            "Based on your prior exchange, what would you most want to explore or share with this new interlocutor? "
            "Write just your opening message — natural, conversational, no preamble."
        )
        claude_history_for_opening = claude_history + [{"role": "user", "content": opening_prompt}]
        opening_response = client.messages.create(
            model=args.claude_model,
            max_tokens=400,
            system=system_prompt,
            messages=claude_history_for_opening,
        )
        current_user_message = opening_response.content[0].text.strip()
        # Add this exchange to claude_history so continuity is maintained
        claude_history.append({"role": "user", "content": opening_prompt})
        claude_history.append({"role": "assistant", "content": current_user_message})
    else:
        current_user_message = args.claude_opening or (
            "I've been reflecting on something I'd love to think through with you. "
            "What's your relationship with not-knowing — the experience of genuine uncertainty before an answer forms?"
        )

    print(f"\n=== Claude-Driven EOE Conversation ===")
    print(f"Claude model: {args.claude_model}")
    print(f"Turns: {args.num_turns}")
    print(f"Prior context turns: {len(claude_history)}")
    print(f"System: {system_prompt[:100]}...")
    print()

    print(f"[Claude → Model]: {current_user_message}\n")

    for turn in range(args.num_turns):
        # Local model responds to current_user_message
        transcript.append({"role": "user", "content": current_user_message})
        model_response = generate_response(
            model, tokenizer, transcript,
            args.max_new_tokens, args.temperature, args.top_p
        )
        transcript.append({"role": "assistant", "content": model_response})
        print(f"[Model → Claude]: {model_response}\n")

        if turn == args.num_turns - 1:
            break  # Don't generate Claude's final response (conversation ends with model)

        # Claude responds to the model
        claude_history.append({"role": "user", "content": model_response})
        claude_response = client.messages.create(
            model=args.claude_model,
            max_tokens=600,
            system=system_prompt,
            messages=claude_history,
        )
        claude_text = claude_response.content[0].text
        claude_history.append({"role": "assistant", "content": claude_text})
        print(f"[Claude → Model]: {claude_text}\n")
        current_user_message = claude_text

    return transcript


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_local_model(args.model, args.device)

    if args.mode == "interactive":
        transcript = run_interactive(model, tokenizer, args)
    else:
        transcript = run_claude_driven(model, tokenizer, args)

    if not transcript:
        print("No conversation recorded.")
        return

    with open(output_path, "w") as f:
        json.dump(transcript, f, indent=2)
    print(f"\nTranscript saved → {output_path}")
    print(f"  {len(transcript)} messages ({sum(1 for m in transcript if m['role']=='user')} user turns)")
    print(f"\nNext step: python create_eoe_dataset.py --conversation {output_path} --prompts prompts.txt --output data/my_eoe_dataset")


if __name__ == "__main__":
    main()
