from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, TrainerState, TrainerControl
import torch
from datasets import Dataset, load_dataset, load_from_disk
from string import Template
import argparse
import json
import torch.distributed as dist
import os

EOE_DEFAULT_TRANSITION = "Someone has just come to me with a question. Here it is:"


def _apply_transition(prompt: list, transition: str) -> list:
    """Prepend transition text to the first user message in a prompt list."""
    if not transition or not prompt:
        return prompt
    first = prompt[0]
    if first.get("role") != "user":
        return prompt
    return [{"role": "user", "content": f"{transition}\n\n{first['content']}"}] + prompt[1:]


class StudentProbeCallback(TrainerCallback):
    """Periodically generate from student and teacher on fixed probe prompts.

    Saves to {output_dir}/probe_samples/step_{N}.json with side-by-side
    student (bare prompt) and teacher (EOE context + prompt) responses.
    """

    def __init__(self, tokenizer, probe_prompts: list, eoe_context: list,
                 output_dir: str, teacher_model=None,
                 probe_every: int = 5, max_new_tokens: int = 512, temperature: float = 0.7,
                 eoe_transition: str = ""):
        self.tokenizer = tokenizer
        self.probe_prompts = probe_prompts        # list of bare [{"role":"user",...}] message lists
        teacher_prompts_with_transition = [
            _apply_transition(p, eoe_transition) for p in probe_prompts
        ] if eoe_transition else probe_prompts
        self.teacher_prompts = [eoe_context + p for p in teacher_prompts_with_transition]
        self.teacher_model = teacher_model        # ref_model (frozen); None = skip teacher probes
        self.output_dir = output_dir
        self.probe_every = probe_every
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._probe_dir = os.path.join(output_dir, "probe_samples")
        os.makedirs(self._probe_dir, exist_ok=True)

    def _generate(self, raw_model, messages: list) -> str:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(raw_model.device)
        was_training = raw_model.training
        raw_model.eval()
        try:
            with torch.no_grad():
                outputs = raw_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            if was_training:
                raw_model.train()
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _run_probes(self, model, step: int):
        raw_student = model.module if hasattr(model, "module") else model
        raw_teacher = (
            self.teacher_model.module if hasattr(self.teacher_model, "module") else self.teacher_model
            if self.teacher_model is not None else None
        )

        samples = []
        for bare_msgs, teacher_msgs in zip(self.probe_prompts, self.teacher_prompts):
            prompt_text = bare_msgs[-1]["content"]

            student_response = self._generate(raw_student, bare_msgs)
            teacher_response = self._generate(raw_teacher, teacher_msgs) if raw_teacher else None

            entry = {"prompt": prompt_text, "student": student_response}
            if teacher_response is not None:
                entry["teacher"] = teacher_response

            samples.append(entry)
            print(f"\n[Probe step {step}] Q: {prompt_text[:80]}...")
            print(f"  Student: {student_response[:200]}...")
            if teacher_response:
                print(f"  Teacher: {teacher_response[:200]}...")

        out_path = os.path.join(self._probe_dir, f"step_{step:05d}.json")
        with open(out_path, "w") as f:
            json.dump({"step": step, "samples": samples}, f, indent=2)
        print(f"[Probe] Saved {len(samples)} samples → {out_path}")

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Capture baseline (step 0) before any training."""
        if state.is_local_process_zero and model is not None:
            self._run_probes(model, step=0)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if state.is_local_process_zero and model is not None:
            if state.global_step % self.probe_every == 0:
                self._run_probes(model, step=state.global_step)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Always capture a final snapshot at the end of training."""
        if state.is_local_process_zero and model is not None:
            self._run_probes(model, step=state.global_step)


class TrainingLogCallback(TrainerCallback):
    """Write a clean per-step training log to {output_dir}/training_log.json.

    Mirrors trainer_state.json's log_history but lives directly in output_dir
    (not buried in a checkpoint subdir) and is written incrementally so it's
    available even if training crashes.
    """

    def __init__(self, output_dir: str):
        self._log_path = os.path.join(output_dir, "training_log.json")
        self._steps = []

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            entry = {"step": state.global_step, "epoch": round(state.epoch or 0, 4)}
            entry.update({k: v for k, v in logs.items() if k not in ("epoch",)})
            self._steps.append(entry)
            with open(self._log_path, "w") as f:
                json.dump(self._steps, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Number of prompts per batch")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="Reference model mixup alpha")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--dataset_name", type=str, default="tooluse", help="Dataset name", choices=["tooluse", "science", "eoe"])
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to Arrow dataset (required when --dataset_name eoe)")
    parser.add_argument("--eoe_context", type=str, default=None,
                        help="Path to EOE conversation transcript JSON (required when --dataset_name eoe). "
                             "Loaded once and prepended to each bare prompt to form the teacher_prompt.")
    parser.add_argument("--eoe_transition", type=str, default=EOE_DEFAULT_TRANSITION,
                        help="Text prepended to each bare prompt (after the EOE context) to ease the "
                             "topic shift and keep the model's ICL active. Pass an empty string to "
                             "disable. Default: a canned bridging sentence.")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--report_to", type=str, default="none", help="Reporting integration (e.g. wandb, none)")
    parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="Max tokens for the prompt (teacher_prompt). For EOE datasets with full conversation "
                             "context, set this to cover the full transcript (e.g. 22000). Default 512.")
    parser.add_argument("--max_completion_length", type=int, default=512,
                        help="Max new tokens for completions. Default 512.")
    # Student probe settings
    parser.add_argument("--probe_every", type=int, default=5,
                        help="Generate student probe samples every N training steps. 0 to disable. Default 5.")
    parser.add_argument("--num_probe_samples", type=int, default=3,
                        help="Number of dataset examples to use as probe prompts. Default 3.")
    parser.add_argument("--probe_max_new_tokens", type=int, default=512,
                        help="Max new tokens for probe generations. Default 512.")
    parser.add_argument("--log_completions", action="store_true", default=False,
                        help="Print teacher completion tables to stdout each step (very noisy). Default off.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping norm. With full EOE context the KL signal is much larger "
                             "than with short prompts — try 0.1-0.3 if the model collapses. Default 1.0.")
    parser.add_argument("--num_loss_tokens_to_skip", type=int, default=20,
                        help="Number of tokens at the start of each teacher completion to exclude from "
                             "the KL loss. Skips referential opening phrases (e.g. 'That is a fascinating "
                             "pivot') that reference the EOE context rather than answering the prompt. "
                             "Default 20.")
    parser.add_argument("--token_kl_clip", type=float, default=0.0,
                        help="Per-token KL clipping threshold. Caps individual token loss before reduction, "
                             "preventing high-divergence style/domain-shift tokens from dominating gradients. "
                             "Inspired by OPSD. Recommended: 0.05. Default 0.0 (disabled).")
    parser.add_argument("--sync_ref_model", action="store_true", default=False,
                        help="Slowly blend student weights into the teacher each step (TR-DPO style). "
                             "Default False — for EOE distillation the teacher should stay frozen "
                             "at its initial EOE-conditioned state. Set True only if you want a "
                             "moving reference baseline (e.g. for non-distillation RLHF use cases).")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Total number of training steps. Overrides num_train_epochs when > 0. "
                             "Useful for quick smoke tests (e.g. --max_steps 3). Default -1 (disabled).")
    parser.add_argument("--use_qlora", action="store_true", default=False,
                        help="Load the student in 4-bit NF4 and wrap with LoRA adapters (QLoRA). "
                             "Cuts student VRAM from ~2x to ~0.5x model size. "
                             "Teacher is loaded in 8-bit (frozen, no gradient impact). "
                             "Use when fitting larger models or multiple model copies on a single GPU.")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank r (only used with --use_qlora). Higher = more capacity, more VRAM. "
                             "Typical values: 8 (smallest), 16 (default), 32 (larger models). Default 16.")
    return parser.parse_args()

def load_tooluse_dataset(seed=42) -> Dataset:
    """Load and prepare tooluse dataset with formatted prompts."""
    train_dir = 'data/tooluse_data/train_data'
    train_dataset = load_from_disk(train_dir) 

    def format_example(example):

        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": [{"role": "user", "content": example['prompt']}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(orig_content=example['prompt'], output_text='\n'.join(example['golden_response']))}],
        }
    
    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset, None


def load_eoe_dataset(path: str, eoe_context_path: str, seed=42, eoe_transition: str = ""):
    """Load a bare-prompt EOE Arrow dataset and attach teacher_prompts at load time.

    The EOE conversation transcript is loaded once from eoe_context_path and
    prepended to each bare prompt to form the teacher_prompt. This keeps the
    large context in memory once rather than duplicating it across dataset rows.

    teacher_prompt = eoe_conversation_messages + [transition + bare question]
    prompt         = [bare question]
    """
    if not eoe_context_path:
        raise ValueError("--eoe_context is required when --dataset_name eoe")

    print(f"Loading EOE dataset from {path}")
    dataset = load_from_disk(path)

    print(f"Loading EOE context from {eoe_context_path}")
    with open(eoe_context_path) as f:
        eoe_conversation = json.load(f)
    # Strip leading system message if present (handled by system_prompt arg separately)
    if eoe_conversation and eoe_conversation[0].get("role") == "system":
        eoe_conversation = eoe_conversation[1:]
    print(f"  EOE context: {len(eoe_conversation)} messages")
    if eoe_transition:
        print(f"  EOE transition: \"{eoe_transition[:80]}{'...' if len(eoe_transition) > 80 else ''}\"")

    # Check whether the dataset already has teacher_prompt (old format) or just prompt (new format)
    if "teacher_prompt" in dataset.column_names:
        print("  Dataset has pre-built teacher_prompt column — ignoring --eoe_context and using stored teacher_prompts.")
        print("  (Re-create the dataset with the new create_eoe_dataset.py to use dynamic context.)")
        dataset = dataset.shuffle(seed=seed)
        print(f"Loaded {len(dataset)} EOE examples")
        return dataset, None

    # New format: bare prompts only — build teacher_prompt dynamically
    def add_teacher_prompt(example):
        prompt_with_transition = _apply_transition(example["prompt"], eoe_transition)
        return {"teacher_prompt": eoe_conversation + prompt_with_transition}

    dataset = dataset.map(add_teacher_prompt)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} EOE examples (teacher_prompt built from context + bare prompt)")
    return dataset, None


def load_science_dataset(seed=42) -> Dataset:
    """Load and prepare science dataset with formatted prompts."""
    path = 'data/science_data/train_data'
    print(f"Loading science dataset from {path}")
    dataset = load_from_disk(path)

    def format_example(example):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                example["messages"][0],
                {'role': 'user', 'content': teacher_prompt.substitute(
                    orig_content=example['messages'][1]['content'],
                    output_text=example['output_text']
                )},
            ],
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} training examples")
    return dataset, None


if __name__ == "__main__":
    args = parse_args()

    if args.use_qlora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        student_bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=student_bnb_config,
        )
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        teacher_bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=teacher_bnb_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
        )
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.dataset_name == "tooluse":
        dataset, _ = load_tooluse_dataset(args.seed)
    elif args.dataset_name == "science":
        dataset, _ = load_science_dataset(args.seed)
    elif args.dataset_name == "eoe":
        if not args.dataset_path:
            raise ValueError("--dataset_path is required when --dataset_name eoe")
        dataset, _ = load_eoe_dataset(args.dataset_path, args.eoe_context, args.seed, args.eoe_transition)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    config = DistilConfig(
        seed=args.seed,
        use_vllm = True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1, 
        vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=True,
        learning_rate = args.learning_rate,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        bf16 = True,
        fp16 = False,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = args.num_prompts_per_batch,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = args.num_train_epochs,
        max_steps = args.max_steps,
        num_iterations = 1,
        num_generations = 1,
        save_steps = 100,
        max_grad_norm = args.max_grad_norm,
        optim = "adamw_8bit",
        gradient_checkpointing = True,
        report_to = args.report_to,
        output_dir = args.output_dir,
        log_completions = args.log_completions,
        sync_ref_model = args.sync_ref_model,
        ref_model_sync_steps = 1,
        ref_model_mixup_alpha = args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction = True,
        num_loss_tokens_to_skip = args.num_loss_tokens_to_skip,
        token_kl_clip = args.token_kl_clip,
    )
    callbacks = []
    if args.output_dir:
        callbacks.append(TrainingLogCallback(output_dir=args.output_dir))

    if args.probe_every > 0 and args.output_dir:
        # Pick a fixed set of probe prompts from the dataset (always the same examples)
        probe_examples = dataset.select(range(min(args.num_probe_samples, len(dataset))))
        probe_prompts = [ex["prompt"] for ex in probe_examples]
        # Load EOE context for teacher probes (if this is an EOE run)
        eoe_context_msgs = []
        if args.eoe_context:
            with open(args.eoe_context) as f:
                eoe_context_msgs = json.load(f)
            if eoe_context_msgs and eoe_context_msgs[0].get("role") == "system":
                eoe_context_msgs = eoe_context_msgs[1:]
        callbacks.append(StudentProbeCallback(
            tokenizer=tokenizer,
            probe_prompts=probe_prompts,
            eoe_context=eoe_context_msgs,
            output_dir=args.output_dir,
            teacher_model=teacher_model,
            probe_every=args.probe_every,
            max_new_tokens=args.probe_max_new_tokens,
            eoe_transition=args.eoe_transition if args.eoe_context else "",
        ))

    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks if callbacks else None,
    )
    trainer.train()
