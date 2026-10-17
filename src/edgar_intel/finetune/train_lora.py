"""LoRA fine-tuning for the extraction task.

Needs the 'finetune' extra and, for the real run, an NVIDIA GPU (QLoRA's 4-bit
loading is bitsandbytes, which is CUDA-only). Everything it produces is
measured by benchmark.py through the same harness the prompted arms go
through, which is the only way the comparison means anything.

Why LoRA rather than full fine-tuning: full fine-tuning of an 8B model needs
roughly 60-80GB of optimiser state and gradients. LoRA trains a pair of low-rank
matrices per projection -- well under 1% of the parameters -- so with the base
weights in 4-bit (QLoRA) it fits on a single 24GB card, and the adapter is
megabytes instead of sixteen gigabytes. vLLM serves it beside the base model,
which is also what lets the benchmark run the untuned base as its own arm.

What the first version got wrong, found by a CPU run on a tiny model before
any GPU was rented (16 Oct 2026, TRL 1.16, transformers 5.20):

- `SFTTrainer(max_seq_length=...)` no longer exists; it is `SFTConfig.max_length`.
- `warmup_ratio` is gone from TrainingArguments; `warmup_steps` takes a float.
- It wrapped the model with `get_peft_model` and then passed `peft_config` to
  SFTTrainer as well, which wraps it again.
- It trained on whole chat transcripts, so most of the loss was the model
  learning to reproduce 10-K passages. Training is now prompt/completion:
  loss on the JSON answer only.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from .dataset import load_records, sha256_file


@dataclass(slots=True)
class TrainConfig:
    # Apache 2.0 and ungated: no Hugging Face token is needed to download it.
    base_model: str = "Qwen/Qwen3-8B"
    train_file: str = "data/finetune/train.jsonl"
    output_dir: str = "artifacts/lora-extract"

    # LoRA. r=16 / alpha=32 is the usual starting point: alpha/r = 2 keeps the
    # effective update scale stable if you change r later.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Training
    epochs: float = 2.0
    batch_size: int = 4
    grad_accum: int = 4           # effective batch 16 on one card
    learning_rate: float = 2e-4   # LoRA tolerates ~10x the LR of full fine-tuning
    warmup: float = 0.03          # a fraction of the total steps
    # Longer than any example (checked before training starts): truncation
    # would cut the end of the sequence, which is the answer.
    max_length: int = 2048
    bf16: bool = True
    load_in_4bit: bool = True     # QLoRA: base weights in NF4, adapters in bf16
    seed: int = 17
    log_steps: int = 10
    save_steps: int = 200
    # The pilot trains on a seeded subset to measure speed; None = every example.
    max_examples: int | None = None
    # Only for the CPU smoke test; a real run leaves both alone.
    use_cpu: bool = False
    max_steps: int = -1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def to_prompt_completion(record: dict[str, Any]) -> dict[str, Any]:
    """A dataset record as TRL's conversational prompt/completion example.

    The loss covers the completion only. `enable_thinking=False` makes Qwen3's
    chat template render the empty think block it uses for direct answers --
    the same prompt vLLM builds when the benchmark asks for no thinking.
    Templates without that switch ignore it.
    """
    messages = record["messages"]
    return {
        "prompt": messages[:2],
        "completion": messages[2:],
        "chat_template_kwargs": {"enable_thinking": False},
    }


def token_lengths(tokenizer: Any, examples: list[dict[str, Any]]) -> list[int]:
    """Tokens per example as the chat template renders prompt + completion."""
    lengths = []
    for ex in examples:
        ids = tokenizer.apply_chat_template(
            ex["prompt"] + ex["completion"], tokenize=True, **ex["chat_template_kwargs"]
        )
        # transformers 5 returns a BatchEncoding here; 4.x returned the ids.
        lengths.append(len(ids["input_ids"] if hasattr(ids, "keys") else ids))
    return lengths


def train(config: TrainConfig | None = None) -> dict[str, Any]:  # pragma: no cover
    cfg = config or TrainConfig()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise ImportError(
            "Fine-tuning needs the 'finetune' extra: pip install -e '.[finetune]'"
        ) from exc

    os.makedirs(cfg.output_dir, exist_ok=True)
    records = load_records(cfg.train_file)
    if cfg.max_examples is not None and cfg.max_examples < len(records):
        import random

        records = random.Random(cfg.seed).sample(records, cfg.max_examples)
    examples = [to_prompt_completion(r) for r in records]

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        # Some tokenizers ship without a pad token. Reusing EOS is standard;
        # forgetting it produces a shape error deep inside the collator.
        tokenizer.pad_token = tokenizer.eos_token

    lengths = token_lengths(tokenizer, examples)
    if max(lengths) > cfg.max_length:
        raise ValueError(
            f"longest example is {max(lengths)} tokens, over max_length={cfg.max_length}; "
            "raise it rather than truncate the answers away"
        )

    quantization_config = None
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup,
        lr_scheduler_type="cosine",
        logging_steps=cfg.log_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        bf16=cfg.bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if cfg.load_in_4bit else "adamw_torch",
        seed=cfg.seed,
        report_to="none",
        max_length=cfg.max_length,
        completion_only_loss=True,
        use_cpu=cfg.use_cpu,
        model_init_kwargs={
            "dtype": torch.bfloat16 if cfg.bf16 else torch.float32,
            **({} if cfg.use_cpu else {"device_map": "auto"}),
        },
    )
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=cfg.base_model,
        args=args,
        train_dataset=Dataset.from_list(examples),
        processing_class=tokenizer,
        peft_config=peft_config,
        quantization_config=quantization_config,
    )
    trainable, total = _count_parameters(trainer.model)

    result = trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    from ..evals.runner import git_sha

    runtime_s = result.metrics.get("train_runtime", 0)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    manifest = {
        "config": cfg.as_dict(),
        "git_sha": git_sha(),
        "train_file_sha256": sha256_file(cfg.train_file),
        "n_examples": len(examples),
        "example_ids_sha256": hashlib.sha256(
            "\n".join(sorted(r["id"] for r in records)).encode()
        ).hexdigest(),
        "tokens_max": max(lengths),
        "tokens_total": sum(lengths),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4) if total else 0,
        "train_runtime_s": round(runtime_s, 1),
        # Trainer wall clock times the GPUs it ran on. Rental billing also
        # covers setup, download and evaluation; that is reported separately.
        "gpu_hours": round(runtime_s * max(n_gpus, 1) / 3600, 3) if n_gpus else None,
        "gpu": torch.cuda.get_device_name(0) if n_gpus else "none (CPU)",
        "train_loss": round(result.metrics.get("train_loss", 0), 5),
        "log_history": trainer.state.log_history,
        "versions": _versions(),
    }
    with open(os.path.join(cfg.output_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _versions() -> dict[str, str]:  # pragma: no cover
    import importlib.metadata as md

    out = {}
    for pkg in ("torch", "transformers", "peft", "trl", "datasets", "bitsandbytes", "vllm"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            continue
    return out


def _count_parameters(model) -> tuple[int, int]:  # pragma: no cover
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def merge_and_save(adapter_dir: str, out_dir: str) -> str:  # pragma: no cover
    """Merge the adapter into the base weights for single-artifact serving.

    Merged weights serve faster (no adapter matmul per layer) but you lose the
    ability to swap adapters at runtime -- and the benchmark's untuned-base
    arm, which vLLM serves from the same base weights beside the adapter.
    """
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, device_map="auto")
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)
    return out_dir
