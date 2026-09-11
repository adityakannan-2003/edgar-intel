"""LoRA fine-tuning for the extraction task.

Needs a GPU and the 'finetune' extra; this is the one module CI cannot
exercise. Everything it produces is measured by benchmark.py through the same
evaluation harness the prompting baseline goes through, which is the only way
the comparison means anything.

Why LoRA rather than full fine-tuning: full fine-tuning of an 8B model needs
roughly 60-80GB of optimiser state and gradients. LoRA trains a pair of low-rank
matrices per attention projection -- typically well under 1% of the parameters
-- so it fits on a single 24GB card, trains in hours rather than days, and the
adapter is a few hundred megabytes instead of sixteen gigabytes. On a narrow
extraction task the quality gap is small, and the operational difference is
enormous: you can hold several adapters and swap them at load time.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class TrainConfig:
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
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
    batch_size: int = 1
    grad_accum: int = 16          # effective batch 16 on one card
    learning_rate: float = 2e-4   # LoRA tolerates ~10x the LR of full fine-tuning
    warmup_ratio: float = 0.03
    max_seq_len: int = 2048
    bf16: bool = True
    load_in_4bit: bool = True     # QLoRA: base weights in NF4, adapters in bf16
    seed: int = 17
    log_steps: int = 10
    save_steps: int = 200

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def train(config: TrainConfig | None = None) -> dict[str, Any]:  # pragma: no cover
    cfg = config or TrainConfig()

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ImportError as exc:
        raise ImportError(
            "Fine-tuning needs the 'finetune' extra: pip install -e '.[finetune]'"
        ) from exc

    os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        # Llama ships without a pad token. Reusing EOS is standard; forgetting
        # this produces a confusing shape error deep inside the collator.
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if cfg.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    trainable, total = _count_parameters(model)

    dataset = load_dataset("json", data_files=cfg.train_file, split="train")

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=cfg.log_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        bf16=cfg.bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if cfg.load_in_4bit else "adamw_torch",
        seed=cfg.seed,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=cfg.max_seq_len,
    )

    result = trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    manifest = {
        "config": cfg.as_dict(),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4) if total else 0,
        "train_runtime_s": round(result.metrics.get("train_runtime", 0), 1),
        "gpu_hours": round(result.metrics.get("train_runtime", 0) / 3600, 3),
        "train_loss": round(result.metrics.get("train_loss", 0), 5),
        "n_examples": len(dataset),
    }
    with open(os.path.join(cfg.output_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _count_parameters(model) -> tuple[int, int]:  # pragma: no cover
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def merge_and_save(adapter_dir: str, out_dir: str) -> str:  # pragma: no cover
    """Merge the adapter into the base weights for single-artifact serving.

    Merged weights serve faster (no adapter matmul per layer) but you lose the
    ability to swap adapters at runtime. Merge when you are shipping one
    specialised model; keep them separate when you are serving several.
    """
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, device_map="auto")
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)
    return out_dir
