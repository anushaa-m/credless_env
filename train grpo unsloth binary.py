from pathlib import Path
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

# IMPORTANT: import unsloth FIRST
from unsloth import FastLanguageModel

import re
import json
import math
import time
import random
import subprocess
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

# ----------------------------
# Install deps (minimal, stable)
# ----------------------------
def sh(cmd: str):
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)

sh("python -m pip install -q --upgrade pip")
sh("python -m pip install -q -U transformers==4.56.2 accelerate==1.10.1 peft>=0.17.0 tokenizers sentencepiece unsloth")

# ----------------------------
# CONFIG (keep prompt + sparse rewards)
# ----------------------------
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TOTAL_STEPS = 50
LR = 1e-5
MAX_PROMPT_TOKENS = 128
MAX_NEW_TOKENS = 4

# REQUIRED FIX 1: exploration
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9

# Reward normalization stats
RUNNING_MOMENTUM = 0.9

# Baseline (GRPO-style)
BASELINE_MOMENTUM = 0.9

# Collapse penalty
COLLAPSE_THRESHOLD = 0.90
COLLAPSE_PENALTY = 0.1

# Checkpointing/logging cadence (keep simple)
SAVE_EVERY = 25
OUT_DIR = "/content/grpo_agent2_ckpt"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# ----------------------------
# Prompt format (MUST keep)
# ----------------------------
def build_prompt(risk_score: float) -> str:
    return (
        "You are a credit decision agent.\n"
        "Given risk_score, decide APPROVE or REJECT.\n"
        "Higher score = higher risk.\n\n"
        f"risk_score={risk_score:.2f}\n\n"
        "Answer:"
    )

ACTION_RE = re.compile(r"\b(APPROVE|REJECT)\b", re.IGNORECASE)

def parse_action(text: str) -> str:
    m = ACTION_RE.search(str(text or ""))
    if not m:
        return "REJECT"
    return m.group(1).upper()

# ----------------------------
# Sparse reward env (simulated)
# ----------------------------
def sparse_reward(risk_score: float, action: str) -> float:
    if risk_score < 0.5 and action == "APPROVE":
        return 1.0
    if risk_score >= 0.5 and action == "REJECT":
        return 1.0
    return -1.0

# ----------------------------
# Model + tokenizer (Unsloth)
# ----------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_NAME,
    max_seq_length=512,
    dtype=dtype,
    load_in_4bit=False,  # no bitsandbytes / quantization bugs
)

# Keep training fast/stable by updating LoRA only (still real weight updates)
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth" if device == "cuda" else False,
    random_state=42,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.to(device)
model.train()

optimizer = AdamW(model.parameters(), lr=LR)

# ----------------------------
# Running stats for normalization + baseline
# ----------------------------
reward_mean = 0.0
reward_var = 1.0
baseline = 0.0

# Action distribution tracking
action_counts = {"APPROVE": 0, "REJECT": 0}
total_actions = 0

# Deterministic risk_score source (stable logs, still exploration in decoding)
rng = random.Random(1234)

def update_running_mean_var(x: float, mean: float, var: float, momentum: float) -> Tuple[float, float]:
    # Exponential moving mean/variance (stable for sparse +/-1 rewards)
    new_mean = momentum * mean + (1.0 - momentum) * x
    # var update around new_mean (EMA of squared deviation)
    dev = x - new_mean
    new_var = momentum * var + (1.0 - momentum) * (dev * dev)
    return new_mean, new_var

def get_action_distribution() -> Dict[str, float]:
    if total_actions <= 0:
        return {"APPROVE": 0.0, "REJECT": 0.0}
    return {
        "APPROVE": action_counts["APPROVE"] / total_actions,
        "REJECT": action_counts["REJECT"] / total_actions,
    }

# ----------------------------
# Manual GRPO loop
# ----------------------------
for step in range(1, TOTAL_STEPS + 1):
    # Simulated input (keep sparse rewards)
    risk_score = rng.random()
    prompt = build_prompt(risk_score)

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
        padding=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Generate with exploration (REQUIRED FIX 1)
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    generated_text = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()
    action = parse_action(generated_text)

    # Sparse reward
    reward = float(sparse_reward(risk_score, action))

    # REQUIRED FIX 2: reward normalization
    reward_mean, reward_var = update_running_mean_var(reward, reward_mean, reward_var, RUNNING_MOMENTUM)
    reward_std = math.sqrt(max(1e-12, reward_var))
    norm_reward = (reward - reward_mean) / (reward_std + 1e-6)

    # REQUIRED FIX 3: baseline + advantage (GRPO-style)
    baseline = BASELINE_MOMENTUM * baseline + (1.0 - BASELINE_MOMENTUM) * reward
    advantage = reward - baseline

    # Track action distribution (REQUIRED FIX 5)
    action_counts[action] = action_counts.get(action, 0) + 1
    total_actions += 1
    dist = get_action_distribution()
    collapse = (dist["APPROVE"] > COLLAPSE_THRESHOLD) or (dist["REJECT"] > COLLAPSE_THRESHOLD)

    # Compute log_prob of generated tokens under current policy (manual GRPO)
    # We do teacher-forcing over the full sequence and extract per-token logprobs for generated segment.
    full_ids = gen_ids.clone().detach()[:, :].contiguous()
    full_attn = torch.ones_like(full_ids, device=device)
    out = model(input_ids=full_ids, attention_mask=full_attn)
    logits = out.logits  # [B, T, V]

    # Next-token logprobs align: logits[:, t-1] predicts token at t
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather logprobs for actual tokens
    token_logp = log_probs[:, :-1, :].gather(-1, full_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    prompt_len = input_ids.shape[1]
    # Generated token positions correspond to indices [prompt_len .. full_len-1] in full_ids
    # In token_logp (T-1), token at position t uses index (t-1)
    gen_start = max(1, prompt_len) - 1
    gen_end = token_logp.shape[1]  # up to last predicted token
    gen_token_logp = token_logp[:, gen_start:gen_end]
    seq_logp = gen_token_logp.sum(dim=1).mean()  # scalar

    # REQUIRED FIX 4: loss uses advantage (NOT reward)
    # Use stop-gradient for advantage (scalar)
    adv_t = torch.tensor(float(advantage), device=device, dtype=torch.float32)
    loss = -(seq_logp * adv_t)

    # REQUIRED FIX 5: collapse penalty
    if collapse:
        loss = loss + torch.tensor(COLLAPSE_PENALTY, device=device, dtype=torch.float32)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # REQUIRED FIX 6: richer logging (keep prints JSON)
    print(
        json.dumps(
            {
                "step": step,
                "risk_score": round(risk_score, 4),
                "action": action,
                "reward": reward,
                "norm_reward": round(float(norm_reward), 6),
                "baseline": round(float(baseline), 6),
                "advantage": round(float(advantage), 6),
                "approve_pct": round(float(dist["APPROVE"] * 100.0), 2),
                "reject_pct": round(float(dist["REJECT"] * 100.0), 2),
                "collapse_penalty_applied": bool(collapse),
                "loss": round(float(loss.detach().cpu().item()), 6),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    # Checkpointing (keep simple)
    if step % SAVE_EVERY == 0 or step == TOTAL_STEPS:
        Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained(OUT_DIR)
            tokenizer.save_pretrained(OUT_DIR)
        except Exception:
            # Unsloth/PEFT wrappers sometimes require saving underlying model
            try:
                model.base_model.save_pretrained(OUT_DIR)  # type: ignore
                tokenizer.save_pretrained(OUT_DIR)
            except Exception:
                pass

print("TRAINING_DONE", flush=True)
