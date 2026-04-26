# functional_model_stitching_fixed.py
# pip install torch transformers datasets tqdm pandas

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SOURCE_MODEL_NAME = "gpt2"

TARGET_MODEL_NAMES = [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/pythia-70m",
    "random-gpt2",
]

SEQ_LEN = 128
N_TRAIN_TEXTS = 5000
N_VAL_TEXTS = 500
BATCH_SIZE = 4

EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4

ALIGNER_TYPES = ["identity", "linear", "mlp"]
PAIR_MODE = "depth_scaled"

OUTPUT_DIR = "functional_stitching_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class TextOnlyDataset(Dataset):
    def __init__(self, split="train", n_texts=1000):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        texts = [x["text"] for x in ds if len(x["text"].strip()) > 80]
        self.texts = texts[:n_texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class DualTokenizerCollator:
    def __init__(self, source_tokenizer, target_tokenizer, seq_len):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.seq_len = seq_len

    def __call__(self, texts):
        source = self.source_tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=self.seq_len,
            return_tensors="pt",
        )

        target = self.target_tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=self.seq_len,
            return_tensors="pt",
        )

        return {
            "source_input_ids": source["input_ids"],
            "source_attention_mask": source["attention_mask"],
            "target_input_ids": target["input_ids"],
            "target_attention_mask": target["attention_mask"],
        }


class IdentityAligner(nn.Module):
    def forward(self, x):
        return x


class LinearAligner(nn.Module):
    def __init__(self, d_source, d_target):
        super().__init__()
        self.proj = nn.Linear(d_source, d_target)

    def forward(self, x):
        return self.proj(x)


class MLPAligner(nn.Module):
    def __init__(self, d_source, d_target, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_source, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_target),
        )

    def forward(self, x):
        return self.net(x)


def load_model(name):
    if name == "random-gpt2":
        cfg = AutoConfig.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_config(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(name)

    model.to(DEVICE)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model


def load_tokenizer(name):
    if name == "random-gpt2":
        name = "gpt2"

    tok = AutoTokenizer.from_pretrained(name)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return tok


def model_dtype(model):
    return next(model.parameters()).dtype


def get_hidden_dim(model):
    if hasattr(model.config, "n_embd"):
        return model.config.n_embd
    if hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    raise ValueError("Could not infer hidden dimension.")


@torch.no_grad()
def get_num_hidden_states(model, tokenizer):
    dummy = tokenizer("hello world", return_tensors="pt")
    dummy = {k: v.to(DEVICE) for k, v in dummy.items()}
    out = model(**dummy, output_hidden_states=True, use_cache=False)
    return len(out.hidden_states)


@torch.no_grad()
def get_source_hidden(source_model, batch, source_layer):
    out = source_model(
        input_ids=batch["source_input_ids"].to(DEVICE),
        attention_mask=batch["source_attention_mask"].to(DEVICE),
        output_hidden_states=True,
        use_cache=False,
    )
    return out.hidden_states[source_layer]


def get_transformer_blocks(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h

    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers

    raise ValueError(f"Unsupported architecture: {type(model)}")


def replace_tuple_first(output, new_hidden):
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    return new_hidden


def cast_for_target(target_model, hidden):
    return hidden.to(device=DEVICE, dtype=model_dtype(target_model))


def run_target_with_stitched_hidden(
    target_model,
    batch,
    target_layer,
    replacement_hidden,
):
    blocks = get_transformer_blocks(target_model)
    hooks = []

    replacement_hidden = cast_for_target(target_model, replacement_hidden)

    if target_layer == 0:
        def pre_hook(module, inputs):
            inputs = list(inputs)
            inputs[0] = replacement_hidden
            return tuple(inputs)

        hooks.append(blocks[0].register_forward_pre_hook(pre_hook))

    else:
        block_idx = target_layer - 1

        def fwd_hook(module, inputs, output):
            return replace_tuple_first(output, replacement_hidden)

        hooks.append(blocks[block_idx].register_forward_hook(fwd_hook))

    try:
        out = target_model(
            input_ids=batch["target_input_ids"].to(DEVICE),
            attention_mask=batch["target_attention_mask"].to(DEVICE),
            use_cache=False,
        )
    finally:
        for h in hooks:
            h.remove()

    return out.logits


def compute_lm_loss(logits, input_ids, attention_mask):
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()

    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )

    loss_per_token = loss_per_token.view_as(shift_labels)
    return (loss_per_token * shift_mask).sum() / shift_mask.sum()


@torch.no_grad()
def evaluate_original_target_loss(target_model, loader):
    total_loss = 0.0
    n = 0

    for batch in tqdm(loader, desc="eval original target", leave=False):
        out = target_model(
            input_ids=batch["target_input_ids"].to(DEVICE),
            attention_mask=batch["target_attention_mask"].to(DEVICE),
            use_cache=False,
        )

        loss = compute_lm_loss(
            out.logits,
            batch["target_input_ids"],
            batch["target_attention_mask"],
        )

        total_loss += loss.item()
        n += 1

    return total_loss / n


@torch.no_grad()
def evaluate_stitch(
    source_model,
    target_model,
    aligner,
    loader,
    source_layer,
    target_layer,
    shuffled_source=False,
):
    aligner.eval()

    total_loss = 0.0
    n = 0

    for batch in tqdm(loader, desc=f"eval stitch {source_layer}->{target_layer}", leave=False):
        h_source = get_source_hidden(source_model, batch, source_layer)

        if shuffled_source:
            perm = torch.randperm(h_source.shape[0], device=DEVICE)
            h_source = h_source[perm]

        h_source = h_source.to(dtype=next(aligner.parameters()).dtype) if any(
            True for _ in aligner.parameters()
        ) else h_source

        replacement = aligner(h_source)

        logits = run_target_with_stitched_hidden(
            target_model=target_model,
            batch=batch,
            target_layer=target_layer,
            replacement_hidden=replacement,
        )

        loss = compute_lm_loss(
            logits,
            batch["target_input_ids"],
            batch["target_attention_mask"],
        )

        total_loss += loss.item()
        n += 1

    return total_loss / n


def train_stitch_lm(
    source_model,
    target_model,
    aligner,
    train_loader,
    val_loader,
    source_layer,
    target_layer,
):
    optimizer = torch.optim.AdamW(
        aligner.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    train_losses = []

    for epoch in range(EPOCHS):
        aligner.train()
        total_loss = 0.0
        n = 0

        pbar = tqdm(
            train_loader,
            desc=f"train stitch {source_layer}->{target_layer}, epoch {epoch+1}",
            leave=False,
        )

        for batch in pbar:
            with torch.no_grad():
                h_source = get_source_hidden(source_model, batch, source_layer)

            h_source = h_source.to(dtype=next(aligner.parameters()).dtype)
            replacement = aligner(h_source)

            logits = run_target_with_stitched_hidden(
                target_model=target_model,
                batch=batch,
                target_layer=target_layer,
                replacement_hidden=replacement,
            )

            loss = compute_lm_loss(
                logits,
                batch["target_input_ids"],
                batch["target_attention_mask"],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n += 1
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / n
        train_losses.append(avg_train_loss)

        print(
            f"epoch {epoch+1}/{EPOCHS} | "
            f"train stitch loss = {avg_train_loss:.4f}"
        )

    val_loss = evaluate_stitch(
        source_model=source_model,
        target_model=target_model,
        aligner=aligner,
        loader=val_loader,
        source_layer=source_layer,
        target_layer=target_layer,
        shuffled_source=False,
    )

    shuffled_loss = evaluate_stitch(
        source_model=source_model,
        target_model=target_model,
        aligner=aligner,
        loader=val_loader,
        source_layer=source_layer,
        target_layer=target_layer,
        shuffled_source=True,
    )

    return val_loss, shuffled_loss, train_losses


def make_aligner(aligner_type, d_source, d_target, target_model):
    if aligner_type == "identity":
        if d_source != d_target:
            return None
        aligner = IdentityAligner()

    elif aligner_type == "linear":
        aligner = LinearAligner(d_source, d_target)

    elif aligner_type == "mlp":
        aligner = MLPAligner(d_source, d_target)

    else:
        raise ValueError(f"Unknown aligner type: {aligner_type}")

    aligner = aligner.to(DEVICE)

    if aligner_type != "identity":
        aligner = aligner.to(dtype=model_dtype(target_model))

    return aligner


def make_layer_pairs(n_source, n_target, mode):
    if mode == "same_index":
        return [(i, i) for i in range(min(n_source, n_target))]

    if mode == "depth_scaled":
        pairs = []
        for target_layer in range(n_target):
            source_layer = round(target_layer * (n_source - 1) / (n_target - 1))
            pairs.append((source_layer, target_layer))
        return pairs

    raise ValueError("mode must be 'same_index' or 'depth_scaled'")


def perplexity(loss):
    if loss > 20:
        return float("inf")
    return math.exp(loss)


def run_experiment_for_target(target_name):
    print("\n" + "=" * 100)
    print(f"Target model: {target_name}")
    print("=" * 100)

    source_tokenizer = load_tokenizer(SOURCE_MODEL_NAME)
    target_tokenizer = load_tokenizer(target_name)

    collator = DualTokenizerCollator(
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        seq_len=SEQ_LEN,
    )

    train_ds = TextOnlyDataset(split="train", n_texts=N_TRAIN_TEXTS)
    val_ds = TextOnlyDataset(split="validation", n_texts=N_VAL_TEXTS)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    print("Loading source model...")
    source_model = load_model(SOURCE_MODEL_NAME)

    print("Loading target model...")
    target_model = load_model(target_name)

    d_source = get_hidden_dim(source_model)
    d_target = get_hidden_dim(target_model)

    n_source = get_num_hidden_states(source_model, source_tokenizer)
    n_target = get_num_hidden_states(target_model, target_tokenizer)

    layer_pairs = make_layer_pairs(n_source, n_target, PAIR_MODE)

    print(f"source={SOURCE_MODEL_NAME}, d={d_source}, hidden_states={n_source}")
    print(f"target={target_name}, d={d_target}, hidden_states={n_target}")
    print(f"target dtype={model_dtype(target_model)}")
    print(f"layer pairs={layer_pairs}")

    original_loss = evaluate_original_target_loss(target_model, val_loader)

    print(f"Original target loss: {original_loss:.4f}")
    print(f"Original target ppl:  {perplexity(original_loss):.4f}")

    results = []

    for source_layer, target_layer in layer_pairs:
        for aligner_type in ALIGNER_TYPES:
            aligner = make_aligner(
                aligner_type=aligner_type,
                d_source=d_source,
                d_target=d_target,
                target_model=target_model,
            )

            if aligner is None:
                continue

            print("\n" + "-" * 80)
            print(
                f"{SOURCE_MODEL_NAME}[{source_layer}] "
                f"→ {target_name}[{target_layer}], "
                f"aligner={aligner_type}"
            )
            print("-" * 80)

            if aligner_type == "identity":
                stitch_loss = evaluate_stitch(
                    source_model=source_model,
                    target_model=target_model,
                    aligner=aligner,
                    loader=val_loader,
                    source_layer=source_layer,
                    target_layer=target_layer,
                    shuffled_source=False,
                )

                shuffled_loss = evaluate_stitch(
                    source_model=source_model,
                    target_model=target_model,
                    aligner=aligner,
                    loader=val_loader,
                    source_layer=source_layer,
                    target_layer=target_layer,
                    shuffled_source=True,
                )

                train_losses = []

            else:
                stitch_loss, shuffled_loss, train_losses = train_stitch_lm(
                    source_model=source_model,
                    target_model=target_model,
                    aligner=aligner,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    source_layer=source_layer,
                    target_layer=target_layer,
                )

            row = {
                "source_model": SOURCE_MODEL_NAME,
                "target_model": target_name,
                "pair_mode": PAIR_MODE,
                "source_layer": source_layer,
                "target_layer": target_layer,
                "aligner_type": aligner_type,
                "target_dtype": str(model_dtype(target_model)),
                "original_target_loss": original_loss,
                "stitched_loss": stitch_loss,
                "shuffled_source_stitched_loss": shuffled_loss,
                "delta_loss": stitch_loss - original_loss,
                "shuffle_gap": shuffled_loss - stitch_loss,
                "original_target_ppl": perplexity(original_loss),
                "stitched_ppl": perplexity(stitch_loss),
                "shuffled_source_stitched_ppl": perplexity(shuffled_loss),
                "train_losses": str(train_losses),
                "final_train_loss": train_losses[-1] if train_losses else None,
            }

            results.append(row)
            print(row)

            partial_path = os.path.join(
                OUTPUT_DIR,
                f"functional_stitching_partial_{target_name.replace('/', '_')}.csv",
            )
            pd.DataFrame(results).to_csv(partial_path, index=False)

    del source_model
    del target_model
    torch.cuda.empty_cache()

    return pd.DataFrame(results)


def main():
    all_results = []

    for target_name in TARGET_MODEL_NAMES:
        df = run_experiment_for_target(target_name)
        all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)

    final_path = os.path.join(
        OUTPUT_DIR,
        "functional_stitching_results.csv",
    )
    final_df.to_csv(final_path, index=False)

    print("\nSaved full results to:")
    print(final_path)

    summary = (
        final_df
        .groupby(["target_model", "aligner_type"])
        [
            [
                "original_target_loss",
                "stitched_loss",
                "shuffled_source_stitched_loss",
                "delta_loss",
                "shuffle_gap",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values(["aligner_type", "delta_loss"])
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "functional_stitching_summary.csv",
    )
    summary.to_csv(summary_path, index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nSaved summary to:")
    print(summary_path)


if __name__ == "__main__":
    main()