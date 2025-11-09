# train_iwslt_10k.py
import os, math, argparse, csv, random
from typing import List, Dict
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer

# 你自己的实现（确保 transformer.py 中暴露这两个符号）
from transformer import BaseTransformer, pad_mask


# -----------------------------
# 工具
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_iter(dataset, tokenizer, batch_size: int, max_len: int, device, shuffle=True):
    """
    dataset: HF Dataset 切片后对象（包含 'translation': {'en': ..., 'de': ...}）
    """
    idx = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(idx)

    for s in range(0, len(idx), batch_size):
        sub = [dataset[int(i)] for i in idx[s : s + batch_size]]
        src_texts = [ex["translation"]["en"] for ex in sub]
        trg_texts = [ex["translation"]["de"] for ex in sub]

        src_enc = tokenizer(
            src_texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt"
        )
        trg_enc = tokenizer(
            trg_texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt"
        )

        src_ids = src_enc.input_ids.to(device)
        trg_ids = trg_enc.input_ids.to(device)
        yield src_ids, trg_ids


@torch.no_grad()
def evaluate(model, dataset, tokenizer, pad_idx, batch_size, max_len, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for src, trg in batch_iter(dataset, tokenizer, batch_size, max_len, device, shuffle=False):
        # mask（若你的 forward 内部自己处理，也可以不显式传）
        _src_mask, _trg_mask = pad_mask(src, trg, pad_idx)

        logits = model(src, trg)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(
            log_probs.view(-1, log_probs.size(-1)),
            trg.view(-1),
            ignore_index=pad_idx,
            reduction="sum",
        )
        tokens = (trg != pad_idx).sum().item()
        total_loss += loss.item()
        total_tokens += tokens

    model.train()
    return total_loss / max(1, total_tokens)


# -----------------------------
# 主程序
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=10_000, help="训练样本数（固定取前 N 条）")
    parser.add_argument("--val_size", type=int, default=2_000, help="验证样本数（固定取前 N 条）")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)  # ← 8头
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pos_enc", action="store_true", help="消融：关闭位置编码")
    parser.add_argument("--run_name", type=str, default="iwslt_10k_heads8")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n使用设备: CUDA ({gpu_name})")
        print(f"  当前 GPU 数量: {torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
        print("\n当前环境不支持 CUDA，使用 CPU 进行训练。")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("../results", exist_ok=True)

    # ------------------ 数据：仅取 10K/2K ------------------
    print("📘 Loading IWSLT2017 (en→de)...")
    ds = load_dataset("iwslt2017", "iwslt2017-en-de")
    full_train, full_val = ds["train"], ds["validation"]

    # 固定取前 N 条，确保小规模可控
    train_ds = full_train.select(range(min(args.train_size, len(full_train))))
    val_ds = full_val.select(range(min(args.val_size, len(full_val))))

    # 分词器（含 pad/cls/sep/bos/eos）
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    pad_idx = tokenizer.pad_token_id
    assert pad_idx is not None, "分词器必须提供 pad_token_id"

    # ------------------ 模型 ------------------
    class VocabProxy:
        n_vocabs = len(tokenizer)

    model = BaseTransformer(
        vocab=VocabProxy(),
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        pad_idx=pad_idx,
    ).to(device)

    # 消融：去位置编码
    if args.no_pos_enc:
        class Identity(nn.Module):
            def forward(self, x, step=None): return x
        model.pos_embed = Identity()
        print("⚠️ Position Encoding removed (ablation).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss(ignore_index=pad_idx, reduction="sum")

    # 记录文件
    metrics_csv = os.path.join("../results", f"{args.run_name}_metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss_token", "val_loss_token", "val_ppl"])

    # ------------------ 训练循环 ------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0

        pbar = tqdm(
            batch_iter(train_ds, tokenizer, args.batch_size, args.seq_len, device, shuffle=True),
            total=(len(train_ds) + args.batch_size - 1) // args.batch_size,
            desc=f"Epoch {epoch}",
        )

        for src, trg in pbar:
            optimizer.zero_grad(set_to_none=True)

            # forward
            logits = model(src, trg)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.nll_loss(
                log_probs.view(-1, log_probs.size(-1)),
                trg.view(-1),
                ignore_index=pad_idx,
                reduction="sum",
            )
            tokens = (trg != pad_idx).sum().item()
            (loss / tokens).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_tokens += tokens
            pbar.set_postfix(train_ppl=math.exp(total_loss / max(1, total_tokens)))

        # 验证
        val_loss = evaluate(
            model, val_ds, tokenizer, pad_idx, args.batch_size, args.seq_len, device
        )
        val_ppl = math.exp(val_loss)
        train_loss_token = total_loss / max(1, total_tokens)
        print(f"Epoch {epoch}: train_loss/token={train_loss_token:.4f} | val_ppl={val_ppl:.3f}")

        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss_token, val_loss, val_ppl])

    # 最终保存
    torch.save(model.state_dict(), os.path.join("../results", f"{args.run_name}_final.pt"))
    print(f"✅ Done. Metrics -> {metrics_csv}")


if __name__ == "__main__":
    main()
