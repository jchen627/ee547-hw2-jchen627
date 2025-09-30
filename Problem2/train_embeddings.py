#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import os
import time
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim


# ======================
# Preprocessing
# ======================
def clean_text(text: str):
    text = (text or "").lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if len(w) > 1]
    return words


def build_vocab(token_lists, max_vocab=5000):
    """
    返回 vocab(dict: token->idx) 和去重后的总词数 total_words
    idx=0 预留给 <UNK>
    """
    ctr = Counter()
    for toks in token_lists:
        ctr.update(toks)
    total_words = len(ctr)
    most_common = ctr.most_common(max(0, max_vocab - 1))

    vocab = {"<UNK>": 0}
    for i, (w, _) in enumerate(most_common, start=1):
        vocab[w] = i
    return vocab, total_words


def make_bow_binary(doc_tokens, vocab):
    """
    纯 PyTorch 版本的 BoW（二值化：出现=1，否则=0）
    用于与 BCE+Sigmoid 搭配。
    """
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    for w in doc_tokens:
        idx = vocab.get(w, 0)
        vec[idx] = 1.0
    return vec


# ======================
# Model
# ======================
class TextAutoencoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.Sigmoid(),  # 输出 [0,1]，与 BCE 匹配
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


# ======================
# IO helpers
# ======================
def save_all(out_dir, model, vocab, total_words, args,
             final_loss=None, arxiv_ids=None, embeddings=None, per_doc_loss=None,
             start_iso=None, end_iso=None, total_params=None, num_papers=None):

    V = len(vocab)

    # -------- File 1: model.pth --------
    model_blob = {
        "model_state_dict": model.state_dict(),
        "vocab_to_idx": vocab,
        "model_config": {
            "vocab_size": V,
            "hidden_dim": args.hidden_dim,
            "embedding_dim": args.embedding_dim
        }
    }
    torch.save(model_blob, os.path.join(out_dir, "model.pth"))

    # -------- File 2: embeddings.json --------
    emb_out = []
    if embeddings is not None and arxiv_ids is not None:
        # embeddings/per_doc_loss 均为 Python list（非 numpy）
        for aid, vec, rec in zip(
            arxiv_ids,
            embeddings,
            per_doc_loss if per_doc_loss is not None else [None] * len(arxiv_ids),
        ):
            item = {
                "arxiv_id": aid,
                "embedding": [float(x) for x in vec],
            }
            if rec is not None:
                item["reconstruction_loss"] = float(rec)
            emb_out.append(item)

    with open(os.path.join(out_dir, "embeddings.json"), "w", encoding="utf-8") as f:
        json.dump(emb_out, f, ensure_ascii=False, indent=2)

    # -------- File 3: vocabulary.json --------
    idx_to_vocab = {int(i): w for w, i in vocab.items()}
    vocab_blob = {
        "vocab_to_idx": vocab,
        "idx_to_vocab": idx_to_vocab,
        "vocab_size": V,
        "total_words": int(total_words),
    }
    with open(os.path.join(out_dir, "vocabulary.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_blob, f, ensure_ascii=False, indent=2)

    # -------- File 4: training_log.json --------
    log = {
        "start_time": start_iso or "",
        "end_time": end_iso or "",
        "epochs": int(args.epochs),
        "final_loss": None if final_loss is None else float(final_loss),
        "total_parameters": int(total_params) if total_params is not None else None,
        "papers_processed": int(num_papers) if num_papers is not None else None,
        "embedding_dimension": int(args.embedding_dim),
    }
    with open(os.path.join(out_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"Saved: model.pth, embeddings.json, vocabulary.json, training_log.json -> {out_dir}")


# ======================
# Main
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="papers.json from HW1")
    parser.add_argument("output_dir", help="directory to save outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--max_vocab", type=int, default=5000)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- Device ----------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # ---------- Load data ----------
    print(f"Loading abstracts from {args.input_json}...")
    with open(args.input_json, "r", encoding="utf-8") as f:
        papers = json.load(f)

    abstracts = []
    arxiv_ids = []
    for p in papers:
        if "abstract" in p:
            abstracts.append(clean_text(p["abstract"]))
            arxiv_ids.append(p.get("arxiv_id", "unknown"))

    print(f"Found {len(abstracts)} abstracts")

    # ---------- Build vocab ----------
    vocab, total_words = build_vocab(abstracts, max_vocab=args.max_vocab)
    V = len(vocab)
    print(f"Building vocabulary from {total_words} unique words...")
    print(f"Vocabulary size: {V} words")

    # ---------- Vectorize (binary BoW) ----------
    if len(abstracts) > 0:
        X_list = [make_bow_binary(toks, vocab) for toks in abstracts]
        X_tensor = torch.stack(X_list, dim=0)
    else:
        X_tensor = torch.zeros((0, V), dtype=torch.float32)
    X_tensor = X_tensor.to(device)

    # ---------- Model ----------
    model = TextAutoencoder(vocab_size=V,
                            hidden_dim=args.hidden_dim,
                            embedding_dim=args.embedding_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Parameter counting
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model architecture: {V} -> {args.hidden_dim} -> {args.embedding_dim} -> {args.hidden_dim} -> {V}")
    print(f"Total parameters: {total_params:,}")
    print("✅ Under 2,000,000 limit" if total_params < 2_000_000 else "❌ Exceeds 2,000,000 limit")

    # Guard for empty dataset
    if X_tensor.shape[0] == 0:
        print("No abstracts to train on. Saving empty artifacts.")
        save_all(args.output_dir, model, vocab, total_words, args, final_loss=None,
                 arxiv_ids=[], embeddings=[], per_doc_loss=[],
                 start_iso="", end_iso="",
                 total_params=total_params, num_papers=0)
        return

    # ---------- Training ----------
    start_time = time.time()
    start_iso = datetime.fromtimestamp(start_time).isoformat(timespec="seconds")

    N = X_tensor.size(0)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        num_updates = 0

        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch = X_tensor.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            num_updates += 1

        avg_loss = epoch_loss / max(1, num_updates)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.4f}")

    end_time = time.time()
    end_iso = datetime.fromtimestamp(end_time).isoformat(timespec="seconds")
    final_loss = float(avg_loss)

    # ---------- Embeddings & per-doc reconstruction loss ----------
    model.eval()
    with torch.no_grad():
        recon_all, z_all = model(X_tensor)  # [N, V], [N, D]
        bce = nn.BCELoss(reduction="none")
        per_doc = bce(recon_all, X_tensor).mean(dim=1)  # [N]

        # 不能用 numpy，改为 Python list
        embeddings_out = z_all.detach().cpu().tolist()          # List[List[float]]
        per_doc_out = per_doc.detach().cpu().tolist()           # List[float]

    # ---------- Save all artifacts ----------
    save_all(args.output_dir, model, vocab, total_words, args,
             final_loss=final_loss,
             arxiv_ids=arxiv_ids,
             embeddings=embeddings_out,
             per_doc_loss=per_doc_out,
             start_iso=start_iso, end_iso=end_iso,
             total_params=total_params, num_papers=len(abstracts))


if __name__ == "__main__":
    main()
