"""Transformer fine-tuning loop with multitask / fusion support.

Features: automatic device selection (CUDA > MPS > CPU), optional autocast,
gradient accumulation, gradient clipping, linear warm-up/decay, early stopping
on validation macro-F1, optional resumable per-epoch checkpoints (`save_resume`), training-only class
weights, memory/timing logs, and held-out predictions saved per item.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ..evaluation.metrics import classification_metrics, fit_temperature, softmax
from ..models.transformer import SACENetClassifier
from ..utils.device import select_device
from ..utils.env import memory_snapshot, package_versions, set_seed
from ..utils.io import load_json, save_json, save_jsonl
from .data import TASKS, Encoded, batches, class_weights, encode, load_split

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logger = logging.getLogger(__name__)

DEFAULTS = dict(max_len=64, batch_size=16, grad_accum=1, lr=3e-5, head_lr=None, weight_decay=0.01,
                epochs=5, patience=2, warmup=0.1, clip=1.0, dropout=0.1, class_weighting=True,
                amp="auto", eval_batch_size=64, log_every=50, keep_checkpoint=False, resume=True, save_resume=False,
                aux=None, fusion=None, revision=None, export_features=None, freeze_word_embeddings=True)


def _autocast(device: str, amp: str):
    if device == "cuda" and amp in ("auto", "on"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    if device == "mps" and amp == "on":
        return torch.autocast("mps", dtype=torch.float16)
    return contextlib.nullcontext()


def load_fusion_features(path: str | Path, ids: list[str]) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    index = {k: i for i, k in enumerate(z["ids"].tolist())}
    feats = np.concatenate([z["prob"][:, None], z["emb"].astype(np.float32)], axis=1)
    return feats[[index[i] for i in ids]]


@torch.no_grad()
def predict(model: SACENetClassifier, data: Encoded, pad_id: int, device: str, batch_size: int = 64,
            task: str = "main", amp: str = "off", return_pooled: bool = False):
    model.eval()
    logits = np.zeros((len(data.ids), model.heads[task].out_features), dtype=np.float32)
    pooled = np.zeros((len(data.ids), model.encoder.config.hidden_size), dtype=np.float32) if return_pooled else None
    for b in batches(data, batch_size, pad_id, shuffle=False, sort_by_length=True):
        with _autocast(device, amp):
            out = model(b["input_ids"].to(device), b["attention_mask"].to(device), task=task,
                        fusion_feats=b["fusion"].to(device) if "fusion" in b else None,
                        return_pooled=return_pooled)
        lg, h = (out if return_pooled else (out, None))
        logits[b["index"]] = lg.float().cpu().numpy()
        if return_pooled:
            pooled[b["index"]] = h.float().cpu().numpy()
    return (logits, pooled) if return_pooled else logits


def _eval_loss(logits: np.ndarray, y: np.ndarray) -> float:
    return float(nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(y)).item())


def train_run(cfg: dict, run_dir: Path) -> dict:
    cfg = {**DEFAULTS, **cfg}
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "metrics.json").exists():
        logger.info("Run %s already complete; skipping", run_dir.name)
        return load_json(run_dir / "metrics.json")
    save_json(cfg, run_dir / "config.json")
    set_seed(cfg["seed"])
    dev = select_device(cfg.get("device", "auto"))
    device = dev.selected
    save_json({"device": dev.to_dict(), "packages": package_versions()}, run_dir / "environment.json")

    task = cfg["task"]
    labels = TASKS[task]["labels"]
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg["revision"])
    pad_id = tok.pad_token_id

    split_data = {}
    for split in ("train", "val", "test"):
        ids, texts, y, _ = load_split(task, split)
        if cfg.get("smoke_n"):   # smoke test only: deterministic subsample
            keep = np.random.default_rng(0).permutation(len(ids))[: cfg["smoke_n"]]
            ids, texts, y = [ids[i] for i in keep], [texts[i] for i in keep], y[keep]
        fus = load_fusion_features(cfg["fusion"]["features"], ids) if cfg["fusion"] else None
        split_data[split] = encode(tok, ids, texts, y, cfg["max_len"], fus)
    if cfg.get("shuffle_train_labels"):
        # Control condition: permute TRAINING labels only (val/test keep gold labels).
        split_data["train"].labels = np.random.default_rng(cfg["seed"] + 4242).permutation(split_data["train"].labels)
    lens = [len(x) for x in split_data["train"].input_ids]
    trunc = float(np.mean([l >= cfg["max_len"] for l in lens]))

    aux = cfg["aux"]
    aux_data = None
    if aux:
        a_ids, a_texts, a_y, _ = load_split(aux["task"], "train")
        if cfg.get("smoke_n"):
            a_ids, a_texts, a_y = a_ids[: cfg["smoke_n"]], a_texts[: cfg["smoke_n"]], a_y[: cfg["smoke_n"]]
        if aux.get("shuffle_labels"):
            # Control condition: identical auxiliary inputs, labels permuted (keeps the class ratio).
            a_y = np.random.default_rng(cfg["seed"] + 999).permutation(a_y)
        aux_data = encode(tok, a_ids, a_texts, a_y, cfg["max_len"])

    heads = {"main": len(labels)}
    if aux:
        heads["aux"] = len(TASKS[aux["task"]]["labels"])
    fusion_dim = split_data["train"].fusion.shape[1] if cfg["fusion"] else 0
    model = SACENetClassifier(cfg["model_name"], heads, cfg["dropout"], fusion_dim=fusion_dim,
                              revision=cfg["revision"],
                              freeze_word_embeddings=cfg["freeze_word_embeddings"]).to(device)

    w_main = class_weights(split_data["train"].labels, len(labels)).to(device) if cfg["class_weighting"] else None
    loss_main = nn.CrossEntropyLoss(weight=w_main)
    loss_aux = None
    if aux:
        w_aux = class_weights(aux_data.labels, heads["aux"]).to(device) if cfg["class_weighting"] else None
        loss_aux = nn.CrossEntropyLoss(weight=w_aux)

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": cfg["lr"]},
                             {"params": head_params, "lr": cfg["head_lr"] or cfg["lr"]}],
                            weight_decay=cfg["weight_decay"], fused=(device in ("cuda", "mps")))
    steps_per_epoch = math.ceil(len(split_data["train"].ids) / cfg["batch_size"] / cfg["grad_accum"])
    total = steps_per_epoch * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup"] * total), total)

    ckpt_dir = run_dir / "checkpoint"
    ckpt_dir.mkdir(exist_ok=True)
    history, best, bad, start_epoch = [], -1.0, 0, 0
    best_state = None
    if cfg["resume"] and (ckpt_dir / "last.pt").exists():
        st = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        history, best, bad, start_epoch = st["history"], st["best"], st["bad"], st["epoch"] + 1
        best_state = st["best_state"]
        logger.info("Resumed %s from epoch %d", run_dir.name, start_epoch)

    t_start = time.perf_counter()
    log_f = (run_dir / "train_log.jsonl").open("a", encoding="utf-8")
    aux_iter = None
    for epoch in range(start_epoch, cfg["epochs"]):
        if bad >= cfg["patience"]:
            break
        model.train()
        t_ep = time.perf_counter()
        run_loss, run_aux, n_b = 0.0, 0.0, 0
        opt.zero_grad(set_to_none=True)
        if aux:
            aux_iter = batches(aux_data, cfg["batch_size"], pad_id, shuffle=True, seed=cfg["seed"] * 100 + epoch)
        for step, b in enumerate(batches(split_data["train"], cfg["batch_size"], pad_id, shuffle=True,
                                         seed=cfg["seed"] * 1000 + epoch)):
            with _autocast(device, cfg["amp"]):
                logits = model(b["input_ids"].to(device), b["attention_mask"].to(device), "main",
                               fusion_feats=b["fusion"].to(device) if "fusion" in b else None)
                loss = loss_main(logits.float(), b["labels"].to(device))
                la = torch.zeros((), device=device)
                if aux:
                    ab = next(aux_iter, None)
                    if ab is None:   # auxiliary set is smaller: cycle it
                        aux_iter = batches(aux_data, cfg["batch_size"], pad_id, shuffle=True,
                                           seed=cfg["seed"] * 100 + epoch + 7 * step)
                        ab = next(aux_iter)
                    al = model(ab["input_ids"].to(device), ab["attention_mask"].to(device), "aux")
                    la = loss_aux(al.float(), ab["labels"].to(device))
                total_loss = (loss + aux["lambda"] * la) if aux else loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
            (total_loss / cfg["grad_accum"]).backward()
            if (step + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["clip"])
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            run_loss += float(loss.item())
            run_aux += float(la.item())
            n_b += 1
            if step % cfg["log_every"] == 0:
                log_f.write(json.dumps({"epoch": epoch, "step": step, "loss": float(loss.item()),
                                        "aux_loss": float(la.item()), "lr": sched.get_last_lr()[0]}) + "\n")
                log_f.flush()
        # ---- epoch-end validation ----
        if device == "mps":       # release cached allocator blocks (lower standing footprint; no effect on results)
            torch.mps.empty_cache()
        val_logits = predict(model, split_data["val"], pad_id, device, cfg["eval_batch_size"])
        yv = split_data["val"].labels
        vm = classification_metrics(yv, softmax(val_logits), labels)
        tr_logits_eval = None
        rec = {"epoch": epoch, "train_loss": run_loss / max(n_b, 1), "aux_loss": run_aux / max(n_b, 1) if aux else None,
               "val_loss": _eval_loss(val_logits, yv), "val_macro_f1": vm["macro_f1"],
               "val_accuracy": vm["accuracy"], "epoch_seconds": time.perf_counter() - t_ep,
               "memory": memory_snapshot()}
        history.append(rec)
        logger.info("[%s] epoch %d train_loss=%.4f val_loss=%.4f val_mF1=%.4f (%.0fs)", run_dir.name, epoch,
                    rec["train_loss"], rec["val_loss"], rec["val_macro_f1"], rec["epoch_seconds"])
        if vm["macro_f1"] > best:
            best, bad = vm["macro_f1"], 0
            # Best weights are kept in host RAM (~0.95 GB for MuRIL) rather than on
            # disk: the audit machine had < 6 GB free disk.
            # Frozen parameters (the token-embedding table, ~600 MB for MuRIL) never change, so
            # only trainable parameters and buffers are copied.
            frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
                          if k not in frozen}
        else:
            bad += 1
        if cfg["save_resume"]:   # opt-in: model + optimiser state (~3x model size on disk)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "history": history, "best": best, "bad": bad, "epoch": epoch,
                        "best_state": best_state}, ckpt_dir / "last.pt")
        del tr_logits_eval
    log_f.close()
    train_seconds = time.perf_counter() - t_start

    # ---- final evaluation with the best checkpoint ----
    missing, unexpected = model.load_state_dict(best_state, strict=False)
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert not unexpected and set(missing) <= frozen, (missing, unexpected)
    del best_state
    out = {"exp_id": cfg["exp_id"], "task": task, "labels": labels, "history": history,
           "best_epoch": int(np.argmax([h["val_macro_f1"] for h in history])),
           "train_seconds": train_seconds, "device": device, "n_parameters": model.n_parameters(),
           "n_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "train_truncation_rate": trunc}
    pred_rows = {}
    eval_splits = ("val",) if cfg.get("skip_test") else ("val", "test")   # tuning runs never touch test
    for split in eval_splits:
        lg = predict(model, split_data[split], pad_id, device, cfg["eval_batch_size"])
        pred_rows[split] = lg
    # Degenerate-run flag: the selected checkpoint predicts (almost) a single class on validation.
    vp = pred_rows["val"].argmax(1)
    out["n_predicted_classes_val"] = int(len(np.unique(vp)))
    out["collapsed"] = bool(np.bincount(vp).max() / len(vp) > 0.98)
    if out["collapsed"]:
        logger.warning("[%s] COLLAPSED: selected checkpoint predicts one class on validation", run_dir.name)
    T = fit_temperature(pred_rows["val"], split_data["val"].labels)
    out["temperature"] = T
    for split, lg in pred_rows.items():
        y = split_data[split].labels
        out[split] = classification_metrics(y, softmax(lg), labels)
        out[split + "_calibrated"] = {k: v for k, v in classification_metrics(y, softmax(lg, T), labels).items()
                                      if k in ("ece", "brier", "nll", "reliability")}
        save_jsonl([{"id": i, "y": int(t), "logits": l.tolist(), "prob": softmax(l[None])[0].tolist()}
                    for i, t, l in zip(split_data[split].ids, y, lg)], run_dir / f"predictions_{split}.jsonl")
    if aux and not cfg.get("skip_test"):
        # Report the auxiliary head on the sarcasm test split too (diagnostic).
        a_ids, a_texts, a_y, _ = load_split(aux["task"], "test")
        a_enc = encode(tok, a_ids, a_texts, a_y, cfg["max_len"])
        al = predict(model, a_enc, pad_id, device, cfg["eval_batch_size"], task="aux")
        out["aux_test"] = classification_metrics(a_y, softmax(al), TASKS[aux["task"]]["labels"])

    if cfg["export_features"]:
        _export_features(model, tok, cfg, device, run_dir)
    for extra in cfg.get("extra_eval", []) or []:
        if cfg["fusion"]:
            # Fusion needs sarcasm-model features, which exist only for MC-Hinglish texts.
            out[f"ext_{extra}"] = {"skipped": "fusion features unavailable for external data"}
            continue
        e_ids, e_texts, e_y, _ = load_split(extra, "test")
        e_enc = encode(tok, e_ids, e_texts, e_y, cfg["max_len"])
        lg = predict(model, e_enc, pad_id, device, cfg["eval_batch_size"])
        save_jsonl([{"id": i, "y": int(t), "logits": l.tolist(), "prob": softmax(l[None])[0].tolist()}
                    for i, t, l in zip(e_ids, e_y, lg)], run_dir / f"predictions_{extra}.jsonl")
        out[f"ext_{extra}"] = {"n": len(e_ids), "saved": f"predictions_{extra}.jsonl"}
    out["memory_end"] = memory_snapshot()
    save_json(out, run_dir / "metrics.json")

    if cfg["keep_checkpoint"]:
        model_dir = run_dir / "checkpoint" / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        # fp16 storage halves disk use; Predictor casts back to fp32 on load.
        torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()},
                   model_dir / "weights.pt")
        tok.save_pretrained(model_dir)
        save_json({"model_name": cfg["model_name"], "revision": cfg["revision"], "heads": heads,
                   "labels": labels, "fusion_dim": fusion_dim, "dropout": cfg["dropout"],
                   "max_len": cfg["max_len"], "temperature": T}, model_dir / "sacenet_config.json")
        (ckpt_dir / "last.pt").unlink(missing_ok=True)
    else:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    del model, opt
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return out


def _export_features(model, tok, cfg, device, run_dir: Path) -> None:
    """Sarcasm model -> P(sarcastic) + pooled embedding for every text of the target tasks."""
    pad_id = tok.pad_token_id
    for target in cfg["export_features"]:
        all_ids, all_texts = [], []
        for split in ("train", "val", "test"):
            try:
                ids, texts, _, _ = load_split(target, split)
            except FileNotFoundError:
                continue
            all_ids += ids
            all_texts += texts
        enc = encode(tok, all_ids, all_texts, np.zeros(len(all_ids), dtype=int), cfg["max_len"])
        lg, emb = predict(model, enc, pad_id, device, cfg["eval_batch_size"], return_pooled=True)
        prob = softmax(lg)[:, 1]
        np.savez_compressed(run_dir / f"features_{target}.npz", ids=np.array(all_ids), prob=prob.astype(np.float32),
                            emb=emb.astype(np.float16))
