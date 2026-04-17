"""Training entry point.

Usage:
    python -m src.train --config configs/unet_resnet34.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Optional

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .augment import train_tf, val_tf
from .dataset import TileDataset
from .losses import get_loss
from .metrics import ConfusionMatrix
from .models import build_model, count_parameters
from .utils import IGNORE_INDEX, NUM_CLASSES, ensure_dir, set_seed


def load_cfg(path: str) -> dict:
    base_path = os.path.join(os.path.dirname(path), "base.yaml")
    cfg: dict = {}
    if os.path.exists(base_path):
        with open(base_path) as f:
            cfg.update(yaml.safe_load(f) or {})
    with open(path) as f:
        cfg.update(yaml.safe_load(f) or {})
    return cfg


def _maybe_init_wandb(cfg: dict):
    if not cfg.get("use_wandb", True):
        return None
    try:
        import wandb  # type: ignore
        wandb.init(
            project=cfg.get("wandb_project", "landcover-seg"),
            name=cfg["run_name"],
            config=cfg,
            mode=cfg.get("wandb_mode", "online"),
        )
        return wandb
    except Exception as e:  # pragma: no cover - optional dep
        print(f"[warn] wandb disabled: {e}")
        return None


def _log(wb, payload: dict) -> None:
    if wb is not None:
        wb.log(payload)


def evaluate(model, loader, device) -> dict:
    model.eval()
    cm = ConfusionMatrix(NUM_CLASSES, IGNORE_INDEX)
    with torch.no_grad(), autocast():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).argmax(1).cpu().numpy()
            cm.update(pred, y.numpy())
    return cm.report()


def train(cfg: dict, cfg_path: str) -> None:
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(cfg["ckpt_dir"])

    model = build_model(cfg["model"], num_classes=NUM_CLASSES).to(device)
    n_params = count_parameters(model) / 1e6

    loss_fn = get_loss(cfg["loss"])
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4)
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = GradScaler()

    train_ds = TileDataset(cfg["data_root"], "train", train_tf(cfg.get("tile_size", 816)))
    val_ds = TileDataset(cfg["data_root"], "val", val_tf())
    train_loader = DataLoader(
        train_ds, batch_size=cfg["bs"], shuffle=True,
        num_workers=cfg.get("workers", 4), pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["bs"], shuffle=False,
        num_workers=cfg.get("workers", 4), pin_memory=True,
    )

    wb = _maybe_init_wandb(cfg)
    print(f"[info] model={cfg['model']} params={n_params:.2f}M  "
          f"loss={cfg['loss']}  train={len(train_ds)} val={len(val_ds)}")

    best_miou, wait = 0.0, 0
    patience = cfg.get("patience", 10)
    t0 = time.time()
    ckpt_path = os.path.join(cfg["ckpt_dir"], f"{cfg['run_name']}_best.pt")
    history: list[dict] = []
    ensure_dir(cfg.get("output_dir", "outputs"))
    hist_path = os.path.join(cfg["output_dir"], f"{cfg['run_name']}_history.csv")

    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg['epochs']}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        sch.step()
        train_loss = running / max(len(train_loader), 1)

        report = evaluate(model, val_loader, device)
        miou = report["mIoU"]
        print(f"epoch {epoch+1:3d} | loss {train_loss:.4f} | val mIoU {miou:.4f} "
              f"| pix_acc {report['pixel_acc']:.4f}")
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_mIoU": miou,
            "val_mDice": report["mDice"],
            "val_pixel_acc": report["pixel_acc"],
            "lr": opt.param_groups[0]["lr"],
            **{f"val_IoU_{k}": v for k, v in report["per_class_IoU"].items()},
        }
        history.append(row)
        with open(hist_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerows(history)
        _log(wb, {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/mIoU": miou,
            "val/mDice": report["mDice"],
            "val/pixel_acc": report["pixel_acc"],
            "lr": opt.param_groups[0]["lr"],
            **{f"val/IoU_{k}": v for k, v in report["per_class_IoU"].items()},
        })

        if miou > best_miou:
            best_miou = miou
            wait = 0
            torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                        "epoch": epoch + 1, "val_mIoU": miou}, ckpt_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"[info] early stop at epoch {epoch+1}")
                break

    train_time_hr = (time.time() - t0) / 3600.0
    summary = {
        "run_name": cfg["run_name"],
        "model": cfg["model"],
        "loss": cfg["loss"],
        "params_M": n_params,
        "best_val_mIoU": best_miou,
        "train_time_hr": train_time_hr,
        "ckpt": ckpt_path,
    }
    ensure_dir(cfg.get("output_dir", "outputs"))
    with open(os.path.join(cfg["output_dir"], f"{cfg['run_name']}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _log(wb, {"summary/" + k: v for k, v in summary.items() if isinstance(v, (int, float))})
    if wb is not None:
        wb.finish()
    print(f"[done] best mIoU={best_miou:.4f}  time={train_time_hr:.2f}h  ckpt={ckpt_path}")


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    train(cfg, args.config)


if __name__ == "__main__":
    main()
