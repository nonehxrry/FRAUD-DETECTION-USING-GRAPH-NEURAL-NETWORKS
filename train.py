"""
train.py — FIXED + IMPROVED VERSION
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from torch_geometric.loader import NeighborLoader

from utils.preprocessing import preprocess_pipeline
from utils.graph_builder import build_graph
from models.gnn_model import FraudGNN, FraudGraphSAGE


# ────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Device: {DEVICE}")


# ────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--transaction', default='data/train_transaction.csv')
    p.add_argument('--identity', default='data/train_identity.csv')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--model', choices=['gat', 'sage'], default='gat')
    p.add_argument('--max_nodes', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--num_neighbors', type=str, default='10,5,5',
                   help='Comma-separated neighbors sampled per layer, e.g. 15,10,5')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--amp', action='store_true', default=True,
                   help='Enable mixed precision on CUDA for lower memory usage')
    p.add_argument('--full_batch', action='store_true',
                   help='Use full-batch training (useful for small demo graphs)')
    p.add_argument('--output', default='models/saved/')
    return p.parse_args()


# ────────────────────────────────────────────────
def make_masks(n, val_ratio=0.15, test_ratio=0.15, seed=42):
    idx = np.arange(n)

    train_idx, temp_idx = train_test_split(
        idx, test_size=val_ratio + test_ratio, random_state=seed
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed
    )

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


# ────────────────────────────────────────────────
def _parse_num_neighbors(spec: str, num_layers: int):
    vals = [int(v.strip()) for v in spec.split(',') if v.strip()]
    if not vals:
        vals = [10]
    if len(vals) < num_layers:
        vals.extend([vals[-1]] * (num_layers - len(vals)))
    return vals[:num_layers]


def _build_loader(data, mask, num_neighbors, batch_size, num_workers, shuffle):
    seed_idx = torch.where(mask)[0]
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=('transaction', seed_idx),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )


def train_epoch(model, loader, optimizer, loss_fn, scaler, use_amp: bool):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(batch.x_dict, batch.edge_index_dict)
            seed_count = batch['transaction'].batch_size
            logits = out[:seed_count]
            labels = batch['transaction'].y[:seed_count]
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * seed_count
        total_examples += seed_count

    return total_loss / max(total_examples, 1)


def train_epoch_full(model, data, optimizer, loss_fn, scaler, use_amp: bool):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        out = model(data.x_dict, data.edge_index_dict)
        mask = data['transaction'].train_mask
        loss = loss_fn(out[mask], data['transaction'].y[mask])

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return float(loss.item())


# ────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(DEVICE, non_blocking=True)
        out = model(batch.x_dict, batch.edge_index_dict)
        seed_count = batch['transaction'].batch_size

        logits = out[:seed_count]
        labels = batch['transaction'].y[:seed_count]

        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels_np)

    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)

    auc = roc_auc_score(labels_np, probs)

    return auc, probs, preds, labels_np


@torch.no_grad()
def evaluate_full(model, data, mask):
    model.eval()

    out = model(data.x_dict, data.edge_index_dict)

    logits = out[mask]
    labels = data['transaction'].y[mask]

    probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    preds = logits.argmax(dim=-1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    auc = roc_auc_score(labels_np, probs)

    return auc, probs, preds, labels_np


# ────────────────────────────────────────────────
def main():
    args = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # 1. Preprocess
    df, encoders, scaler = preprocess_pipeline(
        args.transaction,
        args.identity
    )

    w0, w1 = df.attrs['class_weights']
    class_weights = torch.tensor([w0, w1], dtype=torch.float).to(DEVICE)

    # 2. Build graph
    data = build_graph(df, max_nodes=args.max_nodes)

    # OPTIONAL STABILITY TIP (important for your graph)
    # data = edge_dropout(data, p=0.1)

    # 3. Masks
    n_tx = data['transaction'].x.shape[0]
    train_mask, val_mask, test_mask = make_masks(n_tx)

    data['transaction'].train_mask = train_mask
    data['transaction'].val_mask = val_mask
    data['transaction'].test_mask = test_mask

    num_neighbors = _parse_num_neighbors(args.num_neighbors, args.layers)

    # 4. Model
    in_channels = data['transaction'].x.shape[1]

    if args.model == 'gat':
        model = FraudGNN(
            metadata=data.metadata(),
            in_channels=in_channels,
            hidden_channels=args.hidden,
            num_layers=args.layers,
            heads=args.heads,
            dropout=args.dropout
        ).to(DEVICE)
    else:
        model = FraudGraphSAGE(
            metadata=data.metadata(),
            in_channels=in_channels,
            hidden_channels=args.hidden,
            num_layers=args.layers,
            dropout=args.dropout
        ).to(DEVICE)

    train_loader = val_loader = test_loader = None
    if not args.full_batch:
        train_loader = _build_loader(
            data,
            data['transaction'].train_mask,
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        )
        val_loader = _build_loader(
            data,
            data['transaction'].val_mask,
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False
        )
        test_loader = _build_loader(
            data,
            data['transaction'].test_mask,
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False
        )
    else:
        data = data.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # FIX: verbose removed (PyTorch compatibility)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=5,
        factor=0.5
    )

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    use_amp = bool(args.amp and DEVICE.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 5. Training
    print(f"\n🚀 Training {args.model.upper()} for {args.epochs} epochs...\n")

    best_val_auc = 0
    patience_counter = 0
    patience_limit = 10

    train_losses = []
    val_aucs = []

    for epoch in range(1, args.epochs + 1):

        if args.full_batch:
            loss = train_epoch_full(model, data, optimizer, loss_fn, scaler, use_amp)
            val_auc, _, _, _ = evaluate_full(model, data, val_mask.to(DEVICE))
        else:
            loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, use_amp)
            val_auc, _, _, _ = evaluate(model, val_loader)

        scheduler.step(val_auc)

        train_losses.append(loss)
        val_aucs.append(val_auc)

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), f"{args.output}/best_model.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f}")

        if patience_counter >= patience_limit:
            print("🛑 Early stopping triggered")
            break

    # 6. Test
    print("\n📊 Evaluating on test set...")

    model.load_state_dict(torch.load(f"{args.output}/best_model.pt"))

    if args.full_batch:
        test_auc, test_probs, test_preds, test_labels = evaluate_full(
            model,
            data,
            test_mask.to(DEVICE)
        )
    else:
        test_auc, test_probs, test_preds, test_labels = evaluate(model, test_loader)

    cm = confusion_matrix(test_labels, test_preds)
    fpr, tpr, _ = roc_curve(test_labels, test_probs)

    print("\n" + "="*50)
    print(f"Test AUC-ROC: {test_auc:.4f}")
    print("Confusion Matrix:\n", cm)
    print(classification_report(
        test_labels,
        test_preds,
        target_names=['Legitimate', 'Fraud']
    ))
    print("="*50)

    # 7. Save curves
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(val_aucs)
    plt.title("Val AUC")

    plt.tight_layout()
    plt.savefig(f"{args.output}/curves.png")
    plt.close()

    # 8. Save artifacts
    joblib.dump(encoders, f"{args.output}/encoders.pkl")
    joblib.dump(scaler, f"{args.output}/scaler.pkl")

    np.save(f"{args.output}/test_probs.npy", test_probs)
    np.save(f"{args.output}/test_labels.npy", test_labels)
    np.save(f"{args.output}/fpr.npy", fpr)
    np.save(f"{args.output}/tpr.npy", tpr)
    np.save(f"{args.output}/cm.npy", cm)

    joblib.dump({
        "model": args.model,
        "hidden": args.hidden,
        "layers": args.layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "num_neighbors": num_neighbors,
        "full_batch": args.full_batch,
        "in_channels": in_channels,
        "best_val_auc": best_val_auc,
        "test_auc": test_auc
    }, f"{args.output}/meta.pkl")

    print(f"\n✅ Saved to {args.output}")


# ────────────────────────────────────────────────
if __name__ == "__main__":
    main()