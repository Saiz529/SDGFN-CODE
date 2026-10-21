"""Train SDGFN using five predefined center-grouped validation splits."""

import argparse
import os
import warnings

import msvcrt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from sdgfn_high_order import SDGFNHighOrderBranch
from sdgfn_low_order import SDGFNLowOrderBranch


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CFG = {
    "ROI_NUM": 246,
    "BATCH_SIZE": 8,
    "EPOCHS": 120,
    "FOLDS": 5,
}

IRSSN_CFG = {
    "DROPOUT": 0.35,
    "THRESH": 0.04,
}

CO_CFG = {
    "DROPOUT": 0.3,
    "THRESH": 0.03,
}

TRAIN_CFG = {
    "LR_IR": 11e-5,
    "LR_CO": 13e-5,
    "LR_FUS": 1.1e-4,
    "WD": 2e-5,
}

FOLD_CENTER_SPLIT = [
    {
        "val": [
            "S10", "S16", "S19", "S22", "S23", "S24", "S5", "S9",
        ],
        "train": [
            "S1", "S11", "S12", "S13", "S14", "S15", "S17", "S18",
            "S2", "S20", "S21", "S25", "S3", "S6", "S7", "S8",
        ],
    },
    {
        "val": [
            "S15", "S2", "S22", "S24", "S5", "S8", "S9",
        ],
        "train": [
            "S1", "S10", "S11", "S12", "S13", "S14", "S16", "S17",
            "S18", "S19", "S20", "S21", "S23", "S25", "S3", "S6", "S7",
        ],
    },
    {
        "val": [
            "S10", "S12", "S13", "S17", "S19", "S24", "S6", "S9",
        ],
        "train": [
            "S1", "S11", "S14", "S15", "S16", "S18", "S2", "S20",
            "S21", "S22", "S23", "S25", "S3", "S5", "S7", "S8",
        ],
    },
    {
        "val": [
            "S11", "S12", "S13", "S16", "S18", "S21", "S3",
            "S5", "S6", "S9",
        ],
        "train": [
            "S1", "S10", "S14", "S15", "S17", "S19", "S2",
            "S20", "S22", "S23", "S24", "S25", "S7", "S8",
        ],
    },
    {
        "val": [
            "S12", "S14", "S17", "S21", "S22", "S5", "S9",
        ],
        "train": [
            "S1", "S10", "S11", "S13", "S15", "S16", "S18", "S19",
            "S2", "S20", "S23", "S24", "S25", "S3", "S6", "S7", "S8",
        ],
    },
]


def preprocess_irssn(adj):
    """Symmetrize iRSSN and convert negative weights to attenuated magnitudes."""
    adj = (adj + adj.T) / 2
    pos = np.clip(adj, 0, None)
    neg = np.clip(adj, None, 0)
    adj = pos - 0.5 * neg

    adj[np.abs(adj) < IRSSN_CFG["THRESH"]] = 0
    np.fill_diagonal(adj, 1)
    return adj, adj.copy()


def preprocess_co(adj):
    """Symmetrize and sparsify Ho-iRSSN while retaining edge signs."""
    adj = (adj + adj.T) / 2
    adj[np.abs(adj) < CO_CFG["THRESH"]] = 0
    np.fill_diagonal(adj, 1)
    return adj, adj.copy()


class DualDataset(Dataset):
    """Load paired network matrices for subjects with both input files."""

    def __init__(self, df, irssn_dir, high_order_dir):
        self.df = df.reset_index(drop=True)
        self.valid_indices = []
        self.missing_ids = []
        self.data_cache = []

        for idx in range(len(self.df)):
            subject_id = self.df.iloc[idx]["ID"]
            path1 = os.path.join(irssn_dir, f"iRSSN_{subject_id}.txt")
            path2 = os.path.join(high_order_dir, f"iRSSN_{subject_id}.txt")

            if os.path.exists(path1) and os.path.exists(path2):
                self.valid_indices.append(idx)

                a1 = np.fromfile(
                    path1, sep=" ", dtype=np.float32
                ).reshape(246, 246)
                a2 = np.fromfile(
                    path2, sep=" ", dtype=np.float32
                ).reshape(246, 246)

                a1, x1 = preprocess_irssn(a1)
                a2, x2 = preprocess_co(a2)
                label = 1 if self.df.iloc[idx]["group"] == 1 else 0
                self.data_cache.append((a1, x1, a2, x2, label))
            else:
                self.missing_ids.append(subject_id)

        print(f"Valid samples: {len(self.valid_indices)}")
        if self.missing_ids:
            print(
                f"Skipped subjects with missing files: {len(self.missing_ids)}"
            )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        a1, x1, a2, x2, label = self.data_cache[idx]
        return (
            torch.FloatTensor(a1),
            torch.FloatTensor(x1),
            torch.FloatTensor(a2),
            torch.FloatTensor(x2),
            torch.tensor(label, dtype=torch.long),
        )


class SDGFN(nn.Module):
    """Dual-branch structural graph fusion network."""

    def __init__(self):
        super().__init__()

        self.irssn_net = SDGFNLowOrderBranch(
            dropout=IRSSN_CFG["DROPOUT"],
            num_classes=128,
        )
        self.co_net = SDGFNHighOrderBranch(
            ROInum=246,
            hidden_dim=128,
            layers=3,
            num_classes=128,
            dropout=CO_CFG["DROPOUT"],
        )

        self.gate = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.cls = nn.Linear(128, 2)
        self.cls_ir = nn.Linear(128, 2)
        self.cls_co = nn.Linear(128, 2)

    def forward(self, a1, x1, a2, x2, epoch=None):
        f1 = self.irssn_net(x1, a1)
        f2 = self.co_net(x2, a2)

        f1 = nn.functional.normalize(f1, dim=1)
        f2 = nn.functional.normalize(f2, dim=1)
        feat_cat = torch.cat([f1, f2], dim=1)

        if epoch is not None and epoch < 5:
            g = torch.ones_like(f1) * 0.45
        else:
            g = torch.sigmoid(self.gate(feat_cat) / 0.3)
            g = torch.clamp(g, 0.25, 0.75)

        fused = g * f1 + (1 - g) * f2
        fused = self.fusion(fused)

        out = self.cls(fused)
        out_ir = self.cls_ir(f1)
        out_co = self.cls_co(f2)
        return out, out_ir, out_co, f1, f2, g


def train_epoch(
    model, loader, opt_i, opt_c, opt_f, criterion, scaler, epoch
):
    model.train()
    total_loss = 0
    y_true, y_pred = [], []
    gate_values = []

    for a1, x1, a2, x2, y in loader:
        a1 = a1.to(DEVICE)
        x1 = x1.to(DEVICE)
        a2 = a2.to(DEVICE)
        x2 = x2.to(DEVICE)
        y = y.to(DEVICE)

        opt_i.zero_grad()
        opt_c.zero_grad()
        opt_f.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
            out, out_ir, out_co, f1, f2, g = model(
                a1, x1, a2, x2, epoch
            )
            loss_main = criterion(out, y)
            loss_ir = criterion(out_ir, y)
            loss_co = criterion(out_co, y)

            cosine = nn.functional.cosine_similarity(f1, f2, dim=1)
            loss_orth = torch.clamp(
                torch.abs(cosine) - 0.3, min=0.0
            ).mean()

            alpha = min(0.7, epoch / 50)
            loss = (
                loss_main
                + 0.07 * loss_ir
                + 0.07 * loss_co
                + alpha * 0.03 * loss_orth
            )

        scaler.scale(loss).backward()

        # Preserve the gradient-processing order of the supplied implementation.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)

        scaler.step(opt_i)
        scaler.step(opt_c)
        scaler.step(opt_f)
        scaler.update()

        total_loss += loss.item()
        y_pred.extend(out.argmax(1).cpu().numpy())
        y_true.extend(y.cpu().numpy())
        gate_values.append(g.mean().item())

    return (
        total_loss / len(loader),
        accuracy_score(y_true, y_pred),
        np.mean(gate_values),
    )


@torch.no_grad()
def test_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    y_true, y_pred, y_score = [], [], []

    for a1, x1, a2, x2, y in loader:
        a1 = a1.to(DEVICE)
        x1 = x1.to(DEVICE)
        a2 = a2.to(DEVICE)
        x2 = x2.to(DEVICE)
        y = y.to(DEVICE)

        out, _, _, _, _, _ = model(a1, x1, a2, x2, epoch=None)
        loss = criterion(out, y)
        total_loss += loss.item()

        probability = torch.softmax(out, 1)[:, 1].cpu().numpy()
        y_pred.extend(out.argmax(1).cpu().numpy())
        y_score.extend(probability)
        y_true.extend(y.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception as exc:
        # Retain the original fallback, but explicitly flag it as invalid.
        warnings.warn(
            f"AUC calculation failed: {exc}. The legacy fallback is 0.5; "
            "this value is not a measured AUC.",
            RuntimeWarning,
        )
        auc = 0.5

    if not np.isfinite(auc):
        warnings.warn(
            "AUC is undefined for this validation set.",
            RuntimeWarning,
        )

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    sen = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    pre = tp / (tp + fp + 1e-8)

    # MDD is the positive class. F1 does not affect checkpoint selection.
    denominator = 2 * tp + fp + fn
    f1_score = 2 * tp / denominator if denominator > 0 else 0.0

    return total_loss / len(loader), acc, auc, sen, spec, pre, f1_score


def check_esc():
    """Return whether ESC has been pressed in the Windows console."""
    if msvcrt.kbhit():
        if msvcrt.getch() == b"\x1b":
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SDGFN on predefined center-grouped splits."
    )
    parser.add_argument(
        "--metadata",
        default="data/metadata.xlsx",
        help="Metadata spreadsheet containing ID and group columns.",
    )
    parser.add_argument(
        "--irssn-dir",
        default="data/irssn",
        help="Directory containing low-order network matrices.",
    )
    parser.add_argument(
        "--high-order-dir",
        default="data/high_order",
        help="Directory containing high-order network matrices.",
    )
    parser.add_argument(
        "--save-dir",
        default="checkpoints",
        help="Output directory for checkpoints.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    df = pd.read_excel(args.metadata)
    df["center"] = (
        df["ID"].astype(str).str.split("-").str[0].str.strip()
    )

    print(f"Centers in metadata: {df['center'].nunique()}")
    print(f"Subjects in metadata: {len(df)}")
    print("Press ESC to interrupt the current split.")

    # The original implementation shares one scaler across all splits.
    scaler = torch.amp.GradScaler("cuda")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.01,
        weight=torch.tensor([1.0, 1.45]).to(DEVICE),
    )

    split_metric_list = []

    for fold_idx in range(CFG["FOLDS"]):
        split_number = fold_idx + 1
        split = FOLD_CENTER_SPLIT[fold_idx]
        val_centers = split["val"]
        train_centers = split["train"]

        print(f"\n{'=' * 20} Split {split_number} {'=' * 20}")
        print(f"Training centers: {sorted(train_centers)}")
        print(f"Validation centers: {sorted(val_centers)}")

        train_df = df[
            df["center"].isin(train_centers)
        ].reset_index(drop=True)
        val_df = df[
            df["center"].isin(val_centers)
        ].reset_index(drop=True)

        print(
            f"Metadata counts: training={len(train_df)}, "
            f"validation={len(val_df)}"
        )

        train_dataset = DualDataset(
            train_df, args.irssn_dir, args.high_order_dir
        )
        val_dataset = DualDataset(
            val_df, args.irssn_dir, args.high_order_dir
        )

        # These counts reflect subjects with both network files.
        for name, dataset in (
            ("Training", train_dataset),
            ("Validation", val_dataset),
        ):
            mdd_count = sum(item[4] for item in dataset.data_cache)
            print(
                f"{name}: MDD={mdd_count}, "
                f"HC={len(dataset) - mdd_count}"
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG["BATCH_SIZE"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=CFG["BATCH_SIZE"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        model = SDGFN().to(DEVICE)

        opt_i = optim.AdamW(
            model.irssn_net.parameters(),
            lr=TRAIN_CFG["LR_IR"],
            weight_decay=TRAIN_CFG["WD"],
        )
        opt_c = optim.AdamW(
            model.co_net.parameters(),
            lr=TRAIN_CFG["LR_CO"],
            weight_decay=TRAIN_CFG["WD"],
        )
        opt_f = optim.AdamW(
            [
                *model.gate.parameters(),
                *model.fusion.parameters(),
                *model.cls.parameters(),
                *model.cls_ir.parameters(),
                *model.cls_co.parameters(),
            ],
            lr=TRAIN_CFG["LR_FUS"],
            weight_decay=TRAIN_CFG["WD"],
        )

        best_acc = 0.0
        best_result = None

        for epoch in range(CFG["EPOCHS"]):
            if check_esc():
                print(
                    "Split interrupted. Any previously selected result "
                    "will remain in the summary."
                )
                break

            tr_loss, tr_acc, gate_mean = train_epoch(
                model,
                train_loader,
                opt_i,
                opt_c,
                opt_f,
                criterion,
                scaler,
                epoch,
            )

            va_loss, va_acc, va_auc, sen, spec, pre, f1 = test_epoch(
                model, val_loader, criterion
            )

            print(
                f"Epoch {epoch:03d} | "
                f"TrainAcc={tr_acc:.4f} | "
                f"ValAcc={va_acc:.4f} | "
                f"AUC={va_auc:.4f} | "
                f"F1={f1:.4f} | "
                f"Gate={gate_mean:.2f}"
            )

            if va_acc > best_acc:
                best_acc = va_acc
                best_result = (va_acc, va_auc, sen, spec, pre, f1)

                checkpoint_path = os.path.join(
                    args.save_dir,
                    f"best_fold_{split_number}.pth",
                )
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

        if best_result is not None:
            split_metric_list.append(best_result)

    if not split_metric_list:
        print("No checkpoint results are available for summary.")
        return

    results = np.asarray(split_metric_list)
    metric_names = ("ACC", "AUC", "SEN", "SPEC", "PRE", "F1")

    print("\n" + "=" * 60)
    print(f"Results from {len(results)} included splits")
    print("Unweighted mean ± sample standard deviation")

    for column, name in enumerate(metric_names):
        mean = results[:, column].mean()
        std = (
            np.std(results[:, column], ddof=1)
            if len(results) > 1
            else float("nan")
        )
        print(f"{name:<4} = {mean:.4f} ± {std:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()