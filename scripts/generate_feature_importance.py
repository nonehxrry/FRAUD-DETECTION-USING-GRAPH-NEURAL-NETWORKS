from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch


def _resolve_first(base: Path, names):
    for name in names:
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing any of: {names}")


def main():
    artifact_dir = Path("models/saved")

    model_path = _resolve_first(artifact_dir, ["best_model.pt"])
    scaler_path = _resolve_first(artifact_dir, ["scaler.pkl"])

    scaler = joblib.load(scaler_path)
    state_dict = torch.load(model_path, map_location="cpu")

    # Find learned transaction projection weights from LazyLinear.
    candidates = [
        k for k in state_dict.keys()
        if "node_projections.transaction" in k and k.endswith("weight")
    ]
    if not candidates:
        raise KeyError("Could not find transaction projection weights in state_dict")

    weight_key = sorted(candidates)[0]
    w = state_dict[weight_key]
    if w.ndim != 2:
        raise ValueError(f"Unexpected weight shape for {weight_key}: {tuple(w.shape)}")

    # Mean absolute weight across hidden units as a model-based importance proxy.
    importance = w.abs().mean(dim=0).cpu().numpy().astype(float)

    if hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)
    else:
        feature_names = [f"f_{i}" for i in range(len(importance))]

    n = min(len(feature_names), len(importance))
    feature_names = feature_names[:n]
    importance = importance[:n]

    denom = float(importance.sum()) + 1e-12
    importance = importance / denom

    out_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    out_path = artifact_dir / "feature_importance.csv"
    out_df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
