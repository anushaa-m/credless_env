from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


def write_reward_curve(
    history: Iterable[Mapping[str, Any]],
    *,
    csv_path: str | Path = "rl/reward_curve.csv",
    png_path: str | Path = "rl/reward_curve.png",
) -> dict[str, str]:
    frame = pd.DataFrame(list(history))
    out: dict[str, str] = {}

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    out["csv"] = str(csv_path).replace("\\", "/")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return out

    if not len(frame):
        return out

    if "mean_reward" not in frame.columns:
        return out

    plt.figure(figsize=(8, 4))
    plt.plot(frame["batch_start"], frame["mean_reward"], marker="o", linewidth=2)
    plt.title("GRPO training reward curve")
    plt.xlabel("batch_start")
    plt.ylabel("mean_reward")
    plt.grid(True, alpha=0.3)
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=140)
    plt.close()
    out["png"] = str(png_path).replace("\\", "/")
    return out

