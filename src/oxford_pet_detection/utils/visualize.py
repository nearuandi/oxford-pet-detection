from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_detection_viz(
    image: np.ndarray,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image)

    for b, s, lab in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [float(x) for x in b.tolist()]
        w, h = x2 - x1, y2 - y1
        ax.add_patch(plt.Rectangle((x1, y1), w, h, fill=False, linewidth=2))
        name = class_names[int(lab)] if int(lab) < len(class_names) else str(int(lab))
        ax.text(x1, y1, f"{name} {float(s):.2f}", fontsize=10, bbox=dict(facecolor="white", alpha=0.7))

    ax.axis("off")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)