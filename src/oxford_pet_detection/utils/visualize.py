from pathlib import Path
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np


def save_detection_viz(
    image,
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    class_names: list[str],
    save_path: Path,
) -> None:
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image)

    H, W = image.shape[0], image.shape[1]

    for b, s, lab in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [float(x) for x in b.tolist()]

        x1 = max(0.0, min(x1, W - 1.0))
        x2 = max(0.0, min(x2, W - 1.0))
        y1 = max(0.0, min(y1, H - 1.0))
        y2 = max(0.0, min(y2, H - 1.0))

        w, h = x2 - x1, y2 - y1
        ax.add_patch(plt.Rectangle((x1, y1), w, h, fill=False, linewidth=2))

        lab_i = int(lab)
        name = class_names[lab_i] if lab_i < len(class_names) else str(lab_i)
        ax.text(x1, y1, f"{name} {float(s):.2f}", fontsize=10, bbox=dict(facecolor="white", alpha=0.7))

    ax.axis("off")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)