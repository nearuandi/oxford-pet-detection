from typing import Any


def detection_collate(batch: list[tuple[Any, Any]]):
    images, targets = zip(*batch)
    return list(images), list(targets)