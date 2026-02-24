from typing import Any
from torch import Tensor


# batch = [
#     (image1, target1),
#     (image2, target2),
#     (image3, target3),
# ]
def detection_collate(batch: list[tuple[Any, Any]]):
    # unpack
    # images  = (image1, image2, image3)
    # targets = (target1, target2, target3)
    images, targets = zip(*batch)
    # images  = [Tensor, Tensor, Tensor]
    # targets = [dict, dict, dict]
    return list(images), list(targets)