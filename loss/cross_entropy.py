import torch.nn as nn


class CrossEntropyLoss:
    def __init__(self, use_sigmoid=True, weight=None, reduction='mean', pos_weight=None):
        if use_sigmoid:
            self.criterion = nn.BCEWithLogitsLoss(
                weight=weight,
                reduction=reduction,
                pos_weight=pos_weight)
        else:
            self.criterion = nn.BCELoss(
                weight=weight,
                reduction=reduction
            )

    def __call__(self, preds, targets):
        return self.criterion(preds, targets)
