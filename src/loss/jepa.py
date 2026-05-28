import torch.nn.functional as F
from torch import nn


class JEPALoss(nn.Module):
    """
    I-JEPA feature prediction loss.

    Original I-JEPA normalizes target features over the embedding dimension
    and optimizes Smooth L1 between predictor outputs and target embeddings.
    """

    def __init__(self, act_lambda=0.001, normalize_targets=True):
        super().__init__()
        self.normalize_targets = normalize_targets
        self.act_lambda = act_lambda

    def forward(self, predictions, targets, act_loss=None, **batch):
        if self.normalize_targets:
            targets = F.layer_norm(targets, (targets.size(-1),))

        loss = F.smooth_l1_loss(predictions, targets)
        result = {"loss": loss}

        if act_loss is not None:
            result["act_loss"] = act_loss
            result["mse_loss"] = loss
            result["loss"] = result["loss"] + act_loss * self.act_lambda

        return result
