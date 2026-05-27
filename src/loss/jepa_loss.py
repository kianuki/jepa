import torch
from torch import Tensor
from torch import nn


class JEPALoss(nn.Module):
    def __init__(self, act_lambda=0.01):
        super().__init__()

        self.mse = nn.MSELoss()
        self.act_lambda = act_lambda

    def forward(
        self, pred_target_embeddings, act_loss, target_embeddings, **batch
    ) -> Tensor:

        loss = self.mse(pred_target_embeddings, target_embeddings)
        loss += self.act_lambda * act_loss

        return {"loss": loss}
