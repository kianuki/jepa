import torch.nn.functional as F

from src.metrics.base_metric import BaseMetric


class JEPAMetric(BaseMetric):
    def __init__(self, metric_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_type = metric_type

    def __call__(self, predictions, targets, **batch):
        if self.metric_type == "cosine":
            return F.cosine_similarity(predictions, targets, dim=-1).mean().item()

        if self.metric_type == "mse":
            return F.mse_loss(predictions, targets).item()

        raise ValueError(f"Unknown JEPA metric type: {self.metric_type}")
