import torch.nn.functional as F

from src.metrics.base_metric import BaseMetric


class JEPAMetric(BaseMetric):
    def __init__(self, metric_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_type = metric_type

    def __call__(self, predictions, targets, avg_act_steps=None, **batch):
        if self.metric_type == "cosine":
            return F.cosine_similarity(predictions, targets, dim=-1).mean().item()

        if self.metric_type == "mse":
            return F.mse_loss(predictions, targets).item()
        
        if self.metric_type == "target_std":
            target_std = targets.std(dim=0).mean().item()
            return target_std
        
        if self.metric_type == "avg_act_steps":
            if avg_act_steps is None:
                return 0.0
            return avg_act_steps
        raise ValueError(f"Unknown JEPA metric type: {self.metric_type}")
