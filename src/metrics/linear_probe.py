from src.metrics.base_metric import BaseMetric
from torch import nn
from tqdm import tqdm
import torch

class LinearProbeMetric(BaseMetric):
    def __init__(self, embed_dim, num_classes=100, probe_epochs=10, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.probe_epochs = probe_epochs
        self.classifier = nn.Linear(embed_dim, num_classes)
        self._is_trained = False

    def set_encoder(self, encoder, device):
        self.encoder = encoder
        self.device = device
        self.classifier = self.classifier.to(device)

    def train_probe(self, train_loader, device):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        self.encoder.eval()
        for _ in tqdm(range(self.probe_epochs), desc="Linear Probe", leave=False):
            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["fine_label"].to(device)

                with torch.no_grad():
                    emb = self.encoder(images).mean(dim=1)

                loss = criterion(self.classifier(emb), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._is_trained = True

    def __call__(self, image, fine_label, **batch):
        assert self._is_trained, "Call train_probe() before using metric"

        with torch.no_grad():
            emb = self.encoder(image).mean(dim=1)
            logits = self.classifier(emb)

        acc = (logits.argmax(1) == fine_label).float().mean()
        return acc