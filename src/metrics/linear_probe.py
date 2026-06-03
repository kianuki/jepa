from src.metrics.base_metric import BaseMetric
from torch import nn
from tqdm import tqdm
import torch

class LinearProbeMetric(BaseMetric):
    def __init__(self, embed_dim, num_classes=100, probe_epochs=10, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.probe_epochs = probe_epochs
        self._is_trained = False

    def set_encoder(self, encoder, device):
        self.encoder = encoder
        self.device = device

    def train_probe(self, train_loader, device):
        self.init_classifier()
        self.classifier.train()
        self.encoder.eval()
        
        self._is_trained = False

        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

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
    
    def init_classifier(self):
        self.classifier = nn.Linear(self.embed_dim, self.num_classes).to(self.device)

    def __call__(self, image, fine_label, **batch):
        assert self._is_trained, "Call train_probe() before using metric"

        with torch.no_grad():
            emb = self.encoder(image).mean(dim=1)
            logits = self.classifier(emb)

        acc = (logits.argmax(1) == fine_label).float().mean()
        return acc
