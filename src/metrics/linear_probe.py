from src.metrics.base_metric import BaseMetric
from torch import nn
from tqdm import tqdm
import torch

class LinearProbeMetric(BaseMetric):
    def __init__(self, embed_dim, epoch_len, num_classes=100, probe_epochs=10, lp_batch_size=256, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.probe_epochs = probe_epochs
        self.epoch_len = epoch_len
        self.lp_batch_size = lp_batch_size
        self._is_trained = False

    def set_encoder(self, encoder, device):
        self.encoder = encoder.to(device)
        self.device = device

    def train_probe(self, train_loader):
        self.init_classifier()
        self.classifier = self.classifier.train()
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        self.encoder.eval()
        
        self._is_trained = False

        emb_list = []
        labels_list = []

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(self.device)
            labels = batch["fine_label"].to(self.device)

            with torch.no_grad():
                emb = self.encoder(images).mean(dim=1)
                emb_list.append(emb.detach().cpu())
                labels_list.append(labels.detach().cpu())

            if batch_idx + 1 >= self.epoch_len:
                break

        emb_list = torch.cat(emb_list, dim=0)
        labels_list = torch.cat(labels_list, dim=0)
        
        probe_dataset = torch.utils.data.TensorDataset(emb_list, labels_list)
        probe_loader = torch.utils.data.DataLoader(
                probe_dataset,
                batch_size=self.lp_batch_size,
                shuffle=True,
                drop_last=False)

        for _ in tqdm(range(self.probe_epochs), desc="Linear Probe", leave=False):
            for embs, labels in probe_loader:
                embs, labels = embs.to(self.device), labels.to(self.device)
                loss = criterion(self.classifier(embs), labels)
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

        acc = (logits.argmax(1) == fine_label).float().mean().detach().cpu().item()
        return acc
