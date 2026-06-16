from src.metrics.base_metric import BaseMetric
from torch import nn
from tqdm import tqdm
import torch
import umap
import matplotlib.pyplot as plt
import numpy as np


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
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=5e-3)
        criterion = nn.CrossEntropyLoss()

        self.encoder.eval()
        
        self._is_trained = False

        emb_list = []
        labels_list = []

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(self.device)
            labels = batch["fine_label"].to(self.device)

            with torch.no_grad():
                emb = self.encoder(images)["embeddings"].mean(dim=1)
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
        self.classifier = nn.Sequential(nn.LayerNorm(self.embed_dim),
                                        nn.Linear(self.embed_dim, self.num_classes)).to(self.device)

    def __call__(self, image, fine_label, **batch):
        assert self._is_trained, "Call train_probe() before using metric"

        with torch.no_grad():
            emb = self.encoder(image)["embeddings"].mean(dim=1)
            logits = self.classifier(emb)

        acc = (logits.argmax(1) == fine_label).float().mean().detach().cpu().item()
        return acc
    
    def eval_per_class_umap(self, test_loader):
        assert self._is_trained, "Call train_probe() before using metric"
        
        all_preds = []
        all_labels = []
        all_embs = []
        
        self.encoder.eval()
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)
                labels = batch["fine_label"].to(self.device)
                
                emb = self.encoder(images)["embeddings"].mean(dim=1)
                preds = self.classifier(emb).argmax(1)
                
                all_embs.append(emb.cpu())
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        embs = torch.cat(all_embs)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        res = {}
        for class_idx in range(self.num_classes):
            mask = (all_labels == class_idx)
            if mask.sum() > 0:
                res[f'{class_idx}'] = (all_preds[mask] == all_labels[mask]).float().mean().item()

        reducer = umap.UMAP(n_components=2)
        reduced = reducer.fit_transform(embs)

        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(
            reduced[:, 0], reduced[:, 1],
            c=all_labels, cmap="tab10",
            s=2, alpha=0.5
        )
        plt.colorbar(scatter, ax=ax, ticks=range(self.num_classes))
        ax.set_title("UMAP of encoder embeddings")
        return fig, res

    def plot_per_class_bar(self, per_class):
        classes = list(per_class.keys())
        values = list(per_class.values())
        
        fig, ax = plt.subplots(figsize=(12, 4))
        bars = ax.bar(range(len(classes)), values)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=8)
        
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([c.replace("acc_class_", "") for c in classes])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Per-class accuracy")
        ax.axhline(y=sum(values)/len(values), color="red", linestyle="--", label="mean")
        ax.legend()
        
        plt.tight_layout()
        return fig