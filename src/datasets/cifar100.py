import json
import re
from pathlib import Path

from tqdm import tqdm

from datasets import load_dataset
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class Cifar100(BaseDataset):
    def __init__(self, split, *args, **kwargs):
        self._data_dir = ROOT_PATH / "data" / "dataset_cifar100"
        self._regex = re.compile("[^a-z ]")
        self._dataset = load_dataset(
            "uoft-cs/cifar100",
            cache_dir=self._data_dir,
            split=split,
        )
        index = self._get_or_load_index(split)
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, split):
        index_path = self._data_dir / f"{split}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            img_dir = self._data_dir / "images" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            
            index = []
            for idx, entry in enumerate(tqdm(self._dataset)):
                img_path = img_dir / f"{idx}.png"
                if not img_path.exists():
                    entry["img"].save(img_path)
                
                index.append({
                    "path": str(img_path.absolute()),
                    "fine_label": entry["fine_label"],
                })
            
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        
        return index
