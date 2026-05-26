import torch

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    images = torch.stack([item["image"] for item in dataset_items])
    labels = torch.tensor(
        [item["fine_label"] for item in dataset_items], dtype=torch.long
    )

    return {
        "image": images,
        "fine_label": labels
    }
