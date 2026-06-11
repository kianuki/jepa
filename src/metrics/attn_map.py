import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


class AttentionVisualizer:
    """
    Epoch-level visual logger for I-JEPA encoder PCA maps and predictor attention.

    This class is intentionally not a BaseMetric: it logs images and does not
    produce scalar values for MetricTracker aggregation.
    """

    def __init__(
        self,
        num_images=5,
        seed=42,
        split="test",
        log_every_n_epochs=1,
        enabled=True,
        encoder_layers=None,
        predictor_layers=None,
        alpha=0.55,
        image_name_prefix="attn",
    ):
        self.num_images = num_images
        self.seed = seed
        self.split = split
        self.log_every_n_epochs = log_every_n_epochs
        self.enabled = enabled
        self.encoder_layers = encoder_layers
        self.predictor_layers = predictor_layers
        self.alpha = alpha
        self.image_name_prefix = image_name_prefix

        self._fixed_batch = None
        self._image_paths = None
        self._indices = None

    def should_log(self, epoch, split):
        return (
            self.enabled
            and split == self.split
            and epoch % self.log_every_n_epochs == 0
        )

    def setup_fixed_batch(self, dataset, collate_fn=None, device=None):
        """
        Select fixed dataset items and build fixed masks once.

        Args:
            dataset: dataset with __getitem__. Preferably returns picture_path.
            collate_fn: dataloader/mask collator used to build masks_enc/masks_pred.
            device: optional device to move tensor fields to immediately.
        """
        if len(dataset) < self.num_images:
            raise ValueError(
                f"Cannot sample {self.num_images} images from dataset of size {len(dataset)}"
            )

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        self._indices = torch.randperm(len(dataset), generator=generator)[
            : self.num_images
        ].tolist()

        items = [dataset[idx] for idx in self._indices]
        self._image_paths = [self._resolve_image_path(dataset, item, idx) for item, idx in zip(items, self._indices)]

        if collate_fn is None:
            batch = torch.utils.data.default_collate(items)
        else:
            batch = collate_fn(items)

        self._fixed_batch = self._move_tensor_fields(batch, device)
        return self

    def setup_from_dataloader(self, dataloader, device=None):
        """
        Convenience wrapper around setup_fixed_batch using a dataloader object.
        """
        return self.setup_fixed_batch(
            dataset=dataloader.dataset,
            collate_fn=getattr(dataloader, "collate_fn", None),
            device=device,
        )

    @torch.no_grad()
    def log(self, model, writer, device=None):
        """
        Build and log visualizations using writer.add_image.

        The writer step/mode should be set by the trainer before this call.
        """
        if writer is None:
            return
        if self._fixed_batch is None:
            raise RuntimeError("Call setup_fixed_batch() before log().")

        was_training = model.training
        model.eval()

        batch = self._move_tensor_fields(self._fixed_batch, device)
        images = batch["image"]
        masks_enc = batch["masks_enc"]
        masks_pred = batch["masks_pred"]

        encoder_grid = self._build_encoder_grid(model, images)
        writer.add_image(f"{self.image_name_prefix}/encoder_pca", encoder_grid)

        predictor_grid = self._build_predictor_grid(model, images, masks_enc, masks_pred)
        writer.add_image(f"{self.image_name_prefix}/predictor_attention", predictor_grid)

        if was_training:
            model.train()

    def _build_encoder_grid(self, model, images):
        encoder_out = model.context_encoder(
            images,
            masks=None,
            save_intermediate_embs=True,
        )
        intermediate_embs = encoder_out["intermediate_embs"]
        layer_ids = self._select_layers(intermediate_embs, self.encoder_layers)

        rows = []
        original_images = self._load_original_images()
        for image_idx, original in enumerate(original_images):
            row = [original]
            for layer_id in layer_ids:
                emb = intermediate_embs[layer_id][image_idx]
                pca_image = self._embedding_pca_image(emb, original.size)
                row.append(pca_image)
            rows.append(row)

        headers = ["image"] + [f"enc L{layer_id}" for layer_id in layer_ids]
        return self._make_grid(rows, headers=headers)

    def _build_predictor_grid(self, model, images, masks_enc, masks_pred):
        context_embeddings = model.context_encoder(images, masks=masks_enc)["embeddings"]
        predictor_out = model.predictor(
            context_embeddings,
            masks_x=masks_enc,
            masks=masks_pred,
            return_attn=True,
        )
        attentions = predictor_out.get("attentions", [])

        layer_ids = self._select_sequence_indices(attentions, self.predictor_layers)
        masks_enc_list = self._as_mask_list(masks_enc)
        masks_pred_list = self._as_mask_list(masks_pred)
        num_pred_masks = len(masks_pred_list)
        num_context_tokens = masks_enc_list[0].shape[1]
        num_patches = self._num_patches(model)
        grid_size = int(round(num_patches**0.5))

        rows = []
        original_images = self._load_original_images()
        for image_idx, original in enumerate(original_images):
            row = [original]
            for layer_id in layer_ids:
                heatmap, target_mask_up = self._predictor_attention_map(
                    attentions[layer_id],
                    image_idx=image_idx,
                    num_pred_masks=num_pred_masks,
                    num_context_tokens=num_context_tokens,
                    context_mask=masks_enc_list[0][image_idx],
                    num_patches=num_patches,
                    grid_size=grid_size,
                    image_size=original.size,
                )
                row.append(self._overlay_heatmap(original, heatmap, target_mask_up))
            rows.append(row)

        headers = ["image"] + [f"pred L{layer_id}" for layer_id in layer_ids]
        return self._make_grid(rows, headers=headers)

    def _predictor_attention_map(
        self,
        attn,
        image_idx,
        num_pred_masks,
        num_context_tokens,
        context_mask,
        num_patches,
        grid_size,
        image_size,
    ):
        # attn shape: [B * num_pred_masks, heads, N_context + N_target, N_context + N_target]
        batch_size = context_mask.shape[0]
        per_mask_maps = []
        for pred_idx in range(num_pred_masks):
            batch_idx = pred_idx * batch_size + image_idx
            attn_to_context = attn[batch_idx, :, num_context_tokens:, :num_context_tokens]
            values = attn_to_context.mean(dim=(0, 1))
            full_values = torch.zeros(num_patches, device=attn.device, dtype=values.dtype)
            full_values[context_mask.to(attn.device)] = values
            per_mask_maps.append(full_values)

        patch_values = torch.stack(per_mask_maps, dim=0).mean(dim=0)
        patch_values = self._normalize_tensor(patch_values)

        target_mask = torch.ones(num_patches, device=context_mask.device, dtype=torch.float32)
        target_mask[context_mask.to(torch.long)] = 0.0
        
        t_mask_2d = target_mask.reshape(1, 1, grid_size, grid_size)
        target_mask_up = F.interpolate(
                t_mask_2d,
                size=(image_size[1], image_size[0]),
                mode="nearest").squeeze().detach().cpu().numpy()

        heatmap = patch_values.reshape(1, 1, grid_size, grid_size)
        heatmap = F.interpolate(
            heatmap,
            size=(image_size[1], image_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        return heatmap.squeeze().detach().cpu().numpy(), target_mask_up

    def _embedding_pca_image(self, emb, image_size):
        num_tokens = emb.shape[0]
        grid_size = int(round(num_tokens**0.5))
        if grid_size * grid_size != num_tokens:
            raise ValueError(f"Cannot reshape {num_tokens} tokens into a square grid.")

        x = emb.detach().float().cpu()
        x = x - x.mean(dim=0, keepdim=True)
        _, _, v = torch.pca_lowrank(x, q=3, center=False)
        pca = x @ v[:, :3]
        pca = self._normalize_tensor(pca)
        pca = pca.reshape(grid_size, grid_size, 3).numpy()

        image = Image.fromarray((pca * 255).astype(np.uint8), mode="RGB")
        return image.resize(image_size, Image.Resampling.BICUBIC)

    def _load_original_images(self):
        images = []
        for path in self._image_paths:
            image = Image.open(path).convert("RGB")
            images.append(image)
        return images

    def _overlay_heatmap(self, image, heatmap, target_mask_up=None):
        base = image.convert("RGB")
        heatmap_rgb = self._heatmap_to_rgb(heatmap)
        heatmap_image = Image.fromarray(heatmap_rgb, mode="RGB").resize(
            base.size,
            Image.Resampling.BILINEAR,
        )

        blended = Image.blend(base, heatmap_image, self.alpha)

        if target_mask_up is not None:
            blended_np = np.array(blended)
            blended_np[target_mask_up > 0.5] = [0, 0, 0]
            return Image.fromarray(blended_np)
        
        return blended

    def _heatmap_to_rgb(self, heatmap):
        heatmap = np.asarray(heatmap, dtype=np.float32)
        heatmap = heatmap - heatmap.min()
        denom = heatmap.max()
        if denom > 0:
            heatmap = heatmap / denom

        red = np.clip(2.0 * heatmap, 0.0, 1.0)
        green = np.clip(2.0 * (1.0 - np.abs(heatmap - 0.5)), 0.0, 1.0)
        blue = np.clip(2.0 * (1.0 - heatmap), 0.0, 1.0)
        rgb = np.stack([red, green, blue], axis=-1)
        return (rgb * 255).astype(np.uint8)

    def _make_grid(self, rows, headers=None, pad=4, header_height=18):
        cell_w = max(image.size[0] for row in rows for image in row)
        cell_h = max(image.size[1] for row in rows for image in row)
        n_rows = len(rows)
        n_cols = max(len(row) for row in rows)
        top = header_height if headers is not None else 0

        canvas = Image.new(
            "RGB",
            (n_cols * cell_w + (n_cols + 1) * pad, top + n_rows * cell_h + (n_rows + 1) * pad),
            "white",
        )
        draw = ImageDraw.Draw(canvas)

        if headers is not None:
            for col, header in enumerate(headers):
                x = pad + col * (cell_w + pad)
                draw.text((x, 2), header, fill="black")

        for row_idx, row in enumerate(rows):
            for col_idx, image in enumerate(row):
                x = pad + col_idx * (cell_w + pad)
                y = top + pad + row_idx * (cell_h + pad)
                canvas.paste(image.resize((cell_w, cell_h)), (x, y))

        return canvas

    def _select_layers(self, layer_dict, requested_layers):
        if requested_layers is None:
            keys = sorted(layer_dict.keys())
            if len(keys) <= 4:
                return keys
            return [keys[0], keys[len(keys) // 3], keys[(2 * len(keys)) // 3], keys[-1]]
        return [layer for layer in requested_layers if layer in layer_dict]

    def _select_sequence_indices(self, values, requested_indices):
        if requested_indices is None:
            if len(values) <= 4:
                return list(range(len(values)))
            return [0, len(values) // 3, (2 * len(values)) // 3, len(values) - 1]
        return [idx for idx in requested_indices if 0 <= idx < len(values)]

    def _as_mask_list(self, masks):
        if isinstance(masks, list):
            return masks
        return [m for m in masks]

    def _num_patches(self, model):
        if hasattr(model.context_encoder.patch_embed, "num_patches"):
            return model.context_encoder.patch_embed.num_patches
        return model.predictor.predictor_pos_embed.shape[1]

    def _resolve_image_path(self, dataset, item, idx):
        if "picture_path" in item:
            return item["picture_path"]
        if hasattr(dataset, "_index") and "path" in dataset._index[idx]:
            return dataset._index[idx]["path"]
        raise KeyError("Dataset item must contain picture_path or dataset._index[idx]['path'].")

    def _move_tensor_fields(self, batch, device):
        if device is None:
            return batch

        moved = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    def _normalize_tensor(self, x):
        x = x - x.amin(dim=0, keepdim=True)
        denom = x.amax(dim=0, keepdim=True).clamp_min(1e-8)
        return x / denom
