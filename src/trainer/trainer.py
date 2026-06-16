from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker, epoch: int):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.

        P.S. БАТЧ НА ВХОДЕ:
        {image: tensor [B, C, N, N],
         picture_path: [B],
         fine_label: [B],
         'masks_enc': [1, B] кол-во масок контекста (по умолчанию 1),
         'masks_pred': [num_areas, B] кол-во прямоугольников (1 если выбран random MaskCollator)
         }
        """

        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            self.global_train_step += 1
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.model.update_target_encoder(self.global_train_step, self.total_steps)

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            if met.name == "LinearProbe":
                if epoch % self.n_epoch_probe == 0:
                    metrics.update(met.name, met(**batch))

                    test_loader = self.evaluation_dataloaders.get("test")

                    if test_loader != None:
                        fig, per_class = met.evaluate_per_class_umap(test_loader)
                        self.writer.add_scalars({
                            f"per_class/acc_class_{i}": v 
                            for i, v in per_class.items()
                        })

                        self.writer.add_image("umap", fig)
            else:
                metrics.update(met.name, met(**batch))

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # self.global_step += 1 uncomment only if delete same row in process batch method
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
