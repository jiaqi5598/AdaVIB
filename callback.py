import pytorch_lightning as pl
import torch
import os
import logging

from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path

from utils import count_trainable_parameters, save_json, save_outputs

logging.basicConfig(format='--- %(levelname)s: %(message)s ---',
                    level=logging.DEBUG)


def get_early_stopping_callback(metric, patience):
    return EarlyStopping(
        monitor=f"{metric}",  # val loss not training loss
        mode="min" if "loss" in metric else "max",
        patience=patience,
        verbose=True,
    )


def get_checkpoint_callback(output_dir, metric, save_top_k=1, lower_is_better=False):

    if metric == "f1":
        exp = "{f1:.4f}-{step_count}"
    elif metric == "loss":
        exp = "{loss:.4f}-{step_count}"
    else:
        raise NotImplementedError(
            f"seq2seq callbacks only support rouge2, bleu and loss, got {metric}, You can make your own by adding to this function."
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=exp,
        monitor=metric,
        mode="min" if "loss" in metric else "max",
        save_top_k=save_top_k,
    )
    return checkpoint_callback


class Seq2SeqLoggingCallback(pl.Callback):

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        save_json(pl_module.metrics, pl_module.metrics_save_path)
        save_outputs(pl_module.preds, pl_module.preds_save_path)

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module):
        save_json(pl_module.metrics, pl_module.metrics_save_path)
        save_outputs(pl_module.preds, pl_module.preds_save_path)


class LoggingCallback(pl.Callback):



    def on_batch_end(self, trainer, pl_module):
        lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
        lrs = {f"lr_group_{i}": lr for i, lr in enumerate(lr_scheduler.get_lr())}
        pl_module.logger.log_metrics(lrs)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics

        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Test results *****")
        metrics = trainer.callback_metrics

        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))