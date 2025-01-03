#!/usr/bin/env python

import sys
import os
sys.path.append(os.getcwd())
import argparse
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from pathlib import Path
import pytorch_lightning as pl

from utils import check_output_dir
from transformers import (
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
)

from minigpt4.common.config import Config
from callback import LoggingCallback, Seq2SeqLoggingCallback
from pytorch_lightning.plugins import MixedPrecision
from minigpt4_module import CustomModule

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}

# torch.set_float32_matmul_precision('high')

def generic_train(
    model: pl.LightningModule,
    cfg,
    early_stopping_callback=None,
    logger=False,  # can pass WandbLogger() here
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    ckpt_path=None,
    **extra_train_kwargs
):

    odir = Path(cfg.run_cfg.output_dir)
    odir.mkdir(exist_ok=True)

    # add custom checkpoints
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=cfg.run_cfg.output_dir,
            monitor=None,
        )

    if early_stopping_callback:
        extra_callbacks.append(early_stopping_callback)

    if logging_callback is None:
        logging_callback = LoggingCallback()

    train_params = {}

    _plugins = None
    if cfg.run_cfg.amp:
        _plugins = MixedPrecision(
            precision='16-mixed',
            device="cuda",
            scaler=torch.cuda.amp.GradScaler()
        )

    if cfg.model_cfg.gpus > 1:
        train_params["accelerator"] = "auto"
        train_params["strategy"] = "ddp"

    train_params["accumulate_grad_batches"] = cfg.run_cfg.gradient_accumulation_steps
    train_params["profiler"] = None
    train_params["max_epochs"] = cfg.run_cfg.max_epoch

    trainer = pl.Trainer(
        callbacks=[logging_callback, checkpoint_callback] + extra_callbacks,
        logger=logger,
        enable_checkpointing=True,
        plugins=_plugins,
        **train_params,
    )

    if cfg.run_cfg.do_train:
        trainer.fit(model, ckpt_path=ckpt_path)

    return trainer


def main(cfg):

    pl.seed_everything(cfg.run_cfg.seed)

    Path(cfg.config.run.output_dir).mkdir(exist_ok=True)
    check_output_dir(cfg, expected_items=1)
    cfg.model_cfg.gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    lightning_ckpt = cfg.model_cfg.lightning_ckpt if cfg.model_cfg.lightning_ckpt != '' else None

    if (not cfg.run_cfg.do_train) and (lightning_ckpt is not None):
        model = CustomModule.load_from_checkpoint(lightning_ckpt, cfg=cfg, strict=False)
    else:
        model: CustomModule = CustomModule(cfg)

    if cfg.run_cfg.num_workers > 0:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)

    trainer: pl.Trainer = generic_train(
        model,
        cfg,
        logging_callback=Seq2SeqLoggingCallback(),
        logger=False,
        ckpt_path=lightning_ckpt
    )

    if cfg.run_cfg.do_train:
        return model

    trainer.test(model=model)

    return model


def add_args():

    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg_path", required=False, help="path to configuration file.",
                        default="minigpt4_configs/minigpt4_stage2_finetune.yaml")

    parser.add_argument("--overwrite_output_dir", default=True)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = add_args()
    cfg = Config(args)
    main(cfg)
