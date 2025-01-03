import os
import torch
import pytorch_lightning as pl
import logging
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any

from minigpt4.datasets.datasets.caption_datasets import CaptionDataset
from minigpt4.models.mini_gpt4 import MiniGPT4
from minigpt4.datasets.builders.image_text_pair_builder import COCO2014Builder
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
}

class CustomModule(pl.LightningModule):

    vqa_prompt = '###Human: <Img><ImageHere></Img> '
    max_length = 2000
    num_beams = 1
    min_length = 1

    def __init__(self, cfg, **kwargs):

        super().__init__()

        self.cfg = cfg
        self.output_dir = cfg.run_cfg.output_dir
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.preds_save_path = Path(self.output_dir) / "preds.txt"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"

        self.step_count = 0
        self.metrics = defaultdict(list)
        self.preds: List = []
        self.validation_step_outputs = []
        self.loss_names = ["loss", "ce_loss", "kl_loss"]

        self.model = MiniGPT4.from_config(cfg.model_cfg)

        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.dataset_object = COCO2014Builder(self.model, cfg_path=cfg.cfg_path)

        self.n_obs = None

        self.num_workers = cfg.run_cfg.num_workers

        self.already_saved_batch = False

    def get_dataset(self, type_path) -> CaptionDataset:

        if type_path == "train":
            dataset = self.dataset_object.build_datasets()
        else:
            assert False

        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:

        dataset = self.get_dataset(type_path)

        self.img_ids: Dict = dataset.img_ids

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

        return loader

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train", batch_size=self.cfg.run_cfg.batch_size_train, shuffle=True)

    def _step(self, batch: dict) -> Tuple:  # following minigpt4 forward

        image = batch['image']
        text_input_ids = batch['text_input_ids']
        text_attention_mask = batch['text_attention_mask']

        img_embeds, atts_img, kl_loss, _, _, _ = self.model.encode_img(
            image,
            self.training,
            beta=self.cfg.run_cfg.vib_beta,
            self_adaptive=self.cfg.run_cfg.self_adaptive,
        )

        img_embeds, atts_img = self.model.prompt_wrap(img_embeds, atts_img, self.vqa_prompt)

        batch_size = text_input_ids.size(0)
        sample_size = img_embeds.size(0) // batch_size
        if sample_size != 1:
            text_input_ids = text_input_ids[None, :, :].repeat(sample_size, 1, 1)
            text_attention_mask = text_attention_mask[None, :, :].repeat(sample_size, 1, 1)
            text_input_ids = text_input_ids.reshape(-1, text_input_ids.size(-1))  # [b * sample_size, seq_len]
            text_attention_mask = text_attention_mask.reshape(-1, text_attention_mask.size(-1))

        targets = text_input_ids.masked_fill(
            text_input_ids == self.model.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1] + 1],
                       dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=text_input_ids.dtype,
                         device=text_input_ids.device) * self.model.llama_tokenizer.bos_token_id
        bos_embeds = self.model.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        text_input_embeds = self.model.llama_model.model.embed_tokens(text_input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, text_input_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, text_attention_mask], dim=1)

        outputs = self.model.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )

        loss = outputs.loss

        overall_loss = loss + kl_loss

        return (overall_loss, loss, kl_loss)

    @property
    def pad(self) -> int:
        raise NotImplementedError("pad not implemented")

    def training_step(self, batch, batch_idx) -> Dict:

        overall_loss, loss, kl_loss = self._step(batch)

        self.log('ce_loss', loss, prog_bar=True)  # log training loss
        self.log('kl_loss', kl_loss, prog_bar=True)   # beta * kl loss

        logs = {name: loss for name, loss in zip(self.loss_names, (overall_loss, loss, kl_loss))}

        return {"loss": overall_loss, "log": logs}

    def _optimizer(self, model):

        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                logging.info(f"{n} --- requires_grad: False")
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
            logging.info(f"{n} --- requires_grad: True")

        logging.info("number of trainable parameters: %d" % num_parameters)

        optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(self.cfg.run_cfg.weight_decay),
            },
            {
                "params": p_non_wd,
                "weight_decay": 0
            },
        ]

        beta2 = self.cfg.run_cfg.get("beta2", 0.999)

        optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(self.cfg.run_cfg.init_lr),
            weight_decay=float(self.cfg.run_cfg.weight_decay),
            betas=(0.9, beta2),
        )

        return optimizer


    def setup(self, stage=None):
        if stage == "fit":
            self.dataset_size = len(self.train_dataloader().dataset)
        else:
            raise NotImplementedError

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        effective_batch_size = self.cfg.run_cfg.batch_size_train * self.cfg.run_cfg.gradient_accumulation_steps * num_devices
        return self.dataset_size // effective_batch_size * self.cfg.run_cfg.max_epoch

    def steps_per_epoch(self):
        num_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        effective_batch_size = self.cfg.run_cfg.batch_size_train * self.cfg.run_cfg.gradient_accumulation_steps * num_devices
        return self.dataset_size // effective_batch_size

    def configure_optimizers(self):

        model = self.model
        self.optimizer = self._optimizer(model)
        total_steps = self.total_steps()
        warmup_fraction = float(self.cfg.run_cfg.get("warmup_fraction", 0.1))
        scheduler = get_polynomial_decay_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(warmup_fraction * total_steps),
            num_training_steps=total_steps,
            lr_end=self.cfg.run_cfg.min_lr
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [self.optimizer], [scheduler]

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:

        model_state_dict = checkpoint['state_dict']

        param_req_grad_list = [k for (k, v) in self.model.named_parameters() if v.requires_grad]

        for k in list(model_state_dict.keys()):
            if not any([k.endswith(_p) for _p in param_req_grad_list]):
                del checkpoint['state_dict'][k]




