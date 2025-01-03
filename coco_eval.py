import sys
import os
sys.path.append(os.getcwd())

import argparse
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from pathlib import Path
import pytorch_lightning as pl
from collections import defaultdict
from typing import Dict, List
from utils import pickle_save, check_output_dir
from minigpt4.models.mini_gpt4 import MiniGPT4
from utils import flatten_list

from minigpt4.common.config import Config
from callback import Seq2SeqLoggingCallback
from pytorch_lightning.plugins import MixedPrecision
from minigpt4.datasets.builders.image_text_pair_builder import COCO2014Builder
from torch.utils.data import DataLoader
from minigpt4.conversation.conversation import StoppingCriteriaSub, StoppingCriteriaList


class MiniGPT4EvalModule(pl.LightningModule):

    max_new_tokens = 256
    max_length = 2000
    num_beams = 1
    min_length = 1

    PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
             "The assistant gives helpful, detailed, and polite answers to the user's questions. Please answer the question. " \
             "###Human: <Img><ImageHere></Img> Describe this image.###Assistant:"

    default_val_metric = "loss"

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg
        self.output_dir = cfg.run_cfg.output_dir

        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.preds_save_path = Path(self.output_dir) / "preds.txt"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"

        self.step_count = 0
        self.metrics = defaultdict(list)
        self.preds: List = []
        self.test_step_outputs = []
        self.loss_names = ["loss", "ce_loss", "kl_loss"]

        self.model = MiniGPT4.from_config(cfg.model_cfg)

        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.dataset_object = COCO2014Builder(self.model, cfg_path=cfg.cfg_path)  # cfg -> cfg_path

        self.n_obs = None

        self.num_workers = cfg.run_cfg.num_workers

        self.already_saved_batch = False

        # self.val_metric = self.default_val_metric

    def get_dataset(self):

        datasets = self.dataset_object.build_eval_datasets()

        return datasets

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:

        dataset = self.get_dataset()

        self.img_ids: Dict = dataset.img_ids

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

        return loader

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(f"eval", batch_size=self.cfg.run_cfg.batch_size_eval)

    def forward(self, **kwargs):

        return self.model(**kwargs)

    def get_stopping_criteria(self) -> StoppingCriteriaList:
        stop_words_ids = [torch.tensor([835]).to(self.device),   # self.model.llama_tokenizer.convert_ids_to_tokens([835]) -> _###
                          torch.tensor([2277, 29937]).to(self.device)]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        return stopping_criteria

    def get_context_embeds(self, image):

        image_embs, _, _, _, _, _ = self.model.encode_img(
            image,
            is_training=self.training,
        )

        prompt_segs = self.PROMPT.split('<ImageHere>')

        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]

        assert len(seg_embs) == 2

        seg_embs_before = seg_embs[0].repeat(image_embs.size(0), 1, 1)  # [b, seq_len, dim]
        seg_embs_after = seg_embs[1].repeat(image_embs.size(0), 1, 1)
        context_embs = torch.cat([seg_embs_before, image_embs, seg_embs_after], dim=1)

        # context_embs = torch.stack(context_embs, dim=0)   # [b, seq_len, emb_dim]
        current_max_len = context_embs.shape[1] + self.max_new_tokens
        if current_max_len - self.max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - self.max_length)
        context_embeds = context_embs[:, begin_idx:]

        return context_embeds

    @ staticmethod
    def post_process_output(output_text):

        new_output_text = []

        for _ou in output_text:
            _ou = _ou.replace("<unk>", "").replace("<s>", "").strip()
            new_output_text.append(_ou.split("###")[0].split('Assistant:')[-1].strip())

        return new_output_text

    def _generative_step(self, batch: dict) -> dict:

        image_ids = batch['image_id']
        img_ids = [list(self.img_ids.keys())[i] for i in image_ids]
        image = batch['image']
        context_embs = self.get_context_embeds(image)

        outputs = self.model.llama_model.generate(
            inputs_embeds=context_embs,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=self.get_stopping_criteria(),
            num_beams=self.num_beams,
            min_length=self.min_length,
            output_scores=True,
            return_dict_in_generate=True,
        )

        output_sequences = outputs.sequences
        output_text = self.model.llama_tokenizer.batch_decode(output_sequences, add_special_tokens=False)

        assert len(image_ids) == len(output_text)

        output_text = self.post_process_output(output_text)
        base_metrics = {}
        base_metrics.update(preds=output_text, image_ids=img_ids)
        self.test_step_outputs.append(base_metrics)

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def on_test_epoch_end(self):
        self.step_count += 1
        outputs: List = self.test_step_outputs

        preds: List[str] = flatten_list([x["preds"] for x in outputs])
        img_ids: List[str] = flatten_list([x['image_ids'] for x in outputs])

        for i, c in enumerate(preds):
            self.preds.append(
                {
                    "id": img_ids[i],
                    "question": "Describe this image",
                    'model': preds[i],
                }
            )



def generic_train(
    model: pl.LightningModule,
    cfg,
    logger=False,  # can pass WandbLogger() here
    # extra_callbacks=[],
    logging_callback=None,
    ckpt_path=None,
    **extra_train_kwargs
):

    # init model
    odir = Path(cfg.run_cfg.output_dir)
    odir.mkdir(exist_ok=True)
    train_params = {}

    _plugins = None
    if cfg.run_cfg.amp:
        _plugins = MixedPrecision(
            precision='16-mixed',
            device="cuda",
            scaler=torch.cuda.amp.GradScaler()
        )

    if cfg.model_cfg.gpus > 1:
        train_params["accelerator"] = "auto"  # "ddp"
        train_params["strategy"] = "ddp"

    train_params["profiler"] = None

    trainer = pl.Trainer(
        callbacks=[logging_callback],
        logger=logger,
        enable_checkpointing=True,
        plugins=_plugins,
        **train_params,
    )

    return trainer


def main(cfg):

    pl.seed_everything(cfg.run_cfg.seed)

    Path(cfg.config.run.output_dir).mkdir(exist_ok=True)
    check_output_dir(cfg, expected_items=2)
    cfg.model_cfg.gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    lightning_ckpt = cfg.model_cfg.lightning_ckpt if cfg.model_cfg.lightning_ckpt != '' else None

    EvalModule = MiniGPT4EvalModule

    if (not cfg.run_cfg.do_train) and (lightning_ckpt is not None):
        model = EvalModule.load_from_checkpoint(lightning_ckpt, cfg=cfg, strict=False)
    else:
        model: EvalModule = EvalModule(cfg)

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

    trainer.test(model=model)

    return model


def add_args():

    parser = argparse.ArgumentParser(description="COCO_Eval")
    parser.add_argument("--cfg_path", required=False, help="path to configuration file.",
                        default="minigpt4_configs/minigpt4_coco_eval.yaml")
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