"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import torch
import json
from collections import OrderedDict
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.data_utils import prepare_sample
from PIL import Image
from utils import trim_batch
from torch.utils.data.dataloader import default_collate


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class CaptionDataset(BaseDataset, __DisplMixin):

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, model):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.model = model
        self.model.llama_tokenizer.padding_side = "right"
        self.img_ids = {}

        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = '{:0>12}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = prepare_sample(self.vis_processor(image))  # [3, 64, 64]

        caption = self.text_processor(ann["caption"])

        text = caption + self.model.end_sym
        text = self.model.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.model.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        text_input_ids = text['input_ids'].squeeze(0)  # [seq_len]
        text_attention_mask = text['attention_mask'].squeeze(0)  # [seq_len]

        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "image_id": self.img_ids[ann["image_id"]],
        }

    def collate_fn(self, batch):

        image = torch.stack([x["image"] for x in batch], dim=0)  # [b, 3, 64, 64]
        input_ids = torch.stack([x["text_input_ids"] for x in batch], dim=0)
        attention_mask = torch.stack([x["text_attention_mask"] for x in batch], dim=0)

        input_ids, attention_mask = trim_batch(input_ids, self.model.llama_tokenizer.pad_token_id, attention_mask=attention_mask)
        image_ids_list = [x["image_id"] for x in batch]

        return {
            "image": image,
            "text_input_ids": input_ids,
            "text_attention_mask": attention_mask,
            "image_id": image_ids_list,
        }


class CaptionEvalDataset(BaseDataset, __DisplMixin):

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, model):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.model = model
        self.model.llama_tokenizer.padding_side = "right"
        self.img_ids = {}

        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):
        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = '{:0>12}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")
        image = prepare_sample(self.vis_processor(image))  # [3, 64, 64]

        return {
            "image": image,
            "image_id": self.img_ids[ann["image_id"]],
        }

    def collate_fn(self, batch):

        image = torch.stack([x["image"] for x in batch], dim=0)  # [b, 3, 64, 64]
        image_ids_list = [x["image_id"] for x in batch]

        return {
            "image": image,
            "image_id": image_ids_list,
        }


class PopeEvalDataset(__DisplMixin):

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, model):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__()

        self.model = model
        self.model.llama_tokenizer.padding_side = "right"
        self.img_ids = {}

        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.annotation = []

        for ann_path in ann_paths:
            with open(ann_path, "r") as ann_fp:
                for line in ann_fp.readlines():
                    self.annotation.append(json.loads(line))

        n = 0
        for ann in self.annotation:
            img_id = ann["image"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        self._add_instance_ids()

    def __getitem__(self, index):

        ann = self.annotation[index]
        img_file = ann['image']
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")
        image = prepare_sample(self.vis_processor(image))  # [3, 64, 64]

        question = ann["text"]

        question = self.text_processor(question)
        question = self.model.llama_tokenizer(
            question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.model.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        question_input_ids = question['input_ids'].squeeze(0)  # [seq_len]

        label = ann['label']
        if label == 'no':
            label = 0
        else:
            label = 1

        return {
            "image": image,
            'question_input_ids': question_input_ids,
            'label': label,
            "image_id": self.img_ids[ann["image"]],
        }

    def collate_fn(self, batch):

        image = torch.stack([x["image"] for x in batch], dim=0)

        question_input_ids = torch.stack([x["question_input_ids"] for x in batch], dim=0)
        question_input_ids = trim_batch(
            question_input_ids,
            self.model.llama_tokenizer.pad_token_id
        )

        image_ids_list = [x["image_id"] for x in batch]
        label = [x['label'] for x in batch]

        return {
            "image": image,
            'question_input_ids': question_input_ids,
            "image_id": image_ids_list,
            'label': label
        }

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)
