import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder, load_dataset_config, BaseProcessor
from minigpt4.datasets.datasets.laion_dataset import LaionDataset
from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset
from minigpt4.datasets.datasets.caption_datasets import PopeEvalDataset, CaptionDataset, CaptionEvalDataset


@registry.register_builder("cc_sbu")
class CCSBUBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc_sbu/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("laion")
class LaionBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("coco2014")
class COCO2014Builder(BaseDatasetBuilder):

    train_dataset_cls = CaptionDataset
    eval_dataset_cls = CaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco2014/align.yaml",
    }

    def __init__(self, model, cfg_path):
        super().__init__()
        self.model = model
        self.cfg_path = cfg_path

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building training datasets...")

        self.build_processors(self.cfg_path)

        build_info = self.config.build_info
        storage_path = build_info.storage

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        dataset_cls = self.train_dataset_cls

        datasets = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train.json')],
            vis_root=os.path.join(storage_path, 'image'),
            model=self.model
        )

        return datasets

    def build_eval_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building evaluation datasets...")

        self.build_processors(self.cfg_path)

        build_info = self.config.build_info
        storage_path = build_info.storage

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.eval_dataset_cls

        datasets = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            ann_paths=[os.path.join(storage_path, 'eval.json')],
            vis_root=os.path.join(storage_path, 'image'),
            model=self.model
        )

        return datasets


@registry.register_builder("pope_eval")
class PopeBuilder(BaseDatasetBuilder):

    dataset_cls = PopeEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/pope_eval/align.yaml",
    }

    POPE_PATH = {
        "random": "./pope_coco/random.json",
        "popular": "./pope_coco/popular.json",
        "adversarial": "./pope_coco/adversarial.json",
    }

    def __init__(self, model, cfg_path):
        super().__init__()
        self.model = model
        self.cfg_path = cfg_path

    def build_datasets(self):
        logging.info("Building datasets...")

        self.build_processors(self.cfg_path)

        build_info = self.config.build_info
        eval_storage_path = build_info.storage_eval
        eval_type_path = self.POPE_PATH[build_info.eval_type]

        dataset_cls = self.dataset_cls

        # TODO: only use train as eval
        datasets = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            ann_paths=[eval_type_path],
            vis_root=os.path.join(eval_storage_path, 'image'),
            model=self.model
        )

        return datasets

