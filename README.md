# AdaVIB

This repo contains the code for the paper: **Mitigating Hallucinations in Large Vision-Language Models by Adaptively Constraining Information Flow**. AAAI 2025

## Requirements

This code is built upon [Pytorch-Lightning](https://lightning.ai) framework, tested on Python 3.9, Pytorch 2.3.0 and transformers 4.28.0.

```
pip install -r requirements.txt
```

## Resources

#### 1. Datasets

The [training and evaluation datasets](https://drive.google.com/drive/folders/1DRJhPKGJ8vou1wQPAVjw48cWrDyPkIXS?usp=sharing) are from [ LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) and [POPE](https://github.com/RUCAIBox/POPE). The grounded images of these datasets can be downloaded from [coco2014](https://cocodataset.org/#home). Downloading the train and val images, and put them under the corresponding paths. 

The train images should be put under the path you specified in 

```
minigpt4/configs/datasets/coco2014/align.yaml
```

The val images should be put under the path you specified in 

```
minigpt4/configs/datasets/pope_eval/align.yaml
```

Note that the train and evaluation data sampled from LLaVa-Instruct-150K should be put under the same path as the train images.

#### 2. Models

[**MiniGPT-4**](https://github.com/Vision-CAIR/MiniGPT-4): Download the [Vicuna-7B](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main), [Q-Former](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth) and the [VIT](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth), then putting them under the path you specified in 

```
minigpt4/configs/models/minigpt4.yaml
```

Specifying the path to the corresponding keys:

```
llama_model: <Path to vicuna-7b>
q_former_model: <Path to blip2_pretrained_flant5xxl.pth>
vit_model: <Path to eva_vit_g.pth>
```

Additionally, you need to download the pre-trained vision-language projector of MiniGPT-4 from [here](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing), and put it under the path you specified  in the following files:

```
minigpt4_configs/minigpt4_stage2_finetune.yaml
minigpt4_configs/minigpt4_pope_eval.yaml
minigpt4_configs/minigpt4_coco_eval.yaml
```

The path should be delivered to the ``ckpt`` in ``*.yaml``. For example:

```
ckpt: <Path to the prerained_minigpt4_7b.pth>
```

## Training

Following the default settings in ``minigpt4_configs/minigpt4_stage2_finetune.yaml``, specify ``output_dir`` in this file and run the following to train the model:

```
export CUDA_VISIBLE_DEVICES=0
python finetune.py --cfg_path minigpt4_configs/minigpt4_stage2_finetune.yaml
```

*This code supports multi-GPU training. Simply specify the GPUs' id you used.*

## Inference

Once the training is finished, you will get a ``.ckpt`` file under the ``output_dir``,  a checkpoint of the vision-language projector you trained before. Specify the path to this checkpoint under the following file:

```
minigpt4_configs/minigtp4_pope_eval.yaml
minigpt4_configs/minigtp4_coco_eval.yaml
```

The path of the checkpoint file should be delivered to the ``lightning_ckpt`` in ``minigpt4_configs/*_eval.yaml``, for example:

```
lightning_ckpt: <Path to the trained_projector.ckpt>
```

Run the following command to conduct model inference on the POPE and the COCO, respectively:

```
python pope_eval.py --cfg_path minigpt4_configs/minigpt4_pope_eval.yaml
python coco_eval.py --cfg_path minigpt4_configs/minigpt4_coco_eval.yaml
```

## Acknowledgement

Many thanks to the following projects, this project is partially built upon them.

[MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4); [LURE](https://github.com/YiyangZhou/LURE); [LLaVa](https://github.com/haotian-liu/LLaVA);  [POPE](https://github.com/RUCAIBox/POPE); [CHAIR](https://github.com/LisaAnne/Hallucination); 

## Citation
If you find our work helpful, please use the following citations.
```bibtext
@article{bai2025mitigating,
  title={Mitigating Hallucinations in Large Vision-Language Models by Adaptively Constraining Information Flow},
  author={Bai, Jiaqi and Guo, Hongcheng and Peng, Zhongyuan and Yang, Jian and Li, Zhoujun and Li, Mohan and Tian, Zhihong},
  journal={arXiv preprint arXiv:2502.20750},
  year={2025}
}
```
