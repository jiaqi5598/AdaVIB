import logging
import random
import torch
import torch.nn as nn
import math

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer



@registry.register_model("mini_gpt4")
class MiniGPT4(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        mm_projector_type=None
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        # self.ln_vision -> LayerNorm
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        print('Loading Q-Former Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:

            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                # device_map="auto"
                device_map={'': device_8bit}
            )
            self.llama_model.bfloat16()  # TODO: it solves inf logits issue

        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        # emb2mu
        self.mm_projector_type = mm_projector_type

        if mm_projector_type == "vib":

            self.llama_proj = nn.Linear(
                self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
            )  # don't modify this, because it should be loaded from pre-trained checkpoints.

            self.llama_proj_std = nn.Linear(
                self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
            )

            self.mu_p = nn.Parameter(torch.randn(self.llama_model.config.hidden_size))
            self.std_p = nn.Parameter(torch.randn(self.llama_model.config.hidden_size))

        elif mm_projector_type == "linear":

            self.llama_proj = nn.Linear(
                self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
            )

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def universal_sentence_embedding(self, sentences, mask, sqrt=True):
        '''
        :param sentences: [batch_size, seq_len, hidden_size]
        :param mask: [batch_size, seq_len]
        :param sqrt:
        :return: [batch_size, hidden_size]
        '''
        # need to mask out the padded chars
        sentence_sums = torch.bmm(
            sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)
        ).squeeze(-1)
        divisor = (mask.sum(dim=1).view(-1, 1).float())
        if sqrt:
            divisor = divisor.sqrt()
        sentence_sums /= divisor
        return sentence_sums

    def _emb_entropy(self, inputs_llama, llama_emb_weights):
        batch_size = inputs_llama.size(0)
        prompt_len = inputs_llama.size(1)
        with torch.no_grad():
            _mask = torch.ones([batch_size, prompt_len]).to(inputs_llama.device)
            _pooling_states = self.universal_sentence_embedding(inputs_llama, mask=_mask)
            _prob = torch.matmul(_pooling_states, llama_emb_weights.transpose(0, 1)).softmax(-1)
            _entropy = - (_prob * (_prob + 1e-8).log()).sum(-1)
        return _entropy, _prob

    def estimate(self, emb, emb2mu, emb2std):
        """Estimates mu and std from the given input embeddings."""
        mean = emb2mu(emb)
        std = torch.nn.functional.softplus(emb2std(emb))
        return mean, std

    def kl_div(self, mu_q, std_q, mu_p, std_p):
        k = mu_q.size(-1)
        mu_diff = mu_p - mu_q
        mu_diff_sq = torch.mul(mu_diff, mu_diff)
        logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=-1)
        logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=-1)
        fs = torch.sum(torch.div(std_q ** 2, std_p ** 2), dim=-1) + torch.sum(torch.div(mu_diff_sq, std_p ** 2), dim=-1)
        kl_divergence = (fs - k + logdet_std_p - logdet_std_q) * 0.5
        return kl_divergence

    def reparameterize(self, mu, std, sample_size):
        batch_size = mu.size(0)
        z = torch.randn(sample_size, batch_size, mu.size(1), mu.size(2)).to(mu.device)
        return mu + std * z

    def get_logits(self, z):
        return z.reshape(-1, z.size(-2), z.size(-1))

    def vib_layer(self, query_output_state, is_training, ib_sample_size):
        batch_size = query_output_state.size(0)
        prompt_len = query_output_state.size(1)
        mu, std = self.estimate(query_output_state, self.llama_proj, self.llama_proj_std)

        _mask = torch.ones([batch_size, prompt_len], requires_grad=False).to(mu.device)
        mu_pooling = self.universal_sentence_embedding(mu, _mask)
        std_pooling = self.universal_sentence_embedding(std, _mask)

        mu_p = self.mu_p.view(1, -1).expand(batch_size, -1)
        std_p = torch.nn.functional.softplus(self.std_p.view(1, -1).expand(batch_size, -1))
        kl_loss = self.kl_div(mu_pooling, std_pooling, mu_p, std_p)

        if is_training:
            z = self.reparameterize(mu, std, sample_size=ib_sample_size)
            sampled_logits = self.get_logits(z)
            logits = sampled_logits
        else:
            logits = mu

        return logits, kl_loss

    def _alpha_fn(self, _entropy, v_size):
        return - (_entropy / math.log(v_size)).log()

    def encode_img(self, image, is_training, beta=1., self_adaptive=False, ib_sample_size=3):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        _alpha = None
        if self.mm_projector_type == "vib":

            inputs_llama, kl_loss = self.vib_layer(query_output.last_hidden_state, is_training, ib_sample_size)
            llama_emb_weight = self.llama_model.model.embed_tokens.weight
            _entropy, _prob = self._emb_entropy(inputs_llama, llama_emb_weight)

            if self_adaptive and is_training:
                _alpha = self._alpha_fn(_entropy, v_size=llama_emb_weight.size(0))
                sample_size = _alpha.size(0) // kl_loss.size(0)
                _alpha = _alpha.reshape(sample_size, -1).mean(0)
                kl_loss = beta * (kl_loss * _alpha).mean()
            else:
                kl_loss = beta * kl_loss.mean()

        elif self.mm_projector_type == "linear":

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            llama_emb_weight = self.llama_model.model.embed_tokens.weight
            _entropy, _prob = self._emb_entropy(inputs_llama, llama_emb_weight)
            kl_loss = torch.tensor(0., requires_grad=False)

        else:

            assert False

        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)

        return inputs_llama, atts_llama, kl_loss, _alpha, _entropy, _prob

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model")
        q_former_model = cfg.get("q_former_model")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")
        mm_projector_type = cfg.get("mm_projector_type")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            mm_projector_type=mm_projector_type
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
