import itertools
import json
import re
import pickle
import os
import numpy as np
import torch
import torch.nn.functional as F
import math

from typing import Callable, Dict, List, Iterable
# from rouge_score import rouge_scorer, scoring
# from sacrebleu import corpus_bleu
from torch import nn
from torch.nn.modules.loss import _Loss

try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False


try:
    import nltk
    NLTK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):

    NLTK_AVAILABLE = False

def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def kofn(kofn: str):
    """
    ''     -> 0, 1
    '1of2' -> 0, 2
    '2of2' -> 1, 2
    :param kofn:
    :return:
    """
    if not kofn:
        return 0, 1
    k, n = [int(i) for i in kofn.lower().split('of')]
    assert 1 <= k <= n
    return k-1, n


def universal_sentence_embedding(sentences, mask, sqrt=True):
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


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""

    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)    # [max_seq_len]

    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def shift_tokens_left(input_ids: torch.Tensor, pad_token_id: int, decoder_end_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, :-1] = input_ids[:, 1:].clone()
    shifted_input_ids[:, -1] = decoder_end_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def sequence_loss(logits, targets, xent_fn=None, mask=None):  # masked_src_len=None
    """ functional interface of SequenceLoss"""

    assert logits.size()[:-1] == targets.size()

    target = targets.masked_select(mask)
    logit = logits.masked_select(
        mask.unsqueeze(2).expand_as(logits)
    ).contiguous().view(-1, logits.size(-1))

    if xent_fn:
        loss = xent_fn(logit, target)
    else:
        loss = F.cross_entropy(logit, target)
    assert (not math.isnan(loss.mean().item())
            and not math.isinf(loss.mean().item()))

    return loss

def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def save_txt(content: List, path: str):
    with open(path, "w", encoding='utf-8') as f:
        for _s in content:
            f.write(_s + "\n")

def save_outputs(content: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for _s in content:
            f.write(str(_s) + "\n")


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]

def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    re.sub("<n>", "", x)  # remove pegasus newline char
    assert NLTK_AVAILABLE, "nltk must be installed to separate newlines between sentences. (pip install nltk)"
    return "\n".join(nltk.sent_tokenize(x))


def check_output_dir(cfg, expected_items=0):
    """
    Checks whether to bail out if output_dir already exists and has more than expected_items in it

    `args`: needs to have the following attributes of `args`:
      - output_dir
      - do_train
      - overwrite_output_dir

    `expected_items`: normally 0 (default) - i.e. empty dir, but in some cases a few files are expected (e.g. recovery from OOM)
    """
    if (
        os.path.exists(cfg.config.run.output_dir)
        and len(os.listdir(cfg.config.run.output_dir)) > expected_items
        and cfg.do_train
        and not cfg.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({cfg.config.run.output_dir}) already exists and "
            f"has {len(os.listdir(cfg.config.run.output_dir))} items in it (expected {expected_items} items). "
            "Use --overwrite_output_dir to overcome."
        )

_NON_WORD = re.compile(r'\W+')


def _extractive_normalize(text: str):
    return _NON_WORD.sub(' ', text).lower().strip()
