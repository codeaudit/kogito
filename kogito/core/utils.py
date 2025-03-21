import spacy
import inflect
import itertools
import json
import pickle
from typing import Callable, Dict, Iterable, List
import uuid
import math

import numpy as np
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from torch import nn
from torch.utils.data import Sampler

from transformers import BartTokenizer

IGNORE_WORDS = set(["personx", "persony", "personz", "_", "'", "-"])
ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def vp_present_participle(phrase):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(phrase)
    inflection_engine = inflect.engine()
    return " ".join(
        [
            inflection_engine.present_participle(token.text)
            if token.pos_ == "VERB" and token.tag_ != "VGG"
            else token.text
            for token in doc
        ]
    )


def posessive(word):
    inflection_engine = inflect.engine()
    if inflection_engine.singular_noun(word) is False:
        return "have"
    else:
        return "has"


def article(word):
    return "an" if word[0] in ["a", "e", "i", "o", "u"] else "a"


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start if start != -1 else None


def encode_line(
    tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"
):
    extra_kw = (
        {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
    )
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        **extra_kw,
    )


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def calculate_bleu_score(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {"bleu": corpus_bleu(output_lns, [refs_lns], **kwargs).score}


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size):
        self.data, self.bs = data, batch_size

    def key(self, i):
        return self.data[i]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        idxs = np.random.permutation(len(self.data))
        sz = self.bs * 50
        ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate(
            [sorted(s, key=self.key, reverse=True) for s in ck_idx]
        )
        sz = self.bs
        ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax(
            [self.key(ck[0]) for ck in ck_idx]
        )  # find the chunk with the largest key,
        ck_idx[0], ck_idx[max_ck] = (
            ck_idx[max_ck],
            ck_idx[0],
        )  # then make sure it goes first.
        if len(ck_idx) < 3:
            return iter(np.concatenate(ck_idx))

        sort_idx = np.concatenate(
            [np.concatenate(np.random.permutation(ck_idx[1:-1])), ck_idx[-1]]
        )
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def save_json(content, path):
    with open(path, "w") as f:
        json.dump(content, f, indent=4)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def calculate_rouge(
    output_lns: List[str], reference_lns: List[str], use_stemmer=True
) -> Dict:
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: v.mid.fmeasure for k, v in result.items()}


def freeze_params(model: nn.Module):
    for par in model.parameters():
        par.requires_grad = False


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(
        model_grads
    ), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_uuid(length=8):
    u = str(uuid.uuid4())
    if length is not None:
        u = u[:length]
    return u


def truncate_sequences_dual(sequences, max_length):
    words_to_cut = sum(list(map(len, sequences))) - max_length
    if words_to_cut <= 0:
        return sequences

    words_to_cut_before = math.ceil(words_to_cut / 2.0)
    words_to_cut_after = words_to_cut // 2

    while words_to_cut_before > len(sequences[0]):
        words_to_cut_before -= len(sequences[0])
        sequences = sequences[1:]
    sequences[0] = sequences[0][words_to_cut_before:]

    while words_to_cut_after > len(sequences[-1]):
        words_to_cut_after -= len(sequences[-1])
        sequences = sequences[:-1]
    last = len(sequences[-1]) - words_to_cut_after
    sequences[-1] = sequences[-1][:last]

    return sequences


def pad_ids(arrays, padding, max_length=-1):
    if max_length < 0:
        max_length = max(list(map(len, arrays)))

    arrays = [array + [padding] * (max_length - len(array)) for array in arrays]

    return arrays


def text_to_list(text):
    return [t.strip().strip("'") for t in text.strip("]").strip("[").split(",")]
