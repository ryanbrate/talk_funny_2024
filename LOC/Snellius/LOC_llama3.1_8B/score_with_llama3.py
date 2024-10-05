"""
    A script to return [gpt-2 tokens, list of P(token | preceeding)]
"""
import json
import logging
import pathlib
import re
import typing
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM # LlamaForCausalLM, LlamaTokenizer


BATCH_SIZE = 8
MODEL = "/gpfs/work4/0/khse0643/meta-llama/Meta-Llama-3.1-8B"
SAVE_NAME = "chains_llama3.1_8B"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    # which device are we using?
    print(device)

    # load the model
    print("loading the model ...")
    # model = LlamaForCausalLM.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.to(device)
    # tokenizer = LlamaTokenizer.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # load the quotes
    print("loading the quotes")
    with open("../tuples_LOC/tuples_news.json", "r") as f:
        quotes = json.load(f)

    blacklist_indices = []
    # # open any blacklists present
    # blacklist_fp = pathlib.Path("../quotes_blacklist.json")
    # if blacklist_fp.exists():
    #     print("loading quotes_blacklist.json")
    #     with open(blacklist_fp, "r") as f:
    #         blacklist_indices = set(json.load(f))

    # iterator of spans (i.e., stripped of subtending quotes)
    print("getting non-blacklisted spans and stripping subtending quotation marks")
    spans = iter(
        [
            quote.strip('"“”')
            for i, (url, (quote, manner, speaker)) in enumerate(quotes)
            if i not in blacklist_indices
        ]
    )

    # harvest chains for next delta in batches
    print("working through batches")
    chains = []
    batch = get_batch(spans, BATCH_SIZE)
    while len(batch) > 0:

        # get chained probabilities for the batch, convert from tensor to float
        chains += [[x.item() for x in c] for c in get_chains(batch, tokenizer, model)]
        print(len(chains))

        # increment batch
        batch = get_batch(spans, BATCH_SIZE)

       
    # dump the last delta
    with open(SAVE_NAME + ".json", "w") as f:
        json.dump(chains, f)


def get_batch(quotes: typing.Iterator, batch_size: int) -> list[str]:

    batch = []
    try:
        for _ in range(batch_size):
            batch.append(next(quotes))
        return batch
    except StopIteration:
        return batch


def get_chains(spans: list[str], tokenizer, model):
    """
    get the chained probabilities wrt., each span, i.e.: torch.tensor([P(w1), P(w2|w1), P(w3|w1,w2), ...])
    """
    # encode the spans
    spans_ids = [tokenizer.encode(span) for span in spans]

    # add padding
    max_length = max([len(span_ids) for span_ids in spans_ids])
    pad_token_id = tokenizer.eos_token_id
    input_ids = torch.tensor(
        [
            span_ids + [pad_token_id] * (max_length - len(span_ids))
            for span_ids in spans_ids
        ]
    ).to(device)

    # create attention mask
    attention_mask = torch.tensor(
        [
            [0 if token_id == pad_token_id else 1 for token_id in span_ids]
            for span_ids in input_ids
        ]
    ).to(device)

    # get the chained proababilities
    with torch.no_grad():

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        # logits.shape = (|batch size|, |tokens|, |vocab|)

        probs = F.softmax(logits, dim=-1).cpu()  # transfer to gpu

        chains = [
            [probs[i][j][id_] for j, id_ in enumerate(span_ids)]
            for i, span_ids in enumerate(spans_ids)
        ]

    return chains


if __name__ == "__main__":
    main()
