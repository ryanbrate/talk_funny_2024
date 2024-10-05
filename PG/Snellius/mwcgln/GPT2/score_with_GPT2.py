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
from transformers import GPT2LMHeadModel, GPT2Tokenizer

BATCH_SIZE = 8
MODEL = "gpt2-xl"
SAVE_NAME = "chains_1.6B"
SAVE_AT = 4000 * BATCH_SIZE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    assert SAVE_AT % BATCH_SIZE == 0

    # which device are we using?
    print(device)

    # load the model
    print("loading the model ...")
    model = GPT2LMHeadModel.from_pretrained(MODEL)
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL)

    # open the harvested quotations ...
    print("loading the quotations ...")
    with open("../tuples_PG/quotes_5Jul.json", "r") as f:
        quotes = json.load(f)

    # open any blacklists present
    blacklist_fp = pathlib.Path("../tuples_PG/quotes_blacklist.json")
    if blacklist_fp.exists():
        print("loading quotes_blacklist.json")
        with open(blacklist_fp, "r") as f:
            blacklist_indices = set(json.load(f))

    # What's our last saved quote index that we have output for?
    save_fps = list(pathlib.Path(".").glob(f"{SAVE_NAME}*json"))
    if len(save_fps) > 0:
        i = sorted([int(re.search(r'_(\d+)\.json', str(fp)).groups()[0]) for fp in save_fps])[-1]
    else:
        i = 0


    # iterator of spans (i.e., stripped of subtending quotes)
    print("getting non-blacklisted spans and stripping subtending quotation marks")
    speakers_of_interest = ['man', 'woman', 'child', 'gentleman', 'lady', 'negro', 'Negro']
    spans = iter(
        [
            quote.strip('"“”')
            for i, (id_, p, quote, manner, speaker) in enumerate(quotes)
            if i not in blacklist_indices and speaker in speakers_of_interest
        ]
    )

    # pop spans already included in deltas
    print("popping seen spans")
    spans = islice(spans, i * SAVE_AT, None)

    # harvest chains for next delta in batches
    print("working through batches")
    chains = []
    batch = get_batch(spans, BATCH_SIZE)
    print("i", i)
    while len(batch) > 0:

        # get chained probabilities for the batch, convert from tensor to float
        chains += [[x.item() for x in c] for c in get_chains(batch, tokenizer, model)]
        print(len(chains))

        # increment batch
        batch = get_batch(spans, BATCH_SIZE)

        # progress
        if len(chains) == SAVE_AT:

            # dump the delta
            i += 1
            with open(SAVE_NAME + f"_{i}.json", "w") as f:
                json.dump(chains, f)

            # reset for next delta
            chains = []

    # dump the last delta
    i += 1
    with open(SAVE_NAME + f"_{i}.json", "w") as f:
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
