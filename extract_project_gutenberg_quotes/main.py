"""
    
"""
import json
import os
import pathlib
import pickle
import re
import typing
from itertools import cycle, product

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from Loaders.PG_book import gen_paragraphs, gen_sentences


def main():

    # load compiled patterns
    compiled_patterns_fp = pathlib.Path("patterns.pickle")
    with open(compiled_patterns_fp, "rb") as f:
        patterns: list[re.Pattern] = pickle.load(f)
    print("loaded compiled patterns")

    # load the dictionary for reparing paragraph split words
    with open(
        pathlib.Path("~/english.txt")
        .expanduser()
        .resolve(),
        "r",
        encoding="utf-8",
    ) as f:
        dictionary: set[str] = set([w.strip("\n") for w in f.readlines()])
    print("dictionary loaded")

    ## iterate over literature, and collect quotations
    fps = list(
        gen_dir(
            pathlib.Path("~/PATH_TO/PG_en_PS_fiction_050204").expanduser().resolve(),
            pattern=re.compile(r".+\.txt"),
        )
    )

    pd.concat(
        process_map(
            extract_star,
            zip(fps, cycle([patterns]), cycle([dictionary])),
            chunksize=100,
        )
    ).to_csv("quotes_en_PS_fiction_050224.csv")


def extract_star(t):
    return extract(*t)


def extract(fp, patterns, dictionary) -> pd.DataFrame:
    """Return a csv of extracted quotes for the file."""

    # dfs = []
    # for paragraph in tqdm(paragraphs):
    #     dfs.append(extract_paragraph(paragraph, patterns))

    ## containers storing resolved and unresolved quotes
    quotes = {
        "person": [],
        "manner": [],
        "quote": [],
    }

    for paragraph in gen_paragraphs(fp, dictionary=dictionary):

        # extract all pattern matches from paragraph
        matches = get_matches(paragraph, patterns)

        # remove clashing spans that appear later in matches var
        matches = discard(matches)

        last_person = None
        for (match_object, pattern, quote_i, manner_i, person_i) in matches:

            quote = match_object.groups()[quote_i]

            # directly attatched person
            if person_i:
                person = match_object.groups()[person_i]
                last_person = person
                manner = match_object.groups()[manner_i]
            # no directly attached person
            else:
                # assume last person is speaker
                if last_person:
                    person = "<<" + last_person + ">>"
                    manner = "NA"
                else:
                    person = "NA"
                    manner = "NA"

            # populate
            quotes["person"].append(person)
            quotes["manner"].append(manner)
            quotes["quote"].append(quote)

    df = pd.DataFrame(quotes)
    df['book'] = fp.stem

    return df


def get_matches(paragraph: str, patterns: list) -> list[tuple]:
    """Return a list of (match object, pattern, quote_i, manner_i, person_i) tuples"""

    ## capture all matches for all paragraphs, in order of patters as listed
    matches = []
    for pattern, (quote_i, manner_i, person_i) in patterns:

        # list of all match objects in paragraph
        match_objects = list(re.finditer(pattern, paragraph))

        # capture each match object into matches
        for match_object in match_objects:
            matches.append((match_object, pattern, quote_i, manner_i, person_i))

    return matches


def discard(matches: list[tuple]) -> list[tuple]:
    """Return a list of (match object, pattern, quote_i, manner_i, person_i) tuples

    i.e., we must deal with overlapping captures
    We keep:
        * all non-clashing matched text spans;
        * the earlier capture (in matches) for clashes: i.e., corresponding to higher ranked patterns;
    """

    kept = []

    # containers holding the start and end indices, wrt., the paragraph, of pattern-matched text spans
    starts = []
    ends = []

    # populate starts and ends
    for i, (match_object, pattern, quote_i, manner_i, person_i) in enumerate(matches):
        start, end = match_object.span()
        starts.append(start)
        ends.append(end)

    # discard later matches in clashes
    discarded = []
    for i, (starti, endi) in enumerate(zip(starts, ends)):

        # previously discarded instances should be ignored
        if i not in discarded:

            # consider matches[i] against later matches
            for j in range(i, len(matches)):

                if i != j:

                    startj, endj = starts[j], ends[j]

                    # print('i', matches[i][0].groups(), starti, endi)
                    # print('j', matches[j][0].groups(), startj, endj)

                    # don't clash
                    if (endi < startj) or (starti > endj):
                        pass
                        # print('no clash')
                    else:
                        discarded.append(j)
                        # print('discarded', matches[j][0].groups())

    # return non-discarded
    return [tup for i, tup in enumerate(matches) if i not in discarded]


def gen_dir(
    directory: pathlib.Path, *, pattern: re.Pattern = re.compile(".+")
) -> typing.Generator:
    """Return a generator yielding pathlib.Path objects in a directory,
    optionally matching a pattern.

    Args:
        dir (str): directory from which to retrieve file names [default: script dir]
        pattern (re.Pattern): regex pattern [default: all files]
    """

    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            yield directory / filename
        else:
            continue


if __name__ == "__main__":
    main()
