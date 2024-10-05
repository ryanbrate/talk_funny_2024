"""
loaders for chunks of text from Project Gutenberg books

See the available loader for details of the returned info format: 
    * sentences()
    * paragraphs()
"""

import pathlib
import re
import typing
from math import ceil
from multiprocessing.pool import Pool

import orjson
import pandas as pd

def as_parts_star(t: tuple) -> pd.DataFrame:
    return as_sentences(*t)


def as_parts(fp: pathlib.Path, dictionary: list[str]) -> typing.Generator:
    """Return a df::pd.Dataframe wrt., the passed fp, with cols "label", "text"
    where 'text' is a sentence part, and 'label' is paragraph_i, sentence_part_i
    """
    # build a dict of sentences, labelled as paragraph/ sentence within paragraph
    d = {"label": [], "text": []}
    for paragraph_i, paragraph in enumerate(gen_paragraphs(fp, dictionary=dictionary)):
        for sentence_part_i, sentence_part in enumerate(gen_sentences_parts(paragraph)):
            d["label"].append([paragraph_i, sentence_part_i])
            d["text"].append(sentence_part)

    return pd.DataFrame(d)


def as_sentences_star(t: tuple) -> pd.DataFrame:
    return as_sentences(*t)


def as_sentences(fp: pathlib.Path, dictionary: list[str]) -> typing.Generator:
    """Return a df::pd.Dataframe wrt., the passed fp, with cols "label", "text"
    where 'text' is a sentence, and 'label' is paragraph_i, sentence_i
    """
    # build a dict of sentences, labelled as paragraph/ sentence within paragraph
    d = {"label": [], "text": []}
    for paragraph_i, paragraph in enumerate(gen_paragraphs(fp, dictionary=dictionary)):
        for sentence_i, sentence in enumerate(gen_sentences(paragraph)):
            d["label"].append([paragraph_i, sentence_i])
            d["text"].append(sentence)

    return pd.DataFrame(d)


def as_paragraphs_star(t: tuple) -> pd.DataFrame:
    return as_paragraphs(*t)


def as_paragraphs(fp: pathlib.Path, dictionary: set[str]) -> typing.Generator:
    """Return a df::pd.Dataframe wrt., the passed fp, with cols "label", "text"
    where 'text' is a paragraph
    """
    # build a dict of paragraphs
    d = {"label": [], "text": []}

    for paragraph_i, paragraph in enumerate(gen_paragraphs(fp, dictionary=dictionary)):
        d["label"].append(paragraph_i)
        d["text"].append(paragraph)

        # return a dataframe corresponding to fp
    return pd.DataFrame(d)


def gen_sentences(paragraph: str) -> typing.Generator:
    """Return a generator of sentence strings for paragraph.

    Note: assumes paragraph free of '\n'
    """
    for sentence in re.findall(r"([^\.!?]*[\.!?])\s*", paragraph):
        if sentence:
            yield sentence


def gen_sentences_parts(paragraph: str) -> typing.Generator:
    """Return a generator of sentence part strings for paragraph.

    Note: assumes paragraph free of '\n'
    """
    for sentence_part in re.findall(r"([^\.!?,;:]*[\.!?,;:])\s*", paragraph):
        if sentence_part:
            yield sentence_part


def gen_paragraphs(fp: pathlib.Path, *, dictionary: set[str]) -> typing.Generator:
    """Return a generator of paragraph strings for book at fp.

    Note: dictionary is used to help resolve hyphenatic split words due to formatting
    Note: paragraphs assumed as separated by '\n\n'
    Note: paragraphs cleaned up, removing \n is a way sentitive to hyphens
    """

    # open the doc
    with open(fp, "r", encoding="latin-1") as f:
        doc = f.read()

    # ignore the extraneous PG text, take only the book
    match = re.search(
        r"\*\*\*\s*START OF.+?\*\*\*(.+)\*\*\*\s*END OF",
        doc,
        flags=re.DOTALL,
    )

    if match:

        # true book text
        doc = match.groups()[0]

        # split into presumed paragraphs
        paragraphs = re.split("\n\n\n*", doc)

        # remove empty paragraphs
        paragraphs = [p for p in paragraphs if len(p) != 0]

        # remove newlines (sensitively)
        pattern_split = re.compile(r"([a-zA-Z']+)-\s*\n\s*([a-zA-Z']+)")
        for i, paragraph in enumerate(paragraphs):

            # remove newlines adjacent to hyphenated words
            for x, y in set(re.findall(pattern_split, paragraph)):

                try:
                    if x + y in dictionary:
                        paragraph = re.sub(rf"{x}-\s*\n\s*{y}", f"{x}{y}", paragraph)
                    else:
                        paragraph = re.sub(rf"{x}-\s*\n\s*{y}", f"{x}-{y}", paragraph)
                except:
                    pass

            # remove other newline cases
            paragraph = re.sub(r"\s*\n\s*", r" ", paragraph)

            # strip start and end whitespace
            paragraph = paragraph.strip()

            # re-add amended to paragraph container
            paragraphs[i] = paragraph

        # yield
        for paragraph in paragraphs:
            yield paragraph

    else:

        return
        yield


def gen_dir(
    dir_path: pathlib.Path,
    *,
    pattern: re.Pattern = re.compile(".+"),
    ignore_pattern: typing.Union[re.Pattern, None] = None,
) -> typing.Generator:
    """Return a generator yielding pathlib.Path objects in a directory,
    optionally matching a pattern.

    Args:
        dir (str): directory from which to retrieve file names [default: script dir]
        pattern (re.Pattern): re.search pattern to match wanted files [default: all files]
        ignore (re.Pattern): re.search pattern to ignore wrt., previously matched files
    """

    for fp in filter(lambda fp: re.search(pattern, str(fp)), dir_path.glob("*")):

        # no ignore pattern specified
        if ignore_pattern is None:
            yield fp
        else:
            # ignore pattern specified, but not met
            if re.search(ignore_pattern, str(fp)):
                pass
            else:
                yield fp
