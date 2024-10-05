""" Compile patterns for identifying quotes and attributed speakers.
"""

import pathlib
import pickle
import re

import pandas as pd

compiled_patterns_fp = pathlib.Path("patterns.pickle")
if compiled_patterns_fp.exists():
    print("patterns.pickle already exists ... aborting")
    pass
else:

    print("building patterns.pickle ...")

    # load english words
    english_words_fp = (
        pathlib.Path("~/Projects/English_Words/words_pos.csv").expanduser().resolve()
    )
    df = pd.read_csv(english_words_fp)
    print("\tloaded english words")

    # get determiners
    dets = df.loc[df.loc[:, "pos_tag"] == "DT", "word"].values
    print(f"\tloaded dets: {len(dets)} total")

    # get manners
    manners = df.loc[
        (df.loc[:, "pos_tag"].isin(["VBD", "VBN", "VBZ", "NNS"]))
        & (df.loc[:, "word"].str.isalpha()),
        "word",
    ].values
    print(f"\tloaded manners: {len(manners)} total")
    # e.g., 'VBD':said; 'VBN':sighed; 'VBZ':says, 'NNS': signals

    # # get nouns
    # nouns = df.loc[(df.loc[:, "pos_tag"].isin(['NNS', 'NN'])) & (df.loc[:, 'word'].str.isalpha()), "word"].values
    # print(f"\tloaded manners: {len(nouns)} total")

    # get adj
    adjs = df.loc[
        df.loc[:, "pos_tag"].isin(["NN", "JJR", "JJ", "JJS", "VBG"]), "word"
    ].values
    print(f"\tloaded adjs: {len(adjs)} total")
    # # e.g., JJR, bigger; JJS, smallest ; VBG, moaning ; NN, tall

    #
    # build smarter pattern representations via trie
    #

    def mt(words) -> dict:

        root = dict()
        for word in words:
            current_dict = root
            for letter in word:
                current_dict = current_dict.setdefault(letter, {})
        return root

    def mp(trie, match_group=True) -> re.Pattern:
        """make a pattern from a trie
        e.g., {h:{e:{n:{}, m:{}}}} is (h(?:e(?:n|m)))
        """

        # build basic pattern
        p = (
            str(trie)
            .replace(" ", "")
            .replace(":", "")
            .replace("{}", "")
            .replace(",", "|")
            .replace("{", "(?:")
            .replace("'", "")
            .replace("}", ")?")
        )

        if match_group:
            return "(" + p[3:-2] + ")"
        else:
            return "(?:" + p[3:-2] + ")"

    # manners_p = mp(mt(manners))
    # adjs_p = mp(mt(adjs), False)
    # dets_p = mp(mt(dets), False)
    dets_p = f"(?:{'|'.join(dets)})"
    # print('\tbuilt Trie pattern strings')
    # p = mp(mt(manners))
    # print(re.match(p, 'shouted'))
    # exit(1)

    patterns = [
        #
        # names
        #
        (
            re.compile(
                # rf'["“](.+?),*["”],*\s+({"|".join(manners)})\s+(?:(?:{"|".join(adjs)})\s+)?([A-Z][a-z|.]*(?:\s+[A-Z][a-z|.]*)*)'
                rf'["“](.+?),*["”],*\s+(\w+)(?:\s+[a-z]+)?\s+((?:(?:[A-Z][a-z|.]*)\s+)*(?:[A-Z][a-z]*))',
                flags=re.DOTALL,
            ),  # "blah blah said [shy] Mr.|Mrs.|Dr|Prof.|Ms. Tom [Liam Smith]"
            (0, 1, 2),  # (quote index, said index, person index)
        ),
        #
        # descriptors with a determiner, e.g., 'the engineer'
        #
        (
            re.compile(
                # rf'["“](.+?),*["”],*\s+({"|".join(manners)})\s+(?:{"|".join(dets)})\s+(?:(?:{"|".join(adjs)})\s+)?(\w+)'
                rf'["“](.+?),*["”],?\s+(\w+)\s+{dets_p}\s+(\w+(?:\s+\w+)?)',
                flags=re.DOTALL,
            ),  # "blah blah (said) the|a ([tall] nurse)"
            (0, 1, 2),
        ),
        #
        #  named -- inverted
        #
        (
            re.compile(
                # rf'([A-Z][a-z|.]*(?:\s+[A-Z][a-z|.]*)*)\s+({"|".join(manners)})\s+["“](.+?),*["”]'
                rf'((?:(?:[A-Z][a-z|.]*)\s+)*(?:[A-Z][a-z]*))\s+(\w+)\s+["“](.+?),*["”]',
                flags=re.DOTALL,
            ),  # Tom [Smith] said "blah blah"
            (2, 1, 0),
        ),
        #
        # descriptors, with a determiner, instead of name -- inverted
        #
        (
            re.compile(
                # rf'(?:{"|".join(dets)})\s+(?:(?:{"|".join(adjs)})\s+)?(\w+)\s+({"|".join(manners)})\s+["“](.+?),*["”],*'
                rf'{dets_p}\s+(\w+(?:\s+\w+)?)\s+(\w+)\s+["“](.+?),*["”],*',
                flags=re.DOTALL,
            ),  # the|a [adj] nurse said "blah blah"
            (2, 1, 0),
        ),
        #
        # not directly unattributed quotes
        #
        (
            re.compile(
                '(?:["“](.+?)["”])', flags=re.DOTALL
            ),  # text not directly associated with a speaker
            (0, None, None),
        ),
    ]

    # #  load the common adjectives
    # with open("common_adjectives.txt", "r") as f:
    #     adjs: list[str] = list(filter(lambda s: s != "", f.read().split("\n")))

    # # load the manners of saying "said"
    # with open("manners.txt", "r") as f:
    #     manners: list[str] = list(filter(lambda s: s != "", f.read().split("\n")))

    # patterns = [
    #    #
    #    # names / nicknames
    #    #
    #    (
    #        re.compile(
    #            rf'["“](.+?),*["”],*\s+({"|".join(manners)})\s+(?:{"|".join(adjs)}\s+)*((?:Mr\.\s+|Mrs\.\s+|Dr\.\s+|Prof\.\s+)*[A-Z][a-z|.]*(?:\s+[A-Z][a-z|.]*)*)'
    #        ),  # "blah blah said [Mr.|Mrs.|Dr.|Prof.] Tom [Liam Smith]"
    #        (0, 1, 2),  # (quote index, said index, person index)
    #    ),
    #    #
    #    # descriptors with a determiner, e.g., 'the engineer'
    #    #
    #    (
    #        re.compile(
    #            rf'["“](.+?),*["”],*\s+({"|".join(manners)})\s+(?:the|an|a)\s+(?:{"|".join(adjs)})*\s*(\w+)'
    #        ),  # "blah blah said the|a [adj] nurse"
    #        (0, 1, 2),
    #    ),
    #    #
    #    #  named -- inverted
    #    #
    #    (
    #        re.compile(
    #            rf'(?:{"|".join(adjs)})*\s*([A-Z][a-z]+) ({"|".join(manners)}) ["“](.+?),*["”]'
    #        ),  # Tom [Smith] said "blah blah"
    #        (2, 1, 0),
    #    ),
    #    #
    #    # descriptors, with a determiner, instead of name -- inverted
    #    #
    #    (
    #        re.compile(
    #            rf'(?:the|a|an)\s+(?:{"|".join(adjs)})*\s*(\w+)\s+({"|".join(manners)})\s+["“](.+?),*["”],*'
    #        ),  # the|a nurse said "blah blah"
    #        (2, 1, 0),
    #    ),
    #    (
    #        re.compile(
    #            '(?:["“](.+?)["”])'
    #        ),  # text not directly associated with a speaker
    #        (0, None, None),
    #    ),
    # ]
    print("\tcompiled patterns")

    #
    # testing on import
    #

    # test cases
    test_cases = [
        (0, '"blah blah" said Mr. Tom Jones', [("blah blah", "said", "Mr. Tom Jones")]),
        (
            0,
            '"blah blah" said Prof. James E. Jones',
            [("blah blah", "said", "Prof. James E. Jones")],
        ),
        (0, '"blah blah" shouted tall Tom', [("blah blah", "shouted", "Tom")]),
        (
            0,
            '"blah blah" said Tom Liam Smith',
            [("blah blah", "said", "Tom Liam Smith")],
        ),
        (1, '"blah blah" said the nurse', [("blah blah", "said", "nurse")]),
        (1, '"blah blah" said the able nurse', [("blah blah", "said", "able nurse")]),
        (
            2,
            'Tom S. Smith espoused "blah blah"',
            [("Tom S. Smith", "espoused", "blah blah")],
        ),
        (3, 'the nurse said "blah blah"', [("nurse", "said", "blah blah")]),
        (
            3,
            'the tall engineer said "blah blah"',
            [("tall engineer", "said", "blah blah")],
        ),
        (
            4,
            '"hah!", cried Tom, "what are you doing?"',
            [("hah!",), ("what are you doing?",)],
        ),
    ]

    # run test cases on import
    for i, test_case in enumerate(test_cases, start=1):

        pattern = patterns[test_case[0]][0]
        test_text = test_case[1]
        expected_matches = test_case[2]

        # list of match_objects
        matches: tuple[str] = [
            match.groups() for match in re.finditer(pattern, test_text)
        ]
        assert matches == expected_matches, f"test case {i} failed: {matches}"

    print("\trun test case assertions")

    ## save
    with open(compiled_patterns_fp, "wb") as f:
        pickle.dump(patterns, f)
    print("saved as pickle")
