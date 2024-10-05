import re

import orjson


def main():

    # get raw unix dict
    with open("/usr/share/dict/words", "r") as f:
        words = [line.strip("\n") for line in f]

    # include any lowercased words in their uppercase form too
    all_words = []
    for word in words:
        all_words.append(word)
        if re.match("[a-z]", word[0]):
            all_words.append(word[0].capitalize() + word[1:])

    # # save
    # with open("english.json", "wb") as f:
    #     f.write(orjson.dumps(all_words))

    # save
    with open("english.txt", "w", encoding="utf-8") as f:
        f.writelines([w + "\n" for w in all_words])


if __name__ == "__main__":
    main()
