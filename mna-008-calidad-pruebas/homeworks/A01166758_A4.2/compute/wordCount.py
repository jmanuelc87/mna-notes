import sys

from collections import defaultdict
from compute import Compute


def word_writer(file, words):
    """Writes the words as tsv in a file

    Args:
        file (str): the file
        words (dict): the word count dict
    """
    with open(file, "w") as f:
        for k, v in words.items():
            f.write(f"{k}\t{v}\n")
            print(f"{k}\t{v}")


def generator(file):
    """Reads the file and yields each row as a float

    Args:
        file (str): the file to read from

    Yields:
        float: the value one per each row
    """
    with open(file, "r") as f:
        for row in f:
            try:
                yield row.strip()
            except Exception as e:
                print(f"{e}")


class WordCount(Compute):

    def __init__(self, generator):
        self.generator = generator

    def calculate(self):
        words = defaultdict(int)
        for word in self.generator():
            words[word] += 1
        return words


if __name__ == "__main__":
    file = sys.argv[1]
    wc = WordCount(generator=lambda: generator(file))
    words = wc.calculate()
    word_writer("WordCountResults.txt", words)
