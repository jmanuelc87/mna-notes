import sys

from compute import Compute
from dataclasses import dataclass


def number_writer(file, numCalc):
    """writes the numbers to the file and console

    Args:
        file (str): the file to save the result
        numCalc (NumberConverter): the class that converts the numbers
    """
    with open(file, "w") as f:
        for number in numCalc.calculate():
            f.write(f"{number.dec},{number.hex},{number.bin}\n")
            print(f"{number.dec},{number.hex},{number.bin}")


def generator(file):
    """Reads the file and yields each row as an int

    Args:
        file (str): the file to read from

    Yields:
        int: the value one per each row
    """
    with open(file, "r") as f:
        for row in f:
            try:
                yield int(row)
            except Exception as e:
                print(f"{e}")


@dataclass
class Number:
    dec: int
    hex: str
    bin: str


class NumberConverter(Compute):

    def __init__(self, generator):
        self.generator = generator

    def calculate(self):
        for dec, hex, bin in zip(self.generator(), self.toHex(), self.toBin()):
            yield Number(dec, hex, bin)

    def toHex(self):
        """converts to hexadecimal representation

        Yields:
            str: hexadecimal value
        """
        hexString = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                     "A", "B", "C", "D", "E", "F"]
        for n in self.generator():
            number = n
            counter = 0
            remainder = []
            result = ""
            if number > 0:
                while number != 0:
                    remainder.append(number % 16)
                    number = number // 16
                    counter += 1
                for reverse in remainder[::-1]:
                    result = result + hexString[reverse]
            else:
                result = "0"
            yield result

    def toBin(self):
        """Converts to binary representation

        Yields:
            str: binary value
        """
        for n in self.generator():
            number = n
            result = ""
            while number != 0:
                remainder = number % 2
                number = number // 2
                result = str(remainder) + result
            yield result


if __name__ == "__main__":
    file = sys.argv[1]
    ncalc = NumberConverter(generator=lambda: generator(file))
    number_writer("ConvertionResults.txt", ncalc)
