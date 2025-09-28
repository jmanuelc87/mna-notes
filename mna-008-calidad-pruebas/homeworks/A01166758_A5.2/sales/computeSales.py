import sys
import json
import collections


def reader(file: str):
    with open(file) as f:
        return json.load(f)


def generator(obj: list):
    for item in obj:
        yield item


class SalesCalculator(object):

    def __init__(self, catalogue, prices):
        self.catalogue = catalogue
        self.prices = prices

    def calculate(self):
        cata = reader(self.catalogue)
        pric = reader(self.prices)

        cataDict = collections.defaultdict(float)
        for item in generator(cata):
            cataDict[item['title']] = item['price']

        total = 0.0
        for item in generator(pric):
            price = cataDict[item['Product']]
            total += price * item['Quantity']

        return total


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Arguments aren't complete please follow the guidelines")
        print("python computeSales.py products.json sales.json")
        exit(1)

    catalogue = sys.argv[1]
    prices = sys.argv[2]
    calculator = SalesCalculator(catalogue, prices)
    print(f"Total Price: ${calculator.calculate():.2f}")
