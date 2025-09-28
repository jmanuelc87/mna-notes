import sys
import math
import time

from compute import Compute
from collections import defaultdict
from dataclasses import dataclass


def stat_writer(file, stats):
    """writes descriptive statistics to a file

    Args:
        file (str): the name of the file
        stats (DescriptiveStats): the object that generates statistics
    """
    with open(file, "w") as f:
        f.write(f"mean: {stats.mean}\n")
        f.write(f"median: {stats.median}\n")
        f.write(f"mode: {stats.mode}\n")
        f.write(f"variance: {stats.variance}\n")
        f.write(f"std dev: {stats.stdDev}\n")
        f.write(f"time elapsed: {stats.timeit}\n")

        print(f"mean: {data.mean}")
        print(f"median: {data.median}")
        print(f"mode: {data.mode}")
        print(f"variance: {data.variance}")
        print(f"std dev: {data.stdDev}")
        print(f"time elapsed: {data.timeit}")


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
                yield float(row)
            except Exception as e:
                print(f"{e}")


@dataclass
class DescriptiveStats:
    mean: float
    median: float
    mode: float
    stdDev: float
    variance: float
    timeit: float


class StatsCalculator(Compute):

    def __init__(self, generator):
        self.generator = generator

    def calculate(self):
        start = time.time()
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.variance()
        std = self.stdDev()
        end = time.time() - start
        return DescriptiveStats(mean, median, mode, std, var, end)

    def mean(self):
        """calculates the mean

        Returns:
            float: the mean for the generator distribution
        """
        return sum(self.generator()) / sum(1 for _ in self.generator())

    def median(self):
        """calculates the median

        Returns:
            float: the median for the generator distribution
        """
        num = list(self.generator())
        N = len(num)
        if N % 2 == 0:
            return (num[N // 2] + num[(N // 2) + 1]) / 2
        else:
            return num[(N + 1) // 2]

    def mode(self):
        """calculates the mode

        Returns:
            float: the mode for the generator distribution
        """
        counter = defaultdict(int)
        list_sort = sorted(self.generator())

        for n in list_sort:
            counter[n] += 1

        maxNum, repeatedWith = -1, 0

        for k, v in counter.items():
            if repeatedWith < v:
                repeatedWith = v
                maxNum = k
        return maxNum

    def stdDev(self):
        """calculates the standard deviation

        Returns:
            float: the standard deviation for the generator distribution
        """
        return math.sqrt(self.variance())

    def variance(self):
        """calculates the variance

        Returns:
            float: the variance for the generator distribution
        """
        mean = self.mean()
        numerator = sum([(i - mean) ** 2 for i in self.generator()])
        return numerator / (len(list(self.generator())) - 1)


if __name__ == "__main__":
    file = sys.argv[1]
    stats = StatsCalculator(generator=lambda: generator(file))
    data = stats.calculate()
    stat_writer("StatisticsResults.txt", data)
