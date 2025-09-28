import abc


class Compute(abc.ABC):

    @abc.abstractmethod
    def calculate(self):
        raise NotImplementedError
