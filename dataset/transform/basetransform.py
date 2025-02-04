from abc import ABCMeta, abstractmethod


class BaseTransform(metaclass=ABCMeta):
    def __call__(self, results):

        return self.transform(results)

    @abstractmethod
    def transform(self, results):
        """

        """