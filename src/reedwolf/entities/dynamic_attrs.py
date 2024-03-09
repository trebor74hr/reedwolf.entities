from abc import ABC, abstractmethod


class DynamicAttrsBase(ABC):
    """
    Just to mark objects that are (too) flexible
    all attributes are "existing" returning again objects that are same alike.
    """
    @abstractmethod
    def __getattr__(self, name):
        ...
