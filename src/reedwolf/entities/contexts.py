from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass

from .exceptions import EntitySetupError
from .meta import (
    NoneType,
    DEXP_ATTR_TO_CALLABLE_DICT,
)


# ------------------------------------------------------------
# IContext
# ------------------------------------------------------------

@dataclass
class IContext(ABC):

    @classmethod
    @abstractmethod
    def get_dexp_attr_to_callable_dict(cls) -> DEXP_ATTR_TO_CALLABLE_DICT:
        """
        Should return attribute name -> callable or direct value.
        example:
        return {
            "User": cls.user,
            "Session": cls.session,
            "Now": cls.get_now,
            }

        """
        raise EntitySetupError(owner=cls, msg=f"Function 'get_dexp_attrs_dict' needs to be implemented in {cls}")


@dataclass
class ContextDemo(IContext):
    """
    The IContext instances will be passed to Entity evaluation 
    instances/or subcomponents with concrete data.
    Context will be available as ContextNS namespace (Ctx.) and belonging
    Value expressions.
    Context-s are not obligatory in definition and evaluation. 

    This is plain class, no setup()/Setup() process.
    This class is abstract and one should inherit and override methods.
    """

    # noinspection PyMethodMayBeStatic
    def get_user(self) -> NoneType:
        return None

    # noinspection PyMethodMayBeStatic
    def get_session(self) -> NoneType:
        return None

    # # TODO: should be combined with config.debug_mode
    # def is_debug(self) -> bool:
    #     return False

    # noinspection PyMethodMayBeStatic
    def get_now(self) -> datetime:
        return datetime.now()

    @classmethod
    def get_dexp_attr_to_callable_dict(cls) -> DEXP_ATTR_TO_CALLABLE_DICT:
        return {
            "User": cls.get_user,
            "Session": cls.get_session,
            # "IsDebug" : cls.is_debug_mode,
            "Now": cls.get_now,
            }
