from abc import ABC
from datetime import datetime
from typing import (
        Callable, 
        Any,
        Dict,
        )
from dataclasses import dataclass

from .meta import (
        NoneType,
        TypeInfo,
        )


# ------------------------------------------------------------
# IContext
# ------------------------------------------------------------

@dataclass
class IContext(ABC):
    """
    The IContext instances will be passed to Entity evaluation 
    instances/or subcomponents with concrete data.
    Context will be available as ContextNS namespace (Ctx.) and beloinging
    Value expressions.
    Context-s are not obligatory in definition and evaluation. 

    This is plain class, no setup()/Setup() process.
    This class is abstract and one should inherit and override methods.
    """

    def get_user(self) -> NoneType:
        return None

    def get_session(self) -> NoneType:
        return None

    # # TODO: should be combined with config.debug_mode
    # def is_debug(self) -> bool:
    #     return False

    def get_now(self) -> datetime:
        return datetime.now()

    @classmethod
    def get_type_info(cls) -> TypeInfo:
        return TypeInfo.get_or_create_by_type(cls)

    @classmethod
    def get_dexp_attrname_dict(cls) -> Dict[str, Callable[[], Any]]:
        return {
            "User" : cls.get_user,
            "Session" : cls.get_session,
            # "IsDebug" : cls.is_debug_mode,
            "Now" : cls.get_now,
            }

