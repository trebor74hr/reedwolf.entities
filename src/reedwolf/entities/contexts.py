from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from .exceptions import EntitySetupError
from .meta import (
    NoneType,
    ExpressionsAttributesMap, AttrName,
)


# ------------------------------------------------------------
# IContext
# ------------------------------------------------------------

@dataclass
class IContext(ABC):

    @classmethod
    @abstractmethod
    def get_contextns_attributes(cls) -> ExpressionsAttributesMap:
        """
        Should return attribute name -> FieldName (dataclass/...) name OR callable.
        Example:
          return {
            "SessionId": FieldName("session_id"),
            "Session": cls.get_session,
            "User": MethodName("get_user"),
            "Now": cls.get_now,
            }
        If method then it must have no arguments without default.
        """
        raise EntitySetupError(owner=cls, msg=f"Function 'get_dexp_attrs_dict' needs to be implemented in {cls}")

# class ConfigOverrideMixin:
#
#     def is_debug(self) -> Optional[bool]:
#         return None
#
#     def is_trace(self) -> Optional[bool]:
#         return None


# @dataclass
# class ContextDemo(IContext):
#     """
#     The IContext instances will be passed to Entity evaluation
#     instances/or subcomponents with concrete data.
#     Context will be available as ContextNS namespace (Ctx.) and belonging
#     Value expressions.
#     Context-s are not obligatory in definition and evaluation.
#
#     This is plain class, no setup()/Setup() process.
#     This class is abstract and one should inherit and override methods.
#     """
#     username: str
#     session_id: int
#
#     # noinspection PyMethodMayBeStatic
#     def get_session(self) -> NoneType:
#         return None
#
#     # # TODO: should be combined with settings.debug_mode
#     # def is_debug(self) -> bool:
#     #     return False
#
#     # noinspection PyMethodMayBeStatic
#     def get_now(self) -> datetime:
#         return datetime.now()
#
#     @classmethod
#     def get_contextns_attributes(cls) -> ExpressionsAttributesMap:
#         return {
#             "User": AttrName("username"),
#             "Session": cls.get_session,
#             # "IsDebug" : cls.is_debug_mode,
#             "Now": cls.get_now,
#             }
