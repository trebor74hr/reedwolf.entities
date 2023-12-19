# TODO: this module probably should be merged into expressions - since there is circular dependency - see __getattr__
from abc import ABC, abstractmethod
from typing import (
        Optional,
        )

# ------------------------------------------------------------
# Namespaces - classes and singletons
# ------------------------------------------------------------
# Namespaces are dummy objects/classes to enable different namespaces in
# DotExpression declaration


class DynamicAttrsBase(ABC):
    """
    Just to mark objects that are (too) flexible
    all attributes are "existing" returning again objects that are same alike.
    """
    @abstractmethod
    def __getattr__(self, name):
        ...


class Namespace(DynamicAttrsBase):

    RESERVED_ATTR_NAMES = {"_name", "_is_for_internal_use_only", "_alias", "_GetNameWithAlias"}
    
    # manual_setup: bool = False, 
    def __init__(self, name: str, alias: Optional[str] = None, is_for_internal_use_only: bool = False):
        self._name = name
        # namespace can be used for internal use only, should not be used directly
        self._is_for_internal_use_only: bool = is_for_internal_use_only
        # alias, e.g. for Models. -> M.
        self._alias: Optional[str] = alias

    def __str__(self):
        return f"{self._name}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self._name})"

    def _GetNameWithAlias(self):
        return f"{self._name}{ f' / {self._alias}' if self._alias else ''}"

    def __getattr__(self, attr_name):
        if attr_name in self.RESERVED_ATTR_NAMES:  # , "%r -> %s" % (self._node, attr_name):
            from .exceptions import EntitySetupNameError
            raise EntitySetupNameError(owner=self, msg=f"Namespace attribute {attr_name} is reserved, "
                                                       "choose another name.")

        from .expressions import DotExpression
        return DotExpression(node=attr_name, namespace=self)


# Instances - should be used as singletons
# the only namespace declaration in this module
FunctionsNS = Namespace("Fn")

# internally used
OperationsNS = Namespace("Op", is_for_internal_use_only=True)

# managed models
ModelsNS = Namespace("Models", alias="M")

# # Data/D - can be list, primitive type, object, Option etc.
# #   evaluated from functions or Expressions
# DataNS = Namespace("D")

# Field/F - all components in current container - including chain
#           from current parent to top parent, including all their children (e.g.
#           FieldGroups and their Fields, Validations etc.)
#
#           Difference to ModelsNS:
#             - entity struct could be different from storage models (ModelsNS)
#             - Fields are more suitable for embedding - e.g. when other model could be
#               injected into component or component to be reused in other entity-struct

# TODO: rename to ComponentsNS / C.
FieldsNS = Namespace("Fields", alias="F")

# This - values from a current settings, e.g. iteration of loop, option in select
ThisNS = Namespace("This")  # , manual_setup=True

# Context - see contexts.py
ContextNS = Namespace("Ctx")

# Settings - see settings.py
ConfigNS = Namespace("Cfg")

# aliases
Fn = FunctionsNS
M = ModelsNS
# D = DataNS
F = FieldsNS
This = ThisNS
Ctx = ContextNS
Cfg = ConfigNS

# Collect all namespaces
ALL_NS_OBJECTS = {
        name: obj 
        for name, obj in globals().items()
        if isinstance(obj, Namespace)
        }
ALL_NS_OBJECTS.update({
        obj._name: obj 
        for _, obj in globals().items()
        if isinstance(obj, Namespace)
        })


# # Context - Direct access to managed models underneath and global Entity objects like Validation/FieldGroup etc.
# ContextNS = Namespace("Context")
# Ctx  = ContextNS
