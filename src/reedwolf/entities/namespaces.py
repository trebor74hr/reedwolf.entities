# TODO: this module probably should be merged into expressions - since there is circular dependency - see __getattr__
from dataclasses import dataclass, field
from typing import (
    Optional, List, Any, Union,
)

from .dynamic_attrs import DynamicAttrsBase
from .exceptions import EntityInternalError, EntitySetupNameError
from .utils import UndefinedType


# ------------------------------------------------------------
# Namespaces - classes and singletons
# ------------------------------------------------------------
# Namespaces are dummy objects/classes to enable different namespaces in
# DotExpression declaration


class Namespace(DynamicAttrsBase):

    RESERVED_ATTR_NAMES = {"_name", "_is_for_internal_use_only", "_alias", "_GetNameWithAlias", "_is_dexp_or_ns", "_is_dexp"}
    
    # manual_setup: bool = False, 
    def __init__(self, name: str, alias: Optional[str] = None, is_for_internal_use_only: bool = False):
        self._name = name
        # namespace can be used for internal use only, should not be used directly
        self._is_for_internal_use_only: bool = is_for_internal_use_only
        # alias, e.g. for Models. -> M.
        self._alias: Optional[str] = alias
        self._is_dexp_or_ns = True
        self._is_dexp = False

    def __str__(self):
        return f"{self._name}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self._name})"

    def _GetNameWithAlias(self):
        return f"{self._name}{ f' / {self._alias}' if self._alias else ''}"

    def __getattr__(self, attr_name) -> "DotExpression":
        if attr_name in self.RESERVED_ATTR_NAMES:  # , "%r -> %s" % (self._node, attr_name):
            from .exceptions import EntitySetupNameError
            raise EntitySetupNameError(owner=self, msg=f"Namespace attribute {attr_name} is reserved, "
                                                       "choose another name.")

        from .expressions import DotExpression
        return DotExpression(node=attr_name, namespace=self)

@dataclass
class NamespaceRule:
    namespace: Namespace = field(repr=True)
    deny_root: bool = field(repr=True, default=False)
    deny_root_reason_templ: str = field(repr=False, default="")

    def __post_init__(self):
        if self.deny_root and not self.deny_root_reason_templ:
            raise EntityInternalError(f"{self}: Provide deny_root_reason.")
        elif not self.deny_root and self.deny_root_reason_templ:
            raise EntityInternalError(f"{self}: Attr deny_root_reason should be set only when deny_root")

@dataclass
class MultiNamespace:
    rules: List[NamespaceRule] = field(repr=True, default_factory=list)

    def validate_namespace(self, owner: Any, namespace: Namespace, attr_name: str, is_1st_node: bool):
        found: bool = False
        for rule in self.rules:
            if namespace is rule.namespace:
                if is_1st_node and rule.deny_root:
                    deny_reason = rule.deny_root_reason_templ.format(attr_name=attr_name, namespace=namespace)
                    raise EntitySetupNameError(owner=owner, msg=deny_reason)
                found = True
        if not found:
            raise EntityInternalError(owner=owner, msg=f"'{self}' did match namespace: {namespace}")


# NOTE: dropped, ContextNS / Ctx. should be used instead.
# Instances - should be used as singletons
# the only namespace declaration in this module
# FunctionsNS = Namespace("Fn")

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

# MySettings - see settings.py
ContextNS = Namespace("Ctx")

# aliases
# Fn = FunctionsNS
M = ModelsNS
# D = DataNS
F = FieldsNS
This = ThisNS
Ctx = ContextNS

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


# # MySettings - Direct access to managed models underneath and global Entity objects like Validation/FieldGroup etc.
# ContextNS = Namespace("MySettings")
# Ctx  = ContextNS
