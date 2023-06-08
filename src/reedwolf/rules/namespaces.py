# TODO: this module probably should be merged into expressions - since there is circular depencdency - see __getattr__
from abc import ABC, abstractmethod

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

    RESERVED_ATTR_NAMES = {"_name", "_manual_setup"}

    def __init__(self, name:str, manual_setup:bool = False):
        self._name = name
        # manual_setup == True ==> Setup() for DotExpression-s needs to be
        # called postponed, manually, usually with extrra context - like ThisNS
        self._manual_setup = manual_setup

    def __str__(self):
        return f"{self._name}"

    def __repr__(self):
        return f"NS[{self._name}]"

    def __getattr__(self, aname):
        if aname in self.RESERVED_ATTR_NAMES: # , "%r -> %s" % (self._node, aname):
            from .exceptions import RuleSetupNameError
            raise RuleSetupNameError(owner=self, msg=f"Namespace attribute {aname} is reserved, choose another name.")

        from .expressions import DotExpression
        return DotExpression(node=aname, namespace=self)


# Instances - should be used as singletons
# the only namespace declaration in this module
FunctionsNS = Namespace("Fn")

# internally used
OperationsNS = Namespace("Op")

# managed models
ModelsNS = Namespace("Models")

# # Data/D - can be list, primitive type, object, Option etc.
# #   evaluated from functions or Expressions
# DataNS = Namespace("D")

# Field/F - all componenents in current container - including chain
#           from current owner to top owner, including all their children (e.g.
#           FieldGroups and their Fields, Validations etc.
#
#           Difference to ModelsNS:
#             - rules struct could be diifferent from storage models (ModelsNS)
#             - Fields are more suitable for embedding - e.g. when other model could be
#               injected into component or component to be reused in other rules-struct

# TODO: rename to ComponentsNS / C.
FieldsNS = Namespace("Fields")

# This - values from a current context, e.g. iteration of loop, option in select
ThisNS = Namespace("This", manual_setup=True)

# Context - see contexts.py
ContextNS = Namespace("Ctx")

# Config - see config.py
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


# # Context - Direct access to managed models underneath and global Rules objects like Validation/FieldGroup etc
# ContextNS = Namespace("Context")
# Ctx  = ContextNS

