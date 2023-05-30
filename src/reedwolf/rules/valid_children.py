"""
validations used for children records
e.g. in container cardinality.

TODO: Check if they could be implemented in cleaners instead of cardinality ...
        Extension(
            ...
            cleeaners = [
                Cardinality.Single()
            ]

TODO: implement as normal validations, now there are leftovers ...

"""
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from dataclasses import dataclass, field

from .exceptions import (
        RuleSetupError,
        RuleValidationCardinalityError,
        RuleSetupTypeError,
        )
from .utils import (
        to_int,
        UNDEFINED,
        UndefinedType,
        )
from .meta import (
        TransMessageType,
        )
from .components import (
        ValidationBase,
        )
from .base import (
        ValidationFailure,
        IApplySession,
        )


# class ValidatorBase(SetOwnerMixin, ABC):
#     # name, owner, owner_name
# 
#     def is_finished(self):
#         return bool(self.name)


class IChildrenValidator(ValidationBase, ABC):
    label           : Optional[TransMessageType] = field(repr=False, default=None)


class ICardinalityValidation(IChildrenValidator, ABC): # count

    # def __post_init__(self):
    #     if self.__class__==ICardinalityValidation:
    #         raise RuleSetupError(owner=self, msg="Use subclasses of ICardinalityValidation")

    @abstractmethod
    def validate_setup(self):
        """
        if not ok,
            raises RuleSetupTypeError
        """
        raise NotImplementedError("abstract method")

    def validate(self, items_count:bool, raise_err:bool=True) -> bool:
        """
        takes nr. of items and validates
        if ok, returns True
        if not ok,
            if raise_err -> raises RuleValidationCardinalityError
            else -> return false
        """
        raise NotImplementedError("abstract method")

    def _validate_setup_common(self, allow_none:Optional[bool]=None) -> 'AttrDexpNode':  # noqa: F821
        model_attr_node = self.owner.get_bound_model_attr_node()
        if allow_none is not None:
            if allow_none and not model_attr_node.isoptional():
                raise RuleSetupTypeError(owner=self, msg="Type hint is not Optional and cardinality allows None. Add Optional or set .allow_none=False/min=1+")
            if not allow_none and model_attr_node.isoptional():
                raise RuleSetupTypeError(owner=self, msg="Type hint is Optional and cardinality does not allow None. Remove Optional or set .allow_none=True/min=0")
        return model_attr_node

# ------------------------------------------------------------

class Cardinality: # namespace holder

    @dataclass
    class Single(ICardinalityValidation):
        name            : str
        allow_none      : bool = True

        owner           : Union['ContainerBase', UndefinedType] = field(init=False, default=UNDEFINED, repr=False)  # noqa: F821
        owner_name      : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

        def validate_setup(self):
            model_attr_node = self._validate_setup_common(self.allow_none)
            if model_attr_node.islist():
                raise RuleSetupTypeError(owner=self, msg="Type hint is List and should be single instance. Change to Range/Multi or remove type hint List[]")

        def validate(self, items_count:int, raise_err:bool=True):
            if items_count==0 and not self.allow_none:
                if raise_err:
                    raise RuleValidationCardinalityError(owner=self, msg="Expected exactly one item, got none.")
                return False
            if items_count!=1:
                if raise_err:
                    raise RuleValidationCardinalityError(owner=self, msg="Expected exactly one item, got {items_count}.")
                return False
            return True

    @dataclass
    class Range(ICardinalityValidation):
        """
            at least one (min or max) arg is required
            min=None -> any number (<= max)
            max=None -> any number (>= min)
            min=0    -> same as allow_none in other validators

        """
        name            : str
        min             : Optional[int] = None
        max             : Optional[int] = None

        owner           : Union['ContainerBase', UndefinedType] = field(init=False, default=UNDEFINED, repr=False)  # noqa: F821
        owner_name      : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

        def __post_init__(self):
            if self.min is None and self.max is None:
                raise RuleSetupError(owner=self, msg="Please provide min and/or max")
            if self.min is not None and (to_int(self.min) is None or to_int(self.min)<0):
                raise RuleSetupError(owner=self, msg="Please provide integer min >=0 ")
            if self.max is not None and (to_int(self.max) is None or to_int(self.max)<0):
                raise RuleSetupError(owner=self, msg="Please provide integer max >=0 ")
            if self.min is not None and self.max is not None and self.max<self.min:
                raise RuleSetupError(owner=self, msg="Please provide min <= max")
            if self.max is not None and self.max==1:
                raise RuleSetupError(owner=self, msg="Please provide max>1 or use Single.")

        def validate_setup(self):
            # [] is allowed, and that is not same as None ( allow_none=(self.min==0)) )
            model_attr_node = self._validate_setup_common(allow_none=False)
            if not model_attr_node.islist():
                raise RuleSetupTypeError(owner=self, msg="Type hint is not List and should be. Change to Single or add List[] type hint ")

        def validate(self, items_count:int, raise_err:bool=True):
            if self.min and items_count < self.min:
                if raise_err:
                    raise RuleValidationCardinalityError(owner=self, msg=f"Expected at least {self.min} item(s), got {items_count}.")
                return False
            if self.max and items_count < self.max:
                if raise_err:
                    raise RuleValidationCardinalityError(owner=self, msg=f"Expected at most {self.max} items, got {items_count}.")
                return False
            return True

    @dataclass
    class Multi(ICardinalityValidation):
        " [0,1]:N "
        name            : str
        allow_none      : bool = True

        owner           : Union['ContainerBase', UndefinedType] = field(init=False, default=UNDEFINED, repr=False)  # noqa: F821

        owner_name      : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

        def validate_setup(self):
            model_attr_node = self._validate_setup_common(self.allow_none)
            if not model_attr_node.islist():
                raise RuleSetupTypeError(owner=self, msg="Type hint is not a List and should be. Change to Single or add List[] type hint")

        def validate(self, items_count:int, raise_err:bool=True):
            if items_count==0 and not self.allow_none:
                if raise_err:
                    raise RuleValidationCardinalityError(owner=self, msg="Expected at least one item, got none.")
                return False
            return True


# ------------------------------------------------------------
# other validators
# ------------------------------------------------------------
class IUniqueValidator(IChildrenValidator, ABC):
    ...

    # def __post_init__(self):
    #     if self.__class__==IUniqueValidator:
    #         raise RuleSetupError(owner=self, msg=" Use subclasses of IUniqueValidator")

    # def set_owner(self, owner):
    #     super().set_owner(owner)
    #     if not self.name:
    #         self.name = f"{self.owner.name}__{self.__class__.__name__.lower()}"

class Unique: # namespace holder

    @dataclass
    class Global(IUniqueValidator):
        " globally - e.g. within table "
        name            : str
        fields          : List[str] # TODO: better field specification or dexpr?
        ignore_none     : bool = True

        owner           : Union['ContainerBase', UndefinedType] = field(init=False, default=UNDEFINED, repr=False)  # noqa: F821
        owner_name      : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

        def validate(self, apply_session: IApplySession) -> Optional[ValidationFailure]:
            raise NotImplementedError()

    @dataclass
    class Children(IUniqueValidator):
        " within extension records "
        name            : str
        fields          : List[str] # TODO: better field specification or dexpr?
        ignore_none     : bool = True

        owner           : Union['ContainerBase', UndefinedType] = field(init=False, default=UNDEFINED, repr=False)  # noqa: F821
        owner_name      : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

        def validate(self, apply_session: IApplySession) -> Optional[ValidationFailure]:
            raise NotImplementedError()


# ALT: names for IChildrenValidator
#   class IterationValidator:
#   # db terminology: scalar custom functions, table value custom functions, aggregate custom functions
#   class AggregateValidator:
#   class ExtensionValidator:
#   class ItemsValidator:
#   class MultipleItemsValidator:
#   class ContainerItemsValidator:
